from typing import Any, Union

import gym
import numpy as np
import wandb
from omegaconf import DictConfig
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import sync_envs_normalization
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from wandb.sdk.lib.disabled import RunDisabled
from wandb.sdk.wandb_run import Run

from autorl_landscape.run.eval_stage import EvalStage, FinalEval, FreqEval, LSEval


class LandscapeEvalCallback(BaseCallback):
    """A callback for running evaluations at specific points during training, used for the phase algorithm.

    Args:
        conf: Hydra configuration, including information about how the different evaluations should happen, as well
        as the seed for the eval_env.
        eval_env: The env used for all evaluations. Is always reseeded.
        t_ls: Number of total steps until the landscape evaluation.
        t_final: Number of total steps until the finish.
        ls_model_save_path: Location where the agent should be saved. E.g. {agent}/{env}/{date}/{phase}/agents/{id}
        run: Wandb run, used for logging.
        agent_seed: Seed for the agent. Used when saving the agent for trackably deterministic behaviour.
    """

    def __init__(
        self,
        conf: DictConfig,
        phase_index: int,
        eval_env: Union[gym.Env, VecEnv],
        ls_model_save_path: str,
        run: Union[Run, RunDisabled],
        agent_seed: int,
        ls_conf: dict[str, Any],
    ):
        super().__init__(verbose=1)
        self.eval_env = eval_env
        self.ls_conf = ls_conf

        self.freq_eval_episodes = conf.eval.freq_eval_episodes
        self.ls_eval_episodes = conf.eval.ls_eval_episodes
        self.final_eval_episodes = conf.eval.final_eval_episodes

        # final eval save data:
        self.all_final_returns = np.zeros((conf.eval.final_eval_times, self.final_eval_episodes))
        self.all_final_ep_lengths = np.zeros((conf.eval.final_eval_times, self.final_eval_episodes))

        # Build a schedule for evaluations. The list is sorted by the time step (int) and points to an evaluation
        # (LSEval(), FreqEval(), FinalEval(i)) that should be done at or after that time step (evaluations are only
        # ever done when a rollout is completed).
        t_start = 0 if phase_index < 2 else conf.phases[phase_index - 2]  # don't queue freq evals before phase start
        t_freqs = [t for t in range(0, conf.phases[-1] + 1, conf.eval.freq_eval_interval) if t >= t_start]
        t_finals = (
            np.linspace(
                float(conf.eval.final_eval_start) * int(conf.phases[-1]),
                int(conf.phases[-1]),
                int(conf.eval.final_eval_times),
            )
            .round()
            .astype(int)
        )
        self.eval_schedule: list[tuple[int, EvalStage]] = []  # [(t, eval_type, i (only for final eval!))]
        self.eval_schedule.append((conf.phases[phase_index - 1], LSEval()))
        self.eval_schedule.extend([(t_freq, FreqEval()) for t_freq in t_freqs])
        self.eval_schedule.extend([(t, FinalEval(i)) for i, t in enumerate(t_finals, start=1)])
        self.eval_schedule = sorted(self.eval_schedule, key=lambda tup: tup[0], reverse=True)
        # print(self.eval_schedule)

        self.ls_model_save_path = ls_model_save_path
        self.run = run
        self.agent_seed = agent_seed
        self.eval_seed = conf.seeds.eval

        # histogram bounds:
        self.max_return = conf.viz.max_return
        self.max_ep_length = conf.viz.max_ep_length
        self.hist_bins = conf.viz.hist_bins + 1  # internally used for np.linspace, so one more is needed

    def after_update(self) -> None:
        """Triggered in the learning loop after a rollout, and after updates have been made to the policy.

        Check if any evaluations are scheduled for time steps that have been passed.
        """
        ls_eval, freq_eval, final_eval_i = False, False, 0
        while len(self.eval_schedule) > 0:
            t, _ = self.eval_schedule[-1]
            if t > self.num_timesteps:
                break

            t, e = self.eval_schedule.pop()
            match e:
                case FreqEval():
                    freq_eval = True
                case LSEval():
                    ls_eval = True
                case FinalEval(i):
                    final_eval_i = i
        self.evaluate_policy(ls_eval, freq_eval, final_eval_i)

    def _on_step(self) -> bool:
        """Stop training when all evaluations have been done."""
        with np.printoptions(precision=4, linewidth=500, suppress=True):
            print(
                "{} {} {} {} {} {}".format(
                    self.num_timesteps,
                    self.locals["new_obs"],
                    self.locals["rewards"],
                    self.locals["dones"],
                    self.locals["replay_buffer"].observations.sum(),
                    self.locals["replay_buffer"].pos,
                    # sum([l.sum() for l in self.model.q_net.parameters()]),
                )
            )
        return len(self.eval_schedule) > 0

    def evaluate_policy(self, ls_eval: bool, freq_eval: bool, final_eval_i: int) -> None:
        """Evaluate the policy (that is trained on some configuration with a seed).

        Different kinds of evaluations (ls, freq, final) can happen at the same time and do different things.

        Args:
            ls_eval: Write eval output to ls_eval/mean_{return,ep_length} and checkpoint the model.
            freq_eval: Write eval output to eval/mean_{return,ep_length}.
            final_eval_i: If > 0, write eval output to final_eval_i/mean_{return,ep_length}.
        """
        final_eval = final_eval_i > 0
        if not (freq_eval or ls_eval or final_eval):
            return

        assert self.logger is not None
        # assert isinstance(self.model, CustomDQN) or isinstance(self.model, CustomSAC)

        # Only do landscape eval once after the time step has been reached:
        if ls_eval:
            self.done_ls_eval = True
            print(f"{ls_eval=} at {self.num_timesteps=}")

        # Sync training and eval env if there is VecNormalize
        if self.model.get_vec_normalize_env() is not None:
            try:
                sync_envs_normalization(self.training_env, self.eval_env)
            except AttributeError as e:
                raise AssertionError(
                    "Training and eval env are not wrapped the same way, "
                    "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                    "and warning above."
                ) from e

        # evals can overlap
        eval_episodes = max(
            self.freq_eval_episodes * freq_eval,
            self.ls_eval_episodes * ls_eval,
            self.final_eval_episodes * final_eval,
        )

        # returns and ep_lengths of n_eval_episodes evaluation rollouts/episodes
        self.eval_env.seed(self.eval_seed)
        returns, ep_lengths = evaluate_policy(
            model=self.model,
            env=self.eval_env,
            n_eval_episodes=eval_episodes,
            return_episode_rewards=True,
        )
        returns = np.array(returns)
        ep_lengths = np.array(ep_lengths)

        # each eval gets its own view of the full data
        freq_returns = returns[: self.freq_eval_episodes]
        freq_ep_lengths = ep_lengths[: self.freq_eval_episodes]
        ls_returns = returns[: self.ls_eval_episodes]
        ls_ep_lengths = ep_lengths[: self.ls_eval_episodes]
        final_returns = returns[: self.final_eval_episodes]
        final_ep_lengths = ep_lengths[: self.final_eval_episodes]

        if final_eval:
            self.all_final_returns[final_eval_i - 1] = final_returns
            self.all_final_ep_lengths[final_eval_i - 1] = final_ep_lengths

        # Add to current Logger
        # self.logger.record: logs time-dependent values to line plots
        # self.run.summary: logs just once for a run, save raw data
        log_dict: dict[str, Any] = {}
        for f, s, s_returns, s_ep_lengths in [
            (ls_eval, "ls_eval", ls_returns, ls_ep_lengths),
            (final_eval, f"final_eval_{final_eval_i}", final_returns, final_ep_lengths),
        ]:
            if f:
                # NOTE summary data is the actual samples for the data set:
                self.run.summary[f"{s}/returns"] = s_returns
                self.run.summary[f"{s}/ep_lengths"] = s_ep_lengths
                return_hist = np.histogram(s_returns, bins=np.linspace(0, self.max_return, self.hist_bins))
                ep_length_hist = np.histogram(s_ep_lengths, bins=np.linspace(0, self.max_ep_length, self.hist_bins))
                log_dict[f"{s}/return_hist"] = wandb.Histogram(np_histogram=return_hist)
                log_dict[f"{s}/ep_length_hist"] = wandb.Histogram(np_histogram=ep_length_hist)
                log_dict[f"{s}/mean_return"] = np.mean(s_returns)
                log_dict[f"{s}/mean_ep_length"] = np.mean(s_ep_lengths)
        if freq_eval:
            log_dict["freq_eval/mean_return"] = np.mean(freq_returns)
            log_dict["freq_eval/mean_ep_length"] = np.mean(freq_ep_lengths)
            # TODO may not work for DQN because of exploration rate stuff:
            for hp_name in self.ls_conf.keys():
                log_dict[f"freq_eval/{hp_name}"] = getattr(self.model, hp_name)
            # log_dict["freq_eval/exploration_final_eps"] = self.model.exploration_final_eps
            # log_dict["freq_eval/exploration_initial_eps"] = self.model.exploration_initial_eps
            # log_dict["freq_eval/exploration_rate"] = self.model.exploration_rate
            # log_dict["freq_eval/exploration_fraction"] = self.model.exploration_fraction

        log_dict["time/total_timesteps"] = self.num_timesteps
        self.run.log(log_dict)

        # Save ("checkpoint") the model at the end of the landscape stage:
        if ls_eval:
            if self.verbose > 0:
                print(f"Saving model checkpoint to {self.ls_model_save_path}")
            self.model.custom_save(self.ls_model_save_path, seed=self.agent_seed)
