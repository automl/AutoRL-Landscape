from typing import Any

import logging

import numpy as np
import wandb
from numpy.typing import NDArray
from omegaconf import DictConfig
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import sync_envs_normalization
from wandb.sdk.lib.disabled import RunDisabled
from wandb.sdk.wandb_run import Run

from autorl_landscape.run.eval_stage import EvalStage, FinalEval, FreqEval, LSEval
from autorl_landscape.run.rl_context import make_env


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
        ls_model_save_path: str,
        run: Run | RunDisabled,
        agent_seed: int,
        ls_spec: dict[str, Any],
    ):
        super().__init__(verbose=1)
        self.ls_spec = ls_spec
        self.crashed_at: int | None = None

        # self.freq_eval_episodes = conf.eval.freq_eval_episodes
        # self.ls_eval_episodes = conf.eval.ls_eval_episodes
        # self.final_eval_episodes = conf.eval.final_eval_episodes

        # final eval save data:
        self.all_final_returns = np.zeros((conf.eval.final_eval_times, conf.eval.final_eval_episodes))
        self.all_final_ep_lengths = np.zeros((conf.eval.final_eval_times, conf.eval.final_eval_episodes))

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
        self.eval_schedule.append((conf.phases[phase_index - 1], LSEval(conf.eval.ls_eval_episodes)))
        self.eval_schedule.extend([(t_freq, FreqEval(conf.eval.freq_eval_episodes)) for t_freq in t_freqs])
        self.eval_schedule.extend(
            [(t, FinalEval(conf.eval.final_eval_episodes, i)) for i, t in enumerate(t_finals, start=1)]
        )
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

    def _on_training_start(self) -> None:
        logging.warning(f"{self.run.id=} {self.run.name=}")
        # print("num_timesteps new_obs rewards dones sum(replay_buffer) sum(q_net) sum(q_net_target) exploration_rate")

    def _on_step(self) -> bool:
        """Stop training when all evaluations have been done."""
        # with np.printoptions(linewidth=1000, suppress=True):
        #     logging.warning(
        #         "{} {} {} {} {} {} {}".format(
        #             self.num_timesteps,
        #             self.locals["new_obs"],
        #             self.locals["rewards"],
        #             self.locals["dones"],
        #             self.locals["clipped_actions"],
        #             self.locals["rollout_buffer"].observations.sum(),
        #             sum([layer.sum() for layer in self.model.policy.parameters()]),
        #             # sum([layer.sum() for layer in self.model.q_net_target.parameters()]),
        #             # self.model.exploration_rate,
        #         )
        #     )
        if len(self.eval_schedule) > 0:
            return True

        # Stop training and potentially log crash time:
        self.run.summary["meta.crashed_at"] = self.crashed_at
        return False

    def after_update(self) -> None:
        """Triggered in the learning loop after a rollout, and after updates have been made to the policy.

        Check if any evaluations are scheduled for time steps that have been passed.
        """
        eval_stages: set[EvalStage] = set()
        while len(self.eval_schedule) > 0:
            t, _ = self.eval_schedule[-1]
            if t > self.num_timesteps:
                break

            _, e = self.eval_schedule.pop()
            eval_stages.add(e)
        self.evaluate_policy(eval_stages)

    def on_rollout_error(self, _: Exception) -> None:
        """See whether LS or Final Evals still have to be done. Write nan's for those evaluations."""
        todo_stages = [
            (eval_stage, np.full((eval_stage.num_episodes), np.nan), np.full((eval_stage.num_episodes), np.nan))
            for _, eval_stage in self.eval_schedule
            if not isinstance(eval_stage, FreqEval)
        ]
        self.write_evaluation(todo_stages)
        self.eval_schedule = []
        self.crashed_at = self.num_timesteps
        self._on_step()

    def evaluate_policy(self, eval_stages: set[EvalStage]) -> None:
        """Evaluate the policy (that is trained on some configuration with a seed).

        Different kinds of evaluations (FreqEval, LSEval, FinalEval(i)) can happen at the same time and do different
        things, but all log run info to wandb. At LSEval, the model is additionally "checkpointed" (saved to disk). See
        also `LandscapeEvalCallback.write_evaluation()`.

        Args:
            eval_stages: Denotes which types of evaluations should be done at this point in training time.
        """
        if len(eval_stages) == 0:
            return

        # Evaluate the agent:
        eval_env = make_env(self.model.env.envs[0].spec.id, self.eval_seed)
        eval_env.seed(self.eval_seed)

        # Sync training and eval env if there is VecNormalize
        if self.model.get_vec_normalize_env() is not None:
            try:
                sync_envs_normalization(self.training_env, eval_env)
            except AttributeError as e:
                raise AssertionError(
                    "Training and eval env are not wrapped the same way, "
                    "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                    "and warning above."
                ) from e

        # Evals can overlap, so calculate how many evaluation episodes are needed:
        eval_episodes = max(eval_stage.num_episodes for eval_stage in eval_stages)

        returns, ep_lengths = evaluate_policy(
            model=self.model,
            env=eval_env,
            n_eval_episodes=eval_episodes,
            return_episode_rewards=True,
        )
        returns = np.array(returns)
        ep_lengths = np.array(ep_lengths)

        eval_stages_data: list[tuple[EvalStage, NDArray[Any], NDArray[Any]]] = []
        for eval_stage in eval_stages:
            eval_stages_data.append(
                (eval_stage, returns[0 : eval_stage.num_episodes], ep_lengths[0 : eval_stage.num_episodes])
            )
            if isinstance(eval_stage, LSEval):
                # Save ("checkpoint") the model at the end of the landscape stage:
                if self.verbose > 0:
                    logging.warning(f"Saving model checkpoint to {self.ls_model_save_path}")
                self.model.custom_save(self.ls_model_save_path, seed=self.agent_seed)
        self.write_evaluation(eval_stages_data)

    def write_evaluation(self, eval_stages: list[tuple[EvalStage, NDArray[Any], NDArray[Any]]]) -> None:
        """Save evaluation performance data from a given stage to wandb. Only run this once per `self.num_timesteps`.

        Depending on what evaluation stages are handled, do different things:
        FreqEval: Also log landscape hyperparameter values (useful for debugging)
        LSEval: Save the data that will make up the landscape dataset, and note the exact time when this stage happens.
        FinalEval: Save the data that will make up (part of) the final dataset, and note the exact time when this stage
        happens. Also save this data to `self.all_final_{returns,ep_lengths}` for choosing the best policy at the end of
        the phase.

        Args:
            eval_stages: List of (eval_stage, stage_returns, stage_ep_lengths), i.e. stage meta info and actual
            performance sample data.
        """
        # Add to current logger:
        # self.run.log: logs time-dependent values to line plots
        # self.run.summary: logs just once for a run, save raw data
        log_dict: dict[str, Any] = {}
        for eval_stage, stage_returns, stage_ep_lengths in eval_stages:
            # Always log mean return and episode length:
            log_dict[f"{eval_stage.log_name()}/mean_return"] = np.mean(stage_returns)
            log_dict[f"{eval_stage.log_name()}/mean_ep_length"] = np.mean(stage_ep_lengths)

            if isinstance(eval_stage, FreqEval):
                for hp_name, hp_val in self.model.get_ls_conf(self.ls_spec).items():
                    log_dict[f"freq_eval/{hp_name}"] = hp_val

            if isinstance(eval_stage, LSEval | FinalEval):
                # NOTE summary data is the actual samples for the data set:
                self.run.summary[f"{eval_stage.log_name()}/returns"] = stage_returns
                self.run.summary[f"{eval_stage.log_name()}/ep_lengths"] = stage_ep_lengths
                return_hist = np.histogram(stage_returns, bins=np.linspace(0, self.max_return, self.hist_bins))
                ep_length_hist = np.histogram(stage_ep_lengths, bins=np.linspace(0, self.max_ep_length, self.hist_bins))
                log_dict[f"{eval_stage.log_name()}/return_hist"] = wandb.Histogram(np_histogram=return_hist)
                log_dict[f"{eval_stage.log_name()}/ep_length_hist"] = wandb.Histogram(np_histogram=ep_length_hist)

                self.run.summary[f"{eval_stage.log_name()}/happened_at"] = self.num_timesteps

            if isinstance(eval_stage, FinalEval):
                self.all_final_returns[eval_stage.i - 1] = stage_returns
                self.all_final_ep_lengths[eval_stage.i - 1] = stage_ep_lengths

        log_dict["time/total_timesteps"] = self.num_timesteps
        self.run.log(log_dict)
