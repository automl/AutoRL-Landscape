from typing import Union

import gym
import numpy as np
import wandb
from omegaconf import DictConfig
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import sync_envs_normalization
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from wandb.sdk.lib import RunDisabled
from wandb.sdk.wandb_run import Run


class LandscapeEvalCallback(EvalCallback):
    """
    Like EvalCallback, but also saves evaluation and model at a special (landscape eval) timestep to a custom logging
    output.
    """

    def __init__(
        self,
        conf: DictConfig,
        eval_env: Union[gym.Env, VecEnv],
        t_ls: int,
        t_final: int,
        ls_model_save_path: str,
        conf_idx: int,
        run: Union[Run, RunDisabled],
        verbose: int = 1,
    ):
        super().__init__(
            eval_env,
            None,
            None,
            conf.eval.freq_eval_episodes,
            conf.eval.freq_eval_interval,
            None,
            None,
            deterministic=True,
            render=False,
            verbose=verbose,
            warn=True,
        )
        self.t_ls = t_ls
        self.done_ls_eval = False  # set to true after ls_eval
        self.ls_model_save_path = ls_model_save_path
        self.conf_idx = conf_idx
        self.run = run

        # histogram bounds:
        self.max_return = conf.viz.max_return
        self.max_ep_length = conf.viz.max_ep_length
        self.hist_bins = conf.viz.hist_bins + 1  # internally used for np.linspace, so one more is needed

        # final eval save data:
        self.all_final_returns = np.array([])
        self.all_final_ep_lengths = np.array([])

        # special eval config:
        self.ls_eval_episodes = conf.eval.ls_eval_episodes
        self.final_eval_episodes = conf.eval.final_eval_episodes
        self.t_final_evals = (
            np.linspace(
                int(conf.env.total_timesteps * conf.eval.final_eval_start),
                conf.env.total_timesteps,
                conf.eval.final_eval_times,
                dtype=int,
            )
            + t_final
            - conf.env.total_timesteps
        )
        self.eval_seed = conf.eval.seed

    def _on_training_start(self) -> None:
        self._evaluate(False, True, False, False)

    def _on_step(self) -> bool:
        ls_eval = not self.done_ls_eval and self.n_calls >= self.t_ls
        freq_eval = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0  # or (self.n_calls == 1)
        final_eval = self.n_calls in self.t_final_evals
        final_final = self.t_final_evals[-1] == self.n_calls

        # log_str = f"T={self.num_timesteps}={self.n_calls}"
        # for flag in [freq_eval, ls_eval, final_eval]:
        #     if flag:
        #         log_str += f" {flag=}".split("=")[0]
        # print(log_str)

        self._evaluate(ls_eval, freq_eval, final_eval, final_final)
        return True

    def _evaluate(self, ls_eval: bool, freq_eval: bool, final_eval: bool, final_final: bool) -> None:
        """
        Evaluate the policy (that is trained on some configuration with a seed).

        :param ls_eval: Write eval output to ls_eval/mean_{return,ep_length}.
        :param freq_eval: Write eval output to eval/mean_{return,ep_length}, also optionally keep track of evaluations
        by writing to the log_path if it is set. Also keep track of last_mean_return, success_buffer (?) and best model.
        :param final_eval: Write eval output to final_eval/mean_{return,ep_length}.
        :param final_final: last of the final evals, data can now be written.
        """
        assert self.logger is not None

        if freq_eval or ls_eval or final_eval:
            # Only do landscape eval once
            if ls_eval:
                self.done_ls_eval = True

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
                self.n_eval_episodes * freq_eval,
                self.ls_eval_episodes * ls_eval,
                self.final_eval_episodes * final_eval,
            )

            # returns and ep_lengths of n_eval_episodes evaluation rollouts/episodes
            self.eval_env.seed(self.eval_seed)
            returns, ep_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )
            returns = np.array(returns)
            ep_lengths = np.array(ep_lengths)

            # each eval gets its own view of the full data
            freq_returns = returns[: self.n_eval_episodes]
            freq_ep_lengths = ep_lengths[: self.n_eval_episodes]
            ls_returns = returns[: self.ls_eval_episodes]
            ls_ep_lengths = ep_lengths[: self.ls_eval_episodes]
            final_returns = returns[: self.final_eval_episodes]
            final_ep_lengths = ep_lengths[: self.final_eval_episodes]

            if final_eval:
                self.all_final_returns = np.append(self.all_final_returns, final_returns)
                self.all_final_ep_lengths = np.append(self.all_final_ep_lengths, final_ep_lengths)

            # Add to current Logger
            # self.logger.record: logs time-dependent values to line plots
            # self.run.summary: logs just once for a run, save raw data
            # self.run.log: logs histograms of return data for better visualization
            if ls_eval:
                self.run.summary["ls_eval/returns"] = ls_returns
                self.run.summary["ls_eval/ep_lengths"] = ls_ep_lengths
                return_hist = np.histogram(ls_returns, bins=np.linspace(0, self.max_return, self.hist_bins))
                ep_length_hist = np.histogram(ls_ep_lengths, bins=np.linspace(0, self.max_ep_length, self.hist_bins))
                self.run.log({"ls_eval/return_hist": wandb.Histogram(np_histogram=return_hist)})
                self.run.log({"ls_eval/ep_length_hist": wandb.Histogram(np_histogram=ep_length_hist)})
            if final_final:
                # combine all final evals and log together
                shape = (len(self.t_final_evals), -1)
                self.all_final_returns = np.reshape(self.all_final_returns, shape)
                self.all_final_ep_lengths = np.reshape(self.all_final_ep_lengths, shape)
                self.run.summary["final_eval/returns"] = self.all_final_returns
                self.run.summary["final_eval/ep_lengths"] = self.all_final_ep_lengths
                return_hist = np.histogram(self.all_final_returns, bins=np.linspace(0, self.max_return, self.hist_bins))
                ep_length_hist = np.histogram(
                    self.all_final_ep_lengths, bins=np.linspace(0, self.max_ep_length, self.hist_bins)
                )
                self.run.log({"final_eval/return_hist": wandb.Histogram(np_histogram=return_hist)})
                self.run.log({"final_eval/ep_length_hist": wandb.Histogram(np_histogram=ep_length_hist)})
            for f, s, s_returns, s_ep_lengths in [
                (freq_eval, "freq", freq_returns, freq_ep_lengths),
                (ls_eval, "ls", ls_returns, ls_ep_lengths),
                (final_eval, "final", final_returns, final_ep_lengths),
            ]:
                # always log mean value for easy visualization
                if f:
                    self.logger.record(f"{s}_eval/mean_return", np.mean(s_returns))
                    self.logger.record(f"{s}_eval/mean_ep_length", np.mean(s_ep_lengths))

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if ls_eval:
                if self.verbose > 0:
                    print(f"Saving model checkpoint to {self.ls_model_save_path}")
                self.model.save(self.ls_model_save_path)

        return
