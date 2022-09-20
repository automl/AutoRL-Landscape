from typing import List, Union

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
        self.final_mean_return = -1.0  # TODO this is a hardcoded metric to be used for choosing best conf and seed

        # histogram bounds:
        self.max_return = conf.viz.max_return
        self.max_ep_length = conf.viz.max_ep_length
        self.hist_bins = conf.viz.hist_bins

        # final eval save data:
        self.final_returns: List[np.ndarray] = []
        self.final_ep_lengths: List[np.ndarray] = []

        # special eval config:
        # TODO actually use these
        self.ls_eval_episodes = conf.eval.ls_eval_episodes
        self.final_eval_episodes = conf.eval.final_eval_episodes
        self.t_final_evals = np.linspace(
            int(conf.env.total_timesteps * conf.eval.final_eval_start),
            conf.env.total_timesteps,
            conf.eval.final_eval_times,
            dtype=int,
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

            # returns and ep_lengths of n_eval_episodes evaluation rollouts/episodes
            self.eval_env.seed(self.eval_seed)
            returns, ep_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )
            returns = np.array(returns)
            ep_lengths = np.array(ep_lengths)
            if final_eval:
                self.final_returns.append(returns)
                self.final_ep_lengths.append(ep_lengths)

            mean_return, std_return = float(np.mean(returns)), float(np.std(returns))
            mean_ep_length, std_ep_length = np.mean(ep_lengths), np.std(ep_lengths)
            if freq_eval:
                self.last_mean_return = mean_return

            if self.verbose > 0:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, "
                    f"episode_return={mean_return:.2f} +/- {std_return:.2f}"
                )
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")

            # Add to current Logger
            # self.logger.record: logs time-dependent values to line plots
            # self.run.summary: logs just once for a run
            # self.run.log: logs histograms of return data for better visualization
            if freq_eval:
                self.logger.record("eval/mean_return", float(mean_return))
                self.logger.record("eval/mean_ep_length", mean_ep_length)
            if ls_eval:
                self.run.summary["ls_eval/returns"] = returns
                self.run.summary["ls_eval/ep_lengths"] = ep_lengths
                return_hist = np.histogram(returns, bins=np.linspace(0, self.max_return, self.hist_bins + 1))
                ep_length_hist = np.histogram(ep_lengths, bins=np.linspace(0, self.max_ep_length, self.hist_bins))
                self.run.log({"ls_eval/return_hist": wandb.Histogram(np_histogram=return_hist)})
                self.run.log({"ls_eval/ep_length_hist": wandb.Histogram(np_histogram=ep_length_hist)})
            if final_final:
                # combine all final evals and log together
                final_returns = np.stack(self.final_returns)
                final_ep_lengths = np.stack(self.final_ep_lengths)
                self.run.summary["final_eval/returns"] = final_returns
                self.run.summary["final_eval/ep_lengths"] = final_ep_lengths
                return_hist = np.histogram(final_returns, bins=np.linspace(0, self.max_return, self.hist_bins + 1))
                ep_length_hist = np.histogram(final_ep_lengths, bins=np.linspace(0, self.max_ep_length, self.hist_bins))
                self.run.log({"final_eval/return_hist": wandb.Histogram(np_histogram=return_hist)})
                self.run.log({"final_eval/ep_length_hist": wandb.Histogram(np_histogram=ep_length_hist)})

            # Dump log so the evaluation results are printed with the correct timestep
            if freq_eval or final_eval:
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(self.num_timesteps)

            if ls_eval:
                if self.verbose > 0:
                    print(f"Saving model checkpoint to {self.ls_model_save_path}")
                self.model.save(self.ls_model_save_path)

            if final_eval:
                self.final_mean_return = mean_return

        return
