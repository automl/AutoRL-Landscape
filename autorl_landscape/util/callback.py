from typing import Any, List, Optional, Union

import os

import gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import sync_envs_normalization
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from autorl_landscape.util.comparator import Comparator


class LandscapeEvalCallback(EvalCallback):
    """
    Like EvalCallback, but also saves evaluation and model at a special (landscape eval) timestep to a custom logging
    output.
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        t_ls: int,
        ls_model_save_path: str,
        comp: Comparator,
        conf_idx: int,
        run_id: str,
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super().__init__(
            eval_env,
            callback_on_new_best,
            callback_after_eval,
            n_eval_episodes,
            eval_freq,
            log_path,
            best_model_save_path,
            deterministic,
            render,
            verbose,
            warn,
        )
        self.t_ls = t_ls
        self.done_ls_eval = False  # set to true after ls_eval
        self.best_mean_reward: float
        self.ls_model_save_path = ls_model_save_path
        self.comp = comp
        self.conf_idx = conf_idx
        self.run_id = run_id

    def _on_step(self) -> bool:
        ls_eval = not self.done_ls_eval and self.n_calls >= self.t_ls
        freq_eval = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0

        return self._evaluate(ls_eval=ls_eval, freq_eval=freq_eval, final_eval=False)

    def _on_training_end(self) -> None:
        self._evaluate(ls_eval=False, freq_eval=False, final_eval=True)

    def _evaluate(self, ls_eval: bool, freq_eval: bool, final_eval: bool) -> bool:
        """
        Evaluate the policy (that is trained on some configuration with a seed).

        :param ls_eval: Write eval output to ls_eval/mean_{reward,ep_length}.
        :param freq_eval: Write eval output to eval/mean_{reward,ep_length}, also optionally keep track of evaluations
        by writing to the log_path if it is set. Also keep track of last_mean_reward, success_buffer (?) and best model.
        :param final_eval: Write eval output to final_eval/mean_{reward,ep_length}.
        """
        continue_training = True

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

            # Reset success rate buffer
            self._is_success_buffer: List[Any] = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if freq_eval and self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = float(np.mean(episode_rewards)), float(np.std(episode_rewards))
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            if freq_eval:
                self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, "
                    f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                )
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")

            # Add to current Logger
            if freq_eval:
                self.logger.record("eval/mean_reward", float(mean_reward))
                self.logger.record("eval/mean_ep_length", mean_ep_length)
            if ls_eval:
                self.logger.record("ls_eval/mean_reward", float(mean_reward))
                self.logger.record("ls_eval/mean_ep_length", mean_ep_length)
            if final_eval:
                self.logger.record("final_eval/mean_reward", float(mean_reward))
                self.logger.record("final_eval/mean_ep_length", mean_ep_length)

            if freq_eval and len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose > 0:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if freq_eval and mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            if ls_eval:
                if self.verbose > 0:
                    print(f"Saving model checkpoint to {self.ls_model_save_path}")
                self.model.save(self.ls_model_save_path)

            if final_eval:
                # if self.verbose > 0:
                #     print(f"Saving model checkpoint to {self.ls_model_save_path}")
                # self.model.save(self.ls_model_save_path)
                self.comp.record(self.conf_idx, self.run_id, mean_reward)

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training
