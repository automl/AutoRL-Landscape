from __future__ import annotations

from typing import Any

from pathlib import Path

from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.utils import constant_fn
from stable_baselines3.ppo.ppo import PPO

from autorl_landscape.custom_agents.on_policy_algorithm import custom_learn
from autorl_landscape.run.rl_context import make_env, seed_rl_context


class CustomPPO(PPO):
    """Slightly changed PPO Agent that can be saved and loaded at any point to reproduce learning exactly."""

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> CustomPPO:
        """Like original method from `OnPolicyAlgorithm`, but call `after_update()` callback method."""
        return custom_learn(
            self, total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps, progress_bar
        )

    @classmethod
    def custom_load(cls, save_path: Path, seed: int) -> CustomPPO:
        """Load agent which will perform deterministically with further training, like the originally saved agent."""
        loaded_agent: CustomPPO = CustomPPO.load(save_path / Path("model.zip"))
        with (save_path / Path("env.txt")).open("r") as f:
            env_name = f.read()
        loaded_agent.env = make_env(env_name, seed, loaded_agent.n_envs)
        seed_rl_context(loaded_agent, seed, reset=True)
        return loaded_agent

    def custom_save(self, save_path: Path, seed: int) -> None:
        """Save the agent so that it will perform deterministically with further training.

        Can be loaded afterwards with `custom_load`.
        """
        env_name = self.env.envs[0].spec.id
        self.env = make_env(env_name, seed, self.n_envs)
        seed_rl_context(self, seed, reset=True)
        self.save(f"{save_path}/model.zip")
        with (save_path / Path("env.txt")).open("w") as f:
            f.write(env_name)

    def set_ls_conf(self, ls_spec: dict[str, Any], _: int) -> None:
        """Set up the agent with the wanted configuration. Special handling for learning rate."""
        for hp_name, hp_val in ls_spec.items():
            match hp_name:
                # First, handle special cases:
                case "gamma":
                    self.gamma = hp_val
                    self.rollout_buffer.gamma = hp_val
                case "learning_rate":
                    self.lr_schedule = constant_fn(hp_val)
                case "gae_lambda":
                    self.rollout_buffer.gae_lambda = hp_val
                # Then, the rest:
                case _:
                    if not hasattr(self, hp_name):
                        error_msg = f"Hyperparameter {hp_name} cannot be set for {type(self)}."
                        raise ValueError(error_msg)
                    setattr(self, hp_name, hp_val)

    def get_ls_conf(self, ls_spec: list[str]) -> dict[str, Any]:
        """Read the actually used hyperparameter configuration.

        Args:
            ls_spec: list of hyperparameters to read (should come from the hydra config)
        """
        ls_conf: dict[str, Any] = {}
        for hp_name in ls_spec:
            match hp_name:
                case "learning_rate":
                    ls_conf["learning_rate"] = self.lr_schedule(self._current_progress_remaining)
                case "gae_lambda":
                    ls_conf["gae_lambda"] = self.rollout_buffer.gae_lambda
                case _:
                    if not hasattr(self, hp_name):
                        error_msg = f"Hyperparameter {hp_name} does not exist for {type(self)}."
                        raise ValueError(error_msg)
                    ls_conf[hp_name] = getattr(self, hp_name)
        return ls_conf
