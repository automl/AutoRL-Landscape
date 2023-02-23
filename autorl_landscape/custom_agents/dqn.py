from __future__ import annotations

from typing import Any

import copy
import pickle
from pathlib import Path

from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.dqn.dqn import DQN

from autorl_landscape.custom_agents.off_policy_algorithm import custom_learn


class CustomDQN(DQN):
    """Slightly changed DQN Agent that can be saved and loaded at any point to reproduce learning exactly."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.collect_rollout_last_steps = 0
        self.first_rollout = True

    # def collect_rollouts(
    #     self,
    #     env: VecEnv,
    #     callback: BaseCallback,
    #     train_freq: TrainFreq,
    #     replay_buffer: ReplayBuffer,
    #     action_noise: ActionNoise | None = None,
    #     learning_starts: int = 0,
    #     log_interval: int | None = None,
    # ) -> RolloutReturn:
    #     """Same as original, except for setting num_collected_steps."""
    #     return custom_collect_rollouts(
    #         self, env, callback, train_freq, replay_buffer, action_noise, learning_starts, log_interval
    #     )

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DQN",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> CustomDQN:
        """Like original method from `OffPolicyAlgorithm`, but call `after_update()` callback method."""
        return custom_learn(
            self, total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps, progress_bar
        )

    @classmethod
    def custom_load(cls, save_path: Path, seed: int) -> CustomDQN:
        """Load agent which will perform deterministically with further training, like the originally saved agent."""
        loaded_agent: CustomDQN = CustomDQN.load(save_path / Path("model.zip"))
        loaded_agent.load_replay_buffer(save_path / Path("replay_buffer.pkl"))
        with open(save_path / Path("env.pkl"), "rb") as f:
            loaded_agent.set_env(pickle.load(f), force_reset=False)
        # with open(save_path / Path("custom_hps.json"), "r") as f:
        #     loaded_agent.collect_rollout_last_steps = json.load(f)
        #     loaded_agent.first_rollout = True
        loaded_agent.env.seed(seed)
        loaded_agent.action_space.seed(seed)
        loaded_agent.set_random_seed(seed)
        return loaded_agent

    def __deepcopy__(self, memo: dict[Any, Any]) -> CustomDQN:
        # avoid deepcopy of whole agent:
        save_model = copy.copy(self)
        save_model.replay_buffer = copy.deepcopy(self.replay_buffer, memo)
        save_model.ep_info_buffer = copy.deepcopy(self.ep_info_buffer, memo)
        save_model.ep_success_buffer = copy.deepcopy(self.ep_success_buffer, memo)
        save_model._last_original_obs = copy.deepcopy(self._last_original_obs, memo)
        save_model._last_obs = copy.deepcopy(self._last_obs, memo)
        # save_model._current_progress_remaining = copy.deepcopy(self._current_progress_remaining)
        # save_model._n_calls = copy.deepcopy(self._n_calls)
        save_model.q_net = copy.deepcopy(self.q_net, memo)
        save_model.q_net_target = copy.deepcopy(self.q_net_target, memo)
        # save_model.tau = copy.deepcopy(self.tau)
        # save_model.exploration_rate = copy.deepcopy(self.exploration_rate)
        # save_model._logger = copy.deepcopy(self._logger, memo)
        # save_model._episode_num = copy.deepcopy(self._episode_num)
        # save_model.exploration_initial_eps = copy.deepcopy(self.exploration_initial_eps)
        # save_model.exploration_final_eps = copy.deepcopy(self.exploration_final_eps)
        # save_model.learning_starts = copy.deepcopy(self.learning_starts)
        return save_model

    def custom_save(self, save_path: str, seed: int) -> None:
        """Save the agent so that it will perform deterministically with further training.

        Can be loaded afterwards with `custom_load`.
        """
        self.env.seed(seed)
        self.action_space.seed(seed)
        self.set_random_seed(seed)
        save_model = copy.deepcopy(self)
        print(f"Saving model at {save_model.num_timesteps=}")

        # set exploration before saving, since exploration profile is set up directly on load
        save_model.exploration_initial_eps = self.exploration_final_eps
        save_model.exploration_final_eps = self.exploration_final_eps
        # also set learning_starts thing
        save_model.learning_starts = 0
        # save_model.collect_rollout_last_steps = locals["num_collected_steps"]

        save_model.save(f"{save_path}/model.zip")
        save_model.save_replay_buffer(f"{save_path}/replay_buffer.pkl")
        with open(f"{save_path}/env.pkl", "wb") as f:
            e = save_model.get_env()
            pickle.dump(e, f)
        # with open(f"{save_path}/custom_hps.json", "w") as f:
        #     print(f"{locals['num_collected_steps']=}")
        #     json.dump(locals["num_collected_steps"], f)
