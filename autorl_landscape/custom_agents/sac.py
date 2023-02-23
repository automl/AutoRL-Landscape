from __future__ import annotations

from typing import Any

from pathlib import Path

from stable_baselines3.sac.sac import SAC


class CustomSAC(SAC):
    """Slightly changed DQN Agent that can be saved and loaded at any point to reproduce learning exactly."""

    def __init__(self, kwargs: dict[str, Any]):
        super().__init__(**kwargs)

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

    @classmethod
    def custom_load(cls, save_path: Path, seed: int) -> CustomSAC:
        """Load agent which will perform deterministically with further training, like the originally saved agent."""
        raise NotImplementedError

    # def custom_save(self, locals: Dict[str, Any], save_path: str, seed: int) -> None:
    def custom_save(self, save_path: Path, seed: int) -> None:
        """Save the agent so that it will perform deterministically with further training.

        Can be loaded afterwards with `custom_load`.
        """
        # re-seed the original agent to allow for same results with the loaded agent?
        self.env.seed(seed)
        self.action_space.seed(seed)
        self.set_random_seed(seed)
        raise NotImplementedError
