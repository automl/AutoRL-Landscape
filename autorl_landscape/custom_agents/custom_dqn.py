from typing import Any, Dict, Optional, Tuple, Type, Union

import copy
import json
import pickle
from pathlib import Path

import torch as th
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.type_aliases import (
    GymEnv,
    RolloutReturn,
    Schedule,
    TrainFreq,
    TrainFrequencyUnit,
)
from stable_baselines3.common.utils import should_collect_more_steps
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.dqn.policies import DQNPolicy


class CustomDQN(DQN):
    """Slightly changed DQN Agent that can be saved and loaded at any point to reproduce learning exactly."""

    def __init__(
        self,
        policy: Union[str, Type[DQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 0.0001,
        buffer_size: int = 1000000,
        learning_starts: int = 50000,
        batch_size: int = 32,
        tau: float = 1,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            replay_buffer_class,
            replay_buffer_kwargs,
            optimize_memory_usage,
            target_update_interval,
            exploration_fraction,
            exploration_initial_eps,
            exploration_final_eps,
            max_grad_norm,
            tensorboard_log,
            create_eval_env,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model,
        )
        self.collect_rollout_last_steps = 0
        self.first_rollout = True

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        # CHANGED: After loading the model, set num_collected_steps so that the rollout is continued correctly, with
        # model updates happening at the correct time.
        if self.first_rollout:
            num_collected_steps = self.collect_rollout_last_steps
            self.first_rollout = False

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(
                    num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False
                )

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()

        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    @classmethod
    def custom_load(cls, save_path: Path, seed: Optional[int] = None) -> "CustomDQN":
        """
        Load agent which will perform deterministically with further training, compared to the original agent that was
        saved.
        """
        loaded_agent: CustomDQN = CustomDQN.load(save_path / Path("model.zip"))
        loaded_agent.load_replay_buffer(save_path / Path("replay_buffer.pkl"))
        with open(save_path / Path("env.pkl"), "rb") as f:
            loaded_agent.set_env(pickle.load(f), force_reset=False)
        with open(save_path / Path("custom_hps.json"), "r") as f:
            # loaded_agent.set_custom_hps(pickle.load(f))
            loaded_agent.collect_rollout_last_steps = json.load(f)
            loaded_agent.first_rollout = True
        if seed is not None:
            loaded_agent.env.seed(seed)
            loaded_agent.action_space.seed(seed)
            loaded_agent.set_random_seed(seed)
        return loaded_agent

    def __deepcopy__(self, memo: Dict) -> "CustomDQN":
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

    def custom_save(self, locals: Dict[str, Any], save_path: str, seed: Optional[int] = None) -> None:
        """
        Save agent which will perform deterministically with further training after loading with `custom_load`, compared
        to the original agent that was saved.
        """
        self.env.seed(seed)
        self.action_space.seed(seed)
        self.set_random_seed(seed)
        save_model = copy.deepcopy(self)
        print(f"Saving model at {save_model.num_timesteps=}")

        # rest of collect_rollouts() loop:
        save_model._update_info_buffer(locals["infos"], locals["dones"])
        save_model._store_transition(
            save_model.replay_buffer,
            locals["buffer_actions"],
            locals["new_obs"],
            locals["rewards"],
            locals["dones"],
            locals["infos"],
        )
        save_model._update_current_progress_remaining(save_model.num_timesteps, save_model._total_timesteps)
        save_model._on_step()
        for idx, done in enumerate(locals["dones"]):
            if done:
                # num_collected_episodes += 1
                save_model._episode_num += 1
                if locals["action_noise"] is not None:
                    kwargs = dict(indices=[idx]) if locals["env"].num_envs > 1 else {}
                    locals["action_noise"].reset(**kwargs)

                # Log training infos
                if locals["log_interval"] is not None and save_model._episode_num % locals["log_interval"] == 0:
                    save_model._dump_logs()

        # set exploration before saving, since exploration profile is set up directly on load
        save_model.exploration_initial_eps = 0.04
        save_model.exploration_final_eps = 0.04
        # also set learning_starts thing
        save_model.learning_starts = 0
        # save_model.collect_rollout_last_steps = locals["num_collected_steps"]

        save_model.save(f"{save_path}/model.zip")
        save_model.save_replay_buffer(f"{save_path}/replay_buffer.pkl")
        with open(f"{save_path}/env.pkl", "wb") as f:
            e = save_model.get_env()
            pickle.dump(e, f)
        with open(f"{save_path}/custom_hps.json", "w") as f:
            json.dump(locals["num_collected_steps"], f)
