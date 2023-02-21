from typing import Any

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import (
    RolloutReturn,
    TrainFreq,
    TrainFrequencyUnit,
)
from stable_baselines3.common.utils import should_collect_more_steps
from stable_baselines3.common.vec_env.base_vec_env import VecEnv


def custom_collect_rollouts(
    agent: OffPolicyAlgorithm,
    env: VecEnv,
    callback: BaseCallback,
    train_freq: TrainFreq,
    replay_buffer: ReplayBuffer,
    action_noise: ActionNoise | None = None,
    learning_starts: int = 0,
    log_interval: int | None = None,
) -> RolloutReturn:
    """Same as original from , except for setting num_collected_steps."""
    # Switch to eval mode (this affects batch norm / dropout)
    agent.policy.set_training_mode(False)

    num_collected_steps, num_collected_episodes = 0, 0

    # CHANGED: After loading the model, set num_collected_steps so that the rollout is continued correctly, with
    # model updates happening at the correct time.
    if agent.first_rollout:
        num_collected_steps = agent.collect_rollout_last_steps
        agent.first_rollout = False

    assert isinstance(env, VecEnv), "You must pass a VecEnv"
    assert train_freq.frequency > 0, "Should at least collect one step or episode."

    if env.num_envs > 1:
        assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

    # Vectorize action noise if needed
    if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
        action_noise = VectorizedActionNoise(action_noise, env.num_envs)

    if agent.use_sde:
        agent.actor.reset_noise(env.num_envs)

    callback.on_rollout_start()
    continue_training = True

    while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
        if agent.use_sde and agent.sde_sample_freq > 0 and num_collected_steps % agent.sde_sample_freq == 0:
            # Sample a new noise matrix
            agent.actor.reset_noise(env.num_envs)

        # Select action randomly or according to policy
        actions, buffer_actions = agent._sample_action(learning_starts, action_noise, env.num_envs)

        # Rescale and perform action
        new_obs, rewards, dones, infos = env.step(actions)

        agent.num_timesteps += env.num_envs
        num_collected_steps += 1

        # Give access to local variables
        callback.update_locals(locals())
        # Only stop training if return value is False, not when it is None.
        if callback.on_step() is False:
            return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

        # Retrieve reward and episode length if using Monitor wrapper
        agent._update_info_buffer(infos, dones)

        # Store data in replay buffer (normalized action and unnormalized observation)
        agent._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

        agent._update_current_progress_remaining(agent.num_timesteps, agent._total_timesteps)

        # For DQN, check if the target network should be updated
        # and update the exploration schedule
        # For SAC/TD3, the update is dones as the same time as the gradient update
        # see https://github.com/hill-a/stable-baselines/issues/900
        agent._on_step()

        for idx, done in enumerate(dones):
            if done:
                # Update stats
                num_collected_episodes += 1
                agent._episode_num += 1

                if action_noise is not None:
                    kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                    action_noise.reset(**kwargs)

                # Log training infos
                if log_interval is not None and agent._episode_num % log_interval == 0:
                    agent._dump_logs()

    callback.on_rollout_end()

    return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)


def finish_collect_rollouts(agent: OffPolicyAlgorithm, locals: dict[str, Any]) -> None:
    """Rest of collect_rollouts() loop after callback.on_step() call."""
    agent._update_info_buffer(locals["infos"], locals["dones"])
    agent._store_transition(
        agent.replay_buffer,
        locals["buffer_actions"],
        locals["new_obs"],
        locals["rewards"],
        locals["dones"],
        locals["infos"],
    )
    agent._update_current_progress_remaining(agent.num_timesteps, agent._total_timesteps)
    agent._on_step()
    for idx, done in enumerate(locals["dones"]):
        if done:
            # num_collected_episodes += 1  # TODO why commented?
            agent._episode_num += 1
            if locals["action_noise"] is not None:
                kwargs = dict(indices=[idx]) if locals["env"].num_envs > 1 else {}
                locals["action_noise"].reset(**kwargs)

            # Log training infos
            if locals["log_interval"] is not None and agent._episode_num % locals["log_interval"] == 0:
                agent._dump_logs()
