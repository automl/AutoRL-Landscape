from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import MaybeCallback

from autorl_landscape.util.callback import LandscapeEvalCallback


def custom_learn(
    agent: OffPolicyAlgorithm,
    total_timesteps: int,
    callback: MaybeCallback = None,
    log_interval: int = 4,
    tb_log_name: str = "run",
    reset_num_timesteps: bool = True,
    progress_bar: bool = False,
) -> OffPolicyAlgorithm:
    """Like original method from `OffPolicyAlgorithm`, but call `after_update()` callback method."""
    total_timesteps, callback = agent._setup_learn(
        total_timesteps,
        callback,
        reset_num_timesteps,
        tb_log_name,
        progress_bar,
    )

    assert isinstance(callback, LandscapeEvalCallback)
    callback.on_training_start(locals(), globals())

    while agent.num_timesteps < total_timesteps:
        rollout = agent.collect_rollouts(
            agent.env,
            train_freq=agent.train_freq,
            action_noise=agent.action_noise,
            callback=callback,
            learning_starts=agent.learning_starts,
            replay_buffer=agent.replay_buffer,
            log_interval=log_interval,
        )

        if rollout.continue_training is False:
            break

        if agent.num_timesteps > 0 and agent.num_timesteps > agent.learning_starts:
            # If no `gradient_steps` is specified,
            # do as many gradients steps as steps performed during the rollout
            gradient_steps = agent.gradient_steps if agent.gradient_steps >= 0 else rollout.episode_timesteps
            # Special case when the user passes `gradient_steps=0`
            if gradient_steps > 0:
                agent.train(batch_size=agent.batch_size, gradient_steps=gradient_steps)

        callback.after_update()

    callback.on_training_end()

    return agent


# def custom_collect_rollouts(
#     agent: OffPolicyAlgorithm,
#     env: VecEnv,
#     callback: BaseCallback,
#     train_freq: TrainFreq,
#     replay_buffer: ReplayBuffer,
#     action_noise: ActionNoise | None = None,
#     learning_starts: int = 0,
#     log_interval: int | None = None,
# ) -> RolloutReturn:
#     """Same as original collect_rollouts() from OffPolicyAlgorithm, except for moving the on_step callback.

#     callback.on_step() is basically called at the end of a step.
#     """
#     # Switch to eval mode (this affects batch norm / dropout)
#     agent.policy.set_training_mode(False)

#     num_collected_steps, num_collected_episodes = 0, 0

#     assert isinstance(env, VecEnv), "You must pass a VecEnv"
#     assert train_freq.frequency > 0, "Should at least collect one step or episode."

#     if env.num_envs > 1:
#         assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

#     # Vectorize action noise if needed
#     if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
#         action_noise = VectorizedActionNoise(action_noise, env.num_envs)

#     if agent.use_sde:
#         agent.actor.reset_noise(env.num_envs)

#     callback.on_rollout_start()
#     continue_training = True

#     while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
#         if agent.use_sde and agent.sde_sample_freq > 0 and num_collected_steps % agent.sde_sample_freq == 0:
#             # Sample a new noise matrix
#             agent.actor.reset_noise(env.num_envs)

#         # Select action randomly or according to policy
#         actions, buffer_actions = agent._sample_action(learning_starts, action_noise, env.num_envs)

#         # Rescale and perform action
#         new_obs, rewards, dones, infos = env.step(actions)
#         print(
#             f"{agent.num_timesteps} #steps={num_collected_steps} #eps={num_collected_episodes}"
#             "{new_obs} {sum([l.sum() for l in agent.q_net.parameters()])}"
#         )

#         agent.num_timesteps += env.num_envs
#         num_collected_steps += 1

#         # Retrieve reward and episode length if using Monitor wrapper
#         agent._update_info_buffer(infos, dones)

#         # Store data in replay buffer (normalized action and unnormalized observation)
#         agent._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

#         agent._update_current_progress_remaining(agent.num_timesteps, agent._total_timesteps)

#         # For DQN, check if the target network should be updated
#         # and update the exploration schedule
#         # For SAC/TD3, the update is dones as the same time as the gradient update
#         # see https://github.com/hill-a/stable-baselines/issues/900
#         agent._on_step()

#         for idx, done in enumerate(dones):
#             if done:
#                 # Update stats
#                 num_collected_episodes += 1
#                 agent._episode_num += 1

#                 if action_noise is not None:
#                     kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
#                     action_noise.reset(**kwargs)

#                 # Log training infos
#                 if log_interval is not None and agent._episode_num % log_interval == 0:
#                     agent._dump_logs()

#         # Give access to local variables
#         callback.update_locals(locals())
#         # Only stop training if return value is False, not when it is None.
#         if callback.on_step() is False:
#             return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

#     return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)


# def finish_collect_rollouts(agent: OffPolicyAlgorithm, locals: dict[str, Any]) -> None:
#     """Rest of collect_rollouts() loop after callback.on_step() call."""
#     agent._update_info_buffer(locals["infos"], locals["dones"])
#     agent._store_transition(
#         agent.replay_buffer,
#         locals["buffer_actions"],
#         locals["new_obs"],
#         locals["rewards"],
#         locals["dones"],
#         locals["infos"],
#     )
#     agent._update_current_progress_remaining(agent.num_timesteps, agent._total_timesteps)
#     agent._on_step()
#     for idx, done in enumerate(locals["dones"]):
#         if done:
#             # num_collected_episodes += 1  # TODO why commented?
#             agent._episode_num += 1
#             if locals["action_noise"] is not None:
#                 kwargs = dict(indices=[idx]) if locals["env"].num_envs > 1 else {}
#                 locals["action_noise"].reset(**kwargs)

#             # Log training infos
#             if locals["log_interval"] is not None and agent._episode_num % locals["log_interval"] == 0:
#                 agent._dump_logs()
