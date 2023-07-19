from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import MaybeCallback

from autorl_landscape.run.callback import LandscapeEvalCallback


def custom_learn(
    agent: OffPolicyAlgorithm,
    total_timesteps: int,
    callback: MaybeCallback = None,
    log_interval: int = 4,
    tb_log_name: str = "run",
    reset_num_timesteps: bool = True,
    progress_bar: bool = False,
) -> OffPolicyAlgorithm:
    """Like original method from `OffPolicyAlgorithm`, but call `after_update()` callback method.

    `after_update()` is called after policy updates as well as at the very start.
    """
    total_timesteps, callback = agent._setup_learn(
        total_timesteps,
        callback,
        reset_num_timesteps,
        tb_log_name,
        progress_bar,
    )

    assert isinstance(callback, LandscapeEvalCallback)
    callback.on_training_start(locals(), globals())
    callback.after_update()

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
