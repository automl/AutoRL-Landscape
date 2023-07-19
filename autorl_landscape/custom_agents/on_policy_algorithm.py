import sys
import time

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.utils import safe_mean


def custom_learn(
    agent: OnPolicyAlgorithm,
    total_timesteps: int,
    callback: MaybeCallback = None,
    log_interval: int = 1,
    tb_log_name: str = "OnPolicyAlgorithm",
    reset_num_timesteps: bool = True,
    progress_bar: bool = False,
) -> OnPolicyAlgorithm:
    """Like original method from `OffPolicyAlgorithm`, but call `after_update()` callback method.

    `after_update()` is called after policy updates as well as at the very start.
    """
    iteration = 0

    total_timesteps, callback = agent._setup_learn(
        total_timesteps,
        callback,
        reset_num_timesteps,
        tb_log_name,
        progress_bar,
    )

    callback.on_training_start(locals(), globals())
    callback.after_update()

    while agent.num_timesteps < total_timesteps:
        continue_training = agent.collect_rollouts(
            agent.env,
            callback,
            agent.rollout_buffer,
            n_rollout_steps=agent.n_steps,
        )

        if continue_training is False:
            break

        iteration += 1
        agent._update_current_progress_remaining(agent.num_timesteps, total_timesteps)

        # Display training infos
        if log_interval is not None and iteration % log_interval == 0:
            time_elapsed = max((time.time_ns() - agent.start_time) / 1e9, sys.float_info.epsilon)
            fps = int((agent.num_timesteps - agent._num_timesteps_at_start) / time_elapsed)
            agent.logger.record("time/iterations", iteration, exclude="tensorboard")
            if len(agent.ep_info_buffer) > 0 and len(agent.ep_info_buffer[0]) > 0:
                agent.logger.record(
                    "rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in agent.ep_info_buffer])
                )
                agent.logger.record(
                    "rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in agent.ep_info_buffer])
                )
            agent.logger.record("time/fps", fps)
            agent.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
            agent.logger.record("time/total_timesteps", agent.num_timesteps, exclude="tensorboard")
            agent.logger.dump(step=agent.num_timesteps)

        agent.train()

        callback.after_update()

    callback.on_training_end()

    return agent
