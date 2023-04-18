import gym
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor


def make_env(env_name: str, seed: int, n_envs: int | None = None) -> gym.Env:
    """Quick helper to create a gym env and seed it."""
    if n_envs is not None:
        env = make_vec_env(env_name, n_envs, seed)
    else:
        env = gym.make(env_name)
        env = Monitor(env)
        env.seed(seed)
    return env


def seed_rl_context(agent: BaseAlgorithm, seed: int, reset: bool = False) -> None:
    """Set seeds for a whole RL context (i.e., env and agent/policy)."""
    agent.env.seed(seed)
    agent.action_space.seed(seed)
    agent.action_space.np_random.seed(seed)
    agent.set_random_seed(seed)
    if reset:
        agent.env.reset()
