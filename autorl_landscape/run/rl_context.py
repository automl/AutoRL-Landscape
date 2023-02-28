import gym
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.monitor import Monitor


def make_env(env_name: str, seed: int) -> gym.Env:
    """Quick helper to create a gym env and seed it."""
    env = gym.make(env_name)
    env = Monitor(env)
    env.seed(seed)
    return env


def seed_rl_context(agent: BaseAlgorithm, seed: int) -> None:
    """Set seeds for a whole RL context (i.e., env and agent/policy)."""
    agent.env.seed(seed)
    agent.action_space.seed(seed)
    agent.action_space.np_random.seed(seed)
    agent.set_random_seed(seed)
