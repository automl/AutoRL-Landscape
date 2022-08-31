from abc import abstractmethod
from typing import Dict

import coax
import gym
import wandb
from omegaconf.omegaconf import DictConfig

from autorl_landscape.util.debug import DEBUG


class Agent:
    def __init__(self, cfg: DictConfig, ls_conf: DictConfig) -> None:
        self.cfg = cfg
        self.ls_conf = ls_conf
        self.pi = None

        self.env = coax.wrappers.TrainMonitor(gym.make(cfg.env.name))
        self.eval_env = gym.make(cfg.env.name)

    @abstractmethod
    def train(self, steps: int) -> None:
        pass

    def _training_loop(self) -> None:
        pass

    def evaluate(self) -> None:
        eval_runs = 10
        avg_return = 0.0
        for i in range(eval_runs):
            self.eval_env.seed(42 + i)
            state = self.eval_env.reset()
            done = False
            while not done:
                action = self.pi(state)
                state, reward, done, _ = self.eval_env.step(action)
                avg_return += reward

        avg_return /= eval_runs
        self.log({"Evaluation/avg_return": avg_return})

    def log(self, info: Dict) -> None:
        """A simple logging wrapper that ignores debugging runs."""
        if not DEBUG:
            wandb.log(info)
