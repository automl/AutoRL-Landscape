from typing import List, Optional, Tuple

import os

import numpy as np


class Comparator:
    """
    Given the performance readings of many lanscape configurations of an agent over multiple seeds
    figure out the best (landscape) configuration by calculating the performance over the seeds, then
    saving the best agent of that configuration.
    """

    def __init__(self, phase_path: str, num_confs: int, num_seeds: int, strict: bool = False) -> None:
        self.phase_path = phase_path
        self.mean_rewards: List[Optional[List[Tuple[str, float]]]] = [None] * num_confs
        # [ [(9avaw245, 354.3), (23adfcya, 123.5)] <-- conf 1
        # , [(09av0q2a, 100.9), (098ybcqw, 564.3)] <-- conf 2
        # ]
        self.num_confs = num_confs
        self.num_seeds = num_seeds
        self.strict = strict

    def record(self, conf_idx: int, run_id: str, mean_reward: float):
        """
        Save the reward of a specific agent with a specific (landscape) configuration.

        :param conf_idx: Number of the configuration
        :param run_id: wandb id of the agent
        :param mean_reward: performance value to save for later comparison
        """
        assert conf_idx < self.num_confs
        if self.mean_rewards[conf_idx] is None:
            self.mean_rewards[conf_idx] = []
        conf_mean_rewards = self.mean_rewards[conf_idx]
        conf_mean_rewards.append((run_id, mean_reward))
        assert len(conf_mean_rewards) <= self.num_seeds

    def save_best(self):
        """Find the best configuration and remember its best seed as the best model."""
        if self.strict:
            self._assert_all_done()

        # find best configuration by mean(mean_rewards)
        best_mean_reward = -1.0
        best_conf = -1
        for conf, conf_mean_rewards in enumerate(self.mean_rewards):
            if len(conf_mean_rewards) > 0:
                mean_reward = np.mean([r for (_, r) in conf_mean_rewards])
                if mean_reward > best_mean_reward:
                    best_conf = conf
                    best_mean_reward = mean_reward

        # find best seed of best configuration
        best_id_idx = np.argmax([r for (_, r) in self.mean_rewards[best_conf]])
        best_id = self.mean_rewards[best_conf][best_id_idx][0]

        # try:
        #     os.makedirs(f"{self.phase_path}/best_agent", exist_ok=False)
        # except OSError as e:
        #     raise Exception(
        #         f"best_agent dir for {self.phase_path} already exists. Cancelling saving of best agent."
        #     ) from e
        try:
            os.symlink(f"{self.phase_path}/agents/{best_id}", f"{self.phase_path}/best_agent")
        except Exception as e:
            raise Exception(
                f"best_agent dir for {self.phase_path} already exists. Cancelling saving of best agent."
            ) from e

    def _assert_all_done(self):
        """Check that we have data for all configs and all seeds."""
        assert len(self.mean_rewards) == self.num_confs
        for conf_mean_rewards in self.mean_rewards:
            assert len(conf_mean_rewards) == self.num_seeds
