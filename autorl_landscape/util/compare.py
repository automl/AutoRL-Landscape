from typing import Optional

import os

import numpy as np


def choose_best_conf(run_ids: np.ndarray, final_mean_rewards: np.ndarray, save: Optional[str]) -> str:
    """
    Choose the best (ls) conf (row) as that which produced the best performance results on average.

    Then choose best run from that configuration via maximum.

    Rows in `run_ids` and `final_mean_rewards` correspond to values of the same config, but from different seeds.

    :param run_ids: np string array, wandb ids
    :param final_mean_rewards: performance values corresponding to their respective run_ids
    :param save: if given, a symlink linking back to the found best config's folder is created as best_agent. Path
    should have no trailing slashes
    """
    conf_mean_rewards = np.mean(final_mean_rewards, axis=1)
    best_conf = np.argmax(conf_mean_rewards)
    best_seed = np.argmax(final_mean_rewards[best_conf])
    best_id = run_ids[best_conf, best_seed]

    if save is not None:
        os.symlink(f"agents/{best_id}", f"{save}/best_agent", target_is_directory=True)

    return best_id
