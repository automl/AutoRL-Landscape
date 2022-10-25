from typing import Any, DefaultDict, List, Optional, Tuple

import os
from collections import defaultdict

import numpy as np
import scipy.stats


def construct_2d(indices: np.ndarray, *arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Take some indices and uses them to pack the values into their respective rows.

    indices : np.ndarray
        Row for the return
    arrays : np.ndarray
        One or multiple numpy arrays to `reshape`

    Returns
    -------
    np.ndarray
        The reshaped numpy arrays
    """
    rets: List[np.ndarray] = []
    bins = np.bincount(indices)
    num_rows = np.max(indices) + 1
    assert np.all(bins == bins[0]), "Each index must occur the same number of times!"

    for arr in arrays:
        rows: DefaultDict[int, List[Any]] = defaultdict(list)
        assert len(indices) == len(arr), "Index list must have the same length as all given arrays!"
        for index, val in zip(indices, arr):
            rows[index].append(val)
        ret = np.array([rows[i] for i in range(num_rows)])
        rets.append(ret)

    return tuple(rets)


def choose_best_conf(run_ids: np.ndarray, final_returns: np.ndarray, save: Optional[str]) -> str:
    """
    Choose the best (ls) conf (row) as that which produced the best performance results on average.

    Then choose best run from that configuration via maximum.
    Rows in `run_ids` and `final_mean_rewards` correspond to values of the same config, but from different seeds.
    Note: All configurations should have equal amounts of

    :param results: has tuples of (conf_index, run_id, final_returns)
    :param save: if given, a symlink linking back to the found best config's folder is created as best_agent. Path
    should have no trailing slashes
    """
    # IQMs for each configuration (aggregating over both different seeds and their evaluations)
    assert run_ids.shape == final_returns.shape[0:-1]
    num_confs = final_returns.shape[0]
    conf_iqms = scipy.stats.trim_mean(
        final_returns.reshape(num_confs, -1), proportiontocut=0.25, axis=1
    )  # shape (n_confs,)
    best_conf = np.argmax(conf_iqms)
    # IQMs for each seed of the best configuration (aggregating only over their evaluations)
    seed_iqms = scipy.stats.trim_mean(final_returns[best_conf], proportiontocut=0.25, axis=1)  # shape (n_seeds,)
    best_seed = np.argmax(seed_iqms)

    best_id = run_ids[best_conf, best_seed]

    if save is not None:
        os.symlink(f"agents/{best_id}", f"{save}/best_agent", target_is_directory=True)

    return best_id
