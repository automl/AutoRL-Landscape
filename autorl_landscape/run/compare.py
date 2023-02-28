from typing import Any, DefaultDict, List, Optional, Tuple

import os
from collections import defaultdict

import numpy as np
from numpy.typing import NDArray
from scipy.stats import trim_mean


def construct_2d(indices: NDArray[Any], *arrays: NDArray[Any]) -> Tuple[NDArray[Any], ...]:
    """Take some indices and uses them to pack the values into their respective rows.

    Args:
        indices: Row for the return
        arrays: One or multiple numpy arrays to `reshape`

    Return: The reshaped numpy arrays
    """
    rets: List[NDArray[Any]] = []
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


def iqm(x: NDArray[Any], axis: int | None = None) -> NDArray[Any]:
    """Calculate the interquartile mean (IQM) of x."""
    return trim_mean(x, proportiontocut=0.25, axis=axis)


def choose_best_policy(run_ids: NDArray[Any], final_returns: NDArray[Any], save: Optional[str]) -> str:
    """Choose the best (ls) conf (row) as that which produced the best performance results on average.

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
    conf_iqms = iqm(final_returns.reshape(num_confs, -1), axis=1)  # shape (n_confs,)
    best_conf = np.argmax(conf_iqms)
    # IQMs for each seed of the best configuration (aggregating only over their evaluations)
    seed_iqms = iqm(final_returns[best_conf], axis=1)  # shape (n_seeds,)
    best_seed = np.argmax(seed_iqms)

    best_id: str = run_ids[best_conf, best_seed]

    if save is not None:
        os.symlink(f"agents/{best_id}", f"{save}/best_agent", target_is_directory=True)

    return best_id
