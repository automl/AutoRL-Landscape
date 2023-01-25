import matplotlib.pyplot as plt
import numpy as np
from KDEpy.bw_selection import improved_sheather_jones
from pandas import DataFrame

from autorl_landscape.ls_models.ls_model import LSModel
from autorl_landscape.util.data import split_phases


def get_avg_bandwidth(df: DataFrame, viz: bool) -> float:
    """TODO."""
    # collect performance data (ls return) distributions for all configurations and all phases in a dataset
    phase_strs = sorted(df["meta.phase"].unique())
    ys = []
    for phase_str in phase_strs:
        phase_data, _ = split_phases(df, phase_str)
        model = LSModel(phase_data, np.float64)
        ys.append(model.y)
    ys = np.concatenate(ys, axis=0)

    isjs: list[float] = [improved_sheather_jones(y.reshape(-1, 1)) for y in ys]
    # isjs: list[float] = [silvermans_rule(y.reshape(-1, 1)) for y in ys]

    if not viz:
        return np.mean(isjs)

    fig = plt.figure(figsize=(16, 12))
    ax = fig.gca()
    ax.hist(isjs)
    plt.show()
