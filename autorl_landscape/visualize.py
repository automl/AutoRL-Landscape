from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from autorl_landscape.util.ls_sampler import construct_ls


def visualize_samples(conf: DictConfig) -> None:
    """Visualize with plt to inspect the sampled patterns.

    Args:
        conf: Hydra configuration
    """
    print("VIZ ONLY DOES LR AND GAMMA FOR NOW")
    df = construct_ls(conf)
    fig = plt.figure(figsize=(16, 16))
    fig.tight_layout()
    ax = plt.axes()
    ax.scatter(df["learning_rate"], 1 - df["neg_gamma"])
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("gamma")
    plt.show()


def visualize_data_samples(file: str) -> None:
    """Visualize grid of samples, read from a file."""
    df = pd.read_csv(file, index_col=0)
    # phase_data = df[df["meta.phase"] == "phase_0"]
    fig = plt.figure(figsize=(16, 16))
    fig.tight_layout()
    ax = plt.axes()
    ax.scatter(df["ls.learning_rate"], df["ls.gamma"])
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("gamma")
    plt.show()


def visualize_data(file: str, fit_gp: bool) -> None:
    """Get performance data from file and visualize it.

    Args:
        file: csv data file
        fit_gp: if `True`, fit and visualize GP mean instead of just the raw mean
    """
    df = pd.read_csv(file, index_col=0)
    fig = plt.figure(figsize=(16, 12))
    for i in range(3):
        phase_str = f"phase_{i}"
        phase_data = df[df["meta.phase"] == phase_str]
        if fit_gp:
            X = phase_data[["ls.learning_rate", "ls.gamma"]].to_numpy()
            y = phase_data["ls_eval/mean_return"].to_numpy()
            gpr = GaussianProcessRegressor(RBF()).fit(X, y)
            # gpr = GaussianProcessRegressor().fit(X, y)
            print(f"{gpr.score(X, y)=}")
            preds_mean, _ = gpr.predict(X, return_std=True)
            plot_x = np.log10(phase_data["ls.learning_rate"])
            plot_y = phase_data["ls.gamma"]
            plot_z = preds_mean
            zlabel = "GP Mean"
            title = f"GP mean, fitted on mean performance data for phase {i}"
        else:
            plot_x = np.log10(phase_data["ls.learning_rate"])
            plot_y = phase_data["ls.gamma"]
            plot_z = phase_data["ls_eval/mean_return"]
            zlabel = "LS Mean Return"
            title = f"Raw mean performance data for phase {i}"

        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.plot_trisurf(plot_x, plot_y, plot_z, cmap="viridis", edgecolor="none")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_log_tick_formatter))
        ax.set_zlim3d(0, 500)
        ax.set_xlabel("Learning Rate", fontsize=12)
        ax.set_ylabel("Gamma", fontsize=12)
        ax.set_zlabel(zlabel, fontsize=12)
        ax.set_title(title, fontsize=16)
    plt.show()


def _log_tick_formatter(val: Any, pos: Any = None) -> Any:
    return "{:.2e}".format(10**val)
