from typing import Any

import matplotlib.pyplot as plt
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


def fit_gp(data: pd.DataFrame) -> Any:
    """Put in data, fit GP, get grid of predictions back :)."""
    X = data[["ls.learning_rate", "ls.gamma"]].to_numpy()
    y = data["ls_eval/mean_return"].to_numpy()
    gpr = GaussianProcessRegressor(RBF()).fit(X, y)
    print(f"{gpr.score(X, y)=}")
    return gpr


def visualize_data(file: str) -> None:
    """Visualize performance, given the hyperparameters.

    Args:
        file: csv data file
    """
    df = pd.read_csv(file, index_col=0)
    for i in range(3):
        phase_str = f"phase_{i}"
        phase_data = df[df["meta.phase"] == phase_str]
        gpr = fit_gp(phase_data)
        grid = np.meshgrid(np.logspace(-5, -1, 30), np.linspace(0.8, 0.9999, 30))
        g0 = grid[0].reshape(-1)
        g1 = grid[1].reshape(-1)
        preds_mean, preds_std = gpr.predict(np.stack([g0, g1]).T, return_std=True)

        _ = plt.figure(figsize=(30, 30))
        ax = plt.axes(projection="3d")
        ax.plot_surface(
            grid[0],
            grid[1],
            preds_mean.reshape(grid[0].shape),
            # rstride=1,
            # cstride=1,
            cmap="viridis",
            edgecolor="none",
        )
        # ax.yaxis.set_major_formatter(mticker.FuncFormatter(_log_tick_formatter))
        plt.show()
        break


def _log_tick_formatter(val: Any, pos: Any = None) -> Any:
    return "{:.2e}".format(10**val)
