from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from omegaconf import DictConfig
from pandas import DataFrame

from autorl_landscape.util.data import read_wandb_csv
from autorl_landscape.util.grid_space import grid_space_nd
from autorl_landscape.util.ls_sampler import construct_ls


def visualize_sobol_samples(conf: DictConfig) -> None:
    """Visualize with plt to inspect the sampled patterns.

    Args:
        conf: Hydra configuration
    """
    print("VIZ ONLY DOES LR, GAMMA, AND EXPLORATION RATE FOR NOW")
    df = construct_ls(conf)
    fig = plt.figure(figsize=(16, 16))
    fig.tight_layout()
    ax = fig.add_subplot(1, 1, 1, projection="3d")  # TODO use actual number of phases
    ax.scatter(df["learning_rate"], 1 - df["neg_gamma"], df["exploration_final_eps"])
    # ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("gamma")
    ax.set_zlabel("exploration")
    plt.show()


def visualize_data_samples(file: str) -> None:
    """Visualize grid of samples, read from a file."""
    df = read_wandb_csv(file)
    # phase_data = df[df["meta.phase"] == "phase_0"]
    fig = plt.figure(figsize=(16, 16))
    fig.tight_layout()
    ax = plt.axes()
    ax.scatter(df["ls.learning_rate"], df["ls.gamma"])
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("gamma")
    plt.show()


def visualize_data(file: str) -> None:
    """Get performance (returns) data from file and visualize it.

    Args:
        file: csv data file
        fit_gp: if `True`, fit and visualize GP mean instead of just the raw mean
    """
    df = read_wandb_csv(file)
    fig = plt.figure(figsize=(17, 12))
    for i, phase_str in enumerate(sorted(df["meta.phase"].unique())):
        phase_data = df[df["meta.phase"] == phase_str]
        plot_z = np.array(list(phase_data["ls_eval/returns"]))  # (num_confs * num_seeds, num_evals)
        plot_x = np.repeat(np.log10(np.array(phase_data["ls.learning_rate"])), plot_z.shape[-1], axis=0)
        plot_y = np.repeat(np.array(phase_data["ls.gamma"]), plot_z.shape[-1], axis=0)
        zlabel = "LS Mean Return"
        title = f"Returns for {phase_str}"

        ax = fig.add_subplot(1, 3, i + 1, projection="3d")  # TODO use actual number of phases
        ax.scatter(plot_x, plot_y, plot_z, cmap="viridis", edgecolor="none")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_log_tick_formatter))
        ax.set_zlim3d(0, df["conf.viz.max_return"][0])
        ax.set_xlabel("Learning Rate", fontsize=12)
        ax.set_ylabel("Gamma", fontsize=12)
        ax.set_zlabel(zlabel, fontsize=12)
        ax.set_title(title, fontsize=16)
    plt.show()


def plot_surface_(ax: Axes, pt: DataFrame, kwargs: dict[str, Any]) -> Artist:
    """TODO."""
    grid_length = pt.values.shape[0]
    grid = grid_space_nd(2, grid_length)
    grid_x0 = grid[:, :, 0]
    grid_x1 = grid[:, :, 1]

    return ax.plot_surface(grid_x0, grid_x1, pt.values, **kwargs)


# def visualize_ls_model(file: str, sample_percentage: int, viz_samples: bool, retrain: bool, save: bool) -> None:
#     """Get performance data from file and visualize a GP.

#     GP is either loaded from disk (with the file being a sibling to `file`) or trained, depending on whether the file
#     exists.

#     Args:
#         file: csv data file.
#         sample_percentage: Proportion of data that should be used to fit the GP. Useful for datasets with high amounts
#             of samples. Needs to be in the interval [0, 100].
#         viz_samples: Whether to also visualize the used samples in the plots as dots.
#         retrain: Whether to re-train the GP even if the model already exists on disk.
#         save: Whether to save the trained model to disk.
#     """
#     df = read_wandb_csv(file)
#     fig = plt.figure(figsize=(16, 12))
#     for phase_str in sorted(df["meta.phase"].unique()):
#         i = int(phase_str[-1])
#         phase_data = df[df["meta.phase"] == phase_str].sort_values("meta.conf_index")

#         conf_groups = phase_data.groupby(["meta.conf_index", "ls.learning_rate", "ls.gamma"])
#         y = np.array(list(conf_groups["ls_eval/returns"].sum()))  # sum does concat (256, 100)
#         num_evals = np.array(list(phase_data["ls_eval/returns"])).shape[1]  # hacky
#         model = train_or_load_gps(phase_str, file, retrain, save=save)
#         data = (
#             log_to_unit(np.repeat(np.array(phase_data["ls.learning_rate"]), num_evals), 1000, 0.0001, 0.1),
#             log_to_unit(1 - np.repeat(np.array(phase_data["ls.gamma"]), num_evals), 5, 0.0001, 0.2),
#             y.flatten(),
#         )
#         plot_gp(model, fig, df["meta.phase"].nunique(), i + 1, data)
#     fig.tight_layout()
#     plt.show()


def _log_tick_formatter(val: Any, pos: Any = None) -> Any:
    return "{:.2e}".format(10**val)
