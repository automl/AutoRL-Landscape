from typing import Any

import math
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from numpy.typing import NDArray
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from autorl_landscape.util.data import read_wandb_csv

# def visualize_samples(conf: DictConfig) -> None:
#     """Visualize with plt to inspect the sampled patterns.

#     Args:
#         conf: Hydra configuration
#     """
#     print("VIZ ONLY DOES LR AND GAMMA FOR NOW")
#     df = construct_ls(conf)
#     fig = plt.figure(figsize=(16, 16))
#     fig.tight_layout()
#     ax = plt.axes()
#     ax.scatter(df["learning_rate"], 1 - df["neg_gamma"])
#     ax.set_xscale("log")
#     ax.set_xlabel("learning rate")
#     ax.set_ylabel("gamma")
#     plt.show()


def _log_base(x: NDArray, base: float) -> Any:
    return np.log(x) / np.log(base)


def _invert_ls_scaling(x: NDArray[Any], base: float, lower: float, upper: float) -> Any:
    return _log_base(1 + (((x - lower) * (base - 1)) / (upper - lower)), base)


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
    fig = plt.figure(figsize=(16, 12))
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


def visualize_gp(file: str, sample_percentage: int, viz_samples: bool, retrain: bool) -> None:
    """Get performance data from file and visualize a GP.

    GP is either loaded from disk (with the file being a sibling to `file`) or trained, depending on whether the file
    exists.

    Args:
        file: csv data file.
        sample_percentage: Proportion of data that should be used to fit the GP. Useful for datasets with high amounts
            of samples. Needs to be in the interval [0, 100].
        viz_samples: Whether to also visualize the used samples in the plots as dots.
        retrain: Whether to re-train the GP even if the model already exists on disk.
    """
    grid_size = 100
    df = read_wandb_csv(file)
    fig = plt.figure(figsize=(16, 12))
    for i, phase_str in enumerate(sorted(df["meta.phase"].unique())):
        phase_data = df[df["meta.phase"] == phase_str].sort_values("meta.conf_index")
        df_file = Path(file)
        gp_file = df_file.parent / f"{df_file.stem}_gp_{sample_percentage}_{phase_str}.pkl"

        # y = np.array(list(phase_data["ls_eval/returns"]))  # (num_confs * num_seeds, num_evals) = (256 * 5, 20)
        conf_groups = phase_data.groupby(["meta.conf_index", "ls.learning_rate", "ls.gamma"])
        y = np.array(list(conf_groups["ls_eval/returns"].sum()))  # sum does concat (256, 100)
        y_means = np.mean(y, axis=1)  # (256,)
        y_stds = np.std(y, axis=1)
        x = np.array(list(conf_groups.groups.keys()))[:, 1:]  # ls.learning_rate, ls.gamma
        # scale to interval [0, 1], undoing the logarithmic biases:
        x[:, 0] = _invert_ls_scaling(x[:, 0], 1000, 0.0001, 0.1)  # TODO get inversion params from data
        x[:, 1] = _invert_ls_scaling(1 - x[:, 1], 5, 0.0001, 0.2)
        # use only a percentage of evals if requested:
        # if 0 <= sample_percentage and sample_percentage < 100:
        #     rng = np.random.default_rng(0)
        #     rng.shuffle(y, axis=1)
        #     num_samples = round(y.shape[-1] * sample_percentage * 0.01)
        #     y = y[:, :num_samples]
        #     print(y.shape)
        # x = np.repeat(np.array(phase_data[["ls.learning_rate", "ls.gamma"]]), y.shape[-1], axis=0)
        # y = y.flatten()  # (25600,)

        # load or fit gaussian process:
        if gp_file.exists() and not retrain:
            with open(gp_file, "rb") as f:
                gpr: GaussianProcessRegressor  # type: ignore[no-any-unimported]
                orig_score: float
                gpr, orig_score = pickle.load(f)
        else:
            # RBF with per-dimension length scales
            # gpr = GaussianProcessRegressor(normalize_y=True, random_state=0)
            gpr = GaussianProcessRegressor(
                ConstantKernel(1.0, constant_value_bounds="fixed")
                * Matern([1.0, 1.0], length_scale_bounds=(0.01, 100), nu=0.5),
                normalize_y=True,
                alpha=1e-3 * y_stds**2,
                random_state=0,
            )
            # gpr = GaussianProcessRegressor(RBF([1.0, 1.0]), alpha=y_stds, normalize_y=True, random_state=0)
            gpr.fit(x, y_means)
            orig_score = gpr.score(x, y_means)
            with open(gp_file, "wb") as f:
                pickle.dump((gpr, orig_score), f)

        score = gpr.score(x, y_means)
        assert math.isclose(score, orig_score), "R²-Score of loaded GPR differs from original score!"
        grid_x, grid_y = np.meshgrid(np.linspace(0, 1, num=grid_size), np.linspace(0, 1, num=grid_size))
        grid_x = grid_x.flatten()
        grid_y = grid_y.flatten()
        grid = np.stack((grid_x, grid_y), axis=1)

        preds_mean, preds_std = gpr.predict(grid, return_std=True)
        print(f"{score=} {np.mean(preds_std)=} {gpr.kernel_=}")
        # plot_x = np.log10(phase_data["ls.learning_rate"])
        # plot_y = phase_data["ls.gamma"]
        # plot_z = preds_mean
        zlabel = "GP Mean"
        title = f"GP mean, fitted on mean performance data for {phase_str}"

        ax = fig.add_subplot(1, df["meta.phase"].nunique(), i + 1, projection="3d")
        for z, opacity in ((preds_mean, 1.0), (preds_mean + preds_std, 0.5), (preds_mean - preds_std, 0.5)):
            cmap = "viridis" if opacity == 1.0 else None
            color = (0.5, 0.5, 0.5, opacity) if opacity < 1.0 else None
            ax.plot_surface(
                grid_x.reshape(grid_size, grid_size),
                grid_y.reshape(grid_size, grid_size),
                z.reshape(grid_size, grid_size),
                cmap=cmap,
                color=color,
                edgecolor="none",
                shade=False,
            )
        if viz_samples:
            # ax.scatter(x[:, 0], x[:, 1], y_means)
            num_evals = np.array(list(phase_data["ls_eval/returns"])).shape[1]  # hacky
            ax.scatter(
                _invert_ls_scaling(np.repeat(np.array(phase_data["ls.learning_rate"]), num_evals), 1000, 0.0001, 0.1),
                _invert_ls_scaling(1 - np.repeat(np.array(phase_data["ls.gamma"]), num_evals), 5, 0.0001, 0.2),
                y.flatten(),
                alpha=0.1,
            )
        # ax.xaxis.set_major_formatter(mticker.FuncFormatter(_log_tick_formatter))
        ax.set_zlim3d(0, 500)
        ax.set_xlabel("Learning Rate", fontsize=12)
        ax.set_ylabel("Gamma", fontsize=12)
        ax.set_zlabel(zlabel, fontsize=12)
        ax.set_title(title, fontsize=16)
    fig.tight_layout()
    plt.show()


def _log_tick_formatter(val: Any, pos: Any = None) -> Any:
    return "{:.2e}".format(10**val)
