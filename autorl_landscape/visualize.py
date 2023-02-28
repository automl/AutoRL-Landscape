from typing import Any, Callable

from copy import deepcopy
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpecFromSubplotSpec
from pandas import DataFrame
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.inspection import PartialDependenceDisplay

from autorl_landscape.analyze.visualization import Visualization
from autorl_landscape.ls_models.ls_model import LSModel
from autorl_landscape.util.data import read_wandb_csv
from autorl_landscape.util.grid_space import grid_space_nd

# font sizes:
TITLE_FSIZE = 24
LABEL_FSIZE = 15
TICK_FSIZE = 13
LEGEND_FSIZE = 15
# TITLE_FSIZE = 1
# LABEL_FSIZE = 1
# TICK_FSIZE = 1
# LEGEND_FSIZE = 1
# plt.rc("legend", fontsize=LEGEND_FSIZE)

LABELPAD = 10

TICK_POS = np.linspace(0, 1, 4)
TICK_POS_RETURN = np.linspace(0, 1, 6)

CMAP = sns.color_palette("rocket", as_cmap=True)
CMAP_DIVERGING = {
    "cmap": sns.color_palette("vlag", as_cmap=True),
    "norm": TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=3.5),
}
CMAP_DISCRETIZED = {
    "cmap": sns.color_palette("vlag", as_cmap=True),
    "norm": BoundaryNorm(boundaries=[-0.5, 0.75, 1.25, 4.0], ncolors=255),
}

FIGSIZES = {
    "maps": (30, 20),
    "modalities": (32, 12),
    "graphs": (36, 12),
}


def visualize_nd(
    model: LSModel, fig: Figure, sub_gs: Any, grid_length: int, viz_group: str, phase_str: str
) -> tuple[list[str], list[float]]:
    """Visualize an analysis of the landscape model."""
    # prettify phase title:
    phase_i = int(phase_str.split("_")[-1])
    phase_title = f"Phase {phase_i + 1}"
    match viz_group:
        case "maps":  # x0x1 -> y
            add_model_visualization(model, grid_length)
            titles = list(dict.fromkeys([v.title for v in model.get_viz_infos() if v.viz_group == viz_group]))
            x01s = list(combinations(model.get_ls_dim_names(), 2))  # all 2-combinations of x (ls) dimensions

            nrows = 1 + 1 + len(titles)  # phase title, combined 3d, upper map, middle map, lower map
            height_ratios = [0.25, 1.25] + [1.0] * len(titles)
            ncols = len(x01s)
            width_ratios = [1.0] * len(x01s)
            gs = GridSpecFromSubplotSpec(
                nrows, ncols, subplot_spec=sub_gs, height_ratios=height_ratios, width_ratios=width_ratios
            )

            # main plots:
            for i, title in enumerate(titles, start=2):  # rows
                label_x0 = i == (nrows - 1)  # label on last row
                for j, (x0, x1) in enumerate(x01s):  # columns
                    ax = fig.add_subplot(gs[i, j])
                    viz_single_x0x1y(
                        model,
                        ax=ax,
                        x0=x0,
                        x1=x1,
                        match_titles=[title],
                        grid_length=grid_length,
                        projection="",
                        label_x0=label_x0,
                        label_x1=True,
                    )

            # titles:
            row_titles = ["Combined Landscape Model"] + titles
            ax = fig.add_subplot(gs[0, :])
            ax.text(0.5, 0.5, phase_title, ha="center", va="center", fontsize=TITLE_FSIZE)
            ax.axis("off")

            # 3d plots on top:
            for j, (x0, x1) in enumerate(x01s):
                ax = fig.add_subplot(gs[1, j], projection="3d")
                viz_single_x0x1y(
                    model,
                    ax,
                    [n.capitalize() + " Surface" for n in model.model_layer_names],
                    x0,
                    x1,
                    grid_length,
                    projection="3d",
                    label_x0=True,
                    label_x1=True,
                )

            # colorbar legend:
            # colorbar_ax = fig.add_subplot(gs[:, -1])
            # colorbar_ax.set_yticks(TICK_POS, [self.y_info.tick_formatter(x, None) for x in TICK_POS])
            # colorbar_ax.set_ylabel(self.y_info.name)
        case "modalities":
            titles = list(dict.fromkeys([v.title for v in model.get_viz_infos() if v.viz_group == viz_group]))
            x01s = list(combinations(model.get_ls_dim_names(), 2))  # all 2-combinations of x (ls) dimensions
            print(titles)

            nrows = 1 + len(titles)  # phase title, mosaic, discretized mosaic
            height_ratios = [0.25] + [1.0] * len(titles)
            ncols = len(x01s)
            width_ratios = [1.0] * len(x01s)
            gs = GridSpecFromSubplotSpec(
                nrows, ncols, subplot_spec=sub_gs, height_ratios=height_ratios, width_ratios=width_ratios
            )

            for i, title in enumerate(titles, start=1):  # rows
                for j, (x0, x1) in enumerate(x01s):  # columns
                    ax = fig.add_subplot(gs[i, j])
                    viz_single_x0x1y(
                        model,
                        ax=ax,
                        x0=x0,
                        x1=x1,
                        match_titles=[title],
                        grid_length=grid_length,
                        projection="",
                        label_x0=(i == 2),
                        label_x1=True,
                    )

            # titles on the left:
            row_titles = titles
            ax = fig.add_subplot(gs[0, :])
            ax.text(0.5, 0.5, phase_title, ha="center", va="center", fontsize=TITLE_FSIZE)
            ax.axis("off")

            # colorbar_ax.set_yticks(TICK_POS, [self.y_info.tick_formatter(x, None) for x in TICK_POS])
        case "graphs":  # x0 -> y
            x0s = model.get_ls_dim_names()
            # get unique titles, keeping first appearance order as in self._viz_infos:

            nrows = 1 + len(model.model_layer_names)  # title, PDP + ICE
            height_ratios = [0.25] + [1.0] * len(model.model_layer_names)
            ncols = len(x0s)  # title, *x01s, color legend
            width_ratios = [1.0] * len(x0s)
            gs = GridSpecFromSubplotSpec(
                nrows,
                ncols,
                subplot_spec=sub_gs,
                height_ratios=height_ratios,
                width_ratios=width_ratios,
                wspace=0,
                hspace=0.1,
            )

            # dirty hack so that the sklearn PDP method is happy. Defines an estimator that just uses a specific
            # surface of this LSModel:
            class LayerEstimator(BaseEstimator, RegressorMixin):
                def __init__(self, surface: Callable) -> None:
                    self.surface_ = surface

                def fit(self, x: Any, y: Any) -> None:
                    pass

                def predict(self, x: Any) -> Any:
                    # return self.model_.get_middle(x)
                    return self.surface_(x)

            row_titles = []

            # main plots:
            for i, model_layer_name in enumerate(model.model_layer_names, start=1):
                label_x = i == 3
                label_y = True
                surface = getattr(model, f"get_{model_layer_name}")
                title = f"{model_layer_name.capitalize()} Surface"
                row_titles.append(title)
                model_ = LayerEstimator(surface)
                model_.fit(None, None)
                for j, x0 in enumerate(x0s):  # columns
                    ax = fig.add_subplot(gs[i, j])
                    # x = self.x[:, j]
                    PartialDependenceDisplay.from_estimator(model_, model.x, [j], ax=ax, kind="both")
                    ax = plt.gca()

                    ax.set_aspect("equal", "box")

                    if label_x:
                        x0_ticks = [model.get_dim_info(x0).tick_formatter(x, None) for x in TICK_POS]
                        ax.set_xticks(TICK_POS, x0_ticks, fontsize=TICK_FSIZE)
                        ax.xaxis.set_tick_params(rotation=30)
                        ax.set_xlabel(x0, fontsize=LABEL_FSIZE)
                    else:
                        ax.set_xticks([])
                        ax.set_xlabel("")

                    if label_y:
                        y_ticks = [model.y_info.tick_formatter(x, None) for x in TICK_POS_RETURN]
                        ax.set_yticks(TICK_POS_RETURN, y_ticks, fontsize=TICK_FSIZE)
                        ax.set_ylabel(model.y_info.name, fontsize=LABEL_FSIZE)
                    else:
                        ax.set_yticks([])
                        ax.set_ylabel("")
                    ax.set_ylim(0, 1)
                    label_y = False

            ax = fig.add_subplot(gs[0, :])
            ax.text(0.5, 0.5, phase_title, ha="center", va="center", fontsize=TITLE_FSIZE)
            ax.axis("off")
        case _:
            raise NotImplementedError
    return row_titles, height_ratios


def viz_single_x0x1y(
    model: LSModel,
    ax: plt.Axes,
    match_titles: list[str],
    x0: str,
    x1: str,
    grid_length: int,
    projection: str,
    label_x0: bool = False,
    label_x1: bool = False,
) -> None:
    """Make a single plot for a big visualization."""

    def _to_imshow_x(x):
        return (grid_length - 1) * x

    # plot all Visualizations matching the titles given:
    for viz in [v for v in model.get_viz_infos() if v.title in match_titles]:
        # data = self.unnormalize(viz.xy_norm)
        data = viz.xy_norm
        data_col_names: list[str] = list(data.keys())
        y_col_name = data_col_names[-1]
        match viz.viz_type:
            case "scatter":
                if projection == "3d":
                    pass  # too confusing
                    # artist = ax.scatter(data[x0], data[x1], data[y_col_name])
                else:
                    ax.scatter(_to_imshow_x(data[x0]), _to_imshow_x(data[x1]), **viz.kwargs, zorder=10)
            case "map" | "contour":
                pt = data.pivot_table(values=y_col_name, index=x0, columns=x1, aggfunc=np.mean)
                pt_T = data.pivot_table(values=y_col_name, index=x1, columns=x0, aggfunc=np.mean)
                if projection == "3d":
                    kwargs_ = deepcopy(viz.kwargs)
                    if viz.title in ["Upper Surface", "Lower Surface"]:
                        del kwargs_["cmap"]
                    plot_surface_(ax, pt, kwargs_)
                else:
                    contourf = None
                    kwargs_ = {k: v for k, v in viz.kwargs.items() if k != "color"}
                    kwargs__ = {k: v for k, v in kwargs_.items() if k != "cmap"}
                    if viz.viz_type == "map":
                        ax.imshow(pt_T, origin="lower", **kwargs_, zorder=0)
                    else:
                        contourf = ax.contourf(pt_T, **kwargs_, zorder=0)
                        ax.contour(pt_T, **kwargs__, zorder=0, colors="black", linewidths=0.5)
                        ax.set_aspect("equal", "box")

                    # colorbar legend:
                    if viz.title == "Unimodality":
                        cbar = plt.colorbar(mappable=ScalarMappable(**CMAP_DIVERGING), ax=ax)
                        cbar.ax.set_ylabel(r"$\Phi$", fontsize=LABEL_FSIZE, labelpad=LABELPAD)
                        cbar.ax.yaxis.set_tick_params(labelsize=TICK_FSIZE)
                    elif viz.title == "Unimodality (discretized)":
                        cbar = plt.colorbar(mappable=ScalarMappable(**CMAP_DISCRETIZED), ax=ax)
                        cbar.ax.set_ylabel(
                            "multimodal        uncategorized        unimodal",
                            fontsize=0.8 * LABEL_FSIZE,
                            labelpad=LABELPAD,
                        )
                        cbar.ax.set_yticks([])
                    elif contourf is not None:
                        cbar = plt.colorbar(mappable=ScalarMappable(norm=None, cmap=CMAP), ax=ax)
                        # cbar = plt.colorbar(mappable=contourf, format=self.y_info.tick_formatter, ax=ax)
                        cbar.ax.set_yticks(
                            TICK_POS_RETURN,
                            [model.y_info.tick_formatter(x, None) for x in TICK_POS_RETURN],
                            fontsize=TICK_FSIZE,
                        )
                        cbar.ax.set_ylabel(y_col_name, labelpad=LABELPAD, fontsize=LABEL_FSIZE)
            case _:
                pass
                raise NotImplementedError

    # ticks:
    if projection == "3d":
        x0_ticks = [model.get_dim_info(x0).tick_formatter(x, None) for x in TICK_POS]
        ax.set_xticks(TICK_POS, x0_ticks, fontsize=TICK_FSIZE)
        ax.set_xlabel(x0, labelpad=LABELPAD, fontsize=LABEL_FSIZE)
        x1_ticks = [model.get_dim_info(x1).tick_formatter(x, None) for x in TICK_POS]
        ax.set_yticks(TICK_POS, x1_ticks, fontsize=TICK_FSIZE)
        ax.set_ylabel(x1, labelpad=LABELPAD, fontsize=LABEL_FSIZE)
        y_ticks = [model.y_info.tick_formatter(x, None) for x in TICK_POS_RETURN]
        ax.set_zticks(TICK_POS_RETURN, y_ticks, fontsize=TICK_FSIZE)
        ax.set_zlabel(y_col_name, labelpad=LABELPAD, fontsize=LABEL_FSIZE)
        ax.set_zlim3d(0, 1)
    else:
        if label_x0:
            x0_ticks = [model.get_dim_info(x0).tick_formatter(x, None) for x in TICK_POS]
            ax.set_xticks(_to_imshow_x(TICK_POS), x0_ticks, fontsize=TICK_FSIZE, rotation=30)
            # ax.xaxis.set_tick_params(rotation=30)
            ax.set_xlabel(x0, fontsize=LABEL_FSIZE)
        else:
            ax.set_xticks([])
        if label_x1:
            x1_ticks = [model.get_dim_info(x1).tick_formatter(x, None) for x in TICK_POS]
            ax.set_yticks(_to_imshow_x(TICK_POS), x1_ticks, fontsize=TICK_FSIZE)
            ax.set_ylabel(x1, fontsize=LABEL_FSIZE)
        else:
            ax.set_yticks([])
    return


def add_model_visualization(model: LSModel, grid_length: int) -> None:
    """Add visualizations to the model which visualize just the model itself."""
    num_dims = len(model.dim_info)
    grid = grid_space_nd(num_dims, grid_length).reshape(-1, num_dims)
    for model_layer_name in model.model_layer_names:
        title = f"{model_layer_name.capitalize()} Surface"
        func = getattr(model, f"get_{model_layer_name}")
        if model_layer_name == "middle":
            kwargs = {"cmap": CMAP, "vmin": 0, "vmax": 1}
        else:
            kwargs = {
                "color": (0.5, 0.5, 0.5, 0.3),
                "vmin": 0,
                "vmax": 1,
                "cmap": CMAP,
            }

        model.add_viz_info(
            Visualization(
                title,
                "contour",
                "maps",
                model.build_df(grid, func(grid), "ls_eval/returns"),
                kwargs,
            )
        )

    model.add_viz_info(
        Visualization(
            "Interquantile Height",
            "contour",
            "maps",
            model.build_df(grid, model.get_upper(grid) - model.get_lower(grid), "height of interquantile space"),
            {
                "color": (0.5, 0.5, 0.5, 0.3),
                "vmin": 0,
                "vmax": 1,
                "cmap": CMAP,
            },
        )
    )

    if model.best_conf is not None:
        ancestor_x = np.array(model.best_conf[model.get_ls_dim_names()], dtype=model.dtype).reshape(1, -1)
        ancestor_y = np.array(np.mean(np.array(model.best_conf[model.y_info.name]))).reshape(1, 1)

        # to unit cube:
        for i in range(len(model.dim_info)):
            transformer = model.dim_info[i].ls_to_unit
            ancestor_x[:, i] = transformer(ancestor_x[:, i])
        ancestor_y = model.y_info.ls_to_unit(ancestor_y)

        model._viz_infos.extend(
            [
                Visualization(
                    "Middle Surface",
                    "scatter",
                    "maps",
                    model.build_df(ancestor_x, ancestor_y, "ls_eval/returns"),
                    {
                        "color": "white",
                        "marker": "*",
                        "edgecolors": "black",
                        "s": 100,
                        "label": "best configuration",
                    },
                ),
            ]
        )


def visualize_data_samples(file: str) -> None:
    """Visualize grid of samples, read from a file."""
    df = read_wandb_csv(file)
    # phase_data = df[df["meta.phase"] == "phase_0"]
    fig = plt.figure(figsize=(8, 8))
    fig.tight_layout()
    ax = plt.axes()
    ax.scatter(df["ls.learning_rate"], df["ls.gamma"], color="#2369BC")
    ax.set_xscale("log")
    ax.set_xlabel("learning rate", fontsize=LABEL_FSIZE)
    ax.set_ylabel("gamma", fontsize=LABEL_FSIZE)
    ax.xaxis.set_tick_params(labelsize=TICK_FSIZE)
    ax.yaxis.set_tick_params(labelsize=TICK_FSIZE)
    # plt.show()

    fig_file_part = "images/viz_samples"
    Path(fig_file_part).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{fig_file_part}.pdf", bbox_inches="tight")


def plot_surface_(ax: Axes, pt: DataFrame, kwargs: dict[str, Any]) -> Artist:
    """TODO."""
    grid_length = pt.values.shape[0]
    grid = grid_space_nd(2, grid_length)
    grid_x0 = grid[:, :, 0]
    grid_x1 = grid[:, :, 1]

    return ax.plot_surface(grid_x0, grid_x1, pt.values, **kwargs)


def _log_tick_formatter(val: Any, pos: Any = None) -> Any:
    return "{:.2e}".format(10**val)
