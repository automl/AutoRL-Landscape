from typing import Any

from collections.abc import Callable
from copy import deepcopy
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from omegaconf import DictConfig, OmegaConf
from pandas import DataFrame
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.inspection import PartialDependenceDisplay

from autorl_landscape.analyze.visualization import Visualization
from autorl_landscape.ls_models.ls_model import LSModel
from autorl_landscape.util.data import read_wandb_csv
from autorl_landscape.util.grid_space import grid_space_nd
from autorl_landscape.util.ls_sampler import construct_ls

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
Y_SCALED = 1.0  # make this lower if visualization axis limits are too big in y direction
ZOOM_3D = 0.9

LABELPAD = 10

TICK_POS = np.linspace(0, 1, 4)
TICK_POS_RETURN = np.linspace(0, 1, 6)

CMAP = {
    "cmap": sns.color_palette("rocket", as_cmap=True),
    "norm": None,
}
CMAP_DIVERGING = {
    "cmap": sns.color_palette("vlag", as_cmap=True),
    "norm": TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=3.5),
}
CMAP_CRASHED = {
    "cmap": LinearSegmentedColormap.from_list("", ["#15161e", "#db4b4b"]),  # Tokyo!
    "norm": TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0),
}
CMAP_DISCRETIZED = {
    "cmap": sns.color_palette("vlag", as_cmap=True),
    "norm": BoundaryNorm(boundaries=[-0.5, 0.75, 1.25, 4.0], ncolors=255),
}

FIGSIZES = {
    3: {
        "maps": (70, 18),
        "modalities": (70, 7),
        "graphs": (70, 18),
        "crashes": (70, 4),
    },
    4: {
        "maps": (90, 18),
        "modalities": (90, 7),
        "graphs": (90, 18),
        "crashes": (90, 4),
    },
    "cherry_picked": (4, 4),
}


def _middle_of_boundaries(boundaries: list[float]) -> list[float]:
    """[1, 2, 3, 5] -> [1.5, 2.5, 4]."""
    middles = []
    for lower, upper in zip(boundaries, boundaries[1:], strict=False):
        middles.append((lower + upper) / 2)
    return middles


def _fix_dim_name(dim_name: str) -> str:
    """Fix landscape dimension names for file names.

    Example:
        ls.gamma -> gamma
        abc.def.learning_rate -> learning_rate
        exploration_rate -> exploration_rate
    """
    if "." in dim_name:
        return dim_name.split(".")[-1]
    return dim_name


def _add_colorbar(ax: Axes, cmap: dict[str, Any]) -> Colorbar:
    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes("right", size="5%", pad=0.05)
    return plt.colorbar(mappable=ScalarMappable(**cmap), cax=cbar_ax)


def is_picked(picks: list[tuple[int, int]], phase_number: int, row: int, col: int, ncols: int) -> bool:
    """Calculate global position of a plot in a figure, check if that position is selected. No picks means pick all.

    Args:
        picks: List of selected positions.
        phase_number: Starts with 1. Determines in which big column the plot is positioned.
        row: Local to phase.
        col: Local to phase.
        ncols: Number of columns for this phase's plots.
    """
    if len(picks) == 0:
        return True
    global_row = row
    global_col = (phase_number - 1) * ncols + col
    return (global_row, global_col) in picks


def visualize_cherry_picks(
    model: LSModel, picks: list[tuple[int, int]], grid_length: int, viz_group: str, phase_index: int, save_base: Path
) -> None:
    """Visualize specified plots of an analysis of the landscape model."""

    def make_figure(match_titles: list[str], name: str, x0: str, x1: str, projection: str | None = None) -> None:
        fig, ax = plt.subplots(figsize=FIGSIZES["cherry_picked"], subplot_kw={"projection": projection})
        viz_single_x0x1y(model, ax, match_titles, x0, x1, grid_length, projection, True, True)
        x0_ = _fix_dim_name(x0)
        x1_ = _fix_dim_name(x1)
        if not save_base.is_dir():
            save_base.mkdir(parents=True, exist_ok=True)
        bbox_setting = "tight" if projection is None else None
        fig.savefig(
            save_base / f"{model.get_model_name()}{viz_group}_{name}_phase-{phase_index}_{x0_}_{x1_}.pdf",
            bbox_inches=bbox_setting,
        )

    match viz_group:
        case "maps":  # x0x1 -> y
            add_model_visualization(model, grid_length)
            titles = list(dict.fromkeys([v.title for v in model.get_viz_infos() if v.viz_group == viz_group]))
            x01s = list(combinations(model.get_ls_dim_names(), 2))  # all 2-combinations of x (ls) dimensions

            ncols = len(x01s)

            # main plots:
            for i, title in enumerate(titles, start=1):  # rows
                for j, (x0, x1) in enumerate(x01s):  # columns
                    if is_picked(picks, phase_index, i, j, ncols):
                        name = title.split(" ")[0].lower()
                        make_figure([title], name, x0, x1)

            # 3d plots on top:
            i = 0
            title = "combined"
            for j, (x0, x1) in enumerate(x01s):
                if is_picked(picks, phase_index, i, j, ncols):
                    match_titles = [n.capitalize() + " Surface" for n in model.model_layer_names]
                    make_figure(match_titles, "combined", x0, x1, "3d")

        case "modalities":
            titles = list(dict.fromkeys([v.title for v in model.get_viz_infos() if v.viz_group == viz_group]))
            x01s = list(combinations(model.get_ls_dim_names(), 2))  # all 2-combinations of x (ls) dimensions

            ncols = len(x01s)

            for i, title in enumerate(titles):  # rows
                for j, (x0, x1) in enumerate(x01s):  # columns
                    if is_picked(picks, phase_index, i, j, ncols):
                        name = title.lower().replace(" ", "_").replace("(", "").replace(")", "")
                        make_figure([title], name, x0, x1)

        case "crashes":
            titles = list(dict.fromkeys([v.title for v in model.get_viz_infos() if v.viz_group == viz_group]))
            x01s = list(combinations(model.get_ls_dim_names(), 2))  # all 2-combinations of x (ls) dimensions

            ncols = len(x01s)

            for i, title in enumerate(titles):  # rows
                for j, (x0, x1) in enumerate(x01s):  # columns
                    if is_picked(picks, phase_index, i, j, ncols):
                        name = title.lower()
                        make_figure([title], name, x0, x1)

        case "graphs":  # x0 -> y
            x0s = model.get_ls_dim_names()
            # get unique titles, keeping first appearance order as in self._viz_infos:

            ncols = len(x0s)

            # dirty hack so that the sklearn PDP method is happy. Defines an estimator that just uses a specific
            # surface of this LSModel:
            class LayerEstimator(BaseEstimator, RegressorMixin):
                def __init__(self, surface: Callable[[Any], Any]) -> None:
                    self.surface_ = surface

                def fit(self, x: Any, y: Any) -> None:
                    pass

                def predict(self, x: Any) -> Any:
                    # return self.model_.get_middle(x)
                    return self.surface_(x)

            # main plots:
            for i, model_layer_name in enumerate(model.model_layer_names):
                surface = getattr(model, f"get_{model_layer_name}")
                title = f"{model_layer_name.capitalize()} Surface"
                name = model_layer_name
                model_ = LayerEstimator(surface)
                model_.fit(None, None)
                for j, x0 in enumerate(x0s):  # columns
                    if is_picked(picks, phase_index, i, j, ncols):
                        fig, ax = plt.subplots(figsize=FIGSIZES["cherry_picked"])
                        PartialDependenceDisplay.from_estimator(model_, model.x, [j], ax=ax, kind="both")
                        ax = plt.gca()  # Important!

                        ax.set_aspect("equal", "box")

                        x0_ticks = [model.get_dim_info(x0).tick_formatter(x, None) for x in TICK_POS]
                        ax.set_xticks(TICK_POS, x0_ticks, fontsize=TICK_FSIZE)
                        ax.xaxis.set_tick_params(rotation=30)
                        ax.set_xlabel(x0, fontsize=LABEL_FSIZE)
                        y_ticks = [model.y_info.tick_formatter(x, None) for x in TICK_POS_RETURN]
                        ax.set_yticks(TICK_POS_RETURN, y_ticks, fontsize=TICK_FSIZE)
                        ax.set_ylabel(model.y_info.name, fontsize=LABEL_FSIZE)
                        ax.set_ylim(0, Y_SCALED)

                        x0_ = _fix_dim_name(x0)
                        if not save_base.is_dir():
                            save_base.mkdir(parents=True, exist_ok=True)
                        bbox_setting = "tight"
                        fig.savefig(
                            save_base / f"{model.get_model_name()}{viz_group}_{name}_phase-{phase_index}_{x0_}.pdf",
                            bbox_inches=bbox_setting,
                        )
        case _:
            raise NotImplementedError


def visualize_nd(
    model: LSModel, fig: Figure, sub_gs: Any, grid_length: int, viz_group: str, phase_index: int
) -> tuple[list[str], list[float]]:
    """Visualize an analysis of the landscape model."""
    # prettify phase title:
    phase_title = f"Phase {phase_index}"
    match viz_group:
        case "maps":  # x0x1 -> y
            add_model_visualization(model, grid_length)
            titles = list(dict.fromkeys([v.title for v in model.get_viz_infos() if v.viz_group == viz_group]))
            x01s = list(combinations(model.get_ls_dim_names(), 2))  # all 2-combinations of x (ls) dimensions

            nrows = 1 + 1 + len(titles)  # phase title, combined 3d, upper map, middle map, lower map
            height_ratios = [0.25, 1.75] + [1.0] * len(titles)
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

        case "modalities":
            titles = list(dict.fromkeys([v.title for v in model.get_viz_infos() if v.viz_group == viz_group]))
            x01s = list(combinations(model.get_ls_dim_names(), 2))  # all 2-combinations of x (ls) dimensions

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

        case "crashes":
            titles = list(dict.fromkeys([v.title for v in model.get_viz_infos() if v.viz_group == viz_group]))
            x01s = list(combinations(model.get_ls_dim_names(), 2))  # all 2-combinations of x (ls) dimensions

            nrows = 1 + len(titles)  # phase title, crashes
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
                        label_x0=True,
                        label_x1=True,
                    )

            # titles on the left:
            row_titles = titles
            ax = fig.add_subplot(gs[0, :])
            ax.text(0.5, 0.5, phase_title, ha="center", va="center", fontsize=TITLE_FSIZE)
            ax.axis("off")

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
                    ax.set_ylim(0, Y_SCALED)
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
    projection: str | None,
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
                        cbar = _add_colorbar(ax, CMAP_DIVERGING)
                        cbar.ax.set_ylabel(r"$\Phi$", fontsize=LABEL_FSIZE, labelpad=LABELPAD)
                        cbar.ax.yaxis.set_tick_params(labelsize=TICK_FSIZE)

                    elif viz.title == "Crashes":
                        cbar = _add_colorbar(ax, CMAP_CRASHED)
                        cbar.ax.set_ylabel("Crash Chance", fontsize=LABEL_FSIZE, labelpad=LABELPAD)
                        cbar.ax.yaxis.set_tick_params(labelsize=TICK_FSIZE)

                    elif viz.title == "Unimodality (discretized)":
                        cbar = _add_colorbar(ax, CMAP_DISCRETIZED)
                        cbar.ax.yaxis.set_minor_locator(
                            ticker.FixedLocator(_middle_of_boundaries(CMAP_DISCRETIZED["norm"].boundaries))
                        )
                        cbar.ax.yaxis.set_minor_formatter(ticker.FixedFormatter(["MM", "N/A", "UM"]))
                        cbar.ax.set_yticks([])

                    elif contourf is not None:
                        cbar = _add_colorbar(ax, CMAP)
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
        ax.set_zlim3d(0, Y_SCALED)
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
            kwargs = {"cmap": CMAP["cmap"], "vmin": 0, "vmax": 1}
        else:
            kwargs = {
                "color": (0.5, 0.5, 0.5, 0.3),
                "vmin": 0,
                "vmax": 1,
                "cmap": CMAP["cmap"],
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
            "Interquantile Space Height",
            "contour",
            "maps",
            model.build_df(grid, model.get_upper(grid) - model.get_lower(grid), "upper - lower"),
            {
                "color": (0.5, 0.5, 0.5, 0.3),
                "vmin": 0,
                "vmax": 1,
                "cmap": CMAP["cmap"],
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

    fig_file_part = "figures/viz_samples"
    Path(fig_file_part).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{fig_file_part}.pdf", bbox_inches="tight")


def visualize_landscape_spec(conf: DictConfig) -> None:
    """Visualize landscape spec to inspect the sampled patterns.

    Args:
        conf: Hydra configuration
    """
    df = construct_ls(conf)
    fig = plt.figure(figsize=(16, 16))
    fig.tight_layout()

    ax = fig.add_subplot(1, 5, 1, projection="3d")
    ax.scatter(df["learning_rate"], df["gamma"], df["exploration_final_eps"])
    ax.set_xlabel("learning rate", fontsize=LABEL_FSIZE)
    ax.set_ylabel("gamma", fontsize=LABEL_FSIZE)
    ax.set_zlabel("exploration rate", fontsize=LABEL_FSIZE)

    for i, dim_name in enumerate(["learning_rate", "gamma", "exploration_final_eps"], start=2):
        ax = fig.add_subplot(1, 5, i)
        # ax.scatter(df["learning_rate"], df["gamma"], df["tau"])
        ax.scatter(df[dim_name], np.zeros_like(df[dim_name]))
        ax.set_xlabel(dim_name, fontsize=LABEL_FSIZE)
    ax = fig.add_subplot(1, 5, 5)
    ax.text(
        0.1, 0.5, f"num_confs: {conf.num_confs}\n" + str(OmegaConf.to_yaml(conf.ls)), va="center", fontsize=LABEL_FSIZE
    )
    plt.show()


def plot_surface_(ax: Axes, pt: DataFrame, kwargs: dict[str, Any]) -> Artist:
    """TODO."""
    grid_length = pt.values.shape[0]
    grid = grid_space_nd(2, grid_length)
    grid_x0 = grid[:, :, 0]
    grid_x1 = grid[:, :, 1]

    artist = ax.plot_surface(grid_x0, grid_x1, pt.values, **kwargs)
    ax.set_box_aspect(aspect=None, zoom=ZOOM_3D)
    return artist


def _log_tick_formatter(val: Any, pos: Any = None) -> Any:
    return "{:.2e}".format(10**val)
