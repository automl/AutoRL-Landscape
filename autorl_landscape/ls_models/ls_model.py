from typing import Any, Callable

from ast import literal_eval
from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.ticker import FuncFormatter
from numpy.typing import NDArray
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.inspection import PartialDependenceDisplay

from autorl_landscape.util.compare import iqm
from autorl_landscape.util.grid_space import grid_space_nd
from autorl_landscape.util.ls_sampler import DimInfo
from autorl_landscape.visualize import plot_surface_

TICK_POS = np.linspace(0, 1, 4)
TICK_POS_CBAR = np.linspace(0, 1, 5)
CMAP = sns.color_palette("rocket", as_cmap=True)
CMAP_DIVERGING = {
    "cmap": sns.color_palette("vlag", as_cmap=True),
    "norm": TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=3.5),
}
CMAP_DISCRETIZED = {
    "cmap": sns.color_palette("vlag", as_cmap=True),
    "norm": BoundaryNorm(boundaries=[-0.5, 0.75, 1.25, 4.0], ncolors=255),
}
# PROJECTION = "3d"
PROJECTION = None


@dataclass
class Visualization:
    """Saves information for a plot."""

    title: str
    """Title for the plot (Visualizations with matching titles are drawn on the same `Axes`)"""
    viz_type: str
    """scatter, trisurf, etc."""
    viz_group: str
    """For allocating a Visualization to an image (combination of Visualizations)"""
    # x_samples: NDArray[Any]
    # y_samples: NDArray[Any]
    xy_norm: DataFrame
    """DataFrame including y (output) values for some visualization. May omit x values to assume the default, model.x"""
    # label: str
    kwargs: dict[str, Any]


class LSModel(BaseEstimator):
    """A model for the hyperparameter landscape.

    Args:
        data: df with samples of the phase that should be modelled
        dtype: mostly unused.
        y_col: which performance metric should be used
        y_bounds: used for normalizing scores to [0, 1]
        ancestor: optional, for marking the best policy in visualizations
        ci: percentage of performance values of a single configuration that should be inside lower and upper.
    """

    def __init__(
        self,
        data: DataFrame,
        dtype: type,
        y_col: str = "ls_eval/returns",
        y_bounds: tuple[float, float] | None = None,
        best_conf: Series | None = None,
        ci: float = 0.95,
    ) -> None:
        self.dtype = dtype
        self.model_layer_names = ["upper", "middle", "lower"]
        self.best_conf = best_conf

        self.data = data
        self.y_col = y_col
        self.y_bounds = y_bounds
        self.ci = ci

        # mainly for visualization:
        if y_bounds is not None:
            y_dict = {"type": "Float", "lower": y_bounds[0], "upper": y_bounds[1]}
        else:
            y_dict = {"type": "Float", "lower": 0.0, "upper": float(data[0:1]["conf.viz.max_return"])}
        y_info = DimInfo.from_dim_dict(y_col, y_dict, is_y=True)
        assert y_info is not None
        self.y_info = y_info

        # extract ls dimensions info from first row of data:
        dim_dicts: list[dict[str, dict[str, Any]]] = literal_eval(data[0:1]["conf.ls.dims"][0])
        self.dim_info: list[DimInfo] = []
        """DimInfos for each LS dimension, sorted by name"""
        for d in dim_dicts:
            dim_name, dim_dict = next(iter(d.items()))
            di = DimInfo.from_dim_dict(dim_name, dim_dict)

            if di is not None:
                self.dim_info.append(di)
        self.dim_info = sorted(self.dim_info)  # sorts on dim names

        # group runs with the same configuration:
        conf_groups = data.groupby(["meta.conf_index"] + self.get_ls_dim_names())
        # all groups (configurations):
        self.x = np.array(list(conf_groups.groups.keys()), dtype=self.dtype)[:, 1:]
        """(num_confs, num_ls_dims). LS dimensions are sorted by name"""
        # all evaluations (y values) for a group (configuration):
        y = np.array(list(conf_groups[self.y_info.name].sum()), dtype=self.dtype)

        # scale ls dims into [0, 1] interval:
        for i in range(len(self.dim_info)):
            transformer = self.dim_info[i].ls_to_unit
            self.x[:, i] = transformer(self.x[:, i])
        # scale y into [0, 1] interval:
        self.y = self.y_info.ls_to_unit(y)
        """(num_confs, samples_per_conf)"""

        # just all the single evaluation values, not grouped (but still scaled to [0, 1] interval):
        self.x_samples = np.repeat(self.x, self.y.shape[1], axis=0)
        """(num_confs * samples_per_conf, num_ls_dims)"""
        self.y_samples = self.y.reshape(-1, 1)
        """(num_confs * samples_per_conf, 1)"""

        upper_quantile = 1 - ((1 - ci) / 2)
        lower_quantile = 0 + ((1 - ci) / 2)

        # statistical information about each configuration:
        self.y_iqm = iqm(self.y, axis=1).reshape(-1, 1)
        """(num_confs, 1)"""
        # first, select ci quantile:
        self.y_ci_upper = np.quantile(self.y, upper_quantile, method="median_unbiased", axis=1, keepdims=True)
        """(num_confs, 1)"""
        self.y_ci_lower = np.quantile(self.y, lower_quantile, method="median_unbiased", axis=1, keepdims=True)
        """(num_confs, 1)"""

        self._viz_infos: list[Visualization] = [
            Visualization(
                "Raw Return Samples",
                "scatter",
                "graphs",
                self.build_df(self.x_samples, self.y_samples, "ls_eval/returns"),
                {},
                # {"color": "red"},
            )
        ]

    def get_ls_dim_names(self) -> list[str]:
        """Get the list of hyperparameter landscape dimension names."""
        return [di.name for di in self.dim_info]

    def get_dim_info(self, name: str) -> DimInfo | None:
        """Return matching `DimInfo` to a passed name (can be y_info of any dim_info)."""
        if name == self.y_info.name:
            return self.y_info
        for di in self.dim_info:
            if name == di.name:
                return di
        return None

    def get_upper(self, x: NDArray[Any], assimilate_factor: float = 1.0) -> NDArray[Any]:
        """Return an upper estimate of y at the position(s) x."""
        raise NotImplementedError

    def get_middle(self, x: NDArray[Any]) -> NDArray[Any]:
        """Return some middle (mean, interquartile mean, median, etc.) estimate of y at the position(s) x."""
        raise NotImplementedError

    def get_lower(self, x: NDArray[Any], assimilate_factor: float = 1.0) -> NDArray[Any]:
        """Return an lower estimate of y at the position(s) x."""
        raise NotImplementedError

    def _ci_scale(self, x: NDArray[Any], y: NDArray[Any], assimilate_factor: float = 1.0) -> NDArray[Any]:
        """Assimilate passed y values (assumed to come from `get_upper` or `get_lower`) towards the middle values."""
        if assimilate_factor == 1.0:
            return y

        y_middle = self.get_middle(x)
        return assimilate_factor * y + (1 - assimilate_factor) * y_middle

    def add_viz_info(self, viz_info: Visualization) -> None:
        """Add a visualization to this model."""
        self._viz_infos.append(viz_info)

    def get_viz_infos(self) -> list[Visualization]:
        """Return visualization info(s) for data points used for training the model."""
        return self._viz_infos

    def _add_model_viz(self, grid_length: int) -> None:
        num_dims = len(self.dim_info)
        grid = grid_space_nd(num_dims, grid_length).reshape(-1, num_dims)
        for model_layer_name in self.model_layer_names:
            title = f"{model_layer_name.capitalize()} Surface"
            func = getattr(self, f"get_{model_layer_name}")
            if model_layer_name == "middle":
                kwargs = {"cmap": CMAP, "vmin": 0, "vmax": 1}
            else:
                kwargs = {
                    "color": (0.5, 0.5, 0.5, 0.3),
                    "vmin": 0,
                    "vmax": 1,
                    "cmap": CMAP,
                }

            self.add_viz_info(
                Visualization(
                    title,
                    "contour",
                    "maps",
                    self.build_df(grid, func(grid), "ls_eval/returns"),
                    kwargs,
                )
            )

        self.add_viz_info(
            Visualization(
                "Interquantile Height",
                "contour",
                "maps",
                self.build_df(grid, self.get_upper(grid) - self.get_lower(grid), "height of interquantile space"),
                {
                    "color": (0.5, 0.5, 0.5, 0.3),
                    "vmin": 0,
                    "vmax": 1,
                    "cmap": CMAP,
                },
            )
        )

        if self.best_conf is not None:
            ancestor_x = np.array(self.best_conf[self.get_ls_dim_names()], dtype=self.dtype).reshape(1, -1)
            ancestor_y = np.array(np.mean(np.array(self.best_conf[self.y_info.name]))).reshape(1, 1)

            # to unit cube:
            for i in range(len(self.dim_info)):
                transformer = self.dim_info[i].ls_to_unit
                ancestor_x[:, i] = transformer(ancestor_x[:, i])
            ancestor_y = self.y_info.ls_to_unit(ancestor_y)

            self._viz_infos.extend(
                [
                    Visualization(
                        "Middle Surface",
                        "scatter",
                        "maps",
                        self.build_df(ancestor_x, ancestor_y, "ls_eval/returns"),
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

    def unnormalize(self, df: DataFrame) -> DataFrame:
        """Go over the columns in `df` and transform the values back to the original landscape values. Returns a copy.

        Apply the correct `unit_to_ls` transformer if column name matches, otherwise do nothing.
        """
        ret = df.copy(deep=True)
        for col_name in df:
            if col_name == self.y_info.name:
                ret[col_name] = self.y_info.unit_to_ls(ret[col_name])
                break
            for di in self.dim_info:
                if col_name == di.name:
                    ret[col_name] = di.unit_to_ls(ret[col_name])
                    break
        return ret

    def build_df(self, x: NDArray[Any], y: NDArray[Any], y_axis_label: str) -> DataFrame:
        """Helper to construct a `DataFrame` given x points and y readings and a label for y.

        Labels for x are taken from the model.
        """
        assert x.shape[1] == len(self.dim_info)
        return DataFrame(np.concatenate([x, y], axis=1), columns=self.get_ls_dim_names() + [y_axis_label])

    def _visualize_1d(self, grid_length: int, which: str) -> None:
        raise NotImplementedError

    def _visualize_2d(self, grid_length: int) -> None:
        fig = plt.figure(figsize=(16, 10))
        ax = fig.gca()
        ax.set_zlim3d(0, 1)

        grid_x0, grid_x1 = np.meshgrid(np.linspace(0, 1, num=grid_length), np.linspace(0, 1, num=grid_length))
        grid_x0 = grid_x0.flatten()
        grid_x1 = grid_x1.flatten()
        grid = np.stack((grid_x0, grid_x1), axis=1)  # (-1, num_ls_dims = 2)

        # rescale the ls dims (`grid` is unchanged by this):
        # for dim_i, grid_xi in zip(self.get_ls_dim_names(), (grid_x0, grid_x1)):
        # grid_xi = self.dim_info[dim_i].unit_to_ls(grid_xi).reshape(grid_length, grid_length)
        # grid_xi = np.reshape(self.dim_info[dim_i].unit_to_ls(grid_xi), (grid_length, grid_length))
        grid_x0 = grid_x0.reshape((grid_length, grid_length))
        grid_x1 = grid_x1.reshape((grid_length, grid_length))

        # if viz_model:
        for y_, opacity, label in [
            (self.get_upper(grid), 0.5, "modelled upper CI bound"),
            (self.get_middle(grid), 1.0, "modelled mean"),
            (self.get_lower(grid), 0.5, "modelled lower CI bound"),
        ]:
            cmap = "viridis" if opacity == 1.0 else None
            color = (0.5, 0.5, 0.5, opacity) if opacity < 1.0 else None
            # color = "red"
            surface = ax.plot_surface(
                grid_x0,
                grid_x1,
                y_.reshape(grid_length, grid_length),
                cmap=cmap,
                color=color,
                edgecolor="none",
                shade=True,
                label=label,
            )
            _fix_surface_for_legend(surface)

        viz_infos = self.get_viz_infos()
        for viz in viz_infos:
            match viz.viz_type:
                case "scatter":
                    ax.scatter(
                        viz.x_samples[:, 0],
                        viz.x_samples[:, 1],
                        viz.y_samples[:, 0],
                        label=viz.label,
                        **viz.kwargs,
                    )
                case "trisurf":
                    surface = ax.plot_trisurf(
                        viz.x_samples[:, 0],
                        viz.x_samples[:, 1],
                        viz.y_samples[:, 0],
                        label=viz.label,
                        **viz.kwargs,
                        # edgecolor="none",
                        # shade=False,
                    )
                    _fix_surface_for_legend(surface)
                # case "surface":
                #     surface = ax.plot_surface(
                #         viz.x_samples[:, 0],
                #         viz.x_samples[:, 1],
                #         viz.y_samples[:, 0],
                #         label=viz.label,
                #         **viz.kwargs,
                #     )
                #     _fix_surface_for_legend(surface)
                case _:
                    raise NotImplementedError
        ax.set_xlabel(self.dim_info[0].name, fontsize=12)
        ax.set_ylabel(self.dim_info[1].name, fontsize=12)
        ax.set_zlabel(self.y_info.name, fontsize=12)

        ax.xaxis.set_major_formatter(FuncFormatter(self.dim_info[0].tick_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(self.dim_info[1].tick_formatter))
        ax.zaxis.set_major_formatter(FuncFormatter(self.y_info.tick_formatter))
        return

    def visualize_nd(
        self, fig: Figure, sub_gs: Any, grid_length: int, viz_group: str, phase_str: str
    ) -> tuple[list[str], list[float]]:
        """Visualize e.g. with PCA dim reduction, PCP, marginalizing out dimensions."""
        # prettify phase title:
        phase_i = int(phase_str.split("_")[-1])
        phase_title = f"Phase {phase_i + 1}"
        match viz_group:
            case "maps":  # x0x1 -> y
                self._add_model_viz(grid_length)
                titles = list(dict.fromkeys([v.title for v in self.get_viz_infos() if v.viz_group == viz_group]))
                x01s = list(combinations(self.get_ls_dim_names(), 2))  # all 2-combinations of x (ls) dimensions

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
                        self._viz_single_x0x1y(
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
                ax.text(0.5, 0.5, phase_title, ha="center", va="center", fontsize=16)
                ax.axis("off")

                # 3d plots on top:
                for j, (x0, x1) in enumerate(x01s):
                    ax = fig.add_subplot(gs[1, j], projection="3d")
                    self._viz_single_x0x1y(
                        ax,
                        [n.capitalize() + " Surface" for n in self.model_layer_names],
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
                titles = list(dict.fromkeys([v.title for v in self.get_viz_infos() if v.viz_group == viz_group]))
                x01s = list(combinations(self.get_ls_dim_names(), 2))  # all 2-combinations of x (ls) dimensions
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
                        self._viz_single_x0x1y(
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
                ax.text(0.5, 0.5, phase_title, ha="center", va="center", fontsize=16)
                ax.axis("off")

                # colorbar_ax.set_yticks(TICK_POS, [self.y_info.tick_formatter(x, None) for x in TICK_POS])
            case "graphs":  # x0 -> y
                x0s = self.get_ls_dim_names()
                # get unique titles, keeping first appearance order as in self._viz_infos:

                nrows = 1 + len(self.model_layer_names)  # title, PDP + ICE
                height_ratios = [0.25] + [1.0] * len(self.model_layer_names)
                ncols = len(x0s)  # title, *x01s, color legend
                width_ratios = [1.0] * len(x0s)
                gs = GridSpecFromSubplotSpec(
                    nrows, ncols, subplot_spec=sub_gs, height_ratios=height_ratios, width_ratios=width_ratios
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
                for i, model_layer_name in enumerate(self.model_layer_names, start=1):
                    label_y = True
                    surface = getattr(self, f"get_{model_layer_name}")
                    title = f"{model_layer_name.capitalize()} Surface"
                    row_titles.append(title)
                    model_ = LayerEstimator(surface)
                    model_.fit(None, None)
                    for j, x0 in enumerate(x0s):  # columns
                        ax = fig.add_subplot(gs[i, j])
                        # x = self.x[:, j]
                        PartialDependenceDisplay.from_estimator(model_, self.x, [j], ax=ax, kind="both")
                        ax = plt.gca()

                        ax.set_aspect("equal", "box")

                        x0_ticks = [self.get_dim_info(x0).tick_formatter(x, None) for x in TICK_POS]
                        ax.set_xticks(TICK_POS, x0_ticks)
                        ax.xaxis.set_tick_params(rotation=30)
                        ax.set_xlabel(x0)

                        if label_y:
                            y_ticks = [self.y_info.tick_formatter(x, None) for x in TICK_POS]
                            ax.set_yticks(TICK_POS, y_ticks)
                            ax.set_ylabel(self.y_info.name)
                        else:
                            ax.set_yticks([])
                            ax.set_ylabel("")
                        ax.set_ylim(0, 1)
                        label_y = False

                ax = fig.add_subplot(gs[0, :])
                ax.text(0.5, 0.5, phase_title, ha="center", va="center", fontsize=16)
                ax.axis("off")
            case _:
                raise NotImplementedError
        # plt.tight_layout()
        # plt.show()
        return row_titles, height_ratios

    def _viz_single_x0x1y(
        self,
        ax: plt.Axes,
        match_titles: list[str],
        x0: str,
        x1: str,
        grid_length: int,
        projection: str,
        label_x0: bool = False,
        label_x1: bool = False,
    ) -> None:
        def _to_imshow_x(x):
            return (grid_length - 1) * x

        # plot all Visualizations matching the titles given:
        for viz in [v for v in self.get_viz_infos() if v.title in match_titles]:
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
                            cbar.ax.set_ylabel("PHI")
                        elif viz.title == "Unimodality (discretized)":
                            cbar = plt.colorbar(mappable=ScalarMappable(**CMAP_DISCRETIZED), ax=ax)
                            cbar.ax.set_ylabel("multimodal          uncategorized          unimodal")
                            cbar.ax.set_yticks([])
                        elif contourf is not None:
                            # cbar = plt.colorbar(mappable=ScalarMappable(norm=None, cmap=CMAP), ax=ax)
                            cbar = plt.colorbar(mappable=contourf, ax=ax)
                            cbar.ax.set_yticks(
                                TICK_POS_CBAR, [self.y_info.tick_formatter(x, None) for x in TICK_POS_CBAR]
                            )
                            cbar.ax.set_ylabel(y_col_name)
                        # else:
                        #     cbar = plt.colorbar(mappable=ScalarMappable(norm=None, cmap=CMAP), ax=ax)
                        #     cbar.ax.set_yticks(
                        #         TICK_POS_CBAR, [self.y_info.tick_formatter(x, None) for x in TICK_POS_CBAR]
                        #     )
                        #     cbar.ax.set_ylabel(y_col_name)
                case _:
                    pass
                    raise NotImplementedError

        # ticks:
        if projection == "3d":
            x0_ticks = [self.get_dim_info(x0).tick_formatter(x, None) for x in TICK_POS]
            ax.set_xticks(TICK_POS, x0_ticks)
            ax.set_xlabel(x0)
            x1_ticks = [self.get_dim_info(x1).tick_formatter(x, None) for x in TICK_POS]
            ax.set_yticks(TICK_POS, x1_ticks)
            ax.set_ylabel(x1)
            y_ticks = [self.y_info.tick_formatter(x, None) for x in TICK_POS]
            ax.set_zticks(TICK_POS, y_ticks)
            ax.set_zlabel(y_col_name)
            ax.set_zlim3d(0, 1)
        else:
            if label_x0:
                x0_ticks = [self.get_dim_info(x0).tick_formatter(x, None) for x in TICK_POS]
                ax.set_xticks(_to_imshow_x(TICK_POS), x0_ticks)
                ax.xaxis.set_tick_params(rotation=30)
                ax.set_xlabel(x0)
            else:
                ax.set_xticks([])
            if label_x1:
                x1_ticks = [self.get_dim_info(x1).tick_formatter(x, None) for x in TICK_POS]
                ax.set_yticks(_to_imshow_x(TICK_POS), x1_ticks)
                ax.set_ylabel(x1)
            else:
                ax.set_yticks([])
        return

    # def _viz_single_x0y(
    #     self,
    #     ax: plt.Axes,
    #     match_titles: list[str],
    #     x0: str,
    #     # grid_length: int,
    #     label_x0: bool = False,
    #     label_y: bool = False,
    # ) -> None:
    #     # plot all Visualizations with this title:
    #     for viz in [v for v in self.get_viz_infos() if v.title in match_titles]:
    #         # data = self.unnormalize(viz.xy_norm)
    #         data = viz.xy_norm
    #         data_col_names: list[str] = list(data.keys())
    #         y_col_name = data_col_names[-1]
    #         match viz.viz_type:
    #             case "scatter":
    #                 ax.scatter(data[x0], data[y_col_name])
    #             # case "map":
    #             #     pt = data.pivot_table(values=y_col_name, index=x0, columns=x1, aggfunc=np.mean)
    #             #     pt_T = data.pivot_table(values=y_col_name, index=x1, columns=x0, aggfunc=np.mean)
    #             #     ax.imshow(pt_T, vmin=0, vmax=1, cmap=CMAP, origin="lower")
    #             case _:
    #                 pass
    #                 raise NotImplementedError
    #     if label_x0:
    #         x0_ticks = [self.get_dim_info(x0).tick_formatter(x, None) for x in TICK_POS]
    #         ax.set_xticks(TICK_POS, x0_ticks)
    #         ax.set_xlabel(x0)
    #     else:
    #         ax.set_xticks([])
    #     if label_y:
    #         y_ticks = [self.y_info.tick_formatter(x, None) for x in TICK_POS]
    #         ax.set_yticks(TICK_POS, y_ticks)
    #         ax.set_ylabel(y_col_name)
    #     else:
    #         ax.set_yticks([])


def _fix_surface_for_legend(surface):
    surface._edgecolors2d = surface._edgecolor3d
    surface._facecolors2d = surface._facecolor3d
