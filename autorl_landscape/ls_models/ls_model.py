from typing import Any

from ast import literal_eval
from dataclasses import dataclass
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from numpy.typing import NDArray
from pandas import DataFrame

from autorl_landscape.util.compare import iqm
from autorl_landscape.util.ls_sampler import DimInfo


@dataclass
class Visualization:
    """Saves information for a plot."""

    title: str
    """Title for the plot (Visualizations with matching titles are drawn on the same `Axes`)"""
    viz_type: str
    """scatter, trisurf, etc."""
    visualization_group: str
    """For allocating a Visualization to an image (combination of Visualizations)"""
    # x_samples: NDArray[Any]
    # y_samples: NDArray[Any]
    xy_norm: DataFrame
    """DataFrame including y (output) values for some visualization. May omit x values to assume the default, model.x"""
    # label: str
    kwargs: dict[str, Any]


class LSModel:
    """A model for the hyperparameter landscape."""

    def __init__(
        self,
        data: DataFrame,
        dtype: type,
        y_col: str = "ls_eval/returns",
        y_bounds: tuple[float, float] | None = None,
    ) -> None:
        super().__init__()

        self.dtype = dtype
        self.model_layer_names = ["lower", "middle", "upper"]

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
        for d in dim_dicts:
            dim_name, dim_dict = next(iter(d.items()))
            di = DimInfo.from_dim_dict(dim_name, dim_dict)

            if di is not None:
                self.dim_info.append(di)
        self.dim_info = sorted(self.dim_info)  # sorts on dim names
        """DimInfos for each LS dimension, sorted by name"""

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

        # statistical information about each configuration assuming a normal distribution:
        self.y_mean = np.mean(self.y, axis=1, keepdims=True)
        """(num_confs, 1)"""
        self.y_std = np.std(self.y, axis=1, keepdims=True)
        """(num_confs, 1)"""
        self.y_iqm = iqm(self.y, axis=1).reshape(-1, 1)
        """(num_confs, 1)"""
        self.y_ci_upper = np.quantile(self.y, 0.975, method="median_unbiased", axis=1, keepdims=True)
        # self.y_ci_upper = np.quantile(self.y, 0.8, method="median_unbiased", axis=1, keepdims=True)
        """(num_confs, 1)"""
        self.y_ci_lower = np.quantile(self.y, 0.025, method="median_unbiased", axis=1, keepdims=True)
        # self.y_ci_lower = np.quantile(self.y, 0.2, method="median_unbiased", axis=1, keepdims=True)
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

    def get_upper(self, x: NDArray[Any]) -> NDArray[Any]:
        """Return an upper estimate of y at the position(s) x."""
        raise NotImplementedError

    def get_middle(self, x: NDArray[Any]) -> NDArray[Any]:
        """Return some middle (mean, interquartile mean, median, etc.) estimate of y at the position(s) x."""
        raise NotImplementedError

    def get_lower(self, x: NDArray[Any]) -> NDArray[Any]:
        """Return an lower estimate of y at the position(s) x."""
        raise NotImplementedError

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
            title = f"{model_layer_name.capitalize()} Model"
            func = getattr(self, f"get_{model_layer_name}")
            self.add_viz_info(
                Visualization(
                    title,
                    "map",
                    "maps",
                    self.build_df(grid, func(grid), "ls_eval/returns"),
                    {},
                )
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

    def visualize(self, grid_length: int, which: str) -> None:
        """Visualize the model over the whole landscape.

        Args:
            grid_length: Number of points on the grid on one side.
            which: Which visualization to show.
        """
        num_ls_dims = len(self.dim_info)
        match num_ls_dims:
            case 1:
                self._visualize_1d(grid_length, which)
            # case 2:
            #     self._visualize_2d(grid_length, which)
            case n:
                if n < 1:
                    raise Exception(f"Cannot visualize landscape with {n} dimensions!")
                self._visualize_nd(grid_length, which)

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

    def _visualize_nd(self, grid_length: int, which: str) -> None:
        """Visualize e.g. with PCA dim reduction, PCP, marginalizing out dimensions."""
        self._add_model_viz(grid_length)
        # grid_shape = grid.shape[0:-1]
        fig = plt.figure(figsize=(16, 10))
        match which:
            case "maps" | "peaks":  # x0x1 -> y
                x01s = list(combinations(self.get_ls_dim_names(), 2))  # all 2-combinations of x (ls) dimensions
                titles = list(dict.fromkeys([v.title for v in self.get_viz_infos() if v.visualization_group == which]))

                for i, title in enumerate(titles):  # rows
                    add_title = i == 0  # title on first row
                    label_x0 = i == (len(titles) - 1)  # label on last row
                    for j, (x0, x1) in enumerate(x01s):  # columns
                        ax = plt.subplot2grid((len(titles), len(x01s)), (i, j), fig=fig)
                        self._visualize_single(ax, title, x0, x1, grid_length, label_x0, True, add_title)
            case "graphs":  # x0 -> y
                raise NotImplementedError
            case _:
                raise NotImplementedError
        plt.show()

    def _visualize_single(
        self,
        ax: plt.Axes,
        title: str,
        x0: str,
        x1: str,
        grid_length: int,
        label_x0: bool = False,
        label_x1: bool = False,
        add_title: bool = False,
    ) -> None:
        def to_imshow_x(x):
            return (grid_length - 1) * x

        # plot all Visualizations with this title:
        for viz in [v for v in self.get_viz_infos() if v.title == title]:
            # data = self.unnormalize(viz.xy_norm)
            data = viz.xy_norm
            data_col_names: list[str] = list(data.keys())
            y_col_name = data_col_names[-1]
            match viz.viz_type:
                case "scatter":
                    ax.scatter(to_imshow_x(data[x1]), to_imshow_x(data[x0]))
                case "map":
                    pt = data.pivot_table(values=y_col_name, index=x0, columns=x1, aggfunc=np.mean)
                    # sns.heatmap(pt, vmin=0, vmax=1, ax=ax)
                    ax.imshow(pt)
                case _:
                    pass
                    raise NotImplementedError

        # ticks:
        num_ticks = 5
        ticks_poss = np.linspace(0, 1, num_ticks)
        if label_x0:
            x0_ticks = [self.get_dim_info(x0).tick_formatter(x, None) for x in ticks_poss]
            ax.set_xticks(to_imshow_x(ticks_poss) + 0.5, x0_ticks)
            ax.xaxis.set_tick_params(rotation=30)
            ax.set_xlabel(x0)
        else:
            ax.set_xticks([])
        if label_x1:
            x1_ticks = [self.get_dim_info(x1).tick_formatter(x, None) for x in ticks_poss]
            ax.set_yticks(to_imshow_x(ticks_poss) + 0.5, x1_ticks)
            ax.set_ylabel(x1)
        else:
            ax.set_yticks([])


def grid_space_2d(length: int, dtype: type) -> NDArray[Any]:
    """Make grid in 2d unit cube. Returned array has shape (length ** 2, 2).

    Args:
        length: number of points on one side of the grid, aka. sqrt of the number of points returned.
        dtype: dtype for the grid.
    """
    axis = np.linspace(0, 1, num=length, dtype=dtype)
    grid_x, grid_y = np.meshgrid(axis, axis)
    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()
    return np.stack((grid_x, grid_y), axis=1)


def grid_space_nd(
    num_dims: int, grid_length: int, dtype: type = np.float64, bounds: tuple[float, float] = (0.0, 1.0)
) -> NDArray[Any]:
    """Generate a `num_dims` dimensional grid of shape (*(grid_length,) * num_dims, num_dims)."""
    axis = np.linspace(bounds[0], bounds[1], num=grid_length, dtype=dtype)
    grid_xis = np.meshgrid(*[axis] * num_dims)
    return np.stack(grid_xis).T


def _fix_surface_for_legend(surface):
    surface._edgecolors2d = surface._edgecolor3d
    surface._facecolors2d = surface._facecolor3d
