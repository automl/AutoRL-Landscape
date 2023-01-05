from typing import Any

from ast import literal_eval
from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter
from numpy.typing import NDArray
from pandas import DataFrame

from autorl_landscape.util.compare import iqm
from autorl_landscape.util.ls_sampler import DimInfo


@dataclass
class Visualization:
    """Saves information for a scatter plot."""

    viz_type: str
    x_samples: NDArray[Any]
    y_samples: NDArray[Any]
    label: str
    kwargs: dict[str, Any]
    # color: str | None
    # alpha: float | None
    # marker: str | None


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

        self.data = data
        self.dtype = dtype

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
        conf_groups = self.data.groupby(["meta.conf_index"] + self.get_ls_dim_names())
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
        # self.y_ci_upper = np.quantile(self.y, 0.975, method="median_unbiased", axis=1, keepdims=True)
        self.y_ci_upper = np.quantile(self.y, 0.8, method="median_unbiased", axis=1, keepdims=True)
        """(num_confs, 1)"""
        # self.y_ci_lower = np.quantile(self.y, 0.025, method="median_unbiased", axis=1, keepdims=True)
        self.y_ci_lower = np.quantile(self.y, 0.2, method="median_unbiased", axis=1, keepdims=True)
        """(num_confs, 1)"""

        self._viz_infos: list[Visualization] = []

    def get_ls_dim_names(self) -> list[str]:
        """Get the list of hyperparameter landscape dimension names."""
        return [di.name for di in self.dim_info]

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

    def visualize(self, ax: Axes, grid_length: int, viz_model: bool) -> None:
        """Visualize the model over the whole landscape.

        Args:
            ax: Axes to plot on.
            grid_length: Number of points on the grid on one side.
            viz_model: Whether to visualize the model.
        """
        num_ls_dims = len(self.get_ls_dim_names())
        match num_ls_dims:
            case 1:
                self._visualize_1d(ax, grid_length, viz_model)
            case 2:
                self._visualize_2d(ax, grid_length, viz_model)
            case n:
                if n < 1:
                    raise Exception(f"Cannot visualize landscape with {n} dimensions!")
                self._visualize_nd(ax, grid_length, viz_model)

    def _visualize_1d(self, ax: Axes, grid_length: int = 50, viz_model: bool = True) -> None:
        raise NotImplementedError

    def _visualize_2d(self, ax: Axes, grid_length: int = 50, viz_model: bool = True) -> None:
        # assume we have 3d projection Axes:
        # ax.set_zlim3d(self.y_info.lower, self.y_info.upper)
        ax.set_zlim3d(0, 1)

        grid_x0, grid_x1 = np.meshgrid(np.linspace(0, 1, num=grid_length), np.linspace(0, 1, num=grid_length))
        grid_x0 = grid_x0.flatten()
        grid_x1 = grid_x1.flatten()
        grid = np.stack((grid_x0, grid_x1), axis=1)  # (-1, num_ls_dims = 2)

        # rescale the ls dims (`grid` is unchanged by this):
        # for dim_i, grid_xi in zip(self.get_ls_dim_names(), (grid_x0, grid_x1)):
        # grid_xi = self.dim_info[dim_i].unit_to_ls(grid_xi).reshape(grid_length, grid_length)
        # grid_xi = np.reshape(self.dim_info[dim_i].unit_to_ls(grid_xi), (grid_length, grid_length))
        dim_0, dim_1 = self.get_ls_dim_names()
        # grid_x0 = np.reshape(self.dim_info[dim_0].unit_to_ls(grid_x0), (grid_length, grid_length))
        # grid_x1 = np.reshape(self.dim_info[dim_1].unit_to_ls(grid_x1), (grid_length, grid_length))
        grid_x0 = grid_x0.reshape((grid_length, grid_length))
        grid_x1 = grid_x1.reshape((grid_length, grid_length))
        assert np.all((grid >= 0) & (grid <= 1))

        if viz_model:
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

    def _visualize_nd(self, ax: Axes, grid_length: int = 50, viz_model: bool = True) -> None:
        """Visualize e.g. with PCA dim reduction, PCP, marginalizing out dimensions."""
        raise NotImplementedError


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


def _fix_surface_for_legend(surface):
    surface._edgecolors2d = surface._edgecolor3d
    surface._facecolors2d = surface._facecolor3d
