from abc import ABC, abstractmethod
from typing import Any

from ast import literal_eval
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter
from numpy.typing import NDArray
from pandas import DataFrame

from autorl_landscape.util.ls_sampler import DimInfo


class LSModel(ABC):
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
        # self.y_col = y_col
        self.dtype = dtype
        self.model = None

        # mainly for visualization:
        if y_bounds is not None:
            # self.y_min, self.y_max = y_bounds
            y_dict = {"type": "Float", "lower": y_bounds[0], "upper": y_bounds[1]}
        else:
            # self.y_min = 0.0
            # self.y_max = float(data[0:1]["conf.viz.max_return"])
            y_dict = {"type": "Float", "lower": 0.0, "upper": float(data[0:1]["conf.viz.max_return"])}
        y_info = DimInfo.from_dim_dict(y_col, y_dict, is_y=True)
        assert y_info is not None
        self.y_info = y_info

        # extract ls dimensions info from first row of data:
        dim_dicts: list[dict[str, dict[str, Any]]] = literal_eval(data[0:1]["conf.ls.dims"][0])
        # self.dim_info: dict[str, DimInfo] = {}  # dim_name -> {"type": ..., "lower": ..., ...}
        self.dim_info: list[DimInfo] = []
        # self.converters: dict[str, DimInfo]
        for d in dim_dicts:
            dim_name, dim_dict = next(iter(d.items()))
            di = DimInfo.from_dim_dict(dim_name, dim_dict)

            if di is not None:
                self.dim_info.append(di)
        self.dim_info = sorted(self.dim_info)  # sorts on dim names

        # group runs with the same configuration:
        conf_groups = self.data.groupby(["meta.conf_index"] + self.get_ls_dim_names())
        # all groups (configurations):
        self.x = np.array(list(conf_groups.groups.keys()), dtype=self.dtype)[:, 1:]  # (num_confs, num_ls_dims)
        # all evaluations (y values) for a group (configuration):
        y = np.array(list(conf_groups[self.y_info.name].sum()), dtype=self.dtype)  # (num_confs, 100)

        # scale ls dims into [0, 1] interval:
        for i in range(len(self.dim_info)):
            transformer = self.dim_info[i].ls_to_unit
            self.x[:, i] = transformer(self.x[:, i])
        # scale y into [0, 1] interval:
        self.y = self.y_info.ls_to_unit(y)

        # just all the single evaluation values, not grouped (but still scaled to [0, 1] interval):
        self.x_samples = np.repeat(self.x, self.y.shape[1], axis=0)
        self.y_samples = self.y.reshape(-1, 1)

        # statistical information about each configuration assuming a normal distribution:
        self.y_mean = self.y.mean(axis=1)
        self.y_std = self.y.std(axis=1)

        # self.ls_dims = [k for k in df.keys() if k.startswith("ls.")]
        # for phase_str in sorted(data["meta.phase"].unique()):
        #     phase_df = data[data["meta.phase"] == phase_str].sort_values("meta.conf_index")

    def get_ls_dim_names(self) -> list[str]:
        """Get the list of hyperparameter landscape dimension names."""
        return [di.name for di in self.dim_info]

    @abstractmethod
    def get_upper(self, x: NDArray[Any]) -> NDArray[Any]:
        """Return an upper estimate of y at the position(s) x."""
        raise NotImplementedError

    @abstractmethod
    def get_middle(self, x: NDArray[Any]) -> NDArray[Any]:
        """Return some middle (mean, interquartile mean, median, etc.) estimate of y at the position(s) x."""
        raise NotImplementedError

    @abstractmethod
    def get_lower(self, x: NDArray[Any]) -> NDArray[Any]:
        """Return an lower estimate of y at the position(s) x."""
        raise NotImplementedError

    def visualize(self, ax: Axes, grid_length: int = 51, viz_model: bool = True, viz_samples: bool = True) -> Axes:
        """Visualize the model over the whole landscape.

        Args:
            ax: Axes to plot on.
            grid_length: Number of points on the grid on one side.
            viz_model: Whether to visualize the model.
            viz_samples: Whether to visualize the inputted data samples with scatter.

        Returns:
            The modified axes.
        """
        num_ls_dims = len(self.get_ls_dim_names())
        match num_ls_dims:
            case 1:
                return self._visualize_1d(ax, grid_length, viz_model, viz_samples)
            case 2:
                return self._visualize_2d(ax, grid_length, viz_model, viz_samples)
            case n:
                if n < 1:
                    raise Exception(f"Cannot visualize landscape with {n} dimensions!")
                return self._visualize_nd(ax, grid_length, viz_model, viz_samples)

    def _visualize_1d(self, ax: Axes, grid_length: int = 50, viz_gp: bool = True, viz_data: bool = True) -> Axes:
        raise NotImplementedError

    def _visualize_2d(self, ax: Axes, grid_length: int = 50, viz_gp: bool = True, viz_data: bool = True) -> Axes:
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

        if viz_gp:
            for y_, opacity, label in (
                (self.get_upper(grid), 0.5, "modelled upper CI bound"),
                (self.get_middle(grid), 1.0, "modelled mean"),
                (self.get_lower(grid), 0.5, "modelled lower CI bound"),
            ):
                cmap = "viridis" if opacity == 1.0 else None
                color = (0.5, 0.5, 0.5, opacity) if opacity < 1.0 else None
                ax.plot_surface(
                    grid_x0,
                    grid_x1,
                    y_.reshape(grid_length, grid_length),
                    cmap=cmap,
                    color=color,
                    edgecolor="none",
                    shade=False,
                    label=label,
                )
        if viz_data:
            ax.scatter(
                # self.dim_info[dim_0].unit_to_ls(self.x_samples[:, 0]),
                # self.dim_info[dim_1].unit_to_ls(self.x_samples[:, 1]),
                self.x_samples[:, 0],
                self.x_samples[:, 1],
                self.y_samples[:, 0],
                label="samples",
            )
        ax.set_xlabel(self.dim_info[0].name, fontsize=12)
        ax.set_ylabel(self.dim_info[1].name, fontsize=12)
        ax.set_zlabel(self.y_info.name, fontsize=12)

        ax.xaxis.set_major_formatter(FuncFormatter(self.dim_info[0].tick_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(self.dim_info[1].tick_formatter))
        ax.zaxis.set_major_formatter(FuncFormatter(self.y_info.tick_formatter))
        return ax

    def _visualize_nd(self, ax: Axes, grid_length: int = 50, viz_gp: bool = True, viz_data: bool = True) -> Axes:
        raise NotImplementedError

    @abstractmethod
    def save(self, model_save_path: Path) -> None:
        """Save the model to disk."""
        raise NotImplementedError

    @abstractmethod
    def load(self, model_save_path: Path) -> None:
        """Load the model from disk."""
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
