from typing import Any

from ast import literal_eval

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator

from autorl_landscape.analyze.visualization import Visualization
from autorl_landscape.run.compare import iqm
from autorl_landscape.util.ls_sampler import LSDimension


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
        y_info = LSDimension.from_dim_dict(y_col, y_dict, is_y=True)
        assert y_info is not None
        self.y_info = y_info

        # extract ls dimensions info from first row of data:
        dim_dicts: list[dict[str, dict[str, Any]]] = literal_eval(data[0:1]["conf.ls.dims"][0])
        self.dim_info: list[LSDimension] = []
        """DimInfos for each LS dimension, sorted by name"""
        for d in dim_dicts:
            dim_name, dim_dict = next(iter(d.items()))
            di = LSDimension.from_dim_dict(dim_name, dim_dict)

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

    def get_dim_info(self, name: str) -> LSDimension | None:
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
