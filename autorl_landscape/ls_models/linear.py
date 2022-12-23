from typing import Any

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from autorl_landscape.ls_models.ls_model import LSModel, VizInfo


class LinearLSModel(LSModel):
    """Linear Interpolation Model."""

    def __init__(
        self, data: DataFrame, dtype: type, y_col: str = "ls_eval/returns", y_bounds: tuple[float, float] | None = None
    ) -> None:
        super().__init__(data, dtype, y_col, y_bounds)
        self.iqm_model = LinearNDInterpolator(self.x, self.y_iqm)
        # self.ci_upper_model = LinearNDInterpolator(self.x, self.y_mean + 1.96 * self.y_std)
        # self.ci_lower_model = LinearNDInterpolator(self.x, self.y_mean - 1.96 * self.y_std)
        self.ci_upper_model = LinearNDInterpolator(self.x, self.y_ci_upper)
        self.ci_lower_model = LinearNDInterpolator(self.x, self.y_ci_lower)

        # Use "near" interpolation for the borders where "linear" outputs nans:
        self.iqm_model_nearest = NearestNDInterpolator(self.x, self.y_iqm)
        # self.ci_upper_model_nearest = NearestNDInterpolator(self.x, self.y_mean + 1.96 * self.y_std)
        # self.ci_lower_model_nearest = NearestNDInterpolator(self.x, self.y_mean - 1.96 * self.y_std)
        self.ci_upper_model_nearest = NearestNDInterpolator(self.x, self.y_ci_upper)
        self.ci_lower_model_nearest = NearestNDInterpolator(self.x, self.y_ci_lower)

    def get_upper(self, x: NDArray[Any]) -> NDArray[Any]:
        """Return the interpolated upper estimate of y at the position(s) x."""
        upper = self.ci_upper_model(x)
        upper_nearest = self.ci_upper_model_nearest(x)
        return np.where(np.isnan(upper), upper_nearest, upper)

    def get_middle(self, x: NDArray[Any]) -> NDArray[Any]:
        """Return the interpolated IQM of y at the position(s) x."""
        middle = self.iqm_model(x)
        middle_nearest = self.iqm_model_nearest(x)
        return np.where(np.isnan(middle), middle_nearest, middle)

    def get_lower(self, x: NDArray[Any]) -> NDArray[Any]:
        """Return the interpolated lower estimate of y at the position(s) x."""
        lower = self.ci_lower_model(x)
        lower_nearest = self.ci_lower_model_nearest(x)
        return np.where(np.isnan(lower), lower_nearest, lower)

    def get_sample_viz_infos(self, include_rest: bool) -> list[VizInfo]:
        """Return visualization info(s) for samples used for training the model.

        Since this model uses IQM and quantiles of the samples, no actual samples are actually used for training.

        Args:
            include_rest: Whether to include a `VizInfo` for samples that are not used by the model. Always the first
                VizInfo in the list.
        """
        trainers = [
            VizInfo(self.x, self.y_ci_upper, "97.5%-percentile", "red", 0.75, "v"),
            VizInfo(self.x, self.y_iqm, "interquartile mean", "red", 0.75, "D"),
            VizInfo(self.x, self.y_ci_lower, "2.5%-quantile", "red", 0.75, "^"),
        ]
        rest = [VizInfo(self.x_samples, self.y_samples, "data points", None, 0.025, None)] if include_rest else []
        rest.extend(trainers)
        return rest
