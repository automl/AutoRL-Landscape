from typing import Any

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.stats import trimboth

from autorl_landscape.ls_models.ls_model import LSModel
from autorl_landscape.util.compare import iqm


class LinearLSModel(LSModel):
    """Linear Interpolation Model."""

    def __init__(
        self, data: DataFrame, dtype: type, y_col: str = "ls_eval/returns", y_bounds: tuple[float, float] | None = None
    ) -> None:
        super().__init__(data, dtype, y_col, y_bounds)
        trim_y = trimboth(self.y, 0.025, axis=1)

        self.iqm_model = LinearNDInterpolator(self.x, iqm(self.y, axis=1))
        # self.ci_upper_model = LinearNDInterpolator(self.x, self.y_mean + 1.96 * self.y_std)
        # self.ci_lower_model = LinearNDInterpolator(self.x, self.y_mean - 1.96 * self.y_std)
        self.ci_upper_model = LinearNDInterpolator(self.x, np.max(trim_y, axis=1))
        self.ci_lower_model = LinearNDInterpolator(self.x, np.min(trim_y, axis=1))

        # Use "near" interpolation for the borders where "linear" outputs nans:
        self.iqm_model_nearest = NearestNDInterpolator(self.x, iqm(self.y, axis=1))
        # self.ci_upper_model_nearest = NearestNDInterpolator(self.x, self.y_mean + 1.96 * self.y_std)
        # self.ci_lower_model_nearest = NearestNDInterpolator(self.x, self.y_mean - 1.96 * self.y_std)
        self.ci_upper_model_nearest = NearestNDInterpolator(self.x, np.max(trim_y, axis=1))
        self.ci_lower_model_nearest = NearestNDInterpolator(self.x, np.min(trim_y, axis=1))

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

    def save(self, model_save_path: Path) -> None:
        """Save the model to disk."""
        raise NotImplementedError

    def load(self, model_save_path: Path) -> None:
        """Load the model from disk."""
        raise NotImplementedError
