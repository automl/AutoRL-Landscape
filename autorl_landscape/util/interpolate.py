from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


class LinearInterpolator:
    """Combination of `LinearNDInterpolator` and `NearestNDInterpolator` to deal with nan values."""

    def __init__(self, x: NDArray[Any], y: NDArray[Any]) -> None:
        self.linear = LinearNDInterpolator(x, y)
        self.nearest = NearestNDInterpolator(x, y)

    def __call__(self, x: NDArray[Any]) -> NDArray[Any]:
        """Return the interpolated value `y` at `x`."""
        linear_pred = self.linear(x)
        nearest_pred = self.nearest(x)
        return np.where(np.isnan(linear_pred), nearest_pred, linear_pred)
