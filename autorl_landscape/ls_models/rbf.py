from typing import Any

from numpy.typing import NDArray
from pandas import DataFrame, Series
from scipy.interpolate import RBFInterpolator

from autorl_landscape.ls_models.ls_model import LSModel


class RBFInterpolatorLSModel(LSModel):
    """RBF Interpolated model with variable ci size."""

    def __init__(
        self,
        data: DataFrame,
        dtype: type,
        y_col: str = "ls_eval/returns",
        y_bounds: tuple[float, float] | None = None,
        ancestor: Series | None = None,
        ci: float = 0.95,
    ) -> None:
        super().__init__(data, dtype, y_col, y_bounds, ancestor, ci)
        self.iqm_model = RBFInterpolator(self.x, self.y_iqm, kernel="linear")
        self.ci_upper_model = RBFInterpolator(self.x, self.y_ci_upper, kernel="linear")
        self.ci_lower_model = RBFInterpolator(self.x, self.y_ci_lower, kernel="linear")

    def get_upper(self, x: NDArray[Any]) -> NDArray[Any]:
        """Return the interpolated upper estimate of y at the position(s) x."""
        assert x.shape[-1] == len(self.dim_info)
        return self.ci_upper_model(x.reshape(-1, len(self.dim_info))).reshape(*x.shape[0:-1], 1)

    def get_middle(self, x: NDArray[Any]) -> NDArray[Any]:
        """Return the interpolated IQM of y at the position(s) x."""
        assert x.shape[-1] == len(self.dim_info)
        return self.iqm_model(x.reshape(-1, len(self.dim_info))).reshape(*x.shape[0:-1], 1)

    def get_lower(self, x: NDArray[Any]) -> NDArray[Any]:
        """Return the interpolated lower estimate of y at the position(s) x."""
        assert x.shape[-1] == len(self.dim_info)
        return self.ci_lower_model(x.reshape(-1, len(self.dim_info))).reshape(*x.shape[0:-1], 1)
