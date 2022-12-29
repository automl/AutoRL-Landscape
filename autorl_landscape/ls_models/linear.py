from typing import Any

from numpy.typing import NDArray
from pandas import DataFrame

from autorl_landscape.ls_models.ls_model import LSModel, Visualization
from autorl_landscape.util.interpolate import LinearInterpolator


class LinearLSModel(LSModel):
    """Linear Interpolation Model."""

    def __init__(
        self, data: DataFrame, dtype: type, y_col: str = "ls_eval/returns", y_bounds: tuple[float, float] | None = None
    ) -> None:
        super().__init__(data, dtype, y_col, y_bounds)
        self.iqm_model = LinearInterpolator(self.x, self.y_iqm)
        # self.ci_upper_model = LinearNDInterpolator(self.x, self.y_mean + 1.96 * self.y_std)
        # self.ci_lower_model = LinearNDInterpolator(self.x, self.y_mean - 1.96 * self.y_std)
        self.ci_upper_model = LinearInterpolator(self.x, self.y_ci_upper)
        self.ci_lower_model = LinearInterpolator(self.x, self.y_ci_lower)

        # Use "near" interpolation for the borders where "linear" outputs nans:
        # self.iqm_model_nearest = NearestNDInterpolator(self.x, self.y_iqm)
        # self.ci_upper_model_nearest = NearestNDInterpolator(self.x, self.y_mean + 1.96 * self.y_std)
        # self.ci_lower_model_nearest = NearestNDInterpolator(self.x, self.y_mean - 1.96 * self.y_std)
        # self.ci_upper_model_nearest = NearestNDInterpolator(self.x, self.y_ci_upper)
        # self.ci_lower_model_nearest = NearestNDInterpolator(self.x, self.y_ci_lower)

        self._viz_infos = [
            Visualization(
                "scatter", self.x, self.y_ci_upper, "97.5%-percentile", {"color": "red", "alpha": 0.75, "marker": "v"}
            ),
            Visualization(
                "scatter", self.x, self.y_iqm, "interquartile mean", {"color": "red", "alpha": 0.75, "marker": "D"}
            ),
            Visualization(
                "scatter", self.x, self.y_ci_lower, "2.5%-percentile", {"color": "red", "alpha": 0.75, "marker": "^"}
            ),
            Visualization("scatter", self.x_samples, self.y_samples, "data points", {"alpha": 0.025}),
            # no samples are directly used to train the models
        ]

    def get_upper(self, x: NDArray[Any]) -> NDArray[Any]:
        """Return the interpolated upper estimate of y at the position(s) x."""
        return self.ci_upper_model(x)

    def get_middle(self, x: NDArray[Any]) -> NDArray[Any]:
        """Return the interpolated IQM of y at the position(s) x."""
        return self.iqm_model(x)

    def get_lower(self, x: NDArray[Any]) -> NDArray[Any]:
        """Return the interpolated lower estimate of y at the position(s) x."""
        return self.ci_lower_model(x)
