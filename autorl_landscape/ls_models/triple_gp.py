from typing import Any

import gpflow
from numpy.typing import NDArray
from pandas import DataFrame

from autorl_landscape.ls_models.ls_model import LSModel, VizInfo


class TripleGPModel(LSModel):
    """Triple GP Model."""

    def __init__(
        self,
        data: DataFrame,
        dtype: type,
        y_col: str = "ls_eval/returns",
        y_bounds: tuple[float, float] | None = None,
    ) -> None:
        super().__init__(data, dtype, y_col, y_bounds)
        gpflow.config.set_default_float(dtype)  # WARNING Global!

    def fit(self):
        """Fit the three GPs to IQM, upper and lower CI."""
        self.iqm_model = gpflow.models.GPR((self.x, self.y_iqm), kernel=gpflow.kernels.SquaredExponential())
        self.upper_model = gpflow.models.GPR((self.x, self.y_ci_upper), kernel=gpflow.kernels.SquaredExponential())
        self.lower_model = gpflow.models.GPR((self.x, self.y_ci_lower), kernel=gpflow.kernels.SquaredExponential())

        opt = gpflow.optimizers.Scipy()
        opt.minimize(self.iqm_model.training_loss, self.iqm_model.trainable_variables)
        opt.minimize(self.upper_model.training_loss, self.upper_model.trainable_variables)
        opt.minimize(self.lower_model.training_loss, self.lower_model.trainable_variables)

    def get_upper(self, x: NDArray[Any]) -> NDArray[Any]:
        """Return the upper CI estimate of y at the position(s) x."""
        f_mean, _ = self.upper_model.predict_f(x)
        return f_mean.numpy()

    def get_middle(self, x: NDArray[Any]) -> NDArray[Any]:
        """Return the IQM estimate of y at the position(s) x."""
        f_mean, _ = self.iqm_model.predict_f(x)
        return f_mean.numpy()

    def get_lower(self, x: NDArray[Any]) -> NDArray[Any]:
        """Return the lower CI estimate of y at the position(s) x."""
        f_mean, _ = self.lower_model.predict_f(x)
        return f_mean.numpy()

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
