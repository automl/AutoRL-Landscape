from typing import Any

import gpflow
from gpflow.utilities.traversal import Path
from numpy.typing import NDArray
from pandas import DataFrame

from autorl_landscape.ls_models.ls_model import LSModel


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

    def save(self, model_save_path: Path) -> None:
        """Save the model to disk."""
        return super().save(model_save_path)

    def load(self, model_save_path: Path) -> None:
        """Load the model from disk."""
        return super().load(model_save_path)
