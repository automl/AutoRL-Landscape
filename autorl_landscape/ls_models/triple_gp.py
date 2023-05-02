from typing import Any

import gpflow
from numpy.typing import NDArray
from pandas import DataFrame, Series
import pandas as pd

from autorl_landscape.ls_models.ls_model import LSModel

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

def estimate_model_fit(X, y, k: int = 5, metrics: list[callable] | None = None) -> pd.DataFrame:            
    if metrics is None:
        metrics = [mean_squared_error, mean_absolute_error]

    cv = KFold(n_splits=k, shuffle=True, random_state=0)

    data = []
    for i, (train_index, test_index) in enumerate(cv.split(X=X, y=y)):
        X_i = X[train_index]
        Y_i = y[train_index]
        model = gpflow.models.GPR((X_i, Y_i), kernel=gpflow.kernels.SquaredExponential())
        opt = gpflow.optimizers.Scipy()
        opt.minimize(model.training_loss, model.trainable_variables)
        f_mean, _ = model.predict_f(X[test_index])
        y_pred = f_mean.numpy()
        results = {}
        results["fold"] = i
        for metric in metrics:
            results[metric.__name__] = metric(y[test_index], y_pred)
        data.append(results)
    data = pd.DataFrame(data)

    return data
        


class TripleGPModel(LSModel):
    """Triple GP Model."""

    def __init__(
        self,
        data: DataFrame,
        dtype: type,
        y_col: str = "ls_eval/returns",
        y_bounds: tuple[float, float] | None = None,
        best_conf: Series | None = None,
        ci: float = 0.95,
    ) -> None:
        super().__init__(data, dtype, y_col, y_bounds, best_conf, ci)
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

    def estimate_iqm_fit(self):
        print("-"*50)
        print("Estimate IQM surface fit")
        data = estimate_model_fit(X=self.x, y=self.y_iqm, k=5)
        for c in data.columns:
            if c is not "fold":
                print(c, data[c].mean(), data[c].std())
        return data
        
    def get_upper(self, x: NDArray[Any], assimilate_factor: float = 1.0) -> NDArray[Any]:
        """Return the upper CI estimate of y at the position(s) x."""
        f_mean, _ = self.upper_model.predict_f(x)
        return self._ci_scale(x, f_mean.numpy(), assimilate_factor)

    def get_middle(self, x: NDArray[Any]) -> NDArray[Any]:
        """Return the IQM estimate of y at the position(s) x."""
        f_mean, _ = self.iqm_model.predict_f(x)
        return f_mean.numpy()

    def get_lower(self, x: NDArray[Any], assimilate_factor: float = 1.0) -> NDArray[Any]:
        """Return the lower CI estimate of y at the position(s) x."""
        f_mean, _ = self.lower_model.predict_f(x)
        return self._ci_scale(x, f_mean.numpy(), assimilate_factor)

    @staticmethod
    def get_model_name() -> str:
        """Return name of this model, for naming files and the like."""
        return "igpr_"
