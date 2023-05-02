from typing import Any

from numpy.typing import NDArray
from pandas import DataFrame, Series
from scipy.interpolate import RBFInterpolator

from autorl_landscape.ls_models.ls_model import LSModel

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd


def estimate_model_fit(X, y, k: int = 5, metrics: list[callable] | None = None) -> pd.DataFrame:            
    if metrics is None:
        metrics = [mean_squared_error, mean_absolute_error]

    cv = KFold(n_splits=k, shuffle=True, random_state=0)

    data = []
    dim_info = X.shape[-1]
    for i, (train_index, test_index) in enumerate(cv.split(X=X, y=y)):
        X_i = X[train_index]
        Y_i = y[train_index]
        model = RBFInterpolator(X_i, Y_i, kernel="linear")
        X_t = X[test_index]
        y_pred = model(X_t.reshape(-1, dim_info)).reshape(*X_t.shape[0:-1], 1)
        results = {}
        results["fold"] = i
        for metric in metrics:
            results[metric.__name__] = metric(y[test_index], y_pred)
        data.append(results)
    data = pd.DataFrame(data)

    return data
        


class RBFInterpolatorLSModel(LSModel):
    """RBF Interpolated model with variable ci size."""

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
        self.iqm_model = RBFInterpolator(self.x, self.y_iqm, kernel="linear")
        self.ci_upper_model = RBFInterpolator(self.x, self.y_ci_upper, kernel="linear")
        self.ci_lower_model = RBFInterpolator(self.x, self.y_ci_lower, kernel="linear")

    def estimate_iqm_fit(self):
        print("-"*50)
        print("Estimate IQM surface fit")
        data = estimate_model_fit(X=self.x, y=self.y_iqm, k=5)
        for c in data.columns:
            if c is not "fold":
                print(c, data[c].mean(), data[c].std())
        return data

    def get_upper(self, x: NDArray[Any], assimilate_factor: float = 1.0) -> NDArray[Any]:
        """Return the interpolated upper estimate of y at the position(s) x."""
        assert x.shape[-1] == len(self.dim_info)
        y = self.ci_upper_model(x.reshape(-1, len(self.dim_info))).reshape(*x.shape[0:-1], 1)
        return self._ci_scale(x, y, assimilate_factor)

    def get_middle(self, x: NDArray[Any]) -> NDArray[Any]:
        """Return the interpolated IQM of y at the position(s) x."""
        assert x.shape[-1] == len(self.dim_info)
        return self.iqm_model(x.reshape(-1, len(self.dim_info))).reshape(*x.shape[0:-1], 1)

    def get_lower(self, x: NDArray[Any], assimilate_factor: float = 1.0) -> NDArray[Any]:
        """Return the interpolated lower estimate of y at the position(s) x."""
        assert x.shape[-1] == len(self.dim_info)
        y = self.ci_lower_model(x.reshape(-1, len(self.dim_info))).reshape(*x.shape[0:-1], 1)
        return self._ci_scale(x, y, assimilate_factor)

    @staticmethod
    def get_model_name() -> str:
        """Return name of this model, for naming files and the like."""
        return "ilm_"
