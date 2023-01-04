from typing import Any

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from autorl_landscape.ls_models.ls_model import LSModel


class MockLSModel(LSModel):
    """Sine waves."""

    def __init__(
        self, data: DataFrame, dtype: type, y_col: str = "ls_eval/returns", y_bounds: tuple[float, float] | None = None
    ) -> None:
        super().__init__(data, dtype, y_col, y_bounds)
        self.rng = np.random.default_rng(0)

    def _get_whatever(self, x: NDArray[Any], offset: float) -> NDArray[Any]:
        # return 0.25 * np.sin(20 * x[:, 0]) + offset
        return 0.25 * np.sin(10 * x[:, 0]) + offset + 0.25 * np.sin(10 * x[:, 1]) + 0.01 * self.rng.random()

    def get_upper(self, x: NDArray[Any]) -> NDArray[Any]:
        """High sine curve."""
        return self._get_whatever(x, 0.5)

    def get_middle(self, x: NDArray[Any]) -> NDArray[Any]:
        """Medium sine curve."""
        return self._get_whatever(x, 0.4)

    def get_lower(self, x: NDArray[Any]) -> NDArray[Any]:
        """Low sine curve."""
        return self._get_whatever(x, 0.3)
