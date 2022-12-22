from pandas import DataFrame

from autorl_landscape.ls_models.ls_model import LSModel


class TripleGPModel(LSModel):
    """Triple GP Model."""

    def __init__(
        self, data: DataFrame, dtype: type, y_col: str = "ls_eval/returns", y_bounds: tuple[float, float] | None = None
    ) -> None:
        super().__init__(data, dtype, y_col, y_bounds)
