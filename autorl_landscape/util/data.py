from typing import Any

import ast

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def read_wandb_csv(file: str) -> pd.DataFrame:
    """Read data from csv file, making sure to correctly parse some of the fields.

    Args:
        file: path to csv file
    """
    df = pd.read_csv(
        file,
        index_col=0,
        converters={
            "ls_eval/returns": ast.literal_eval,
            "ls_eval/ep_lengths": ast.literal_eval,
        },
    )
    return df


def broadcast_1d(a: NDArray[Any], shape: tuple[int, ...]) -> NDArray[Any]:
    """Broadcasts (n,) to (n, m), copying values such that elements in a row are all equal.

    Args:
        a: (n,)-shaped array
        shape: (n, m)
    """
    return np.broadcast_to(a.reshape((1, -1)).T, shape)
