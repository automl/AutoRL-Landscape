import ast

import pandas as pd


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


# def train_or_load_gpr(x: NDArray[Any], y: NDArray[Any]) -> GaussianProcessClassifier:
#     gpr = GaussianProcessClassifier(RBF())
#     gpr.fit(x, y)
#     return gpr
