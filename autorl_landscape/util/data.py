import ast
from pathlib import Path

import pandas as pd


def read_wandb_csv(file: Path) -> pd.DataFrame:
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


def split_phases(df: pd.DataFrame, phase_str: str) -> tuple[pd.DataFrame, pd.Series | None]:
    """TODO."""
    phase_data = df[df["meta.phase"] == phase_str].sort_values("meta.conf_index")
    ancestor_id: str = phase_data["meta.ancestor"][0]
    if ancestor_id == "None":  # first phase has no ancestor
        ancestor = None
    else:
        ancestor = df.loc[Path(ancestor_id).stem]

    return phase_data, ancestor
