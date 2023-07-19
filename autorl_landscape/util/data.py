import ast
from pathlib import Path

import pandas as pd


def read_wandb_csv(file: Path) -> pd.DataFrame:
    """Read data from csv file, making sure to correctly parse some of the fields.

    Args:
        file: path to csv file
    """
    return pd.read_csv(
        file,
        index_col=0,
        converters={
            "ls_eval/returns": ast.literal_eval,
            "ls_eval/ep_lengths": ast.literal_eval,
            "final_eval_3/returns": ast.literal_eval,
            "final_eval_3/ep_lengths": ast.literal_eval,
        },
    )


def split_phases(df: pd.DataFrame, phase_index: int) -> tuple[pd.DataFrame, pd.Series | None]:
    """Returns data belonging to the specified phase, as well as that phase's chosen best configuration."""
    phase_data = df[df["meta.phase"] == phase_index].sort_values("meta.conf_index")
    # Get best configuration by looking at data from next phase:
    phase_data_next = df[df["meta.phase"] == phase_index + 1].sort_values("meta.conf_index")
    if len(phase_data_next) == 0:
        best_conf = None
    else:
        best_conf_id: str = phase_data_next["meta.ancestor"][0]
        best_conf = df.loc[Path(best_conf_id).stem]
    # ancestor_id: str = phase_data["meta.ancestor"][0]
    # if ancestor_id == "None":  # first phase has no ancestor
    #     ancestor = None
    # else:
    #     ancestor = df.loc[Path(ancestor_id).stem]

    return phase_data, best_conf
