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
            "final_eval_3/returns": ast.literal_eval,
            "final_eval_3/ep_lengths": ast.literal_eval,
        },
    )
    return df


def split_phases(df: pd.DataFrame, phase_str: str) -> tuple[pd.DataFrame, pd.Series | None]:
    """Returns data belonging to the specified phase, as well as that phase's chosen best configuration."""
    phase_data = df[df["meta.phase"] == phase_str].sort_values("meta.conf_index")
    # query best configuration by looking at data from next phase:
    phase_i = int(phase_str.split("_")[-1])
    phase_str_ = f"phase_{phase_i + 1}"
    phase_data_ = df[df["meta.phase"] == phase_str_].sort_values("meta.conf_index")
    if len(phase_data_) == 0:
        best_conf = None
    else:
        best_conf_id: str = phase_data_["meta.ancestor"][0]
        best_conf = df.loc[Path(best_conf_id).stem]
    # ancestor_id: str = phase_data["meta.ancestor"][0]
    # if ancestor_id == "None":  # first phase has no ancestor
    #     ancestor = None
    # else:
    #     ancestor = df.loc[Path(ancestor_id).stem]

    return phase_data, best_conf
