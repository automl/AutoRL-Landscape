from typing import Any, Iterable

from datetime import datetime
from pathlib import Path

import pandas as pd
import wandb
from wandb.apis.public import Run


def download_data(entity_name: str, project_name: str, experiment_tag: str) -> None:
    """Extract data from the local wandb server into a file."""
    api = wandb.Api()

    runs: Iterable[Run] = api.runs(path=f"{entity_name}/{project_name}")
    ids, vals = [], []
    for run in runs:
        if experiment_tag in run.tags:
            ids.append(run.id)
            vals.append({"name": run.name, **run.config, **run.summary})

    if len(ids) == 0:
        print("could not find any runs with the given experiment tag!")
        return

    print("download done")
    vals = [_flatten_dict(v) for v in vals]
    df = pd.DataFrame(vals, index=ids)
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = Path(f"data/{entity_name}_{project_name}/{experiment_tag}_{date_str}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        df.to_csv(file)


def _flatten_dict(d: dict[Any, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    items: list[Any] = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
