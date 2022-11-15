from typing import Any, Iterable

from datetime import datetime
from pathlib import Path

import pandas as pd
import wandb
from wandb.apis.public import Run


def download_data(entity_name: str, project_name: str) -> None:
    """Extract data from the local wandb server into a file."""
    # api = wandb.Api({"base_url": "localhost"})  # TODO local installation not 100% working:
    # https://docs.wandb.ai/guides/self-hosted/setup/on-premise-baremetal
    api = wandb.Api()
    projects = api.projects()
    entity_name = projects[0].entity

    runs: Iterable[Run] = api.runs(path=f"{entity_name}/{project_name}")
    # names, ids, configs, summaries = [], [], [], []
    ids, vals = [], []
    for run in runs:
        # ids.append(run.id)
        # names.append(run.name)
        # configs.append(run.config)
        # summaries.append(run.summary)
        ids.append(run.id)
        vals.append({"name": run.name, **run.config, **run.summary})

    print("download done")
    vals = [_flatten_dict(v) for v in vals]
    # configs = [_flatten_dict(c) for c in configs]
    # summaries = [_flatten_dict(s) for s in summaries]
    df = pd.DataFrame(vals, index=ids)
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = Path(f"data/{entity_name}/{project_name}_{date_str}.csv")
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
