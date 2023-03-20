from typing import Any

from datetime import datetime
from enum import Enum, auto
from itertools import product
from pathlib import Path

import hydra
import numpy as np
import pytest
from numpy.typing import NDArray
from omegaconf import DictConfig

from autorl_landscape.run.train import train_agent


class ConfType(Enum):
    LOWER = auto()
    ZOO_OPTIMAL = auto()


EXPERIMENTS = ["dqn_cartpole", "sac_hopper"]
ZOO_OPTIMAL_CONFS = {
    # From zoo:
    "dqn_cartpole": {
        "learning_rate": 2.3e-3,
        "gamma": 0.99,
        "exploration_final_eps": 0.04,
    },
    # Zoo has no specific settings for the used hyperparameters, so just use defaults:
    "sac_hopper": {
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "tau": 0.005,
    },
}
DEFUNCT_CONFS = {
    "dqn_cartpole": {
        "learning_rate": 1.0,
        "gamma": 0.1,  # ???
        "exploration_final_eps": 1.0,
    },
    "sac_hopper": {
        "learning_rate": 1.0,
        "gamma": 0.1,  # ???
        "tau": 0.5,  # might crash?
    },
}
hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize(config_path="../conf", version_base="1.1")


def train_agent_(
    conf: DictConfig,
    phase_index: int,
    timestamp: str,
    ancestor: Path | None,
    ls_conf: dict[str, Any],
    phase_path: str,
) -> tuple[int, str, NDArray[Any]]:
    seed = 12345
    return train_agent(conf, phase_index, timestamp, ancestor, ls_conf, seed, 54321, phase_path)


def build_ls_conf(conf: DictConfig, experiment_name: str, conf_type: ConfType) -> dict[str, Any]:
    ls_conf: dict[str, Any] = {}
    dims = [(next(iter(d)), d[next(iter(d))]) for d in conf.ls.dims]
    for dim_name, dim_spec in dims:
        if conf_type == ConfType.LOWER:
            ls_conf[dim_name] = dim_spec.lower
        else:
            ls_conf[dim_name] = ZOO_OPTIMAL_CONFS[experiment_name][dim_name]
    return ls_conf


def compose_experiment_config(experiment_name: str) -> DictConfig:
    overrides: list[str] = ["slurm=debug", "phases=20k", "num_confs=0", "num_seeds=0"]
    match experiment_name:
        case "dqn_cartpole":
            overrides.extend(["combo=dqn_cartpole", "ls=dqn"])
        case "sac_hopper":
            overrides.extend(["combo=sac_hopper", "ls=sac"])
        case e:
            raise ValueError(e)

    conf = hydra.compose("config", overrides)
    conf.wandb.project = None
    conf.wandb.entity = None
    conf.wandb.experiment_tag = None
    return conf


@pytest.mark.parametrize(("experiment_name", "conf_type"), product(EXPERIMENTS, ConfType))
def test_same_conf(experiment_name: str, conf_type: ConfType) -> None:
    """If agents are configured the same, they should produce the same results."""
    conf = compose_experiment_config(experiment_name)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    phase_path = f"phase_results/{conf.agent.name}/{conf.env.name}/{timestamp}/phase_1"

    ls_conf = build_ls_conf(conf, experiment_name, conf_type)

    # Original agent (ancestor):
    _, run_id, orig_res = train_agent_(conf, 1, timestamp, None, ls_conf, phase_path)
    ancestor = Path(f"{phase_path}/agents/{run_id}").resolve().relative_to(Path.cwd())

    # Loaded agent:
    _, _, loaded_res = train_agent_(conf, 2, timestamp, ancestor, ls_conf, phase_path)
    print(f"{experiment_name=}")
    print(f"{conf_type=}")
    assert np.all(orig_res == loaded_res)


@pytest.mark.parametrize("experiment_name", EXPERIMENTS)
def test_slightly_different(experiment_name: str) -> None:
    """If agents are configured the same exept for one hyperparameter, they should produce different results."""
    conf = compose_experiment_config(experiment_name)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    phase_path = f"tests/phase_results/{conf.agent.name}/{conf.env.name}/{timestamp}/phase_1"

    ls_conf = build_ls_conf(conf, experiment_name, ConfType.LOWER)

    # Original agent (ancestor):
    _, run_id, orig_res = train_agent_(conf, 1, timestamp, None, ls_conf, phase_path)
    ancestor = Path(f"{phase_path}/agents/{run_id}").resolve().relative_to(Path.cwd())

    dims = [(next(iter(d)), d[next(iter(d))]) for d in conf.ls.dims]
    for dim_name, dim_spec in dims:
        # Make a slightly different configuration:
        ls_conf = build_ls_conf(conf, experiment_name, ConfType.LOWER)
        ls_conf[dim_name] = dim_spec.upper
        # Loaded agent:
        _, _, loaded_res = train_agent_(conf, 2, timestamp, ancestor, ls_conf, phase_path)

        print(f"{experiment_name=}")
        print(f"{dim_name=}")
        assert not np.all(orig_res == loaded_res)


@pytest.mark.parametrize("experiment_name", EXPERIMENTS)
def test_zoo_vs_defunct_confs(experiment_name: str) -> None:
    """Test hyperparameter stuff by checking if defunct configurations yield worse results than good ones."""
    conf = compose_experiment_config(experiment_name)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    phase_path = f"tests/phase_results/{conf.agent.name}/{conf.env.name}/{timestamp}/phase_1"

    ls_conf_zoo = build_ls_conf(conf, experiment_name, ConfType.ZOO_OPTIMAL)

    # Original agent (ancestor):
    _, _, good_res = train_agent_(conf, 1, timestamp, None, ls_conf_zoo, phase_path)

    dims = [(next(iter(d)), d[next(iter(d))]) for d in conf.ls.dims]
    for dim_name, _ in dims:
        # Make a slightly different configuration that should be defunct:
        ls_conf_defunct = build_ls_conf(conf, experiment_name, ConfType.ZOO_OPTIMAL)
        ls_conf_defunct[dim_name] = DEFUNCT_CONFS[experiment_name][dim_name]
        # Loaded agent:
        _, _, bad_res = train_agent_(conf, 1, timestamp, None, ls_conf_defunct, phase_path)

        print(f"{experiment_name=}")
        print(f"{dim_name=}")

        if np.any(np.isnan(good_res)) or np.any(np.isnan(bad_res)):
            print("FOUND NAN")
            assert np.sum(np.isnan(good_res)) < np.sum(np.isnan(bad_res))
        else:
            assert np.sum(good_res) > np.sum(bad_res)
