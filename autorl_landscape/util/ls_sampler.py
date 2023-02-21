from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from omegaconf import DictConfig
from scipy.stats.qmc import Sobol

from autorl_landscape.util.grid_space import grid_space_nd


def construct_ls(conf: DictConfig) -> pd.DataFrame:
    """Build landscape according to the passed config and returns a `pd.DataFrame` with samples from it.

    :param conf: `DictConfig` with ls dims, num of samples and optimal zoo hyperparameters for Constant dimensions.
    :return: `pd.DataFrame` with conf.num_confs entries.
    """
    dims = [(next(iter(d)), d[next(iter(d))]) for d in conf.ls.dims]
    num_random_dims = len([1 for (_, dim_args) in dims if dim_args["type"] != "Constant"])
    # Landscape Hyperparameters:
    match conf.ls["type"]:
        case "Sobol":
            # Sobol sampling is best done with 2 ** n samples:
            assert np.log2(conf.num_confs).is_integer(), "conf.num_confs needs to be a power of 2."

            sampler = Sobol(num_random_dims, seed=conf.seeds.ls)
            samples = sampler.random_base2(int(np.log2(conf.num_confs)))

            configs = _unit_samples_to_ls(conf, samples, dims)
            return pd.DataFrame(configs, columns=[dim_name for (dim_name, _) in dims])
        case "Grid":
            # Grid sampling should be done with n ** num_random_dims samples:
            grid_length = round(conf.num_confs ** (1 / num_random_dims))
            assert grid_length**num_random_dims == conf.num_confs
            samples = grid_space_nd(num_random_dims, int(grid_length)).reshape(conf.num_confs, num_random_dims)

            configs = _unit_samples_to_ls(conf, samples, dims)
            return pd.DataFrame(configs, columns=[dim_name for (dim_name, _) in dims])
        case _:
            raise Exception(f"{conf.ls.type=} is not a known landscape type.")


def unit_to_float(x: NDArray[Any], lower: float, upper: float) -> NDArray[Any]:
    """For Sobol ls, type, Float dim type. Maps [0, 1] interval to [lower, upper]."""
    return (x * (upper - lower)) + lower


def float_to_unit(x: NDArray[Any], lower: float, upper: float) -> NDArray[Any]:
    """For Sobol ls, type, Float dim type. Maps to [0, 1] interval."""
    return (x - lower) / (upper - lower)


def _log_base(x: NDArray[Any], base: float) -> NDArray[Any]:
    return np.log(x) / np.log(base)  # type: ignore[no-any-return]


def unit_to_log(x: NDArray[Any], base: float, lower: float, upper: float) -> NDArray[Any]:
    """For Sobol ls type, Log dim type. Maps [0, 1] interval to [lower, upper] with a logarithmic base."""
    return ((base**x - 1) / (base - 1)) * (upper - lower) + lower


def log_to_unit(x: NDArray[Any], base: float, lower: float, upper: float) -> NDArray[Any]:
    """For Sobol ls type, Log dim type. Maps to [0, 1] interval."""
    return _log_base(1 + (((x - lower) * (base - 1)) / (upper - lower)), base)


def _unit_samples_to_ls(conf: DictConfig, samples: NDArray[Any], dims: list[tuple[Any, Any]]) -> NDArray[Any]:
    configs = np.zeros((conf.num_confs, len(dims)), dtype="O")
    # Transform the sampled sobol numbers into actual configurations:
    s = 0  # sobol dim index
    for i, (dim_name, dim_args) in enumerate(dims):
        if dim_args["type"] == "Integer":
            configs[:, i] = np.round((samples[:, s] * (dim_args["upper"] - dim_args["lower"])) + dim_args["lower"])
            s += 1
        elif dim_args["type"] == "Float":
            # configs[:, i] = (samples[:, s] * (dim_args["upper"] - dim_args["lower"])) + dim_args["lower"]
            configs[:, i] = unit_to_float(samples[:, s], dim_args["lower"], dim_args["upper"])
            s += 1
        elif dim_args["type"] == "Log":
            configs[:, i] = unit_to_log(samples[:, s], dim_args["base"], dim_args["lower"], dim_args["upper"])
            s += 1
        elif dim_args["type"] == "Constant":
            assert hasattr(conf.agent.zoo_optimal_ls, dim_name), "You need to set default values for constant hps!"
            value = conf.agent.zoo_optimal_ls[dim_name]
            configs[:, i] = value
        else:
            raise Exception(f"Unknown ls dimension type: {dim_args['type']}")
    return configs
