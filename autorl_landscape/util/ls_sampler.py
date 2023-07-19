from __future__ import annotations

from typing import Any, Callable

from dataclasses import dataclass
from numbers import Number

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
            error_msg = f"{conf.ls.type=} is not a known landscape type."
            raise ValueError(error_msg)


def unit_to_float(x: NDArray[Any], lower: float, upper: float) -> NDArray[Any]:
    """For Sobol ls, type, Float dim type. Maps [0, 1] interval to [lower, upper]."""
    return (x * (upper - lower)) + lower


def float_to_unit(x: NDArray[Any], lower: float, upper: float) -> NDArray[Any]:
    """For Sobol ls, type, Float dim type. Maps to [0, 1] interval."""
    return (x - lower) / (upper - lower)


# def _log_base(x: NDArray[Any], base: float) -> NDArray[Any]:
#     return np.log(x) / np.log(base)  # type: ignore[no-any-return]


# def unit_to_log(x: NDArray[Any], base: float, lower: float, upper: float) -> NDArray[Any]:
#     """For Sobol ls type, Log dim type. Maps [0, 1] interval to [lower, upper] with a logarithmic base."""
#     return ((base**x - 1) / (base - 1)) * (upper - lower) + lower


# def log_to_unit(x: NDArray[Any], base: float, lower: float, upper: float) -> NDArray[Any]:
#     """For Sobol ls type, Log dim type. Maps to [0, 1] interval."""
#     return _log_base(1 + (((x - lower) * (base - 1)) / (upper - lower)), base)


def unit_to_natural_log(x: NDArray[Any], lower: float, upper: float) -> NDArray[Any]:
    """For Sobol ls type, Log dim type. Maps [0, 1] interval to [lower, upper] with a logarithmic base."""
    return np.exp(unit_to_float(x, np.log(lower), np.log(upper)))


def natural_log_to_unit(x: NDArray[Any], lower: float, upper: float) -> NDArray[Any]:
    """For Sobol ls type, Log dim type. Maps to [0, 1] interval."""
    return (np.log(x) - np.log(lower)) / (np.log(upper) - np.log(lower))


def unit_to_flipped_natural_log(x: NDArray[Any], lower: float, upper: float) -> NDArray[Any]:
    """TODO."""
    # return lower + upper - unit_to_natural_log(1 - x, lower, upper)
    return 1 - np.exp(unit_to_float(1 - x, np.log(1 - upper), np.log(1 - lower)))


def flipped_natural_log_to_unit(x: NDArray[Any], lower: float, upper: float) -> NDArray[Any]:
    """TODO."""
    # return 1 - natural_log_to_unit(x, 1 - upper, 1 - lower)
    return 1 - ((np.log(1 - x) - np.log(1 - upper)) / (np.log(1 - lower) - np.log(1 - upper)))


def _unit_samples_to_ls(conf: DictConfig, samples: NDArray[Any], dims: list[tuple[Any, Any]]) -> NDArray[Any]:
    configs = np.zeros((conf.num_confs, len(dims)), dtype="O")
    # Transform the sampled sobol numbers into actual configurations:
    for i, (dim_name, dim_args) in enumerate(dims):
        d = LSDimension.from_dim_dict(dim_name, dim_args)
        configs[:, i] = d.unit_to_ls(samples[:, i])
    return configs


Transformer = Callable[[NDArray[Any]], NDArray[Any]]
Formatter = Callable[[Any, Any], str]

TICK_PRECISION = 4


@dataclass
class LSDimension:
    """Saves information about a hyperparameter landscape dimension."""

    name: str
    # dim_type: str
    # lower: float
    # upper: float
    unit_to_ls: Transformer
    ls_to_unit: Transformer
    tick_formatter: Formatter

    # base: float | None = None

    @classmethod
    def from_dim_dict(cls, dim_name: str, dim_dict: dict[str, Any], is_y: bool = False) -> LSDimension:
        """Constructs a `DimInfo` object given a name and dictionary as found in the exported data."""
        prefix = "" if is_y else "ls."
        dim_name_ = prefix + dim_name
        # if dim_name.startswith("neg_"):
        #     dim_name_ = prefix + dim_name.split("neg_")[-1]

        #     def negator(x: NDArray[Any]) -> NDArray[Any]:
        #         return 1 - x

        # else:
        #     dim_name_ = prefix + dim_name

        #     def negator(x: NDArray[Any]) -> NDArray[Any]:
        #         return x

        tick_precision = 0 if is_y else TICK_PRECISION
        match dim_dict["type"]:
            # case "Constant":
            #     return None  # ignore these, they are not really ls dims
            case "Float":
                lower = dim_dict["lower"]
                upper = dim_dict["upper"]
                u2f: Transformer = lambda x: unit_to_float(x, lower, upper)
                f2u: Transformer = lambda x: float_to_unit(x, lower, upper)
                fmt: Formatter = lambda val, _: f"{LSDimension._round(u2f(val), tick_precision)}"
                di = cls(dim_name_, u2f, f2u, fmt)
            case "Log":
                lower = dim_dict["lower"]
                upper = dim_dict["upper"]
                u2l: Transformer = lambda x: unit_to_natural_log(x, lower, upper)
                l2u: Transformer = lambda x: natural_log_to_unit(x, lower, upper)
                fmt: Formatter = lambda val, _: f"{LSDimension._round(u2l(val), tick_precision)}"
                di = cls(dim_name_, u2l, l2u, fmt)
            case "FlippedLog":
                lower = dim_dict["lower"]
                upper = dim_dict["upper"]
                # u2l: Transformer = lambda x: 1 - unit_to_natural_log(x, 1 - upper, 1 - lower)
                # l2u: Transformer = lambda x: 1 - natural_log_to_unit(x, 1 - upper, 1 - lower)
                u2l: Transformer = lambda x: unit_to_flipped_natural_log(x, lower, upper)
                l2u: Transformer = lambda x: flipped_natural_log_to_unit(x, lower, upper)
                fmt: Formatter = lambda val, _: f"{LSDimension._round(u2l(val), tick_precision)}"
                di = cls(dim_name_, u2l, l2u, fmt)
            case invalid_dim_name:
                error_msg = f"{invalid_dim_name=} found!"
                raise ValueError(error_msg)
        return di

    def __lt__(self, other: object) -> bool:
        """Sort based on dimension name."""
        if not isinstance(other, LSDimension):
            raise NotImplementedError
        return self.name < other.name

    @staticmethod
    def _round(number: Number, ndigits: int) -> Number:
        """Round to int if 0 precision, else normal `round` behaviour."""
        if ndigits == 0:
            return int(number)
        return round(number, ndigits)
