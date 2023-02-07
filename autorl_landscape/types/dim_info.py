from __future__ import annotations

from typing import Any, Callable

from dataclasses import dataclass

from numpy.typing import NDArray

from autorl_landscape.util.ls_sampler import (
    float_to_unit,
    log_to_unit,
    unit_to_float,
    unit_to_log,
)

Transformer = Callable[[NDArray[Any]], NDArray[Any]]
Formatter = Callable[[Any, Any], str]

TICK_PRECISION = 4


@dataclass
class DimInfo:
    """Saves information about a hyperparameter landscape dimension."""

    name: str
    dim_type: str
    lower: float
    upper: float
    unit_to_ls: Transformer
    ls_to_unit: Transformer
    tick_formatter: Formatter

    base: float | None = None

    @classmethod
    def from_dim_dict(cls, dim_name: str, dim_dict: dict[str, Any], is_y: bool = False) -> DimInfo | None:
        """Constructs a `DimInfo` object given a name and dictionary as found in the exported data."""
        prefix = "" if is_y else "ls."
        if dim_name.startswith("neg_"):
            dim_name_ = prefix + dim_name.split("neg_")[-1]

            def negator(x: NDArray[Any]) -> NDArray[Any]:
                return 1 - x

        else:
            dim_name_ = prefix + dim_name

            def negator(x: NDArray[Any]) -> NDArray[Any]:
                return x

        tick_precision = 0 if is_y else TICK_PRECISION
        match dim_dict["type"]:
            case "Constant":
                return None  # ignore these, they are not really ls dims
            case "Float":
                lower = dim_dict["lower"]
                upper = dim_dict["upper"]
                u2f: Transformer = lambda x: negator(unit_to_float(x, lower, upper))
                f2u: Transformer = lambda x: float_to_unit(negator(x), lower, upper)
                fmt: Formatter = lambda val, _: f"{round_(u2f(val), tick_precision)}"
                di = cls(dim_name_, "Float", lower, upper, u2f, f2u, fmt)
            case "Log":
                lower = dim_dict["lower"]
                upper = dim_dict["upper"]
                b = dim_dict["base"]
                u2l: Transformer = lambda x: negator(unit_to_log(x, b, lower, upper))
                l2u: Transformer = lambda x: log_to_unit(negator(x), b, lower, upper)
                fmt: Formatter = lambda val, _: f"{round_(u2l(val), tick_precision)}"
                di = cls(dim_name_, "Log", lower, upper, u2l, l2u, fmt, b)
            case weird_val:
                raise Exception(f"Weird dimension type {weird_val} found!")
        return di

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, DimInfo):
            raise NotImplementedError
        return self.name < other.name


def round_(number: Any, ndigits: int) -> Any:
    """Round to int if 0 precision, else normal `round` behaviour."""
    if ndigits == 0:
        return int(number)
    return round(number, ndigits)
