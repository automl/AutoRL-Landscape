import tensorflow_probability as tfp


def build_truncated_normal(low: float, high: float) -> type:
    """Build a TruncatedNormal class that has the same constructor as a Normal class."""

    class SpecificTruncatedNormal(tfp.distributions.TruncatedNormal):  # type: ignore[no-any-unimported]
        def __init__(
            self,
            loc: float,
            scale: float,
            validate_args: bool = False,
            allow_nan_stats: bool = True,
            name: str = "SpecificTruncatedNormal",
        ) -> None:
            super().__init__(loc, scale, low, high, validate_args, allow_nan_stats, name)

    return SpecificTruncatedNormal
