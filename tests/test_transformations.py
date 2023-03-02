import numpy as np

from autorl_landscape.util.ls_sampler import float_to_unit, natural_log_to_unit, unit_to_float, unit_to_natural_log


def test_inverse_functions() -> None:
    """See if unit -> LS and LS -> unit transformations work by trying unit -> LS -> unit and LS -> unit -> LS."""
    grid_length = 11
    unit_grid = np.linspace(0, 1, grid_length)
    # bound_pairs = [(0.0001, 0.1), (0.1, 0.2), (0.8, 0.9999)]
    bound_pairs = [(0.1, 0.2), (0.8, 0.9999)]
    function_pairs = [
        (unit_to_float, float_to_unit),
        (unit_to_natural_log, natural_log_to_unit),
        # (inverse_unit_to_natural_log, inverse_natural_log_to_unit),
    ]
    for lb, ub in bound_pairs:
        ls_grid = np.linspace(lb, ub, grid_length)
        for u2ls, ls2u in function_pairs:
            # for x in unit_grid:
            #     v = u2ls(x, lb, ub)
            #     x_ = ls2u(v, lb, ub)
            #     assert isclose(x, x_)
            # for x in ls_grid:
            #     v = ls2u(x, lb, ub)
            #     x_ = u2ls(x, lb, ub)
            #     assert isclose(x, x_)
            assert np.allclose(unit_grid, ls2u(u2ls(unit_grid, lb, ub), lb, ub))
            assert np.allclose(ls_grid, u2ls(ls2u(ls_grid, lb, ub), lb, ub))
