from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import (
    binary_dilation,
    generic_filter,
    maximum_filter,
    minimum_filter,
)

from autorl_landscape.ls_models.ls_model import LSModel, Visualization


def count_peaks_model(
    model: LSModel,
    method_name: str,
    num_dims: int,
    grid_length: int,
    bounds: tuple[float, float],
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], int, int]:
    """Count local minima and maxima, given a model and method of it."""
    func: Callable[[NDArray[Any]], NDArray[Any]] = getattr(model, method_name)
    grid = grid_space_nd(num_dims, grid_length, bounds=bounds)
    y_grid = func(grid).squeeze()
    assert y_grid.shape == grid.shape[0:-1]

    min_mask, max_mask, min_count, max_count = count_peaks(y_grid, num_dims)
    model.add_viz_info(
        Visualization(
            "scatter",
            grid[min_mask],
            y_grid[min_mask].reshape(-1, 1),
            "local minima",
            {"color": "red", "alpha": 0.75, "marker": "v"},
        )
    )
    model.add_viz_info(
        Visualization(
            "scatter",
            grid[max_mask],
            y_grid[max_mask].reshape(-1, 1),
            "local maxima",
            {"color": "red", "alpha": 0.75, "marker": "^"},
        )
    )
    return min_mask, max_mask, min_count, max_count


def count_peaks(y_grid: NDArray[Any], num_dims: int) -> tuple[NDArray[np.bool_], NDArray[np.bool_], int, int]:
    """Count local minima and maxima of given data."""
    # find plateaus (these could be local minima, maxima, or saddle points):
    def any_equal_to_middle(x: NDArray[Any]):
        middle = x[len(x) // 2]
        return np.sum(x == middle) >= 2

    unequal_y = np.min(y_grid) - 1
    plateau_mask = generic_filter(y_grid, any_equal_to_middle, output=np.bool_, size=3, mode="constant", cval=unequal_y)

    # find simple minima and maxima:
    min_mask = (minimum_filter(y_grid, size=3, mode="reflect") == y_grid) & ~plateau_mask
    max_mask = (maximum_filter(y_grid, size=3, mode="reflect") == y_grid) & ~plateau_mask
    min_count = np.sum(min_mask)
    max_count = np.sum(max_mask)

    # handle each plateau separately, check full surroundings:
    structure = np.ones(shape=(3,) * num_dims)
    plat_ys = np.where(plateau_mask, y_grid, unequal_y)
    for plat_y in np.unique(plat_ys)[1:]:
        plateau_mask = plat_ys == plat_y
        surrounding_mask = binary_dilation(plateau_mask, structure=structure) & ~plateau_mask
        plateau_height = y_grid[plateau_mask][0]
        surrounding = y_grid[surrounding_mask]
        if np.all(surrounding > plateau_height):
            min_mask[plateau_mask] = True
            min_count += 1
        elif np.all(surrounding < plateau_height):
            max_mask[plateau_mask] = True
            max_count += 1
    return min_mask, max_mask, min_count, max_count


def grid_space_nd(
    num_dims: int, grid_length: int, dtype: type = np.float64, bounds: tuple[float, float] = (0.0, 1.0)
) -> NDArray[Any]:
    """Generate a `num_dims` dimensional grid of shape (*(grid_length,) * num_dims, num_dims)."""
    axis = np.linspace(bounds[0], bounds[1], num=grid_length, dtype=dtype)
    grid_xis = np.meshgrid(*[axis] * num_dims)
    return np.stack(grid_xis).T
