from typing import Any

import numpy as np
from numpy.typing import NDArray


def grid_space_2d(length: int, dtype: type) -> NDArray[Any]:
    """Make grid in 2d unit cube. Returned array has shape (length ** 2, 2).

    Args:
        length: number of points on one side of the grid, aka. sqrt of the number of points returned.
        dtype: dtype for the grid.
    """
    axis = np.linspace(0, 1, num=length, dtype=dtype)
    grid_x, grid_y = np.meshgrid(axis, axis)
    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()
    return np.stack((grid_x, grid_y), axis=1)


def grid_space_nd(
    num_dims: int, grid_length: int, dtype: type = np.float64, bounds: tuple[float, float] = (0.0, 1.0)
) -> NDArray[Any]:
    """Generate a `num_dims` dimensional grid of shape (*(grid_length,) * num_dims, num_dims)."""
    axis = np.linspace(bounds[0], bounds[1], num=grid_length, dtype=dtype)
    grid_xis = np.meshgrid(*[axis] * num_dims)
    return np.stack(grid_xis).T
