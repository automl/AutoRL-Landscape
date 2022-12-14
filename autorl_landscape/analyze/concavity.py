from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay

from autorl_landscape.ls_models.ls_model import LSModel


def reject_concavity(model: LSModel, grid_length: int = 51) -> bool:
    """Akin to Pushak and Hoos, 2022."""
    grid_x0, grid_x1 = np.meshgrid(np.linspace(0, 1, num=grid_length), np.linspace(0, 1, num=grid_length))
    grid_x0 = grid_x0.flatten()
    grid_x1 = grid_x1.flatten()
    x = np.stack((grid_x0, grid_x1), axis=1)  # (-1, num_ls_dims)

    y_lower = model.get_lower(x).reshape(-1, 1)
    y_upper = model.get_upper(x).reshape(-1, 1)
    lower = np.concatenate((x, y_lower), axis=1)
    upper = np.concatenate((x, y_upper), axis=1)

    # hull = ConvexHull(np.concatenate((grid, lower.reshape(-1, 1)), axis=1))
    hull = Delaunay(lower)
    b = in_hull(upper, hull)
    print(b)
    print(b.sum())
    return b.any()


# https://stackoverflow.com/questions/16750618/
# whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl
def in_hull(points: NDArray[Any], hull: Delaunay) -> NDArray[np.bool_]:
    """Test if points in `p` are in `hull`.

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is a scipy.spatial.Delaunay object.

    """
    return hull.find_simplex(points) >= 0  # find_simplex returns -1 if the point is not in a simplex
