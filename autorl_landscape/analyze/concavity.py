from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay

from autorl_landscape.ls_models.ls_model import LSModel


def find_biggest_nonconcave(model_builder: Callable[[float], LSModel], grid_length: int = 51) -> float:
    """For the `RBFInterpolatorLSModel`, find biggest CI where concavity can still be rejected."""

    def predicate(ci: float) -> bool:
        model = model_builder(ci=ci)
        return reject_concavity(model, grid_length)

    return binary_search(predicate)


def binary_search(predicate: Callable[[float], bool]) -> float:
    """Search for highest percentage in [0, 100] where the predicate holds. return that percentage in [0, 1].

    The predicate needs to be monotonically decreasing.
    """
    low = 0
    high = 100
    p_low = predicate(low / 100)
    p_high = predicate(high / 100)
    match p_low, p_high:
        case (True, False):  # good
            return _binary_search(low, high, predicate)
        case (False, True):
            raise Exception("Predicate definitely does not decrease monotonically!")
        case (False, False):  # i.e., IQM function is concave, can never reject concavity
            return 0.0
        case (True, True):  # i.e., even biggest interval does not allow for concavity
            return 1.0
        # case _:
        #     return -1.0


def _binary_search(low: int, high: int, predicate: Callable[[float], bool]) -> float:
    """Search for highest percentage in [0, 100] where the predicate holds. return that percentage in [0, 1].

    When calling this function, `predicate(low / 100)` should be `True` while `predicate(high / 100)` should be `False`.
    """
    if low + 1 == high:
        return low / 100
    middle = (low + high) // 2  # rounds down
    if predicate(middle / 100) is True:
        return _binary_search(middle, high, predicate)
    else:
        return _binary_search(low, middle, predicate)


def reject_concavity(model: LSModel, grid_length: int = 51) -> bool:
    """Akin to Pushak and Hoos, 2022."""
    grid_x0, grid_x1 = np.meshgrid(np.linspace(0, 1, num=grid_length), np.linspace(0, 1, num=grid_length))
    # TODO meshgrid N-dims
    grid_x0 = grid_x0.flatten()
    grid_x1 = grid_x1.flatten()
    x = np.stack((grid_x0, grid_x1), axis=1)  # (-1, num_ls_dims)

    y_lower = model.get_lower(x).reshape(-1, 1)
    y_upper = model.get_upper(x).reshape(-1, 1)
    lower = np.concatenate((x, y_lower), axis=1)
    upper = np.concatenate((x, y_upper), axis=1)

    # hull = ConvexHull(np.concatenate((grid, lower.reshape(-1, 1)), axis=1))
    hull = Delaunay(lower)
    b = any_in_hull(upper, hull)
    # print(b)
    # print(b.sum())
    # if any of the upper limit points are found in the lower hull, the concave function that is the upper half of the
    # lower hull does not lie within the confidence bounds at this point. Thus, we can then reject concavity.
    return bool(b.any())


# https://stackoverflow.com/questions/16750618/
# whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl
def any_in_hull(points: NDArray[Any], hull: Delaunay) -> NDArray[np.bool_]:
    """Test if points in `p` are in `hull`.

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` Delaunay tesselation representing a convex hull (Delaunay tesselations are always convex)

    """
    return hull.find_simplex(points) >= 0  # find_simplex returns -1 if the point is not in a simplex
