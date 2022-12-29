from typing import Any

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

ATOL = 1e-14


@dataclass
class Simplex:
    """For use with `SimplexInterpolator`, saves corner indices and a "function" to calculate y at a position x."""

    inds: NDArray[Any]
    fn: NDArray[Any]
    """Multiply fn with [x1, x2, ..., xn, 1] to get y(x)"""


class SimplexInterpolator:
    """Maps n-dimensional x to 1-dimensional y.

    In the n+1-dimensional space, x is modelled as a "surface" made up of n-simplices (hyperplanes). The height of the
    surface at any x position is the returned y value.

    For example, if x is 2-dimensional, we construct a surface out of 2-simplices, which are triangles.

    Args:
        x: (num_points, n)-shaped
        y: (num_points, 1)-shaped
        simplices_inds: (num_facets, n + 1)-shaped indices indexing x, y of corners of each simplex
    """

    def __init__(self, x: NDArray[Any], y: NDArray[Any], simplices_inds: NDArray[Any]) -> None:
        # dimension checks:
        num_points = x.shape[0]
        assert y.shape[0] == num_points and y.shape[1] == 1
        assert np.min(simplices_inds) >= 0 and np.max(simplices_inds) < num_points
        self.n = x.shape[1]
        assert simplices_inds.shape[1] == self.n + 1

        self.x = x
        """(num_points, n)"""
        self.y = y
        """(num_points, 1)"""
        self.simplices = [
            Simplex(simplex_inds, self._build_simplex_function(simplex_inds)) for simplex_inds in simplices_inds
        ]
        """(num_facets, n + 1) indices indexing x, y of corners of each simplex"""

    def __call__(self, points: NDArray[Any]) -> NDArray[Any]:
        """Return the interpolated value `y` at `x`."""
        # for every point:
        ret = np.zeros((points.shape[0], 1))
        for i, point in enumerate(points):
            # find the simplex that x is in:
            found = False
            for simplex in self.simplices:
                # simplex = self.x[simplex_corners]
                simplex_ = self._get_simplex_corners(simplex.inds)
                if belongs_to_simplex(point, simplex_):
                    ret[i] = self._apply_simplex_fn(simplex.fn, point)
                    found = True
            if not found:
                ret[i] = float("nan")
        return ret

    def _apply_simplex_fn(self, fn: NDArray[Any], point: NDArray[Any]):
        return np.dot(fn, np.concatenate([point, [1]]))

    def _build_simplex_function(self, simplex_inds: NDArray[Any]) -> Any:
        """Hyperplane function for a simplex."""
        a = self._get_simplex_corners(simplex_inds)
        b = np.ones((a.shape[1],))
        x = np.dot(np.linalg.inv(a), b)
        v_y = x[-1]
        x[-1] = 1 / v_y
        x[0:-1] = (-x[0:-1]) / v_y
        # TODO if running into singular matrix errors, add 1 to y values and later subtract...
        return x
        # https://math.stackexchange.com/questions/2723294/how-to-determine-the-equation-of-the-hyperplane-that-contains-several-points
        # Accepted answer, null space method thing
        # or?:
        # https://math.libretexts.org/Bookshelves/Calculus/The_Calculus_of_Functions_of_Several_Variables_(Sloughter)/01%3A_Geometry_of_R/1.04%3A_Lines_Planes_and_Hyperplanes
        # Example 1.4.6

    def _get_simplex_corners(self, simplex_inds: NDArray[Any]) -> NDArray[Any]:
        """Go from indices to actual points in the (n + 1)-dimensional space."""
        return np.concatenate([self.x, self.y], axis=1)[simplex_inds]


def belongs_to_simplex(point: NDArray[Any], simplex: NDArray[Any]) -> bool:
    """Check whether a (`n`-dimensional) x-position is inside a (`n + 1`-dimensional) simplex, ignoring its height y."""
    # check dimensions:
    point_dim = point.shape[0]
    assert simplex.shape == (point_dim + 1, point_dim + 1)

    # translated from https://discourse.julialang.org/t/find-simplex-that-contains-point-in-triangulation/36181:
    a = simplex
    a[:, -1] = 1  # reuse y space for our ones-column
    a = a.T
    b = np.concatenate([point, [1]])
    x = np.linalg.solve(a, b)
    return bool(np.all(x >= ATOL))


if __name__ == "__main__":
    # small check:
    S = np.array([[0, 0, 0, 100], [1, 0, 0, 100], [0.5, 1, 0, 100], [0.5, 0.5, 1, 100]])  # 100's are ignored
    print(belongs_to_simplex(np.array([0.5, 0.5, 0.5]), S))  # True
    print(belongs_to_simplex(np.array([1, 1, 1]), S))  # False
    si = SimplexInterpolator(
        np.array([[1, 1], [2, 1], [2, 2], [3, 2]]),
        np.array([1, 1, 1, 2]).reshape(-1, 1),
        np.array([[0, 1, 2], [1, 2, 3]]),
    )
    # si = SimplexInterpolator(
    #     np.array([[4, 0, -1], [1, 2, 3], [0, -1, 2], [-1, 1, -1]]),
    #     np.array([0, -1, 0, 1]).reshape(-1, 1),
    #     np.array([[0, 1, 2, 3]]),
    # )
    print(si.simplices)
    print(si.simplices[0].fn @ np.array([1.5, 1, 1]))
    print(si.simplices[1].fn @ np.array([2.5, 2, 1]))
    print(si(np.array([[1.5, 1], [2.5, 2], [1, 1]])))
