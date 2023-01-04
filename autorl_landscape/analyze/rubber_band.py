from typing import Any

from dataclasses import dataclass
from itertools import compress

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import label
from scipy.spatial import ConvexHull

from autorl_landscape.ls_models.ls_model import LSModel, Visualization

ATOL = 1e-14
LOWER_ANCHOR_Y = -1000.0
UPPER_ANCHOR_Y = 1001.0


class RubberBand:
    """Calculate the rubber band (or rather rubber surface) between the lower and upper estimate of an `LSModel`.

    Consider:
    - lower half of the convex hull of the upper confidence bounds
    - upper half of the convex hull of the lower confidence bounds

    Only one is true about:
    - the convex hull halves have vertices at same x
    - the convex hulls overlap

    We want to calculate the "upper (optimistic) rubber band" or "lower (pessimistic) rubber band" between the halves.
    If the hulls do not overlap, the rubber band will be equal to the upper or lower hull.
    """

    def __init__(self, model: LSModel, side: str, grid_length: int = 51) -> None:
        self.model = model
        self.grid_length = grid_length
        assert side in ["lower", "upper"]
        self.side = side

        # build convex hulls:
        grid_x0, grid_x1 = np.meshgrid(np.linspace(0, 1, num=grid_length), np.linspace(0, 1, num=grid_length))
        # TODO meshgrid N-dims
        assert len(self.model.dim_info) == 2
        grid_shape = grid_x0.shape
        grid_x0 = grid_x0.flatten()
        grid_x1 = grid_x1.flatten()
        x = np.stack((grid_x0, grid_x1), axis=1)  # (-1, num_ls_dims)

        y_lower = model.get_lower(x).reshape(-1, 1)
        y_upper = model.get_upper(x).reshape(-1, 1)
        lower_points = np.concatenate((x, y_lower), axis=1)
        upper_points = np.concatenate((x, y_upper), axis=1)

        # start with convex hull of all lower/upper points:
        if side == "lower":
            relevant_points = lower_points
            ch_points, ch_simplices = get_half_of_convex_hull(relevant_points, "upper")
        else:
            relevant_points = upper_points
            ch_points, ch_simplices = get_half_of_convex_hull(relevant_points, "lower")
        rb = SimplexInterpolator(ch_points[:, 0:-1], ch_points[:, -1].reshape(-1, 1), ch_simplices)
        i = 0
        while True:
            print(side)
            self.model.add_viz_info(
                Visualization(
                    "trisurf",
                    rb.points[:, 0:-1],
                    rb.points[:, -1].reshape(-1, 1),
                    f"rubber band {i}",
                    {"color": "blue", "shade": True, "triangles": [s.inds for s in rb.simplices]},
                )
            )
            # find "bad" spots of the rubber band and group them by direct (non-diagonal) adjacency:
            oob_mask = rb.get_oob_mask(x, y_lower, y_upper).reshape(grid_shape)
            print(np.sum(oob_mask))
            if np.sum(oob_mask) <= 0 or i > 10:
                break
            clusters, num_clusters = label(oob_mask)
            if side == "lower":
                for j in range(1, num_clusters + 1):
                    cluster_mask = (clusters == j).reshape(-1)
                    relevant_points = upper_points[cluster_mask]
                    _ = rb.replace_simplices(relevant_points, side)
                # all_relevant_points = upper_points[oob_mask.reshape(-1)]
                side = "upper"
            else:
                for j in range(1, num_clusters + 1):
                    cluster_mask = (clusters == j).reshape(-1)
                    relevant_points = lower_points[cluster_mask]
                    _ = rb.replace_simplices(relevant_points, side)
                # all_relevant_points = lower_points[oob_mask.reshape(-1)]
                side = "lower"

            # self.model.add_viz_info(
            #     Visualization(
            #         "scatter",
            #         all_relevant_points[:, 0:-1],
            #         all_relevant_points[:, -1].reshape(-1, 1),
            #         f"oob points {i}",
            #         {"color": "blue"},
            #     )
            # )
            i += 1

        print(_is_valid_rubber_band(rb, model, x))
        return


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

        # self.x = x
        # """(num_points, n)"""
        # self.y = y
        # """(num_points, 1)"""
        self.points = np.concatenate([x, y], axis=1)
        """(num_points, n + 1)"""
        self.simplices = [
            Simplex(simplex_inds, self._build_simplex_function(simplex_inds)) for simplex_inds in simplices_inds
        ]
        """(num_facets, n + 1) indices indexing x, y of corners of each simplex"""

    def __call__(self, points: NDArray[Any]) -> NDArray[Any]:
        """Return the interpolated value `y` at `x`."""
        # for every point:
        ret = np.zeros((points.shape[0], 1))
        for i, point in enumerate(points):
            simplices = self._get_including_simplices(point)
            if not simplices:
                ret[i] = float("nan")
            else:
                ret[i] = np.mean([self._apply_simplex_fn(s[1].fn, point) for s in simplices])
        return ret

    def get_oob_mask(self, x: NDArray[Any], y_lower: NDArray[Any], y_upper: NDArray[Any]) -> NDArray[np.bool_]:
        """For given x positions and corresponding lower and upper y values, check that we are in bounds."""
        y_si = self(x)
        return (y_si < y_lower - ATOL) | (y_si > y_upper + ATOL)

    def replace_simplices(self, points: NDArray[Any], side: str) -> Any:
        """Delete simplices at points and replace with new ones from a new convex hull of the points."""
        # delete including simplices, remembering all the corners:
        touched_corner_inds: set[int] = set()
        including_simplices_mask = np.full((len(self.simplices),), fill_value=False, dtype=np.bool_)
        for i, simplex in enumerate(self.simplices):
            for point in points[:, 0:-1]:
                if belongs_to_simplex(point, self._get_simplex_corners(simplex.inds)):
                    touched_corner_inds.update(simplex.inds)
                    including_simplices_mask[i] = True
                    # del self.simplices[i]
                    break
        self.simplices = list(compress(self.simplices, ~including_simplices_mask))

        # collect corners that are not dangling:
        # corners that still have a simplex:
        incident_corner_inds = set(np.array([s.inds for s in self.simplices]).flatten())
        # corners on the outside border of the landscape (should never be removed):
        border_corner_inds = set(np.where(np.any((self.points[:, 0:-1] == 0) | (self.points[:, 0:-1] == 1), axis=1))[0])

        # corners that should be included in the new hull:
        surrounding_corner_inds = touched_corner_inds & (incident_corner_inds | border_corner_inds)
        surrounding_corners = self.points[list(surrounding_corner_inds)]

        # build new convex half from the points and add those points and simplices to the model:
        hull_input = np.concatenate([surrounding_corners, points], axis=0)
        hull_points, hull_simplices = get_half_of_convex_hull(hull_input, side)
        hull_simplices += self.points.shape[0]
        self.points = np.concatenate([self.points, hull_points], axis=0)
        for hull_simplex in hull_simplices:
            self._add_simplex(hull_simplex)
        return hull_input

    def _get_including_simplices(self, point: NDArray[Any]) -> list[tuple[int, Simplex]]:
        """Return the simplex that `point` is in, or `None` if the point is not included in any simlex."""
        ret: list[tuple[int, Simplex]] = []
        for i, simplex in enumerate(self.simplices):
            if belongs_to_simplex(point, self._get_simplex_corners(simplex.inds)):
                ret.append((i, simplex))
        # assert len(ret) <= 2, "1 means inside a simplex, 2 means on border of 2 simplices, more means on corner"
        return ret

    def _apply_simplex_fn(self, fn: NDArray[Any], point: NDArray[Any]):
        return np.dot(fn, np.concatenate([point, [1]]))

    def _build_simplex_function(self, simplex_inds: NDArray[Any]) -> NDArray[Any] | None:
        """Hyperplane function for a Simplex. Returns `None` if the Simplex is degenerated.

        Degenerated Simplices are those that are invalid, i.e. corners are on a line (hyperplane).
        """
        a = self._get_simplex_corners(simplex_inds)

        # NOTE not sure if this is correct:
        # if np.linalg.det(a) == 0:
        #     print(f"Found weird Simplex {a}")
        #     return None

        # if all of the points lie on one edge of the landscape, we have a degenerated edge simplex which we reject:
        if np.any(np.all(a[:, 0:-1] == 0, axis=0)) or np.any(np.all(a[:, 0:-1] == 1, axis=0)):
            return None

        # if y values of all corners of the simplex are equal, we can return a constant "function":
        if np.all(a[:, -1] == a[0, -1]):
            x = np.zeros((a.shape[1],))
            x[-1] = a[0, -1]
            return x

        b = np.ones((a.shape[1],))
        x = np.dot(np.linalg.inv(a), b)
        v_y = x[-1]
        x[-1] = 1 / v_y
        x[0:-1] = (-x[0:-1]) / v_y
        # TODO if running into singular matrix errors, add 1 to y values and later subtract...
        if np.inf in x or -np.inf in x:
            pass
        return x
        # https://math.stackexchange.com/questions/2723294/how-to-determine-the-equation-of-the-hyperplane-that-contains-several-points
        # Accepted answer, null space method thing
        # or?:
        # https://math.libretexts.org/Bookshelves/Calculus/The_Calculus_of_Functions_of_Several_Variables_(Sloughter)/01%3A_Geometry_of_R/1.04%3A_Lines_Planes_and_Hyperplanes
        # Example 1.4.6

    def _get_simplex_corners(self, simplex_inds: NDArray[Any]) -> NDArray[Any]:
        """Go from indices to actual points in the (n + 1)-dimensional space."""
        return self.points[simplex_inds]

    def _add_simplex(self, corners: NDArray[Any]) -> None:
        """Adds a simplex to the set if it is not degenerated."""
        fn = self._build_simplex_function(corners)
        if fn is not None:
            self.simplices.append(Simplex(corners, fn))

    # def add_points(self, points: NDArray[Any]) -> None:
    #     """Add points to the simplex (buggy)."""
    #     assert points.shape[1] == self.n + 1
    #     for point in points:
    #         if any(np.all(np.equal(point, self.points), axis=1)):
    #             print(f"Point {point} already registered in SimplexInterpolator.")
    #             continue

    #         simplices = self._get_including_simplices(point[0:-1])
    #         match len(simplices):
    #             case 0:
    #                 print(f"Point {point} is out of bounds of the SimplexInterpolator.")
    #                 continue

    #             # easy case, point is interior to another simplex:
    #             case 1:
    #                 # remove old simplex:
    #                 i, simplex = simplices[0]
    #                 del self.simplices[i]

    #                 # add point and (n + 1) new simplices:
    #                 self.points = np.concatenate([self.points, point.reshape(1, -1)], axis=0)
    #                 corners_set = simplex.inds
    #                 for j in range(len(corners_set)):
    #                     corners_mask = np.full_like(corners_set, fill_value=True, dtype=np.bool_)
    #                     corners_mask[j] = False
    #                     corners = np.concatenate([corners_set[corners_mask], [len(self.points) - 1]], axis=0)
    #                     # self.simplices.append(Simplex(corners, self._build_simplex_function(corners)))
    #                     self._add_simplex(corners)

    #             case 2:
    #                 # remove old simplices:
    #                 (i0, simplex0), (i1, simplex1) = simplices
    #                 del self.simplices[max(i0, i1)]
    #                 del self.simplices[min(i0, i1)]

    #                 # add point and (2 * (n choose (n - 1))) new simplices:
    #                 self.points = np.concatenate([self.points, point.reshape(1, -1)], axis=0)
    #                 inter_corners_set = np.intersect1d(simplex0.inds, simplex1.inds)
    #                 exter_corners_set = np.setdiff1d(np.union1d(simplex0.inds, simplex1.inds), inter_corners_set)
    #                 assert len(inter_corners_set) == len(simplex0.inds) - 1
    #                 assert len(exter_corners_set) == 2
    #                 for j in range(len(inter_corners_set)):
    #                     corners_mask = np.full_like(inter_corners_set, fill_value=True, dtype=np.bool_)
    #                     corners_mask[j] = False

    #                     for exter_corner in exter_corners_set:
    #                         corners = np.concatenate(
    #                             [inter_corners_set[corners_mask], [exter_corner, len(self.points) - 1]], axis=0
    #                         )
    #                         # self.simplices.append(Simplex(corners, self._build_simplex_function(corners)))
    #                         self._add_simplex(corners)

    #             case x:
    #                 print(f"Point {point} is in bounds of {x} Simplices.")


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
    return bool(np.all(x >= -ATOL))
    # return bool(np.all(x >= 0.0))  # WARNING might have rounding issues!


def get_half_of_convex_hull(points: NDArray[Any], side: str) -> tuple[NDArray[Any], NDArray[Any]]:
    """Construct a convex hull around the points and return either the lower or upper half of it."""
    assert side in ["lower", "upper"]
    anchor = UPPER_ANCHOR_Y if side == "lower" else LOWER_ANCHOR_Y
    points_ = np.concatenate((points, _get_anchor_points(points.shape[1] - 1, anchor)), axis=0)
    hull = ConvexHull(points_)

    # add all points of the upper half (aka. uh) of the convex hull:
    hull_inds = hull.vertices
    other_half_inds = np.where(points_[:, -1] == anchor)
    this_half_inds = np.setdiff1d(hull_inds, other_half_inds)
    this_half_points = points_[this_half_inds]
    ind_converter_lookup = {uhi: i for i, uhi in enumerate(this_half_inds)}
    ind_converter = np.vectorize(lambda x: ind_converter_lookup[x])

    # find all simplices of the upper half of the convex hull:
    this_half_simplices_mask = np.all(np.isin(hull.simplices, this_half_inds), axis=1)
    this_half_simplices = hull.simplices[this_half_simplices_mask]
    this_half_simplices = ind_converter(this_half_simplices)

    return this_half_points, this_half_simplices


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


def _is_valid_rubber_band(rb: SimplexInterpolator, model: LSModel, x: NDArray[Any]) -> bool:
    rb_pred = rb(x)
    upper = model.get_upper(x)
    lower = model.get_lower(x)
    # above_upper = ((rb_pred > upper) & (~np.isclose(rb_pred, upper))).flatten()
    # below_lower = ((rb_pred < lower) & (~np.isclose(rb_pred, lower))).flatten()
    above_upper = (rb_pred > upper + ATOL).flatten()
    below_lower = (rb_pred < lower - ATOL).flatten()
    return (True not in above_upper) and (True not in below_lower)
    # model.add_viz_info(
    #     Visualization(
    #         "scatter",
    #         x[above_upper],
    #         rb(x[above_upper]),
    #         f"rb is above upper model here ({np.sum(above_upper)})",
    #         {"color": "black", "marker": "^"},
    #     )
    # )
    # model.add_viz_info(
    #     Visualization(
    #         "scatter",
    #         x[below_lower],
    #         rb(x[below_lower]),
    #         f"rb is below lower model here ({np.sum(below_lower)})",
    #         {"color": "black", "marker": "v"},
    #     )
    # )
    # return False


def _get_anchor_points(num_ls_dims: int, y_val: float) -> NDArray[Any]:
    """For `n` dims, returns `2 ** n` points at the corners of the n-dimensional hypercube."""
    ls = np.array([-0.1, 1.1])
    grid = np.meshgrid(*[ls] * num_ls_dims)
    grid_flat: list[NDArray[Any]] = []
    for g in grid:
        grid_flat.append(g.flatten())
    grid_ = np.stack(grid_flat, axis=1)  # (-1, num_ls_dims)

    # cat y column on the right:
    grid_xy = np.concatenate((grid_, np.full((grid_.shape[0], 1), y_val)), axis=1)
    return grid_xy
