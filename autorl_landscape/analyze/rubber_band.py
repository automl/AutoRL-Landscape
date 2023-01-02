from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull, Delaunay

from autorl_landscape.ls_models.ls_model import LSModel, Visualization
from autorl_landscape.util.simplex import ATOL, SimplexInterpolator

LOWER_ANCHOR_Y = -10.0
UPPER_ANCHOR_Y = 11.0


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
        grid_x0 = grid_x0.flatten()
        grid_x1 = grid_x1.flatten()
        x = np.stack((grid_x0, grid_x1), axis=1)  # (-1, num_ls_dims)

        y_lower = model.get_lower(x).reshape(-1, 1)
        y_upper = model.get_upper(x).reshape(-1, 1)
        lower = np.concatenate((x, y_lower), axis=1)
        upper = np.concatenate((x, y_upper), axis=1)
        # add "anchor points" to the convex hulls so that we can easily differentiate between lower and upper halves:
        lower = np.concatenate((lower, _get_anchor_points(len(self.model.dim_info), LOWER_ANCHOR_Y)), axis=0)
        upper = np.concatenate((upper, _get_anchor_points(len(self.model.dim_info), UPPER_ANCHOR_Y)), axis=0)

        self.lower_hull = ConvexHull(lower)
        self.lower_delaunay = Delaunay(lower)
        self.upper_hull = ConvexHull(upper)
        self.upper_delaunay = Delaunay(upper)
        match side:
            case "lower":
                # add all points of the upper half of the lower convex hull:
                hull_inds = self.lower_hull.vertices
                lower_half_inds = np.where(lower[:, -1] == LOWER_ANCHOR_Y)
                upper_half_inds = np.setdiff1d(hull_inds, lower_half_inds)
                rb_points = lower[upper_half_inds]
                ind_converter_lookup = {uhi: i for i, uhi in enumerate(upper_half_inds)}
                ind_converter = np.vectorize(lambda x: ind_converter_lookup[x])

                # find all simplices of the upper half of the lower convex hull:
                upper_half_simplices_mask = np.all(np.isin(self.lower_hull.simplices, upper_half_inds), axis=1)
                rb_simplices = self.lower_hull.simplices[upper_half_simplices_mask]
                rb_simplices = ind_converter(rb_simplices)
                # rb_points = lower[np.unique(self.lower_delaunay.convex_hull.flatten())]
            case "upper":
                # same as above, but flipped basically:

                # add all points of the lower half of the upper convex hull:
                hull_inds = self.upper_hull.vertices
                upper_half_inds = np.where(upper[:, -1] == UPPER_ANCHOR_Y)
                lower_half_inds = np.setdiff1d(hull_inds, upper_half_inds)
                rb_points = upper[lower_half_inds]
                ind_converter_lookup = {uhi: i for i, uhi in enumerate(lower_half_inds)}
                ind_converter = np.vectorize(lambda x: ind_converter_lookup[x])

                # find all simplices of the lower half of the upper convex hull:
                lower_half_simplices_mask = np.all(np.isin(self.upper_hull.simplices, lower_half_inds), axis=1)
                rb_simplices = self.upper_hull.simplices[lower_half_simplices_mask]
                rb_simplices = ind_converter(rb_simplices)
                # rb_points = upper[np.unique(self.upper_delaunay.convex_hull.flatten())]
            case x:
                raise NotImplementedError

        rb = SimplexInterpolator(rb_points[:, 0:-1], rb_points[:, -1].reshape(-1, 1), rb_simplices)
        print(_is_valid_rubber_band(rb, model, x))

        self.model.add_viz_info(
            Visualization(
                "trisurf",
                rb.points[:, 0:-1],
                rb.points[:, -1].reshape(-1, 1),
                "rubber band before",
                {"color": "blue", "shade": True, "triangles": [s.inds for s in rb.simplices]},
            )
        )

        points_mask = np.full(rb_points.shape[0], fill_value=True, dtype=np.bool_)
        # add all points that lie in the intersection of both hulls (these don't need to be part of the convex hull):
        all_points = np.concatenate((lower, upper), axis=0)
        intersection_mask = (self.lower_delaunay.find_simplex(all_points) >= 0) & (
            self.upper_delaunay.find_simplex(all_points) >= 0
        )
        rb.add_points(all_points[intersection_mask])
        print(_is_valid_rubber_band(rb, model, x))
        # rb_points = np.concatenate((rb_points, all_points[intersection_inds]), axis=0)
        # rb_points = np.unique(rb_points, axis=0)

        # minimize points of the rubber band, see if it is still valid:
        # while True:
        #     num_deactivated = np.sum(points_mask == False)
        #     for point_mask, i in enumerate(points_mask):
        #         if point_mask == False:  # point is already deactivated and fine
        #             continue
        #         points_mask[i] = False
        #         # use a linear interpolator to model the rubber band as a function:
        #         rb = LinearInterpolator(rb_points[points_mask, 0:-1], rb_points[points_mask, -1].reshape(-1, 1))
        #         if not _is_valid_rubber_band(rb, model, x):
        #             points_mask[i] = True  # undo removal
        #         else:
        #             print("removed a point")
        #     # if no points could be deactivated we are done:
        #     if num_deactivated == np.sum(points_mask == False):
        #         break

        rb_points = rb_points[points_mask]
        # show inaccurate triangulization of points on convex hull/rubber band:
        self.model.add_viz_info(
            Visualization(
                "trisurf",
                rb.points[:, 0:-1],
                rb.points[:, -1].reshape(-1, 1),
                "rubber band",
                {"color": "yellow", "shade": True, "triangles": [s.inds for s in rb.simplices]},
            )
        )
        # show actual convex hull (correct triangles) (does not work after changing points though):
        # self.model.add_viz_info(
        #     Visualization(
        #         "trisurf",
        #         lower[:, 0:-1],
        #         lower[:, -1].reshape(-1, 1),
        #         "rubber band",
        #         {"color": "yellow", "shade": True, "triangles": self.lower_delaunay.convex_hull},
        #     )
        # )
        return


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
