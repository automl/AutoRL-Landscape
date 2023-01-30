import numpy as np
from scipy.interpolate import NearestNDInterpolator

from autorl_landscape.ls_models.ls_model import CMAP_DIVERGING, LSModel, Visualization
from autorl_landscape.util.grid_space import grid_space_nd


def check_modality(model: LSModel, grid_length: int) -> None:
    """Check per-configuration data for uni-modality using FTU."""
    from pyfolding import FTU  # lazy import

    phis = np.zeros((model.y.shape[0], 1))
    for i, y in enumerate(model.y):
        ftu = FTU(y, routine="c++")
        phis[i] = ftu.folding_statistics  # FTU indicator for uni-modality; > 1 indicates uni-modality

    phi_interp = NearestNDInterpolator(model.x, phis)

    phi_discrete = np.copy(phis)
    phi_discrete[phi_discrete >= 1.0] = 2.0
    phi_discrete[phi_discrete < 1.0] = 0.0

    phi_discrete_interp = NearestNDInterpolator(model.x, phi_discrete)

    num_dims = len(model.dim_info)
    grid = grid_space_nd(num_dims, grid_length).reshape(-1, num_dims)

    for title, interp in [("Unimodality", phi_interp), ("Unimodality (discretized)", phi_discrete_interp)]:
        model.add_viz_info(
            Visualization(
                title,
                "map",
                "modalities",
                model.build_df(grid, interp(grid), "Φ"),
                CMAP_DIVERGING,
            )
        )
