import numpy as np
from scipy.interpolate import NearestNDInterpolator

from autorl_landscape.ls_models.ls_model import LSModel
from autorl_landscape.types.visualization import Visualization
from autorl_landscape.util.grid_space import grid_space_nd
from autorl_landscape.visualize import CMAP_DIVERGING

P_VALUE_CUTOFF = 0.05


def check_modality(model: LSModel, grid_length: int) -> None:
    """Check per-configuration data for uni-modality using FTU."""
    from pyfolding import FTU  # lazy import

    phis = np.zeros((model.y.shape[0], 1))
    p_values = np.zeros((model.y.shape[0], 1))
    for i, y in enumerate(model.y):
        ftu = FTU(y, routine="c++")
        phis[i] = ftu.folding_statistics  # FTU indicator for uni-modality; > 1 indicates uni-modality
        p_values[i] = ftu.p_value

    phis[p_values >= 0.05] = 1.0
    phi_interp = NearestNDInterpolator(model.x, phis)

    phi_discrete = np.copy(phis)
    phi_discrete[phi_discrete > 1.0] = 3.5
    phi_discrete[phi_discrete < 1.0] = 0.0

    print("Modality analysis:")
    print(f"% unimodal (PHI > 1): {np.sum(phi_discrete > 1.0) / phi_discrete.size}")
    print(f"% multimodal (PHI < 1): {np.sum(phi_discrete < 1.0) / phi_discrete.size}")
    print(f"% uncategorized (p-value >= 0.05): {np.sum(phi_discrete == 1.0) / phi_discrete.size}")

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
