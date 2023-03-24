import numpy as np
from scipy.interpolate import NearestNDInterpolator

from autorl_landscape.analyze.visualization import Visualization
from autorl_landscape.ls_models.ls_model import LSModel
from autorl_landscape.util.grid_space import grid_space_nd
from autorl_landscape.visualize import CMAP_CRASHED


def check_crashing(model: LSModel, grid_length: int) -> None:
    """Visualize how likely it is for a configuration to crash during training."""
    chance_crash_per_conf = np.mean(model.crashed, axis=1, keepdims=True)
    print(model.x[np.any(model.crashed, axis=1)])
    crashed_interp = NearestNDInterpolator(model.x, chance_crash_per_conf)

    num_dims = len(model.dim_info)
    grid = grid_space_nd(num_dims, grid_length).reshape(-1, num_dims)

    model.add_viz_info(
        Visualization("Crashes", "map", "crashes", model.build_df(grid, crashed_interp(grid), "crashes"), CMAP_CRASHED)
    )
