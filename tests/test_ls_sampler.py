import hydra
import matplotlib.pyplot as plt
import numpy as np
import pytest

from autorl_landscape.util.ls_sampler import construct_ls

hydra.initialize(config_path="test_conf", version_base="1.1")


def test_build_sobol_det() -> None:
    conf = hydra.compose("sobol")
    df = construct_ls(conf)
    df2 = construct_ls(conf)
    assert np.all(df == df2)


def test_build_configspace_det() -> None:
    conf = hydra.compose("configspace")
    df = construct_ls(conf)
    df2 = construct_ls(conf)
    assert np.all(df == df2)


def test_partial_sobol() -> None:
    """Check that constant dimensions don't change the sampled dimensions."""
    conf_pure = hydra.compose("only_sobol")  # 4 dims, none constant
    conf_dirty = hydra.compose("sobol")  # 4 non-constant dims, interspersed with constant dims.
    # I.e. df_dirty[1::2] are all the non-constant dims
    df_pure = construct_ls(conf_pure)
    df_dirty = construct_ls(conf_dirty)
    assert np.all(df_pure == df_dirty[["nn_width", "nn_length", "learning_rate", "neg_gamma"]])


@pytest.mark.skip(reason="Visualization")
def test_sobol_pattern() -> None:
    """Visualize with plt to inspect the sampled patterns."""
    df = construct_ls(hydra.compose("sobol_viz"))
    fig = plt.figure(figsize=(16, 16))
    fig.tight_layout()
    ax = plt.axes(projection="3d")
    ax.scatter(df.large, df.small, df.log)
    plt.show()
