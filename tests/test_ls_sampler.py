import hydra
import numpy as np

from autorl_landscape.util.ls_sampler import construct_ls

hydra.initialize(config_path="test_conf", version_base="1.1")


def test_build_sobol_det() -> None:
    """Checks that Sobol type sampler is seeded correctly."""
    conf = hydra.compose("sobol")
    df = construct_ls(conf)
    df2 = construct_ls(conf)
    assert np.all(df == df2)


def test_build_configspace_det() -> None:
    """Checks that ConfigSpace type sampler is seeded correctly."""
    conf = hydra.compose("configspace")
    df = construct_ls(conf)
    df2 = construct_ls(conf)
    assert np.all(df == df2)


def test_partial_sobol() -> None:
    """Check that constant dimensions don't change the sampled dimensions."""
    # TODO test this with all kinds of dims
    conf_pure = hydra.compose("only_sobol")  # 4 dims, none constant
    conf_dirty = hydra.compose("sobol")  # 4 non-constant dims, interspersed with constant dims.
    # I.e. df_dirty[1::2] are all the non-constant dims
    df_pure = construct_ls(conf_pure)
    df_dirty = construct_ls(conf_dirty)
    assert np.all(df_pure == df_dirty[["nn_width", "nn_length", "learning_rate", "neg_gamma"]])


def test_categorical() -> None:
    """Create lots of samples, then check whether the configurations only have the category values."""
    df = construct_ls(hydra.compose("categorical.yaml"))
    assert np.all(np.isin(df["nn_width"], [16, 32, 64, 128, 256]))
    assert np.all(np.isin(df["foo"], ["foo", "bar", "baz"]))
