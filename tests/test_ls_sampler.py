import hydra
import numpy as np

from autorl_landscape.util.ls_sampler import construct_ls

hydra.initialize(config_path="test_conf", version_base="1.1")


def test_build_sobol_det():
    conf = hydra.compose("sobol")
    df = construct_ls(conf)
    df2 = construct_ls(conf)
    assert np.all(df == df2)


def test_build_configspace_det():
    conf = hydra.compose("configspace")
    df = construct_ls(conf)
    df2 = construct_ls(conf)
    assert np.all(df == df2)
