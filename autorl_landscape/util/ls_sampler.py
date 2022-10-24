import numpy as np
import pandas as pd
from ConfigSpace import Categorical, ConfigurationSpace, Float, Uniform
from omegaconf import DictConfig
from scipy.stats.qmc import Sobol


def construct_ls(conf: DictConfig) -> pd.DataFrame:
    """
    Build landscape according to the passed config and returns a `pd.DataFrame` with samples from it.

    :param conf: `DictConfig` with ls dims, num of samples and optimal zoo hyperparameters for Constant dimensions.
    :return: `pd.DataFrame` with conf.ls.num_samples entries.
    """
    # Landscape Hyperparameters:
    if conf.ls["type"] == "ConfigSpace":
        dims = []
        for dim_name, dim_args in [(next(iter(d)), d[next(iter(d))]) for d in conf.ls.dims]:
            if dim_args["type"] == "Categorical":
                dims.append(Categorical(dim_name, dim_args["items"], ordered=True))
            elif dim_args["type"] == "Float":
                dims.append(Float(dim_name, (dim_args.lower, dim_args.upper), distribution=Uniform(), log=dim_args.log))
            elif dim_args["type"] == "Constant":
                zoo_value = conf.agent.zoo_optimal_ls[dim_name]
                dims.append(Categorical(dim_name, [zoo_value], ordered=True))
            else:
                raise Exception(f"Unknown ls dimension type: {dim_args['type']}")

        cs = ConfigurationSpace(seed=conf.ls.seed)
        cs.add_hyperparameters(dims)
        return pd.DataFrame([cs.sample_configuration() for _ in range(conf.ls.num_samples)])

    elif conf.ls["type"] == "Sobol":
        # Sobol sampling is best done with 2 ** n samples:
        assert np.log2(conf.ls.num_samples).is_integer()
        # TODO: if constant dims are there, sample only for the other dims
        # (since the constant values would be overridden anyways)

        dims = [(next(iter(d)), d[next(iter(d))]) for d in conf.ls.dims]
        configs = np.zeros((conf.ls.num_samples, len(dims)), dtype="O")
        sampler = Sobol(len([1 for (_, dim_args) in dims if dim_args["type"] != "Constant"]), seed=conf.ls.seed)
        samples = sampler.random_base2(int(np.log2(conf.ls.num_samples)))

        # Transform the sampled sobol numbers into actual configurations:
        s = 0  # sobol dim index
        for i, (dim_name, dim_args) in enumerate(dims):
            if dim_args["type"] == "Integer":
                configs[:, i] = np.round((samples[:, s] * (dim_args["upper"] - dim_args["lower"])) + dim_args["lower"])
                s += 1
            elif dim_args["type"] == "Categorical":
                num_categories = len(dim_args["items"])
                # convert samples to indices:
                indices = np.floor(samples[:, s] * num_categories).astype(int)
                # TODO configs does only take numbers atm
                configs[:, i] = np.array(dim_args["items"])[indices]
            elif dim_args["type"] == "Log":
                configs[:, i] = (dim_args["base"] ** samples[:, s] - 1) / (dim_args["base"] - 1)
                configs[:, i] = (configs[:, i] * (dim_args["upper"] - dim_args["lower"])) + dim_args["lower"]
                s += 1
            elif dim_args["type"] == "Constant":
                value = conf.agent.zoo_optimal_ls[dim_name]
                configs[:, i] = value
            else:
                raise Exception(f"Unknown ls dimension type: {dim_args['type']}")
        return pd.DataFrame(configs, columns=[dim_name for (dim_name, _) in dims])
    else:
        raise Exception(f"{conf.ls.type=} is not a known landscape type.")
