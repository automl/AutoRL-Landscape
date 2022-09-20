from typing import Any, Dict, List, Optional, Tuple

import gym
import numpy as np
import submitit
import wandb
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Float, Uniform
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.sac.sac import SAC
from wandb.integration.sb3 import WandbCallback

from autorl_landscape.util.callback import LandscapeEvalCallback
from autorl_landscape.util.compare import choose_best_conf


def make_env(env_name: str, seed: int) -> gym.Env:
    """Quick helper to create a gym env and seed it."""
    env = gym.make(env_name)
    env = Monitor(env)
    env.seed(seed)
    return env


def run_phase(
    conf: DictConfig,
    t_ls: int,
    date_str: str,
    phase_str: str,
    init_agent: Optional[str] = None,
) -> None:
    """
    Train a number of sampled configurations, evaluating and saving all agents at t_ls env steps.
    If initial_agent is given, start with its progress instead of training from 0.
    After this, train all agents until t_final env steps and evaluate here to choose the best configuration.
    """
    # Landscape Hyperparameters:
    dims: List[Any] = []
    for dim_name, dim_args in [(next(iter(a)), a[next(iter(a))]) for a in conf.ls.dims]:
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

    # path for saving agents of the current phase
    phase_path = f"phase_results/{conf.agent.name}/{conf.env.name}/{date_str}/{phase_str}"

    executor = submitit.AutoExecutor(folder="submitit", cluster="local")
    jobs = []
    executor.update_parameters(timeout_min=10, slurm_partition="dev", gpus_per_node=1)

    for conf_idx in range(conf.ls.num_samples):

        # Sample or set the configuration
        if conf.ls.use_zoo_optimal_ls:
            if conf.ls.num_samples > 1:
                raise Exception("When using optimal ls hyperparameters, set num_samples to 1!")
            print("WARNING: Using optimal ls configuration instead of sampling...")
            c = Configuration(cs, conf.agent.zoo_optimal_ls)
        else:
            c = cs.sample_configuration()

        ls_conf = {
            # [256, 256] translates to three layers:
            # Linear(i, 256), relu
            # Linear(256, 256), relu
            # Linear(256, o)
            "policy_kwargs": {"net_arch": [c["nn_width"]] * (c["nn_length"] - 1)},
            "learning_rate": c["learning_rate"],
            "gamma": 1 - c["neg_gamma"],
        }
        ls_conf_readable = {
            "nn_width": c["nn_width"],
            "nn_length": c["nn_length"],
            "learning_rate": c["learning_rate"],
            "gamma": 1 - c["neg_gamma"],
        }
        del c

        job = executor.submit(
            _train_agent, init_agent, conf, ls_conf, ls_conf_readable, date_str, phase_str, t_ls, conf_idx, phase_path
        )
        jobs.append(job)

    # print(jobs)
    results = [job.result() for job in jobs]
    run_ids, final_scores = zip(*results)
    run_ids = np.array(run_ids)
    final_scores = np.array(final_scores)
    print(f"{run_ids=}")
    print(f"{final_scores=}")

    choose_best_conf(run_ids, final_scores, save=phase_path)

    return


def _train_agent(
    init_agent: Optional[str],
    conf: DictConfig,
    ls_conf: Dict[str, Any],
    ls_conf_readable: Dict[str, Any],
    date_str: str,
    phase_str: str,
    t_ls: int,
    conf_idx: int,
    phase_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train an agent over multiple seeds, evaluating ls_eval and final_eval.

    :param conf: Base configuration for agent, env, etc.
    :param ls_conf: Setting of the hyperparameters from the landscape
    :param ls_conf_readable: ls_conf but for logging
    :param date_str: Timestamp to distinguish this whole run (not just the current phase!), for saving
    :param phase_str: e.g. "phase_0", for saving
    :param t_ls: For `LandscapeEvalCallback`
    :param conf_idx: For `LandscapeEvalCallback`
    :param phase_path: e.g. "phase_results/{conf.agent.name}/{conf.env.name}/{date_str}/{phase_str}"
    """
    run_ids = np.full((len(conf.seeds),), "", dtype=np.dtype("<U8"))
    final_scores = np.zeros((len(conf.seeds),))
    # for one configuration, train multiple agents
    for i, seed in enumerate(conf.seeds):
        # Environment Creation
        env = make_env(conf.env.name, seed)
        eval_env = make_env(conf.env.name, seed)

        # setup wandb
        run = wandb.init(
            project="checking9",
            config={
                "ls": ls_conf_readable,
                "conf": OmegaConf.to_object(conf),
                "meta": {
                    "timestamp": date_str,
                    "phase": phase_str,
                    "seed": seed,
                },
            },
            sync_tensorboard=True,
            monitor_gym=False,
            save_code=False,
        )
        assert run is not None

        # Basic agent configuration:
        agent_kwargs = {
            "env": env,
            "verbose": 0,  # Higher than 0 breaks the console output logging with too long keys!
            "tensorboard_log": f"runs/{run.id}",
            "seed": seed,
        }

        # Agent Selection:
        agent_name = conf.agent.name
        agent: Optional[BaseAlgorithm]
        if agent_name == "DQN":
            agent = DQN(**agent_kwargs, **conf.agent.hps, **ls_conf)
        elif agent_name == "SAC":
            agent = SAC(**agent_kwargs, **conf.agent.hps, **ls_conf)
        else:
            raise Exception("unknown agent")

        # Load existing parameters of agent:
        if init_agent is not None:
            agent.set_parameters(init_agent)

        wandb_callback = WandbCallback(
            gradient_save_freq=1,
            # model_save_path=f"models/{run.id}",
            verbose=1,
        )
        landscape_eval_callback = LandscapeEvalCallback(
            conf=conf,
            eval_env=eval_env,
            t_ls=t_ls,
            ls_model_save_path=f"{phase_path}/agents/{run.id}/model.zip",
            conf_idx=conf_idx,
            run=run,
            # eval_freq=conf.eval.eval_freq,
            # n_eval_episodes=conf.env.n_eval_episodes,
        )
        # Wandb Logging and Evaluation
        callbacks = CallbackList([wandb_callback, landscape_eval_callback])

        agent.learn(total_timesteps=conf.env.total_timesteps, callback=callbacks)

        run.finish()
        final_scores[i] = np.mean(landscape_eval_callback.all_final_returns)  # TODO actual metric for comparing runs
        run_ids[i] = run.id
    return run_ids, final_scores
