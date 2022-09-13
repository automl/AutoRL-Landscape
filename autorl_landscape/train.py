from typing import Any, Optional

import gym
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
from autorl_landscape.util.comparator import Comparator

# from autorl_landscape.util.debug import DEBUG


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
    initial_agent: Optional[Any] = None,  # TODO
) -> None:
    """
    Train a number of sampled configurations, evaluating and saving all agents at t_ls env steps.
    If initial_agent is given, start with its progress instead of training from 0.
    After this, train all agents until t_final env steps and evaluate here to choose the best configuration.
    """
    # Landscape Hyperparameters:
    nn_width = Categorical(name="nn_width", items=[16, 32, 64, 128, 256], ordered=True)
    nn_length = Categorical(name="nn_length", items=[2, 3, 4, 5], ordered=True)
    lr = Float(name="learning_rate", bounds=(0.0001, 0.1), distribution=Uniform(), log=True)
    # neg_gamma = 1 - gamma, such that log-uniform (reciprocal) distribution can be used
    neg_gamma = Float(name="neg_gamma", bounds=(0.0001, 0.8), distribution=Uniform(), log=True)

    cs = ConfigurationSpace(seed=conf.cs.seed)
    cs.add_hyperparameters([nn_width, nn_length, lr, neg_gamma])

    # Comparator selects the best agent for use in the subsequent phase:
    phase_path = f"phase_results/{conf.agent.name}/{conf.env.name}/{date_str}/{phase_str}"
    comp = Comparator(phase_path, num_confs=conf.cs.num_samples, num_seeds=len(conf.seeds))

    for conf_idx in range(conf.cs.num_samples):
        # Sample or set the configuration
        if conf.cs.use_zoo_optimal_ls:
            if conf.cs.num_samples > 1:
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

        # for one configuration, train multiple agents
        for seed in conf.seeds:
            # Environment Creation
            env = make_env(conf.env.name, seed)
            eval_env = make_env(conf.env.name, seed)

            # setup wandb
            run = wandb.init(
                project="checking5",
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
                "verbose": 0,
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

            # Wandb Logging and Evaluation
            callbacks = CallbackList(
                [
                    # LandscapeEval(
                    #     after=t_ls,
                    #     save_path=f"phase_results/{conf.agent.name}/{conf.env.name}/{date_str}/phase_{phase}/agents/{run.id}",
                    #     verbose=1,
                    # ),
                    WandbCallback(
                        gradient_save_freq=100,
                        # model_save_path=f"models/{run.id}",
                        verbose=7,
                    ),
                    LandscapeEvalCallback(
                        eval_env=eval_env,
                        t_ls=t_ls,
                        comp=comp,
                        conf_idx=conf_idx,
                        run_id=run.id,
                        ls_model_save_path=f"{phase_path}/agents/{run.id}/model.zip",
                        eval_freq=conf.env.eval_freq,
                        n_eval_episodes=conf.env.n_eval_episodes,
                        log_path=None,
                        # log_path=f"logs/{run.id}",
                        deterministic=True,
                        render=False,
                    ),
                ]
            )

            agent.learn(total_timesteps=conf.env.total_timesteps, callback=callbacks)

            run.finish()
    comp.save_best()
