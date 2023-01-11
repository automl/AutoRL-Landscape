from typing import Any, Dict, Optional, Tuple

from pathlib import Path

import gym
import submitit
import wandb
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.monitor import Monitor

from autorl_landscape.custom_agents.custom_dqn import CustomDQN
from autorl_landscape.util.callback import LandscapeEvalCallback
from autorl_landscape.util.compare import choose_best_conf, construct_2d
from autorl_landscape.util.ls_sampler import construct_ls
from autorl_landscape.util.schedule import schedule


def make_env(env_name: str, seed: int) -> gym.Env:
    """Quick helper to create a gym env and seed it."""
    env = gym.make(env_name)
    env = Monitor(env)
    env.seed(seed)
    return env
    # wrapped = Monitor(env)
    # wrapped.seed(seed)
    # return wrapped


def run_phase(
    conf: DictConfig,
    t_ls: int,
    t_final: int,
    date_str: str,
    phase_str: str,
    ancestor: Optional[Path] = None,
) -> None:
    """Train a number of sampled configurations, evaluating and saving all agents at t_ls env steps.

    If initial_agent is given, start with its progress instead of training from 0. After this, train
    all agents until t_final env steps and evaluate here to choose the best configuration.

    Args:
        conf (DictConfig): Configuration for the experiment
        t_ls (int): Number of steps from this phase's start until this phase's landscape evaluation
        t_final (int): Number of steps from this phase's start until this phase's end
        date_str (str): Timestamp that is equal for all phases of this experiment, used for saving
        phase_str (str): e.g. phase_{i}, used for saving

    Returns:
        None
    """
    # path for saving agents of the current phase
    phase_path = f"phase_results/{conf.agent.name}/{conf.env.name}/{date_str}/{phase_str}"

    executor = submitit.AutoExecutor(folder="submitit", cluster=conf.slurm.cluster)
    tasks = []
    executor.update_parameters(**conf.slurm.update_parameters)

    for conf_idx, c in construct_ls(conf).iterrows():  # NOTE iterrows() changes datatypes, we get only np.float64
        # NOTE set hyperparameters
        ls_conf = {
            # [256, 256] translates to three layers:
            # Linear(i, 256), relu
            # Linear(256, 256), relu
            # Linear(256, o)
            "policy_kwargs": {"net_arch": [int(c["nn_width"])] * int(c["nn_length"] - 1)},
            "learning_rate": c["learning_rate"],
            "gamma": 1 - c["neg_gamma"],
            "exploration_final_eps": c["exploration_final_eps"],
        }
        ls_conf_readable = {
            "nn_width": c["nn_width"],
            "nn_length": c["nn_length"],
            "learning_rate": c["learning_rate"],
            "gamma": 1 - c["neg_gamma"],
            "exploration_final_eps": c["exploration_final_eps"],
        }
        del c

        for seed in range(conf.seeds.agent, conf.seeds.agent + conf.num_seeds):
            task = (
                ancestor,
                conf,
                ls_conf,
                ls_conf_readable,
                seed,
                date_str,
                phase_str,
                t_ls,
                t_final,
                conf_idx,
                phase_path,
            )
            tasks.append(task)

    results = schedule(executor, _train_agent, tasks, num_parallel=conf.slurm.num_parallel, polling_rate=10)

    # conf_indices, run_ids, final_scores = zip(*results)
    # run_ids = np.array(run_ids)
    # final_scores = np.array(final_scores)
    run_ids, final_returns = construct_2d(*zip(*results))

    best = choose_best_conf(run_ids, final_returns, save=phase_path)

    print(f"-- {phase_str.upper()} REPORT --")
    print(f"{run_ids=}")
    print(f"{final_returns=}")
    print(f"Best run: {best}\n")

    return


def _train_agent(
    ancestor: Optional[Path],
    conf: DictConfig,
    ls_conf: Dict[str, Any],
    ls_conf_readable: Dict[str, Any],
    seed: int,
    date_str: str,
    phase_str: str,
    t_ls: int,
    t_final: int,
    conf_index: int,
    phase_path: str,
) -> Tuple[int, str, NDArray[Any]]:
    """Train an agent, evaluating ls_eval and final_eval.

    :param ancestor: Path to a saved trained agent from which learning shall be commenced
    :param conf: Base configuration for agent, env, etc.
    :param ls_conf: Setting of the hyperparameters from the landscape
    :param ls_conf_readable: ls_conf but for logging
    :param seed: seed for the Agent, for verifying performance of a configuration over multiple
    random initializations.
    :param date_str: Timestamp to distinguish this whole run (not just the current phase!), for
    saving
    :param phase_str: e.g. "phase_0", for saving
    :param t_ls: For `LandscapeEvalCallback`
    :param t_final: For `LandscapeEvalCallback`
    :param conf_idx: For `LandscapeEvalCallback`
    :param phase_path: e.g. "phase_results/{conf.agent.name}/{conf.env.name}/{date_str}/{phase_str}"

    :return: wandb id of the run and all collected final performance values of the run. I.e. shape is
    (conf.combo.eval.final_eval_episodes * conf.combo.final_eval_times,)
    """
    # run_ids = np.full((len(conf.seeds),), "", dtype=np.dtype("<U8"))
    # final_scores = np.zeros((len(conf.seeds), conf.combo.final_eval_episodes * conf.combo.final_eval_times))
    # for one configuration, train multiple agents

    # Environment Creation
    env = make_env(conf.env.name, seed)
    eval_env = make_env(conf.env.name, seed)

    # setup wandb
    project_root = Path(__file__).parent.parent
    run = wandb.init(
        project=conf.wandb.project,
        config={
            "ls": ls_conf_readable,
            "conf": OmegaConf.to_object(conf),
            "meta": {
                "timestamp": date_str,
                "phase": phase_str,
                "seed": seed,
                "ancestor": str(ancestor),
                "conf_index": conf_index,
            },
        },
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=False,
        dir=project_root,
    )
    assert run is not None

    # Basic agent configuration:
    agent_kwargs = {
        "env": env,
        "verbose": 0,  # WARNING Higher than 0 breaks the console output logging with too long keys
        "tensorboard_log": f"runs/{run.id}",
        "seed": seed,
    }

    # AgentClass: type
    if conf.agent.name == "DQN":
        AgentClass = CustomDQN
    # elif conf.agent.name == "SAC":
    #     AgentClass = SAC
    else:
        raise Exception("unknown agent")
    # Agent Instantiation:
    if ancestor is None:
        agent = AgentClass(**agent_kwargs, **conf.agent.hps, **ls_conf)  # type: ignore
    else:
        agent = AgentClass.custom_load(save_path=ancestor, seed=seed)
        # NOTE set hyperparameters:
        agent.learning_rate = ls_conf["learning_rate"]
        agent.gamma = ls_conf["gamma"]
        print(f"{ls_conf['exploration_final_eps']=}")
        agent.exploration_rate = ls_conf["exploration_final_eps"]
        agent.exploration_final_eps = ls_conf["exploration_final_eps"]
        agent.exploration_initial_eps = ls_conf["exploration_final_eps"]
        agent.exploration_schedule = lambda _: ls_conf["exploration_final_eps"]

    landscape_eval_callback = LandscapeEvalCallback(
        conf=conf,
        eval_env=eval_env,
        t_ls=t_ls,
        t_final=t_final,
        ls_model_save_path=f"{phase_path}/agents/{run.id}",
        run=run,
        agent_seed=seed,
    )
    agent.learn(total_timesteps=t_final, callback=landscape_eval_callback, reset_num_timesteps=False)

    run.finish()
    return conf_index, run.id, landscape_eval_callback.all_final_returns.reshape(-1)
