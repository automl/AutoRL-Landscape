from typing import Any, Optional, Tuple

from pathlib import Path

import submitit
import wandb
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf

from autorl_landscape.custom_agents.dqn import CustomDQN
from autorl_landscape.custom_agents.sac import CustomSAC
from autorl_landscape.run.rl_context import make_env
from autorl_landscape.util.callback import LandscapeEvalCallback
from autorl_landscape.util.compare import choose_best_conf, construct_2d
from autorl_landscape.util.ls_sampler import construct_ls
from autorl_landscape.util.schedule import schedule

# def make_env(env_name: str, seed: int) -> gym.Env:
#     """Quick helper to create a gym env and seed it."""
#     env = gym.make(env_name)
#     env = Monitor(env)
#     env.seed(seed)
#     return env


def run_phase(conf: DictConfig, phase_index: int, timestamp: str, ancestor: Optional[Path] = None) -> None:
    """Train a number of sampled configurations, evaluating and saving all agents at t_ls env steps.

    If initial_agent is given, start with its progress instead of training from 0. After this, train
    all agents until t_final env steps and evaluate here to choose the best configuration.

    Args:
        conf: Configuration for the experiment
        phase_index: Number naming the current phase. For the first phase, this is 1
        timestamp: Timestamp that is equal for all phases of this experiment, used for saving
        ancestor: If present, leads to the agent which `seeds` this phase.
    """
    # base directory for saving agents of the current phase
    phase_path = f"phase_results/{conf.agent.name}/{conf.env.name}/{timestamp}/phase_{phase_index}"

    executor = submitit.AutoExecutor(folder="submitit", cluster=conf.slurm.cluster)
    tasks = []
    executor.update_parameters(**conf.slurm.update_parameters)

    for conf_index, c in construct_ls(conf).iterrows():  # NOTE iterrows() changes datatypes, we get only np.float64
        # set hyperparameters:
        ls_conf = {
            "learning_rate": c["learning_rate"],
            "gamma": 1 - c["neg_gamma"],
            # "exploration_final_eps": c["exploration_final_eps"],
        }

        for seed in range(conf.seeds.agent, conf.seeds.agent + conf.num_seeds):
            task = (conf, phase_index, timestamp, ancestor, ls_conf, seed, conf_index, phase_path)
            tasks.append(task)

    results = schedule(executor, train_agent, tasks, num_parallel=conf.slurm.num_parallel, polling_rate=10)

    # conf_indices, run_ids, final_scores = zip(*results)
    # run_ids = np.array(run_ids)
    # final_scores = np.array(final_scores)
    run_ids, final_returns = construct_2d(*zip(*results))

    best = choose_best_conf(run_ids, final_returns, save=phase_path)

    print(f"-- PHASE {phase_index} REPORT --")
    print(f"{run_ids=}")
    print(f"{final_returns=}")
    print(f"Best run: {best}\n")

    return


def train_agent(
    conf: DictConfig,
    phase_index: int,
    timestamp: str,
    ancestor: Optional[Path],
    ls_conf: dict[str, Any],
    seed: int,
    conf_index: int,
    phase_path: str,
) -> Tuple[int, str, NDArray[Any]]:
    """Train an agent, evaluating ls_eval and final_eval.

    Args:
        conf: Base configuration for agent, env, etc.
        phase_index: Number naming the current phase. For the first phase, this is 1
        timestamp: Timestamp to distinguish this whole run (not just the current phase!), for saving
        ancestor: Path to a saved trained agent from which learning shall be commenced
        ls_conf: Setting of the hyperparameters from the landscape
        seed: seed for the Agent, for verifying performance of a configuration over multiple random initializations.
        conf_index: For `LandscapeEvalCallback`
        phase_path: e.g. "phase_results/{conf.agent.name}/{conf.env.name}/{date_str}/{phase_str}"

    Returns:
        wandb id of the run and all collected final performance values of the run. I.e. shape is
            (conf.combo.eval.final_eval_episodes * conf.combo.final_eval_times,)
    """
    env = make_env(conf.env.name, seed)
    eval_env = make_env(conf.env.name, seed)

    # Setup wandb:
    assert type(conf.wandb.experiment_tag) == str  # should hold?
    project_root = Path(__file__).parent.parent
    run = wandb.init(
        project=conf.wandb.project,
        tags=[conf.wandb.experiment_tag],
        config={
            "ls": ls_conf,
            "conf": OmegaConf.to_object(conf),
            "meta": {
                "timestamp": timestamp,
                "phase": phase_index,
                "seed": seed,
                "ancestor": str(ancestor),
                "conf_index": conf_index,
            },
        },
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=False,
        dir=project_root,
        mode=conf.wandb.mode,
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
        Agent = CustomDQN
    elif conf.agent.name == "SAC":
        Agent = CustomSAC
    else:
        raise Exception("unknown agent")
    # Agent Instantiation:
    if ancestor is None:
        agent = Agent(**agent_kwargs, **conf.agent.hps, **ls_conf)  # type: ignore
    else:
        agent = Agent.custom_load(save_path=ancestor, seed=seed)
        # NOTE set hyperparameters:
        agent.learning_rate = ls_conf["learning_rate"]
        agent.gamma = ls_conf["gamma"]
        # print(f"{ls_conf['exploration_final_eps']=}")
        # agent.exploration_rate = ls_conf["exploration_final_eps"]
        # agent.exploration_final_eps = ls_conf["exploration_final_eps"]
        # agent.exploration_initial_eps = ls_conf["exploration_final_eps"]
        # agent.exploration_schedule = lambda _: ls_conf["exploration_final_eps"]

    landscape_eval_callback = LandscapeEvalCallback(
        conf, phase_index, eval_env, f"{phase_path}/agents/{run.id}", run, seed
    )
    # NOTE total_timesteps setting is too high here for all phases after the first. However, we simply stop learning
    # runs after all needed data has been colleted, through the callback's _on_step() method.
    agent.learn(total_timesteps=conf.phases[-1], callback=landscape_eval_callback, reset_num_timesteps=False)

    run.finish()
    return conf_index, run.id, landscape_eval_callback.all_final_returns.reshape(-1)
