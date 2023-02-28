from typing import Any, Optional, Tuple

from pathlib import Path

import wandb
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf

from autorl_landscape.custom_agents.dqn import CustomDQN
from autorl_landscape.custom_agents.sac import CustomSAC
from autorl_landscape.run.callback import LandscapeEvalCallback
from autorl_landscape.run.rl_context import make_env


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

    match conf.agent.name:
        case "DQN":
            Agent = CustomDQN
        case "SAC":
            Agent = CustomSAC
        case _:
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
