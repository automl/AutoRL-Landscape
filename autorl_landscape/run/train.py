from typing import Any

from collections.abc import Iterable
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
    ancestor: Path | None,
    ls_conf: dict[str, Any],
    seed: int,
    conf_index: int,
    phase_path: str,
) -> tuple[int, str, NDArray[Any]]:
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

    # Quick fix to accept both single tags and lists of tags (because of resume):
    experiment_tags: list[str] = []
    match conf.wandb.experiment_tag:
        case str():
            experiment_tags = [conf.wandb.experiment_tag]
        case Iterable():
            experiment_tags = list(conf.wandb.experiment_tag)
        case _:
            msg = f"Received unexpected {conf.wandb.experiment_tag=}"
            raise ValueError(msg)

    # Setup wandb:
    project_root = Path(__file__).parent.parent.parent
    run = wandb.init(
        project=conf.wandb.project,
        tags=experiment_tags,
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
    if run is None:
        error_msg = "Wandb run not initialized correctly!"
        raise Exception(error_msg)

    # Basic agent configuration:
    agent_kwargs = {
        "env": env,
        "verbose": 0,  # WARNING Higher than 0 breaks the console output logging with too long keys
        "tensorboard_log": f"runs/{run.id}",
        "seed": seed,
        # For smaller replay buffers on disk:
        "optimize_memory_usage": True,
        "replay_buffer_kwargs": {
            "handle_timeout_termination": False,
        },
    }

    match conf.agent.name:
        case "DQN":
            agent_class = CustomDQN
        case "SAC":
            agent_class = CustomSAC
        case _:
            error_msg = f"Unknown agent type {conf.agent.name}."
            raise ValueError(error_msg)

    # Agent Instantiation:
    if ancestor is None:
        agent = agent_class(**agent_kwargs, **conf.agent.hps)
    else:
        agent = agent_class.custom_load(save_path=ancestor, seed=seed)
    agent.set_ls_conf(ls_conf, phase_index)

    landscape_eval_callback = LandscapeEvalCallback(
        conf,
        phase_index,
        eval_env,
        f"{phase_path}/agents/{run.id}",
        run,
        seed,
        ls_conf,
    )
    # NOTE total_timesteps setting is too high here for all phases after the first. However, we simply stop learning
    # runs after all needed data has been colleted, through the callback's _on_step() method.
    try:
        agent.learn(total_timesteps=conf.phases[-1], callback=landscape_eval_callback, reset_num_timesteps=False)
    except ValueError as e:
        landscape_eval_callback.on_rollout_error(e)

    run.finish()
    return conf_index, run.id, landscape_eval_callback.all_final_returns.reshape(-1)
