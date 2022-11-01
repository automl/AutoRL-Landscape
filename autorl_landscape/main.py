import sys
from datetime import datetime
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf

from autorl_landscape.train import run_phase
from autorl_landscape.util.ls_sampler import construct_ls


# @hydra.main(version_base=None, config_path="conf", config_name="config")
def main() -> None:
    """Choose to either start the phases or visualize the landscape samples."""
    hydra.initialize(config_path="../conf", version_base="1.1")

    if len(sys.argv) < 2:
        _print_help()
    match sys.argv[1]:
        case "run":
            conf = hydra.compose("config", overrides=sys.argv[2:])
            print(OmegaConf.to_yaml(conf))
            start_phases(conf)
        case "viz":
            conf = hydra.compose("config", overrides=sys.argv[2:])
            print(OmegaConf.to_yaml(conf))
            visualize_samples(conf)
        case _:
            _print_help()


def visualize_samples(conf: DictConfig) -> None:
    """Visualize with plt to inspect the sampled patterns.

    Args:
        conf: Hydra configuration
    """
    print("VIZ ONLY DOES LR AND GAMMA FOR NOW")
    df = construct_ls(conf)
    fig = plt.figure(figsize=(16, 16))
    fig.tight_layout()
    ax = plt.axes()
    ax.scatter(df["learning_rate"], 1 - df["neg_gamma"])
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("gamma")
    plt.show()


def start_phases(conf: DictConfig) -> None:
    """Run the experiment with the given configuration.

    Args:
        conf: Hydra configuration
    """
    # remember starting time of this run for saving all phase data:
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # TODO still not quite sure what this does but error message seems to be gone
    # wandb.tensorboard.patch(root_logdir="...")

    if conf.phases is not None:
        phases = conf.phases + [conf.total_timesteps]
        if not all(x < y for x, y in zip(phases, phases[1:])):
            raise Exception(f"Phases need to be strictly increasing. Got: {phases}")
        if conf.phases[-1] >= int(conf.total_timesteps * conf.eval.final_eval_start):
            raise Exception(
                "Last phase(s) too long! Not enough timesteps for final evaluation.\n"
                + f"Last phase start: {conf.phases[-1]}.\n"
                + f"final_eval start: {conf.eval.final_eval_start}"
            )

        last_t_phase = 0
        original_total_timesteps = conf.total_timesteps
        ancestor = None
        for i, t_phase in enumerate(phases):
            phase_str = f"phase_{i}"
            run_phase(
                conf=conf,
                t_ls=t_phase - last_t_phase,
                t_final=original_total_timesteps - last_t_phase,
                date_str=date_str,
                phase_str=phase_str,
                ancestor=ancestor,
            )
            ancestor = (
                Path(f"phase_results/{conf.agent.name}/{conf.env.name}/{date_str}/{phase_str}/best_agent")
                .resolve()
                .relative_to(Path.cwd())
            )
            last_t_phase = t_phase
    else:
        # a rudimentary way to just run the agent without any phase stuff
        run_phase(
            conf=conf,
            t_ls=conf.total_timesteps,
            t_final=conf.total_timesteps,
            date_str=date_str,
            phase_str="phase_0",
            ancestor=None,
        )


def _print_help() -> None:
    help_text = f"""Usage: {sys.argv[0]} [COMMAND] [HYDRA_OVERRIDES]

    Commands:
        run: Run experiments
        viz: Visualize landscape samples
    """
    print(help_text)
    exit()
