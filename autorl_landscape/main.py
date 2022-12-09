import argparse
from datetime import datetime
from pathlib import Path

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from autorl_landscape.train import run_phase
from autorl_landscape.util.download import download_data
from autorl_landscape.visualize import (
    visualize_data,
    visualize_data_samples,
    visualize_gp,
)


# @hydra.main(version_base=None, config_path="conf", config_name="config")
def main() -> None:
    """Choose to either start the phases or visualize the landscape samples."""
    # Parse non-hydra commandline arguments:
    parser = argparse.ArgumentParser(prog="phases")
    subparsers = parser.add_subparsers()
    subparsers.required = True

    # phases run ...
    parser_run = subparsers.add_parser("run")
    parser_run.add_argument("overrides", nargs="*", help="Hydra overrides")
    parser_run.set_defaults(func="run")

    # phases viz ...
    parser_viz = subparsers.add_parser("viz")
    viz_subparsers = parser_viz.add_subparsers()
    viz_subparsers.required = True

    # phases viz samples ...
    parser_viz_samples = viz_subparsers.add_parser("samples")
    # parser_viz_samples.add_argument("overrides", nargs="*", help="Hydra overrides")
    parser_viz_samples.add_argument("file", help="csv file containing data of all runs")
    parser_viz_samples.set_defaults(func="viz_samples")

    # phases viz gp ...
    parser_viz_gp = viz_subparsers.add_parser("gp")
    parser_viz_gp.add_argument("file", help="csv file containing data of all runs")
    parser_viz_gp.add_argument(
        "--sample-percentage",
        "--sample",
        dest="sample_percentage",
        required=True,
        type=int,
        help="Percentage of eval episodes that should be used",
    )
    parser_viz_gp.add_argument("--viz-samples", action="store_true", dest="viz_samples", help="Also visualize samples")
    parser_viz_gp.add_argument(
        "--retrain", action="store_true", dest="retrain", help="Re-train GP, even if trained model exists on disk"
    )
    parser_viz_gp.add_argument("--save", action="store_true", dest="save", help="Save the trained model to disk")
    parser_viz_gp.set_defaults(func="viz_gp")

    # phases viz data ...
    parser_viz_data = viz_subparsers.add_parser("data")
    parser_viz_data.add_argument("file", help="csv file containing data of all runs")
    parser_viz_data.set_defaults(func="viz_data")

    # phases dl ...
    api = wandb.Api()
    projects = api.projects()
    entity_name = projects[0].entity  # NOTE hacky
    project_names = [p.name for p in projects]
    parser_dl = subparsers.add_parser("dl")
    parser_dl.add_argument("project_name", type=str, choices=project_names)
    parser_dl.set_defaults(func="dl")

    args = parser.parse_args()
    match args.func:
        case "run":
            start_phases(_prepare_hydra(args))
        case "viz_samples":
            # visualize_samples(_prepare_hydra(args))
            visualize_data_samples(args.file)
        case "viz_gp":
            visualize_gp(args.file, args.sample_percentage, args.viz_samples, args.retrain, args.save)
        case "viz_data":
            visualize_data(args.file)
        case "dl":
            download_data(entity_name, args.project_name)
        case _:
            pass


def _prepare_hydra(args: argparse.Namespace) -> DictConfig:
    hydra.initialize(config_path="../conf", version_base="1.1")
    conf = hydra.compose("config", overrides=args.overrides)
    print(OmegaConf.to_yaml(conf))
    return conf


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
