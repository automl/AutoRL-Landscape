import argparse
from datetime import datetime
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf

from autorl_landscape.analyze.concavity import reject_concavity
from autorl_landscape.ls_models.heteroskedastic_gp import HSGPModel
from autorl_landscape.ls_models.linear import LinearLSModel
from autorl_landscape.train import run_phase
from autorl_landscape.util.data import read_wandb_csv
from autorl_landscape.util.download import download_data
from autorl_landscape.visualize import visualize_data, visualize_data_samples

ENTITY = "kwie98"


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
    parser_viz_gp.add_argument("--viz-samples", action="store_true", dest="viz_samples", help="Also visualize samples")
    parser_viz_gp.add_argument("--save", action="store_true", dest="save", help="Save the trained model to disk")
    parser_viz_gp.add_argument("--load", action="store_true", dest="load", help="Load the trained model from disk")
    parser_viz_gp.add_argument("--grid-length", dest="grid_length", type=int, default=51)
    parser_viz_gp.set_defaults(func="viz_gp")

    # phases viz linear ...
    parser_viz_linear = viz_subparsers.add_parser("linear")
    parser_viz_linear.add_argument("file", help="csv file containing data of all runs")
    parser_viz_linear.add_argument(
        "--viz-samples", action="store_true", dest="viz_samples", help="Also visualize samples"
    )
    parser_viz_linear.add_argument("--save", action="store_true", dest="save", help="Save the trained model to disk")
    parser_viz_linear.add_argument("--load", action="store_true", dest="load", help="Load the trained model from disk")
    parser_viz_linear.add_argument("--grid-length", dest="grid_length", type=int, default=51)
    parser_viz_linear.set_defaults(func="viz_linear")

    # phases viz triple-gp ...
    parser_viz_triple_gp = viz_subparsers.add_parser("triple-gp")
    parser_viz_triple_gp.add_argument("file", help="csv file containing data of all runs")
    parser_viz_triple_gp.add_argument(
        "--viz-samples", action="store_true", dest="viz_samples", help="Also visualize samples"
    )
    parser_viz_triple_gp.add_argument(
        "--retrain", action="store_true", dest="retrain", help="Re-train GP, even if trained model exists on disk"
    )
    parser_viz_triple_gp.add_argument("--save", action="store_true", dest="save", help="Save the trained model to disk")
    parser_viz_triple_gp.add_argument("--grid-length", dest="grid_length", type=int, default=51)
    parser_viz_triple_gp.set_defaults(func="viz_triple_gp")

    # phases viz data ...
    parser_viz_data = viz_subparsers.add_parser("data")
    parser_viz_data.add_argument("file", help="csv file containing data of all runs")
    parser_viz_data.set_defaults(func="viz_data")

    # phases ana ...
    parser_ana = subparsers.add_parser("ana")
    ana_subparsers = parser_ana.add_subparsers()
    ana_subparsers.required = True

    # phases ana concavity ...
    parser_ana_concavity = ana_subparsers.add_parser("concavity")
    parser_ana_concavity.add_argument("--data", help="csv file containing data of all runs", required=True)
    parser_ana_concavity.add_argument("--model", type=str, choices=["hsgp", "linear"], required=True)
    parser_ana_concavity.add_argument("--grid-length", dest="grid_length", type=int, default=51)
    parser_ana_concavity.set_defaults(func="ana_concavity")

    # phases dl ...
    entity_name = ENTITY
    parser_dl = subparsers.add_parser("dl")
    parser_dl.add_argument("project_name", type=str)
    parser_dl.set_defaults(func="dl")

    args = parser.parse_args()
    match args.func:
        case "run":
            start_phases(_prepare_hydra(args))
        case "viz_samples":
            visualize_data_samples(args.file)
        case "viz_gp":
            file = Path(args.file)
            df = read_wandb_csv(file)
            fig = plt.figure(figsize=(16, 12))
            phase_strs = sorted(df["meta.phase"].unique())
            for i, phase_str in enumerate(phase_strs):
                model_folder = file.parent / f"{file.stem}_hsgp_{phase_str}"
                phase_data = df[df["meta.phase"] == phase_str].sort_values("meta.conf_index")
                hsgp = HSGPModel(phase_data, 10, np.float64, "ls_eval/returns", None)
                if args.load:
                    hsgp.load(model_folder)
                else:
                    hsgp.fit(100, verbose=True)
                if args.save and not args.load:
                    hsgp.save(model_folder)

                ax = fig.add_subplot(1, len(phase_strs), i + 1, projection="3d")
                _ = hsgp.visualize(ax, grid_length=args.grid_length, viz_model=True, viz_samples=args.viz_samples)
            plt.show()
        case "viz_linear":
            file = Path(args.file)
            df = read_wandb_csv(file)
            fig = plt.figure(figsize=(16, 12))
            phase_strs = sorted(df["meta.phase"].unique())
            for i, phase_str in enumerate(phase_strs):
                model_folder = file.parent / f"{file.stem}_hsgp_{phase_str}"
                phase_data = df[df["meta.phase"] == phase_str].sort_values("meta.conf_index")
                hsgp = LinearLSModel(phase_data, np.float64, "ls_eval/returns", None)
                if args.load:
                    hsgp.load(model_folder)
                if args.save and not args.load:
                    hsgp.save(model_folder)

                ax = fig.add_subplot(1, len(phase_strs), i + 1, projection="3d")
                _ = hsgp.visualize(ax, grid_length=args.grid_length, viz_model=True, viz_samples=args.viz_samples)
            plt.show()
        case "viz_triple_gp":
            raise NotImplementedError
        case "viz_data":
            visualize_data(args.file)
        case "ana_concavity":
            file = Path(args.data)
            df = read_wandb_csv(file)
            phase_strs = sorted(df["meta.phase"].unique())
            for i, phase_str in enumerate(phase_strs):
                phase_data = df[df["meta.phase"] == phase_str].sort_values("meta.conf_index")
                match args.model:
                    case "hsgp":
                        model = HSGPModel(phase_data, 10, np.float64, "ls_eval/returns", None)
                        model_folder = file.parent / f"{file.stem}_hsgp_{phase_str}"
                        model.load(model_folder)
                    case "linear":
                        model = LinearLSModel(phase_data, np.float64, "ls_eval/returns", None)
                    case _:
                        parser.print_help()
                        return
                reject_concavity(model, grid_length=args.grid_length)
        case "dl":
            download_data(entity_name, args.project_name)
        case _:
            parser.print_help()
            return


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
