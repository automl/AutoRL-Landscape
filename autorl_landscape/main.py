from typing import Any, Iterable, TypeVar

import argparse
from datetime import datetime
from itertools import zip_longest
from pathlib import Path

import hydra
import numpy as np
from matplotlib.artist import Artist
from matplotlib.backend_bases import PickEvent
from matplotlib.figure import Figure
from omegaconf import DictConfig, OmegaConf

from autorl_landscape.ls_models.heteroskedastic_gp import HSGPModel
from autorl_landscape.ls_models.ls_model import LSModel
from autorl_landscape.ls_models.mock import MockLSModel
from autorl_landscape.ls_models.rbf import RBFInterpolatorLSModel
from autorl_landscape.ls_models.triple_gp import TripleGPModel
from autorl_landscape.train import run_phase
from autorl_landscape.util.data import read_wandb_csv
from autorl_landscape.util.download import download_data
from autorl_landscape.visualize import (
    visualize_data,
    visualize_data_samples,
    visualize_sobol_samples,
)

ENTITY = "kwie98"
DEFAULT_GRID_LENGTH = 51
MODELS = ["hsgp", "linear", "triple-gp", "mock"]
VISUALIZATION_GROUPS = ["maps", "peaks", "graphs"]
T = TypeVar("T")


# @hydra.main(version_base=None, config_path="conf", config_name="config")
def main() -> None:
    """Choose to either start the phases or visualize the landscape samples."""
    # Parse non-hydra commandline arguments:
    parser = argparse.ArgumentParser(prog="phases")
    subparsers = parser.add_subparsers()
    subparsers.required = True

    # phases run ...
    parser_run = subparsers.add_parser("run", help="run an experiment using the hydra config in conf/")
    parser_run.add_argument("overrides", nargs="*", help="Hydra overrides")
    parser_run.set_defaults(func="run")

    # phases viz ...
    parser_viz = subparsers.add_parser("viz", help="visualize different views of a hyperparameter landscape dataset")
    viz_subparsers = parser_viz.add_subparsers()
    viz_subparsers.required = True

    # phases viz sobol ...
    parser_viz_sobol = viz_subparsers.add_parser("sobol", help="view the sobol space as defined in the hydra config")
    parser_viz_sobol.add_argument("overrides", nargs="*", help="Hydra overrides")
    parser_viz_sobol.set_defaults(func="viz_sobol")

    # phases viz samples ...
    parser_viz_samples = viz_subparsers.add_parser(
        "samples", help="view the sobol space from a hyperparameter landscape dataset"
    )
    # parser_viz_samples.add_argument("overrides", nargs="*", help="Hydra overrides")
    parser_viz_samples.add_argument("data", help="csv file containing data of all runs")
    parser_viz_samples.set_defaults(func="viz_samples")

    # phases viz hsgp ...
    parser_viz_hsgp = viz_subparsers.add_parser(
        "hsgp", help="view a GP model trained on a hyperparameter landscape dataset"
    )
    _add_model_viz_args(parser_viz_hsgp)
    parser_viz_hsgp.set_defaults(func="viz_hsgp")

    # phases viz linear ...
    parser_viz_linear = viz_subparsers.add_parser(
        "linear", help="view a linear interpolation of select data from a hyperparameter landscape dataset"
    )
    _add_model_viz_args(parser_viz_linear)
    parser_viz_linear.set_defaults(func="viz_linear")

    # phases viz triple-gp ...
    parser_viz_triple_gp = viz_subparsers.add_parser(
        "triple-gp", help="view a triple-GP model trained on a hyperparameter landscape dataset"
    )
    _add_model_viz_args(parser_viz_triple_gp)
    parser_viz_triple_gp.set_defaults(func="viz_triple_gp")

    # phases viz data ...
    parser_viz_data = viz_subparsers.add_parser(
        "data", help="view all datapoints from a hyperparameter landscape dataset"
    )
    parser_viz_data.add_argument("data", help="csv file containing data of all runs")
    parser_viz_data.set_defaults(func="viz_data")

    # phases ana ...
    parser_ana = subparsers.add_parser(
        "ana", help="analyze different aspects of data from a hyperparameter landscape dataset"
    )
    ana_subparsers = parser_ana.add_subparsers()
    ana_subparsers.required = True

    # phases ana maps ... (surfaces and peaks of surfaces)
    parser_ana_maps = ana_subparsers.add_parser("maps", help="")
    parser_ana_maps.add_argument("--data", help="csv file containing data of all runs", required=True)
    parser_ana_maps.add_argument("--model", type=str, choices=MODELS, required=True)
    parser_ana_maps.add_argument("--grid-length", dest="grid_length", type=int, default=DEFAULT_GRID_LENGTH)
    parser_ana_maps.set_defaults(func="maps")

    # # phases ana rb ...
    # parser_ana_rb = ana_subparsers.add_parser("rb", help="")
    # parser_ana_rb.add_argument("side", type=str, choices=["lower", "upper"], help="which rubber band to calculate")
    # parser_ana_rb.add_argument("--data", help="csv file containing data of all runs", required=True)
    # parser_ana_rb.add_argument("--model", type=str, choices=MODELS, required=True)
    # parser_ana_rb.add_argument("--grid-length", dest="grid_length", type=int, default=DEFAULT_GRID_LENGTH)
    # parser_ana_rb.set_defaults(func="ana_rb")

    # phases ana peaks ... (map FTU PHI's)
    parser_ana_peaks = ana_subparsers.add_parser("peaks", help="")
    # parser_ana_peaks.add_argument(
    #     "layer", type=str, choices=["lower", "middle", "upper"], help="which function of the model to analyze"
    # )
    parser_ana_peaks.add_argument("--data", help="csv file containing data of all runs", required=True)
    parser_ana_peaks.add_argument("--model", type=str, choices=MODELS, required=True)
    parser_ana_peaks.add_argument("--grid-length", dest="grid_length", type=int, default=DEFAULT_GRID_LENGTH)
    parser_ana_peaks.set_defaults(func="peaks")

    # phases ana modalities ... (scatter all samples and highlight the found modes per conf)
    parser_ana_modalities = ana_subparsers.add_parser("modalities", help="")
    parser_ana_modalities.add_argument("--data", help="csv file containing data of all runs", required=True)
    parser_ana_modalities.add_argument("--grid-length", dest="grid_length", type=int, default=DEFAULT_GRID_LENGTH)
    parser_ana_modalities.set_defaults(func="modalities", model=None)

    # phases dl ...
    parser_dl = subparsers.add_parser("dl")
    parser_dl.add_argument("project_name", type=str)
    parser_dl.set_defaults(func="dl")

    # handle args:
    args = parser.parse_args()
    match args.func:
        case "run":
            start_phases(_prepare_hydra(args))
        case "viz_samples":
            visualize_data_samples(args.data)
        case "viz_sobol":
            visualize_sobol_samples(_prepare_hydra(args))
        case "viz_hsgp" | "viz_linear" | "viz_triple_gp":
            file = Path(args.data)
            df = read_wandb_csv(file)
            phase_strs = sorted(df["meta.phase"].unique())
            for phase_str in phase_strs:
                phase_data = df[df["meta.phase"] == phase_str].sort_values("meta.conf_index")
                match args.func:
                    case "viz_hsgp":
                        model_folder = file.parent / f"{file.stem}_hsgp_{phase_str}"
                        model = HSGPModel(phase_data, 10, np.float64, "ls_eval/returns", None)
                        if args.load:
                            model.load(model_folder)
                        else:
                            model.fit(100, verbose=True)
                            # model.fit(2, verbose=True)
                        if args.save and not args.load:
                            model.save(model_folder)
                    case "viz_linear":
                        model = RBFInterpolatorLSModel(phase_data, np.float64, "ls_eval/returns", None)
                    case "viz_triple_gp":
                        model = TripleGPModel(phase_data, np.float64, "ls_eval/returns", None)
                        model.fit()
                    case _:
                        parser.print_help()
                        return
                model.visualize(grid_length=args.grid_length, which="maps")
            # _add_legend(fig)
            # plt.show()
        case "viz_data":
            visualize_data(args.data)
        case "maps" | "peaks" | "modalities":
            # lazily import ana-only deps:
            from autorl_landscape.analyze.concavity import reject_concavity
            from autorl_landscape.analyze.distributions import check_modality
            from autorl_landscape.analyze.peaks import find_peaks_model

            file = Path(args.data)
            df = read_wandb_csv(file)
            phase_strs = sorted(df["meta.phase"].unique())
            for phase_str in phase_strs:
                phase_data = df[df["meta.phase"] == phase_str].sort_values("meta.conf_index")
                match args.model:
                    case "hsgp":
                        model = HSGPModel(phase_data, 10, np.float64, "ls_eval/returns", None)
                        model_folder = file.parent / f"{file.stem}_hsgp_{phase_str}"
                        model.load(model_folder)
                    case "linear":
                        model = RBFInterpolatorLSModel(phase_data, np.float64, "ls_eval/returns", None)
                    case "triple-gp":
                        model = TripleGPModel(phase_data, np.float64, "ls_eval/returns", None)
                        model.fit()
                    case "mock":
                        model = MockLSModel(phase_data, np.float64, "ls_eval/returns", None)
                    case _:
                        model = LSModel(phase_data, np.float64, "ls_eval/returns", None)
                match args.func:
                    case "maps":
                        find_peaks_model(model, len(model.dim_info), args.grid_length, bounds=(0, 1))
                    case "peaks":
                        check_modality(model)
                    case "graphs":
                        check_modality(model)
                    case "ci_graphs":
                        reject_concavity(model, grid_length=args.grid_length)
                        raise NotImplementedError
                    case _:
                        parser.print_help()
                        return
                model.visualize(grid_length=args.grid_length, which=args.func)
            # _add_legend(fig)
            # plt.show()
        case "dl":
            download_data(ENTITY, args.project_name)
        case _:
            parser.print_help()
            return


def _prepare_hydra(args: argparse.Namespace) -> DictConfig:
    hydra.initialize(config_path="../conf", version_base="1.1")
    conf = hydra.compose("config", overrides=args.overrides)
    print(OmegaConf.to_yaml(conf))
    return conf


def _add_model_viz_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("data", help="csv file containing data of all runs")
    # group_samples = parser.add_mutually_exclusive_group()
    # group_samples.add_argument(
    #     "--viz-samples", action="store_true", dest="viz_samples", help="Visualize all performance samples"
    # )
    # group_samples.add_argument(
    #     "--viz-model-samples",
    #     action="store_true",
    #     dest="viz_model_samples",
    #     help="Only visualize samples that were used to train the model",
    # )
    group_sl = parser.add_mutually_exclusive_group()
    group_sl.add_argument("--save", action="store_true", dest="save", help="Save the trained model to disk")
    group_sl.add_argument("--load", action="store_true", dest="load", help="Load the trained model from disk")
    parser.add_argument("--grid-length", dest="grid_length", type=int, default=DEFAULT_GRID_LENGTH)
    # parser.add_argument("--which", type=str, choices=VISUALIZATION_GROUPS, help="Which specific viz's to show")


def _add_legend(fig: Figure, hide_at_start: bool = True) -> None:
    legend = fig.legend(handles=fig.axes[0].collections)
    fig_artss = _transpose([ax.collections for ax in fig.axes])
    leg_to_fig: dict[Artist, list[Any]] = {}
    for leg_text, fig_arts in zip(legend.get_texts(), fig_artss):
        if hide_at_start:
            for fig_art in fig_arts:
                if fig_art is not None:
                    fig_art.set_visible(False)
            leg_text.set_alpha(0.2)
        leg_text.set_picker(True)
        leg_to_fig[leg_text] = fig_arts

    def on_pick(event: PickEvent):
        leg_text = event.artist
        fig_artists = leg_to_fig[leg_text]
        for fig_artist in fig_artists:
            if fig_artist is not None:
                visible = not fig_artist.get_visible()
                fig_artist.set_visible(visible)
        leg_text.set_alpha(1.0 if visible else 0.2)
        fig.canvas.draw()

    fig.canvas.mpl_connect("pick_event", on_pick)


def _transpose(ll: Iterable[Iterable[T]]) -> list[list[T]]:
    """Transposes lists of lists (or other things you can iterate over)."""
    return list(map(list, zip_longest(*ll, fillvalue=None)))


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
