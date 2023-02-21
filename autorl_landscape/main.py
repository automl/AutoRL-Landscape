from typing import Iterable, TypeVar

import argparse
from datetime import datetime
from itertools import zip_longest
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from omegaconf import DictConfig, OmegaConf

from autorl_landscape.ls_models.ls_model import LSModel
from autorl_landscape.ls_models.rbf import RBFInterpolatorLSModel
from autorl_landscape.ls_models.triple_gp import TripleGPModel
from autorl_landscape.train import run_phase
from autorl_landscape.util.data import read_wandb_csv, split_phases
from autorl_landscape.util.download import download_data, get_all_tags
from autorl_landscape.visualize import (
    FIGSIZES,
    LEGEND_FSIZE,
    TITLE_FSIZE,
    visualize_data_samples,
    visualize_nd,
)

# ENTITY = "kwie98"
DEFAULT_GRID_LENGTH = 251
MODELS = ["hsgp", "rbf", "triple-gp", "mock"]
VISUALIZATION_GROUPS = ["maps", "peaks", "graphs"]

T = TypeVar("T")


# @hydra.main(version_base=None, config_path="conf", config_name="config")
def main() -> None:
    """Choose to either start the phases or visualize the landscape samples."""
    # Parse non-hydra commandline arguments:
    parser = argparse.ArgumentParser(prog="phases")
    subparsers = parser.add_subparsers()
    subparsers.required = True
    parser.add_argument("--savefig", action="store_true", dest="savefig", help="Save figures instead of showing them")

    # phases run ...
    parser_run = subparsers.add_parser("run", help="run an experiment using the hydra config in conf/")
    parser_run.add_argument("overrides", nargs="*", help="Hydra overrides")
    parser_run.set_defaults(func="run")

    # phases viz ...
    parser_viz = subparsers.add_parser("viz", help="visualize different views of a hyperparameter landscape dataset")
    viz_subparsers = parser_viz.add_subparsers()
    viz_subparsers.required = True

    # phases viz samples ...
    parser_viz_samples = viz_subparsers.add_parser(
        "samples", help="view the sobol space from a hyperparameter landscape dataset"
    )
    # parser_viz_samples.add_argument("overrides", nargs="*", help="Hydra overrides")
    parser_viz_samples.add_argument("data", help="csv file containing data of all runs")
    parser_viz_samples.set_defaults(func="viz_samples")

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

    # phases ana modalities ...
    parser_ana_modalities = ana_subparsers.add_parser("modalities", help="")
    parser_ana_modalities.add_argument("--data", help="csv file containing data of all runs", required=True)
    parser_ana_modalities.add_argument("--grid-length", dest="grid_length", type=int, default=DEFAULT_GRID_LENGTH)
    parser_ana_modalities.set_defaults(func="modalities", model=None)

    # phases ana concavity ...
    parser_ana_concavity = ana_subparsers.add_parser("concavity", help="")
    parser_ana_concavity.add_argument("--data", help="csv file containing data of all runs", required=True)
    parser_ana_concavity.add_argument("--model", type=str, choices=MODELS, required=True)
    parser_ana_concavity.add_argument("--grid-length", dest="grid_length", type=int, default=DEFAULT_GRID_LENGTH)
    parser_ana_concavity.set_defaults(func="concavity", model=None)

    # phases ana graphs ...
    parser_ana_graphs = ana_subparsers.add_parser("graphs", help="")
    parser_ana_graphs.add_argument("--data", help="csv file containing data of all runs", required=True)
    parser_ana_graphs.add_argument("--model", type=str, choices=MODELS, required=True)
    parser_ana_graphs.set_defaults(func="graphs", grid_length=DEFAULT_GRID_LENGTH, model=None)

    # phases dl ...
    parser_dl = subparsers.add_parser("dl")
    parser_dl.add_argument("entity_name", type=str)
    parser_dl.add_argument("project_name", type=str)
    parser_dl.add_argument("experiment_tag", type=str)
    parser_dl.set_defaults(func="dl")

    # handle args:
    args = parser.parse_args()
    match args.func:
        case "run":
            start_phases(_prepare_hydra(args))
        case "viz_samples":
            visualize_data_samples(args.data)
        # case "viz_data":
        #     visualize_data(args.data)
        case "maps" | "modalities" | "graphs":
            # lazily import ana-only deps:
            from autorl_landscape.analyze.modalities import check_modality
            from autorl_landscape.analyze.peaks import find_peaks_model

            file = Path(args.data)
            df = read_wandb_csv(file)
            phase_strs = sorted(df["meta.phase"].unique())

            fig = plt.figure(figsize=FIGSIZES[args.func])
            global_gs = fig.add_gridspec(1, 1 + len(phase_strs))
            for phase_str, sub_gs in zip(phase_strs, [gs for gs in global_gs][1:]):
                phase_data, best_conf = split_phases(df, phase_str)
                match args.model:
                    case "rbf":
                        model = RBFInterpolatorLSModel(phase_data, np.float64, "ls_eval/returns", None, best_conf)
                    case "triple-gp":
                        model = TripleGPModel(phase_data, np.float64, "ls_eval/returns", None, best_conf)
                        model.fit()
                    case _:
                        model = LSModel(phase_data, np.float64, "ls_eval/returns", None, best_conf)
                match args.func:
                    case "maps":
                        if not isinstance(model, RBFInterpolatorLSModel):
                            find_peaks_model(model, len(model.dim_info), args.grid_length, bounds=(0, 1))
                        add_legend = True
                    case "modalities":
                        check_modality(model, args.grid_length)
                        add_legend = False
                    case "graphs":
                        add_legend = False
                    case _:
                        parser.print_help()
                        return
                fig_file_part = None
                if args.savefig:
                    if args.model in MODELS:
                        fig_file_part = f"images/ana-{args.func}-{args.model}"
                    else:
                        fig_file_part = f"images/ana-{args.func}"
                row_titles, height_ratios = visualize_nd(
                    model, fig, sub_gs, args.grid_length, viz_group=args.func, phase_str=phase_str
                )
            plot_figure(fig, global_gs, fig_file_part, row_titles, height_ratios, add_legend)
        case "concavity":
            from autorl_landscape.analyze.concavity import find_biggest_nonconcave

            file = Path(args.data)
            df = read_wandb_csv(file)
            phase_strs = sorted(df["meta.phase"].unique())
            for phase_str in phase_strs:
                phase_data, _ = split_phases(df, phase_str)
                match args.model:
                    case "rbf":
                        model = RBFInterpolatorLSModel(phase_data, np.float64, "ls_eval/returns")
                    case "triple-gp":
                        model = TripleGPModel(phase_data, np.float64, "ls_eval/returns")
                        model.fit()
                    case _:
                        pass
                print(f"{phase_str}:")
                smallest_rejecting_ci = find_biggest_nonconcave(model, args.grid_length)
                print(f"Concavity can be rejected for squeezes stronger than k_{{max}} = {smallest_rejecting_ci:.2f}")
        case "dl":
            download_data(args.entity_name, args.project_name, args.experiment_tag)
        case _:
            parser.print_help()
            return


def plot_figure(
    fig: Figure,
    global_gs: GridSpec,
    fig_file_part: str | None,
    row_titles: list[str],
    height_ratios: list[float],
    add_legend: bool = True,
) -> None:
    """Show or save the figure after adding row titles."""
    # add titles:
    title_gs = GridSpecFromSubplotSpec(1 + len(row_titles), 1, global_gs[0], height_ratios=height_ratios)
    for i, row_title in enumerate(row_titles, start=1):
        ax = fig.add_subplot(title_gs[i, 0])
        ax.text(1.0, 0.5, row_title, ha="right", va="center", fontsize=TITLE_FSIZE)
        ax.axis("off")

    if add_legend:
        # get unique labels in whole figure:
        foo = [a.get_legend_handles_labels() for a in fig.axes]
        handles, labels = [sum(f, []) for f in _transpose(foo)]
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), loc="center right", fontsize=LEGEND_FSIZE)

    if fig_file_part is None:
        plt.show()
    else:
        Path(fig_file_part).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{fig_file_part}.pdf", bbox_inches="tight")


def _prepare_hydra(args: argparse.Namespace) -> DictConfig:
    hydra.initialize(config_path="../conf", version_base="1.1")
    conf = hydra.compose("config", overrides=args.overrides)
    print(OmegaConf.to_yaml(conf))
    return conf


def _transpose(ll: Iterable[Iterable[T]]) -> list[list[T]]:
    """Transposes lists of lists (or other things you can iterate over)."""
    return list(map(list, zip_longest(*ll, fillvalue=None)))


def start_phases(conf: DictConfig) -> None:
    """Run the experiment with the given configuration.

    Args:
        conf: Hydra configuration
    """
    # check whether given tag is unused (to not mess up other experiments):
    tags = get_all_tags(conf.wandb.entity, conf.wandb.project)
    assert type(conf.wandb.experiment_tag) == str
    assert conf.wandb.experiment_tag not in tags, f"Use a unique experiment tag for new experiments! Used: {tags}"

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


# def _add_legend(fig: Figure, hide_at_start: bool = True) -> None:
#     legend = fig.legend(handles=fig.axes[0].collections)
#     fig_artss = _transpose([ax.collections for ax in fig.axes])
#     leg_to_fig: dict[Artist, list[Any]] = {}
#     for leg_text, fig_arts in zip(legend.get_texts(), fig_artss):
#         if hide_at_start:
#             for fig_art in fig_arts:
#                 if fig_art is not None:
#                     fig_art.set_visible(False)
#             leg_text.set_alpha(0.2)
#         leg_text.set_picker(True)
#         leg_to_fig[leg_text] = fig_arts

#     def on_pick(event: PickEvent):
#         leg_text = event.artist
#         fig_artists = leg_to_fig[leg_text]
#         for fig_artist in fig_artists:
#             if fig_artist is not None:
#                 visible = not fig_artist.get_visible()
#                 fig_artist.set_visible(visible)
#         leg_text.set_alpha(1.0 if visible else 0.2)
#         fig.canvas.draw()

#     fig.canvas.mpl_connect("pick_event", on_pick)
