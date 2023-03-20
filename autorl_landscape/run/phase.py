from typing import Any, TypeVar

import pickle
import time
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import submitit
import tqdm
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from pandas import DataFrame

from autorl_landscape.run.compare import choose_best_policy, construct_2d
from autorl_landscape.run.train import train_agent
from autorl_landscape.util.data import split_phases
from autorl_landscape.util.ls_sampler import construct_ls

CONF_DICT_KEY_START = "conf."


def start_phases(conf: DictConfig) -> None:
    """Run the experiment with the given configuration.

    Args:
        conf: Hydra configuration
    """
    # remember starting time of this run for saving all phase data:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if len(conf.phases) > 1 and conf.phases[-2] >= int(conf.phases[-1] * conf.eval.final_eval_start):
        raise Exception(
            "Last phase(s) too short! Not enough timesteps for all final evaluations.\n"
            f"Last phase start: {conf.phases[-2]}.\n"
            f"final_eval start: {conf.eval.final_eval_start}"
        )

    if not all(x < y for x, y in zip(conf.phases, conf.phases[1:])):
        error_msg = f"Phases need to be strictly increasing. Got: {conf.phases}"
        raise ValueError(error_msg)

    ancestor = None
    for phase_index, _ in enumerate(conf.phases, start=1):
        phase(conf, phase_index, timestamp, ancestor)
        ancestor = (
            Path(f"phase_results/{conf.agent.name}/{conf.env.name}/{timestamp}/phase_{phase_index}/best_agent")
            .resolve()
            .relative_to(Path.cwd())
        )


def resume_phases(incomplete_data: DataFrame) -> None:
    """Try to complete a crashed experiment from local and downloaded wandb data.

    - extract the used DictConfig from the data
    - figure out the last phase that is complete in the csv
    """
    # Construct the hydra config:
    keys = [col for col in incomplete_data.columns if col.startswith(CONF_DICT_KEY_START)]
    # This has keys like "conf.slurm.cluster" and values like {"hqa3bj2a": nan}:
    conf_dict_scuffed: dict[str, Any] = incomplete_data[0:1][keys].to_dict()
    # We need a list like ["slurm.cluster=nan", ...]:
    conf_list = [f"{k[len(CONF_DICT_KEY_START):]}={list(v.values())[0]}" for k, v in conf_dict_scuffed.items()]
    conf = OmegaConf.from_dotlist(conf_list)
    # Fix None becoming "nan":
    if conf.slurm.cluster == "nan":
        conf.slurm.cluster = None

    resume_tag = f"resume_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    match conf.wandb.experiment_tag:
        case str(tag):
            conf.wandb.experiment_tag = [tag, resume_tag]
        case Iterable(tags):
            conf.wandb.experiment_tag = [*tags, resume_tag]
        case _:
            msg = f"Received unexpected {conf.wandb.experiment_tag=}"
            raise ValueError(msg)

    # Check last phase in the data, whether all runs are there:
    last_complete_phase = None
    last_complete_phase_ids = None
    phase_indices = sorted(incomplete_data["meta.phase"].unique(), reverse=True)
    runs_per_phase = conf.num_confs * conf.num_seeds
    for phase_index in phase_indices:
        phase_data, _ = split_phases(incomplete_data, phase_index)
        if len(phase_data) == runs_per_phase:
            last_complete_phase = phase_index
            last_complete_phase_ids = phase_data.index
            break

    if last_complete_phase is None or last_complete_phase_ids is None:
        error_msg = "Data does not include a complete phase."
        raise ValueError(error_msg)

    # Extract original timestamp from the data:
    original_timestamps = incomplete_data["meta.timestamp"].unique()
    if len(original_timestamps) != 1 or not isinstance(original_timestamps[0], str):
        error_msg = f"Data has incorrect timestamp: {original_timestamps}"
        raise ValueError(error_msg)
    original_timestamp: str = original_timestamps[0]

    phase_path = f"phase_results/{conf.agent.name}/{conf.env.name}/{original_timestamp}/phase_{last_complete_phase}"
    if not (Path(phase_path) / Path("best_agent")).exists():
        print(f"Did not find {phase_path}/best_agent symlink. Trying to select best config from submitit results...")
        ids_to_results: dict[str, tuple[int, NDArray[Any]] | None] = {
            wandb_id: None for wandb_id in last_complete_phase_ids
        }

        pathlist = (Path(__file__).parent.parent.parent / Path("submitit")).glob("*_result.pkl")  # Yes this is ugly
        for path in pathlist:
            with path.open("rb") as f:
                try:
                    _, (conf_index, wandb_id, arr) = pickle.load(f)  # FIXME
                except ValueError:  # could not unpack tuple
                    continue
                if not isinstance(wandb_id, str):
                    continue  # unusable file
                if wandb_id in ids_to_results:
                    if not (isinstance(conf_index, int) and isinstance(arr, np.ndarray)):
                        msg = f"results for {conf_index=} {wandb_id=} ({path=}) are incorrect: {arr} {arr.shape}"
                        raise ValueError(msg)
                    ids_to_results[wandb_id] = (conf_index, arr)

        missing_ids = [wandb_id for wandb_id, result in ids_to_results.items() if result is None]
        if len(missing_ids) > 0:
            error_msg = f"Could not find results for ids {missing_ids}"
            raise ValueError(error_msg)

        # Like in phase():
        results = [(conf_index, wandb_id, arr) for wandb_id, (conf_index, arr) in ids_to_results.items()]
        run_ids, final_returns = construct_2d(*zip(*results))
        choose_best_policy(
            run_ids,
            final_returns,
            save=phase_path,
        )
    else:
        print(f"Found {phase_path}/best_agent")

    # Like in start_phases():
    ancestor = Path(f"{phase_path}/best_agent").resolve().relative_to(Path.cwd())
    for phase_index, _ in enumerate(conf.phases, start=1):
        if phase_index <= last_complete_phase:
            print(f"Skipping phase {phase_index}")
            continue
        phase(conf, phase_index, original_timestamp, ancestor)
        ancestor = (
            Path(f"phase_results/{conf.agent.name}/{conf.env.name}/{original_timestamp}/phase_{phase_index}/best_agent")
            .resolve()
            .relative_to(Path.cwd())
        )


def phase(conf: DictConfig, phase_index: int, timestamp: str, ancestor: Path | None = None) -> None:
    """Train a number of sampled configurations, evaluating and saving all agents at t_ls env steps.

    If initial_agent is given, start with its progress instead of training from 0. After this, train
    all agents until t_final env steps and evaluate here to choose the best configuration.

    Args:
        conf: Configuration for the experiment
        phase_index: Number naming the current phase. For the first phase, this is 1
        timestamp: Timestamp that is equal for all phases of this experiment, used for saving
        ancestor: If present, leads to the agent which `seeds` this phase.
    """
    # Base directory for saving agents of the current phase:
    phase_path = f"phase_results/{conf.agent.name}/{conf.env.name}/{timestamp}/phase_{phase_index}"

    executor = submitit.AutoExecutor(folder="submitit", cluster=conf.slurm.cluster)
    tasks = []
    executor.update_parameters(**conf.slurm.update_parameters)

    for conf_index, c in construct_ls(conf).iterrows():  # NOTE iterrows() changes datatypes, we get only np.float64
        # set hyperparameters:
        # ls_conf = {
        #     "learning_rate": c["learning_rate"],
        #     "gamma": 1 - c["neg_gamma"],
        #     # "exploration_final_eps": c["exploration_final_eps"],
        # }
        ls_conf = dict(c)

        for seed in range(conf.seeds.agent, conf.seeds.agent + conf.num_seeds):
            task = (conf, phase_index, timestamp, ancestor, ls_conf, seed, conf_index, phase_path)
            tasks.append(task)

    results = schedule_runs(executor, train_agent, tasks, num_parallel=conf.slurm.num_parallel, polling_rate=10)

    # conf_indices, run_ids, final_scores = zip(*results)
    # run_ids = np.array(run_ids)
    # final_scores = np.array(final_scores)
    run_ids, final_returns = construct_2d(*zip(*results))

    best = choose_best_policy(run_ids, final_returns, save=phase_path)

    print(f"-- PHASE {phase_index} REPORT --")
    print(f"{run_ids=}")
    print(f"{final_returns=}")
    print(f"Best run: {best}\n")


T = TypeVar("T")


def schedule_runs(
    executor: submitit.AutoExecutor,
    fn: Callable[..., T],
    fn_args: Sequence[tuple[Any, ...]],  # [(x, y, z, ...)]
    num_parallel: int,
    polling_rate: float = 1.0,
) -> list[T]:
    """Start a limited number of jobs on the submitit cluster, returning results of all jobs when all jobs finish.

    executor : submitit.AutoExecutor
        The SLURM (or local) job executor
    fn : Callable
        The function that is the task.
    fn_args : Sequence[Tuple[Any, ...]]
        A list of tuples. Each tuple represents the arguments to one function call.
    num_parallel : int
        How many jobs to have running at the same time.

    Returns:
    List
        The return values of all jobs.
    """
    if len(fn_args) == 0:
        return []
    if num_parallel < 1:
        raise ValueError
    running_jobs: dict[int, submitit.Job[T]] = {}
    results: dict[int, Any] = {}
    next_job = 0  # index of the job to be started next
    with tqdm.tqdm(total=len(fn_args)) as prog_bar:
        while True:
            # prog_bar.update(next_job)
            # Check running jobs for finished jobs, then save their results:
            for i, job in list(running_jobs.items()):
                if job.done():
                    results[i] = job.result()
                    prog_bar.n = len(results)
                    prog_bar.refresh()
                    del running_jobs[i]

            # Add new jobs:
            if len(running_jobs) > num_parallel:  # that's kind of the point
                error_msg = "Too many running jobs!"
                raise Exception(error_msg)

            # not too many at once, don't overflow index when close to done with all jobs:
            num_new_jobs = min(num_parallel - len(running_jobs), len(fn_args) - next_job)
            if num_new_jobs > 0:
                new_tasks = fn_args[next_job : next_job + num_new_jobs]
                new_jobs = executor.map_array(fn, *zip(*new_tasks))
                if len(new_jobs) != num_new_jobs:
                    error_msg = "Did not get correct number of new jobs!"
                    raise Exception(error_msg)
                for i in range(num_new_jobs):
                    running_jobs[next_job + i] = new_jobs[i]
                next_job += num_new_jobs

            # Do this until we have collected all results:
            if len(results) >= len(fn_args):
                break
            time.sleep(polling_rate)
    return [results[i] for i in range(len(fn_args))]
