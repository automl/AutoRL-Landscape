from typing import Any, Callable, Sequence, Tuple, TypeVar

import time
from datetime import datetime
from pathlib import Path

import submitit
import tqdm
from omegaconf import DictConfig

from autorl_landscape.run.compare import choose_best_policy, construct_2d
from autorl_landscape.run.train import train_agent
from autorl_landscape.util.download import get_all_tags
from autorl_landscape.util.ls_sampler import construct_ls


def start_phases(conf: DictConfig) -> None:
    """Run the experiment with the given configuration.

    Args:
        conf: Hydra configuration
    """
    # check whether given tag is unused (to not mess up other experiments):
    tags = get_all_tags(conf.wandb.entity, conf.wandb.project)
    assert type(conf.wandb.experiment_tag) == str
    if conf.wandb.experiment_tag != "debug":
        assert conf.wandb.experiment_tag not in tags, f"Use a unique experiment tag for new experiments! Used: {tags}"

    # remember starting time of this run for saving all phase data:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # TODO still not quite sure what this does but error message seems to be gone
    # wandb.tensorboard.patch(root_logdir="...")

    if len(conf.phases) > 1:
        if conf.phases[-2] >= int(conf.phases[-1] * conf.eval.final_eval_start):
            raise Exception(
                "Last phase(s) too short! Not enough timesteps for all final evaluations.\n"
                f"Last phase start: {conf.phases[-2]}.\n"
                f"final_eval start: {conf.eval.final_eval_start}"
            )

    if not all(x < y for x, y in zip(conf.phases, conf.phases[1:])):
        raise Exception(f"Phases need to be strictly increasing. Got: {conf.phases}")

    ancestor = None
    for phase_index, _ in enumerate(conf.phases, start=1):
        phase(conf, phase_index, timestamp, ancestor)
        ancestor = (
            Path(f"phase_results/{conf.agent.name}/{conf.env.name}/{timestamp}/phase_{phase_index}/best_agent")
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
        ls_conf = {
            "learning_rate": c["learning_rate"],
            "gamma": 1 - c["neg_gamma"],
            # "exploration_final_eps": c["exploration_final_eps"],
        }

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
    fn_args: Sequence[Tuple[Any, ...]],  # [(x, y, z, ...)]
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

    Returns
    -------
    List
        The return values of all jobs.
    """
    if len(fn_args) == 0:
        return []
    assert num_parallel > 0
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
            assert len(running_jobs) <= num_parallel  # that's kind of the point
            # not too many at once, don't overflow index when close to done with all jobs:
            num_new_jobs = min(num_parallel - len(running_jobs), len(fn_args) - next_job)
            if num_new_jobs > 0:
                new_tasks = fn_args[next_job : next_job + num_new_jobs]
                new_jobs = executor.map_array(fn, *zip(*new_tasks))
                assert len(new_jobs) == num_new_jobs
                for i in range(num_new_jobs):
                    running_jobs[next_job + i] = new_jobs[i]
                next_job += num_new_jobs

            # Do this until we have collected all results:
            if len(results) >= len(fn_args):
                break
            time.sleep(polling_rate)
    return [results[i] for i in range(len(fn_args))]
