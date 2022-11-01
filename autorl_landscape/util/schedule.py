from typing import Any, Callable, ParamSpec, Sequence, Tuple, TypeVar

import time

import submitit
import tqdm

T = TypeVar("T")
P = ParamSpec("P")


def schedule(
    executor: submitit.AutoExecutor,
    fn: Callable[P, T],
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
