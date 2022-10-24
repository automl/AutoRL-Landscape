from typing import Any, Dict, List, Sequence, Tuple

import time

import submitit


def schedule(
    executor: submitit.AutoExecutor,
    tasks: Sequence[Tuple],  # [(func, x, y, z, ...)]
    num_parallel: int,
    polling_rate: float = 1.0,
) -> List:
    """Start a limited number of jobs on the submitit cluster, returning results of all jobs when all jobs finish.

    executor : submitit.AutoExecutor
        The SLURM (or local) job executor
    tasks : List[Tuple[Callable, ...]]
        A list of tasks. Each task is a tuple with the first entry being the function to call, the rest being its
        arguments.
    num_parallel : int
        How many jobs to have running at the same time.

    Returns
    -------
    List
        The return values of all jobs.
    """
    assert num_parallel > 0
    running_jobs: Dict[int, submitit.Job] = {}
    results: Dict[int, Any] = {}
    next_job = 0  # index of the job to be started next
    while True:
        # Check running jobs for finished jobs, then save their results:
        for i, job in list(running_jobs.items()):
            if job.done():
                results[i] = job.result()
                del running_jobs[i]  # sus?

        # Add new jobs:
        while len(running_jobs) < num_parallel and next_job < len(tasks):
            running_jobs[next_job] = executor.submit(*tasks[next_job])
            time.sleep(2)
            next_job += 1

        # Do this until we have collected all results:
        if len(results) >= len(tasks):
            break
        time.sleep(polling_rate)
    return [results[i] for i in range(len(tasks))]
