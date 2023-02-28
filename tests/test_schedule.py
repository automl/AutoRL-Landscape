from typing import Tuple

import time

import submitit

from autorl_landscape.util.schedule import schedule_runs


def test_null() -> None:
    """Tests running the scheduler with nothing to do."""
    executor = submitit.AutoExecutor(folder="test_submitit", cluster="local")
    executor.update_parameters(timeout_min=1000, slurm_partition="dev", gpus_per_node=1)

    res: list = schedule_runs(executor, lambda x: x, [], 10, 1)
    assert res == []


def test_many_jobs() -> None:
    """Tests high parallel setting."""
    executor = submitit.AutoExecutor(folder="test_submitit", cluster="local")
    executor.update_parameters(timeout_min=1000, slurm_partition="dev", gpus_per_node=1)

    schedule_runs(executor, time.sleep, [(1,) for _ in range(5)], num_parallel=1000)


def test_adding_many_at_same_time() -> None:
    """High parallel, not so high polling rate."""
    executor = submitit.AutoExecutor(folder="test_submitit", cluster="local")
    executor.update_parameters(timeout_min=1000, slurm_partition="dev", gpus_per_node=1)

    schedule_runs(executor, time.sleep, [(0.1,) for _ in range(20)], num_parallel=5, polling_rate=1)


def test_return_order() -> None:
    """Tests that return values are in the expected order."""
    executor = submitit.AutoExecutor(folder="test_submitit", cluster="local")
    executor.update_parameters(timeout_min=1000, slurm_partition="dev", gpus_per_node=1)

    def subtract(a: int, b: int) -> int:
        return a - b

    num_tasks = 20
    ret = schedule_runs(executor, subtract, [(2 * i, i) for i in range(num_tasks)], num_parallel=10, polling_rate=1)
    assert ret == list(range(num_tasks))


def check_timing() -> None:
    """Not a test. If only 2 jobs can run at the same time, 10 sleep(1) jobs have to take at least 5 seconds."""
    executor = submitit.AutoExecutor(folder="test_submitit", cluster="local")
    executor.update_parameters(timeout_min=1000, slurm_partition="dev", gpus_per_node=1)

    def f(t: float) -> Tuple[float, float, float]:
        t_start = time.perf_counter()
        time.sleep(t)
        return (t, t_start, time.perf_counter())

    t_0 = time.perf_counter()
    schedule_runs(executor, f, [(t,) for t in range(5, 0, -1)], 2, 0.001)  # TODO maybe reverse range?
    t_1 = time.perf_counter()
    # assert t_1 - t_0 >= 0.5  # TODO no overhead when starting dummy job first?
    print(t_1 - t_0)
