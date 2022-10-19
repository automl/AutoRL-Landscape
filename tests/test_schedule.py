import time

import submitit

from autorl_landscape.util.schedule import schedule


def test_null() -> None:
    executor = submitit.AutoExecutor(folder="test_submitit", cluster="local")
    executor.update_parameters(timeout_min=1000, slurm_partition="dev", gpus_per_node=1)

    res = schedule(executor, [], 10, 1)
    assert res == []
    return


def test_many_jobs() -> None:
    """Tests high parallel setting."""
    executor = submitit.AutoExecutor(folder="test_submitit", cluster="local")
    executor.update_parameters(timeout_min=1000, slurm_partition="dev", gpus_per_node=1)

    schedule(executor, [(time.sleep, 1) for _ in range(5)], num_parallel=1000)
    return


def check_timing() -> None:
    """Not a test. If only 2 jobs can run at the same time, 10 sleep(1) jobs have to take at least 5 seconds."""
    executor = submitit.AutoExecutor(folder="test_submitit", cluster="local")
    executor.update_parameters(timeout_min=1000, slurm_partition="dev", gpus_per_node=1)

    def f(t: float):
        t_start = time.perf_counter()
        time.sleep(t)
        return (t, t_start, time.perf_counter())

    t_0 = time.perf_counter()
    schedule(executor, [(f, t) for t in range(5, 0, -1)], 2, 0.001)  # TODO maybe reverse range?
    t_1 = time.perf_counter()
    # assert t_1 - t_0 >= 0.5  # TODO no overhead when starting dummy job first?
    print(t_1 - t_0)
    return
