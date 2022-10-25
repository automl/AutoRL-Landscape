import time
from multiprocessing import Pool

import numpy as np
import pytest
import submitit
import wandb


def wandb_dummy(index: int) -> None:
    run = wandb.init(
        project="test_overwhelm",
        config={
            "index": index,
        },
        sync_tensorboard=False,
        monitor_gym=False,
        save_code=False,
    )
    for i in range(20):
        wandb.log({"useless_data": np.random.random(1024), "x": i, "y": np.sqrt(i) + 0.1 * np.random.random()})
        time.sleep(1)
    run.finish()


@pytest.mark.skip(reason="Load Check")
def test_many_wandb_jobs() -> None:
    """Checks how much we can load wandb (i.e. how many concurrent jobs)."""
    executor = submitit.AutoExecutor(folder="test_submitit", cluster="local")
    executor.update_parameters(timeout_min=1000, slurm_partition="dev", gpus_per_node=1)

    num_tasks = 2
    # num_parallel = 50
    # polling_rate = 5
    with Pool(50) as p:
        p.map(wandb_dummy, [i for i in range(num_tasks)])
    print("hello")
    # schedule(executor, wandb_dummy, [(i,) for i in range(num_tasks)], num_parallel, polling_rate)
