from datetime import datetime

import hydra
from omegaconf import DictConfig

from autorl_landscape.train import run_phase


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf: DictConfig) -> None:
    # remember starting time of this run for saving all phase data:
    date_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    if not all(x < y for x, y in zip(conf.phases, conf.phases[1:])):
        raise Exception(f"Phases need to be strictly increasing. Got: {conf.phases}")

    last_t_phase = 0
    for i, t_phase in enumerate(conf.phases):
        phase_str = f"phase_{i}"
        run_phase(conf=conf, t_ls=t_phase - last_t_phase, date_str=date_str, phase_str=phase_str, initial_agent=None)


if __name__ == "__main__":
    main()
