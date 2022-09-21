from datetime import datetime

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from autorl_landscape.train import run_phase


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf: DictConfig) -> None:
    print(OmegaConf.to_yaml(conf))
    # remember starting time of this run for saving all phase data:
    date_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    wandb.tensorboard.patch(root_logdir="...")

    phases = conf.phases + [conf.env.total_timesteps]
    if not all(x < y for x, y in zip(phases, phases[1:])):
        raise Exception(f"Phases need to be strictly increasing. Got: {phases}")
    if conf.phases[-1] >= int(conf.env.total_timesteps * conf.eval.final_eval_start):
        raise Exception(
            "Last phase(s) too long! Not enough timesteps for final evaluation.\n"
            + f"Last phase start: {conf.phases[-1]}.\n"
            + f"final_eval start: {conf.eval.final_eval_start}"
        )

    last_t_phase = 0
    original_total_timesteps = conf.env.total_timesteps
    init_agent = None
    for i, t_phase in enumerate(phases):
        phase_str = f"phase_{i}"
        run_phase(
            conf=conf,
            t_ls=t_phase - last_t_phase,
            t_final=original_total_timesteps - last_t_phase,
            date_str=date_str,
            phase_str=phase_str,
            init_agent=init_agent,
        )
        init_agent = f"phase_results/{conf.agent.name}/{conf.env.name}/{date_str}/{phase_str}/best_agent/model.zip"
        last_t_phase = t_phase
        # TODO next phases run for too long, maybe hack global_step?
    return


if __name__ == "__main__":
    main()
