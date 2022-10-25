from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from autorl_landscape.train import run_phase


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf: DictConfig) -> None:
    """Run the experiment with the configuration from the `conf/` directory."""
    print(OmegaConf.to_yaml(conf))
    # remember starting time of this run for saving all phase data:
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # TODO still not quite sure what this does but error message seems to be gone
    # wandb.tensorboard.patch(root_logdir="...")

    if conf.phases is not None:
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
            init_agent = (
                Path(f"phase_results/{conf.agent.name}/{conf.env.name}/{date_str}/{phase_str}/best_agent")
                .resolve()
                .relative_to(Path.cwd())
            )
            last_t_phase = t_phase
    else:
        # a rudimentary way to just run the agent without any phase stuff
        run_phase(
            conf=conf,
            t_ls=conf.env.total_timesteps,
            t_final=conf.env.total_timesteps,
            date_str=date_str,
            phase_str="phase_0",
            init_agent=None,
        )


if __name__ == "__main__":
    main()
