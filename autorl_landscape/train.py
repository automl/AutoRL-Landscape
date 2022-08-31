from typing import Optional

import wandb
from ConfigSpace import Categorical, ConfigurationSpace, Float, Uniform
from omegaconf import DictConfig

from autorl_landscape.agents.agent import Agent
from autorl_landscape.agents.sac import SoftActorCritic
from autorl_landscape.agents.simple import SimpleAgent
from autorl_landscape.util.debug import DEBUG


def train(cfg: DictConfig) -> None:
    """Train a number of sampled configurations."""
    # Landscape Hyperparameters:
    nn_width = Categorical(name="neural net width", items=[16, 32, 64, 128], ordered=True)
    nn_length = Categorical(name="neural net length", items=[2, 3, 4, 5], ordered=True)
    lr = Float(name="learning rate", bounds=(0.0001, 0.1), distribution=Uniform(), log=True)
    # neg_gamma = 1 - gamma, such that log-uniform (reciprocal) distribution can be used
    neg_gamma = Float(name="negated gamma", bounds=(0.0001, 0.8), distribution=Uniform(), log=True)

    cs = ConfigurationSpace()
    cs.add_hyperparameters([nn_width, nn_length, lr, neg_gamma])

    for run in range(5):
        c = cs.sample_configuration()
        ls_conf = DictConfig(
            {
                "nn_width": c["neural net width"],
                "nn_length": c["neural net length"],
                "learning_rate": c["learning rate"],
                "gamma": 1 - c["negated gamma"],
            }
        )

        # Agent Selection:
        agent_name = cfg.agent.name
        agent: Optional[Agent] = None
        if agent_name == "SimpleAgent":
            agent = SimpleAgent(cfg, ls_conf)
        elif agent_name == "SoftActorCritic":
            agent = SoftActorCritic(cfg, ls_conf)
        else:
            raise Exception("unknown agent")

        if not DEBUG:
            wandb.init(
                project="checking",
                # job_type=f"Thing {run // 2}",
                config={
                    "landscape": ls_conf,
                    "configuration": cfg,
                },
            )

        for _ in range(5):
            agent.train(steps=1000)
            agent.evaluate()
        wandb.finish()
