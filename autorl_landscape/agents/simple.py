import coax
from coax.value_losses import mse
from jax.numpy import DeviceArray
from omegaconf import DictConfig
from optax import adam

from autorl_landscape.agents.agent import Agent
from autorl_landscape.util.networks import neural_net


class SimpleAgent(Agent):
    def __init__(self, cfg: DictConfig, ls_conf: DictConfig) -> None:
        super().__init__(cfg, ls_conf)

        # value function and its derived policy
        self.q = coax.Q(self._q_func, self.env, random_seed=cfg.seed)
        self.pi = coax.BoltzmannPolicy(self.q, temperature=0.1)

        # target network
        self.q_target = self.q.copy()

        # experience tracer
        self.tracer = coax.reward_tracing.NStep(n=1, gamma=ls_conf["gamma"])
        self.buffer = coax.experience_replay.SimpleReplayBuffer(capacity=100000)

        # updater
        self.qlearning = coax.td_learning.QLearning(
            self.q,
            q_targ=self.q_target,
            loss_function=mse,
            optimizer=adam(ls_conf["learning_rate"]),
        )

    def _q_func(self, S: DeviceArray, is_training: bool) -> DeviceArray:
        """type-2 q-function: s -> q(s,.)"""
        net = neural_net(
            out_size=self.env.action_space.n, length=self.ls_conf["nn_length"], width=self.ls_conf["nn_width"]
        )
        return net(S)

    def train(self, steps: int) -> None:
        while self.env.T < steps:
            # an episode
            s = self.env.reset()

            for t in range(self.env.spec.max_episode_steps):
                # a step
                if self.env.T >= steps:
                    break

                a = self.pi(s)
                s_next, r, done, _ = self.env.step(a)

                # extend last reward as asymptotic best-case return (as if it would continue for ever)
                if t == self.env.spec.max_episode_steps - 1:
                    assert done
                    # gamma + gamma^2 + gamma^3 + ... = 1 / (1 - gamma)
                    r = 1 / (1 - self.tracer.gamma)

                # trace rewards and add transition to replay buffer
                self.tracer.add(s, a, r, done)
                while self.tracer:
                    self.buffer.add(self.tracer.pop())

                # learn
                if len(self.buffer) >= 100:
                    transition_batch = self.buffer.sample(batch_size=32)
                    metrics = self.qlearning.update(transition_batch)
                    self.log(metrics)

                    # self.env.record_metrics(metrics)

                # sync target network
                self.q_target.soft_update(self.q, tau=0.01)

                if done:
                    break

                s = s_next
