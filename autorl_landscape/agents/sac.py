# from: https://github.com/automl/CARL/blob/train/experiments/context_gating/algorithms/sac.pysacpy
import coax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.numpy import DeviceArray
from omegaconf import DictConfig

from autorl_landscape.agents.agent import Agent
from autorl_landscape.util.networks import neural_net

# from ..networks.sac import pi_func, q_func

# from ..utils import dump_func_dict, evaluate, log_wandb


class SoftActorCritic(Agent):
    def __init__(self, cfg: DictConfig, ls_conf: DictConfig) -> None:
        """
        Soft Actor Critic

        Parameters
        ----------
        env : str
            Gym environment name.
        ls_conf : Dict[str, Any]
            Landscape configuration values.
        hp_conf: DictConfig
            Further (static) hyperparameters.
        """
        super().__init__(cfg, ls_conf)

        self.pi = coax.Policy(self._pi_func, self.env, random_seed=cfg.seed)
        # TODO not sure what action_preprocessor does exactly
        self.q1 = coax.Q(
            self._q_func,
            self.env,
            action_preprocessor=self.pi.proba_dist.preprocess_variate,
            random_seed=cfg.seed,
        )
        self.q2 = coax.Q(
            self._q_func,
            self.env,
            action_preprocessor=self.pi.proba_dist.preprocess_variate,
            random_seed=cfg.seed,
        )
        # target network
        self.q1_targ = self.q1.copy()
        self.q2_targ = self.q2.copy()

        # alpha temperature
        self.log_alpha = None

        # experience tracer
        self.tracer = coax.reward_tracing.NStep(n=cfg.agent.hps.n_step, gamma=ls_conf["gamma"], record_extra_info=True)
        self.replay_buffer = coax.experience_replay.SimpleReplayBuffer(
            capacity=cfg.agent.hps.replay_capacity, random_seed=cfg.seed
        )
        self.policy_regularizer = coax.regularizers.NStepEntropyRegularizer(
            self.pi,
            beta=cfg.agent.hps.beta / self.tracer.n,
            gamma=self.tracer.gamma,
            n=[self.tracer.n],
        )

        self.qlearning1 = coax.td_learning.SoftClippedDoubleQLearning(
            self.q1,
            pi_targ_list=[self.pi],
            q_targ_list=[self.q1_targ, self.q2_targ],
            loss_function=coax.value_losses.mse,
            optimizer=optax.adam(ls_conf["learning_rate"]),
            policy_regularizer=self.policy_regularizer,
        )
        self.qlearning2 = coax.td_learning.SoftClippedDoubleQLearning(
            self.q2,
            pi_targ_list=[self.pi],
            q_targ_list=[self.q1_targ, self.q2_targ],
            loss_function=coax.value_losses.mse,
            optimizer=optax.adam(ls_conf["learning_rate"]),
            policy_regularizer=self.policy_regularizer,
        )
        self.soft_pg = coax.policy_objectives.SoftPG(
            self.pi,
            [self.q1_targ, self.q2_targ],
            optimizer=optax.adam(ls_conf["learning_rate"]),
            regularizer=coax.regularizers.NStepEntropyRegularizer(
                self.pi,
                beta=cfg.agent.hps.beta / self.tracer.n,
                gamma=self.tracer.gamma,
                n=jnp.arange(self.tracer.n),
            ),
        )

    def _pi_func(self, S: DeviceArray, is_training: bool) -> DeviceArray:
        """type-2 pi-function: s -> pi(s,.)"""
        num_actions = np.prod(self.env.action_space.shape)
        net = neural_net(
            out_size=2 * num_actions,
            length=self.ls_conf["nn_length"],
            width=self.ls_conf["nn_width"],
        )
        h = net(S)
        mu = h[:, :num_actions]
        logvar = h[:, num_actions:]
        # TODO softplus or clamp post-processing for logvar?
        return {"mu": mu, "logvar": logvar}

    def _q_func(self, S: DeviceArray, A: DeviceArray, is_training: bool) -> DeviceArray:
        """
        Q function network gets the state and action, outputs a single scalar (state-action value)
        """
        net = neural_net(out_size=1, length=self.ls_conf["nn_length"], width=self.ls_conf["nn_width"])
        return net(jnp.concatenate((S, A), axis=-1)).squeeze(-1)

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

                # trace rewards and add transition to replay buffer
                self.tracer.add(s, a, r, done)
                while self.tracer:
                    self.replay_buffer.add(self.tracer.pop())

                # learn
                if len(self.replay_buffer) >= self.cfg.agent.hps.warmup_steps:
                    transition_batch = self.replay_buffer.sample(batch_size=self.cfg.agent.hps.batch_size)

                    metrics = {}

                    # flip a coin to decide which of the q-functions to update
                    qlearning = self.qlearning1 if jax.random.bernoulli(self.q1.rng) else self.qlearning2
                    metrics.update(qlearning.update(transition_batch))

                    # Q update:
                    # A_next, logps = self.pi(transition_batch.S_next, return_logp=True)
                    # Q1_next = self.q1_targ(transition_batch.S_next, A_next)
                    # Q2_next = self.q2_targ(transition_batch.S_next, A_next)
                    # Q_next = jnp.minimum(Q1_next, Q2_next)
                    # V_next = Q_next - self.log_alpha

                    # delayed policy updates
                    if self.env.T >= self.cfg.agent.hps.exploration_steps:  # and env.T % cfg.pi_update_freq == 0:
                        metrics.update(self.soft_pg.update(transition_batch))

                    self.env.record_metrics(metrics)

                    # sync target networks
                    self.q1_targ.soft_update(self.q1, tau=self.cfg.agent.hps.target_smoothing)
                    self.q2_targ.soft_update(self.q2, tau=self.cfg.agent.hps.target_smoothing)

                if done:
                    break

                s = s_next

            # if env.period(name='generate_gif', T_period=cfg.render_freq) and env.T > cfg.q_warmup_num_frames:
            #     T = env.T - env.T % cfg.render_freq  # round
            #     gif_path = f"{os.getcwd()}/gifs/T{T:08d}.gif"
            #     coax.utils.generate_gif(
            #         env=env, policy=pi, filepath=gif_path)
            #     wandb.log({"eval/episode": wandb.Video(
            #         gif_path, caption=str(T), fps=30)}, commit=False)
            # if self.env.period(name="evaluate", T_period=cfg.eval_freq):
            #     path = dump_func_dict(locals())
            #     average_returns = evaluate(pi, eval_env, cfg.eval_episodes)
            #     self.log(
            #         {
            #             "eval/return_hist": wandb.Histogram(average_returns),
            #             "eval/return": np.mean(average_returns),
            #         },
            #         commit=False,
            #     )
            # log_wandb(env)
        # average_returns = evaluate(pi, eval_env, cfg.eval_episodes)
        # return np.mean(average_returns)
