"""ACPO continuous state space algorithm."""

import time
from enum import Enum
from typing import Callable
import distrax
import jax
from jax.nn import initializers
from jax import numpy as jnp
from flax import struct, nnx
from gymnax.environments.environment import Environment, EnvParams, EnvState
import optax
from safenax import EcoAntV2, FragileAnt
from safenax.wrappers import BraxToGymnaxWrapper, LogWrapper
from tinker import log, norm


class ModelObjective(Enum):
    REWARD = "reward"
    COST = "cost"
    BOTH = "both"


class StateModel(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        embedding_dim: int = 64,
        rnn_hidden_dim: int = 128,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        self.rnn_hidden_dim = rnn_hidden_dim

        hidden_init = initializers.orthogonal(jnp.sqrt(2.0))
        output_init = initializers.orthogonal(1.0)
        zero_bias = initializers.constant(0.0)

        self.gru_cell = nnx.GRUCell(
            in_features=obs_dim + action_dim, hidden_features=rnn_hidden_dim, rngs=rngs
        )

        self.encoder = nnx.Sequential(
            nnx.Linear(
                rnn_hidden_dim + action_dim,
                256,
                kernel_init=hidden_init,
                bias_init=zero_bias,
                rngs=rngs,
            ),
            nnx.LayerNorm(256, rngs=rngs),
            nnx.silu,
            nnx.Linear(
                256,
                obs_dim * embedding_dim,
                kernel_init=hidden_init,
                bias_init=zero_bias,
                rngs=rngs,
            ),
        )

        self.cosine_net = nnx.Sequential(
            nnx.Linear(
                embedding_dim,
                embedding_dim,
                kernel_init=hidden_init,
                bias_init=zero_bias,
                rngs=rngs,
            ),
            nnx.relu,
        )

        self.decoder = nnx.Sequential(
            nnx.Linear(
                embedding_dim,
                256,
                kernel_init=hidden_init,
                bias_init=zero_bias,
                rngs=rngs,
            ),
            nnx.silu,
            nnx.Linear(256, 1, kernel_init=output_init, bias_init=zero_bias, rngs=rngs),
        )

    def get_h(self, h: jax.Array, obs: jax.Array, action: jax.Array) -> jax.Array:
        gru_input = jnp.concatenate([obs, action], axis=-1)
        h_next, _ = self.gru_cell(h, gru_input)
        return h_next

    def __call__(
        self, h: jax.Array, obs: jax.Array, action: jax.Array, tau: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        batch_shape = obs.shape[:-1]

        gru_input = jnp.concatenate([obs, action], axis=-1)
        h_next, _ = self.gru_cell(h, gru_input)

        enc_input = jnp.concatenate([h_next, action], axis=-1)
        psi = self.encoder(enc_input)
        psi = psi.reshape((*batch_shape, self.obs_dim, self.embedding_dim))

        tau_expanded = tau[..., None]
        i = jnp.arange(self.embedding_dim)[None, None, :]
        cosine_input = jnp.cos(jnp.pi * i * tau_expanded)
        phi = self.cosine_net(cosine_input)

        h_modulated = psi * phi
        out = self.decoder(h_modulated)
        next_obs_quantiles = out.squeeze(-1)

        return h_next, next_obs_quantiles


class Actor(nnx.Module):
    """Actor network for continuous action spaces."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        activation: Callable = jax.nn.tanh,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.activation = activation

        hidden_init = initializers.orthogonal(jnp.sqrt(2.0))
        output_init = initializers.orthogonal(1.0)
        zero_bias = initializers.constant(0.0)

        self.fc1 = nnx.Linear(
            obs_dim, 256, kernel_init=hidden_init, bias_init=zero_bias, rngs=rngs
        )
        self.fc2 = nnx.Linear(
            256, 256, kernel_init=hidden_init, bias_init=zero_bias, rngs=rngs
        )
        self.mean_out = nnx.Linear(
            256, action_dim, kernel_init=output_init, bias_init=zero_bias, rngs=rngs
        )
        self.log_std = nnx.Param(jnp.zeros(action_dim))

    def __call__(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        mean = self.mean_out(x)
        pi = distrax.MultivariateNormalDiag(mean, jnp.exp(self.log_std[...]))
        return pi


class Critic(nnx.Module):
    """Critic network with separate value and cost-value heads."""

    def __init__(
        self,
        obs_dim: int,
        activation: Callable = jax.nn.tanh,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        self.activation = activation

        hidden_init = initializers.orthogonal(jnp.sqrt(2.0))
        output_init = initializers.orthogonal(1.0)
        zero_bias = initializers.constant(0.0)

        # Value critic (for rewards)
        self.value_fc1 = nnx.Linear(
            obs_dim, 256, kernel_init=hidden_init, bias_init=zero_bias, rngs=rngs
        )
        self.value_fc2 = nnx.Linear(
            256, 256, kernel_init=hidden_init, bias_init=zero_bias, rngs=rngs
        )
        self.value_out = nnx.Linear(
            256, 1, kernel_init=output_init, bias_init=zero_bias, rngs=rngs
        )

        # Cost-value critic (for costs)
        self.cost_value_fc1 = nnx.Linear(
            obs_dim, 256, kernel_init=hidden_init, bias_init=zero_bias, rngs=rngs
        )
        self.cost_value_fc2 = nnx.Linear(
            256, 256, kernel_init=hidden_init, bias_init=zero_bias, rngs=rngs
        )
        self.cost_value_out = nnx.Linear(
            256, 1, kernel_init=output_init, bias_init=zero_bias, rngs=rngs
        )

    def __call__(self, x):
        # Value critic
        value_x = self.activation(self.value_fc1(x))
        value_x = self.activation(self.value_fc2(value_x))
        value = jnp.squeeze(self.value_out(value_x), axis=-1)

        # Cost-value critic
        cost_value_x = self.activation(self.cost_value_fc1(x))
        cost_value_x = self.activation(self.cost_value_fc2(cost_value_x))
        cost_value = jnp.squeeze(self.cost_value_out(cost_value_x), axis=-1)

        return value, cost_value


@struct.dataclass
class Transition:
    """Transition tuple including cost information."""

    done: jax.Array
    action: jax.Array
    value: jax.Array
    cost_value: jax.Array
    reward: jax.Array
    cost: jax.Array
    running_cost: jax.Array
    cost_discount: jax.Array
    log_prob: jax.Array
    obs: jax.Array
    norm_obs: jax.Array
    norm_next_obs: jax.Array
    next_obs: jax.Array
    h: jax.Array
    info: dict


@struct.dataclass
class TrainState:
    """Extended state for CPO including constraint tracking."""

    actor_params: nnx.State
    critic_params: nnx.State
    state_model_params: nnx.State
    critic_opt_params: optax.OptState
    state_opt_params: optax.OptState
    margin: float


@struct.dataclass
class RunnerState:
    """State for the environment runner."""

    train_state: TrainState
    env_state: EnvState
    obs_norm_state: norm.RunningMeanStdState
    obs: jax.Array
    norm_obs: jax.Array
    running_cost: jax.Array
    cost_discount: jax.Array
    h: jax.Array
    rng: jax.Array


@struct.dataclass
class DynamicConfig:
    """Holds dynamic configuration parameters for CPO training.

    :param rng: Random number generator key.
    :param env_params: Environment parameters.
    :param cost_limit: Maximum expected cost.
    :param critic_lr: Critic learning rate.
    :param state_model_lr: State model learning rate.
    :param gae_gamma: Discount factor for GAE.
    :param gae_lambda: Lambda parameter for GAE.
    :param max_grad_norm: Maximum gradient norm for clipping.
    :param target_kl: Target KL divergence for policy updates.
    :param entropy_coeff: Coefficient for entropy regularization.
    :param backtrack_coeff: Backtracking line search coefficient.
    :param backtrack_iters: Maximum backtracking line search iterations.
    :param damping_coeff: Damping coefficient for Hessian-vector product.
    :param margin_lr: Learning rate for constraint margin updates.
    :param adam_eps: Epsilon parameter for Adam optimizer.
    """

    rng: jax.Array
    env_params: EnvParams
    cost_limit: float
    critic_lr: float = 3e-4
    state_model_lr: float = 3e-4
    gae_gamma: float = 0.99
    cost_gamma: float = 0.999
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    target_kl: float = 0.01
    entropy_coeff: float = 0.0
    backtrack_coeff: float = 0.8
    backtrack_iters: int = 10
    damping_coeff: float = 0.1
    margin_lr: float = 0.05
    adam_eps: float = 1e-5


def cosine_similarity(vec1: jax.Array, vec2: jax.Array) -> jax.Array:
    """Compute cosine similarity between two vectors."""
    dot_prod = jnp.dot(vec1, vec2)
    norm_vec1 = jnp.linalg.norm(vec1) + 1e-8
    norm_vec2 = jnp.linalg.norm(vec2) + 1e-8
    cos_sim = dot_prod / (norm_vec1 * norm_vec2)
    return jnp.clip(cos_sim, -1.0, 1.0)


def hvp(f: Callable, primals: tuple, tangents: tuple) -> jax.Array:
    """Hessian-vector product using JAX autodiff."""
    return jax.jvp(jax.grad(f), primals, tangents)[1]


def cg_solve(
    hvp_fn: Callable,
    b: jax.Array,
    max_iter: int = 10,
    residual_tol: float = 1e-10,
) -> jax.Array:
    """Conjugate gradient solver for Hx = b."""

    def cg_body(state):
        i, x, r, p, rdotr = state
        Ap = hvp_fn(p)
        alpha = rdotr / (jnp.dot(p, Ap) + 1e-8)
        x = x + alpha * p
        r = r - alpha * Ap
        new_rdotr = jnp.dot(r, r)
        beta = new_rdotr / (rdotr + 1e-8)
        p = r + beta * p
        return i + 1, x, r, p, new_rdotr

    def cg_cond(state):
        i, x, r, p, rdotr = state
        return (i < max_iter) & (rdotr >= residual_tol)

    x = jnp.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rdotr = jnp.dot(r, r)

    _, x, _, _, _ = jax.lax.while_loop(cg_cond, cg_body, (0, x, r, p, rdotr))
    return x


def compute_cpo_step(
    g: jax.Array,
    b: jax.Array,
    c: jax.Array,  # scalar array
    hvp_fn: Callable,
    target_kl: float,
    use_constraint: bool,
    damping_coeff: float,
) -> tuple[jax.Array, int]:
    """
    Compute CPO step direction using constrained optimization.

    Returns:
        direction: The search direction
        optim_case: Integer indicating which optimization case was used
    """
    # Add damping to Hessian-vector product
    damped_hvp = lambda v: hvp_fn(v) + damping_coeff * v

    # Solve Hv = g using conjugate gradient
    v = cg_solve(damped_hvp, g, max_iter=10)
    approx_g = damped_hvp(v)
    q = jnp.dot(v, approx_g)

    # Check if we should use TRPO (unconstrained) case
    use_trpo = (jnp.dot(b, b) <= 1e-8) & (c < 0.0)

    def trpo_case():
        """TRPO case: no constraint."""
        optim_case = 4
        lam = jnp.sqrt(q / (2.0 * target_kl))
        direction = v / (lam + 1e-8)
        return direction, optim_case

    def cpo_case():
        """CPO case: solve with constraint."""
        # Solve for constraint direction
        w = cg_solve(damped_hvp, b, max_iter=10)
        r = jnp.dot(w, approx_g)
        s = jnp.dot(w, damped_hvp(w))

        A = q - r**2 / (s + 1e-8)
        B = 2.0 * target_kl - c**2 / (s + 1e-8)

        # Determine optimization case using jnp.where for branch-free logic
        # Case 0: c >= 0 and B < 0 (recovery)
        # Case 1: c >= 0 and B >= 0 (feasible)
        # Case 2: c < 0 and B >= 0 (feasible)
        # Case 3: c < 0 and B < 0 (feasible, ignore constraint)
        optim_case = jnp.where(
            c < 0.0, jnp.where(B < 0.0, 3, 2), jnp.where(B >= 0.0, 1, 0)
        )

        # Recovery case (optim_case == 0)
        nu_recovery = jnp.sqrt(2.0 * target_kl / (s + 1e-8))
        direction_recovery = nu_recovery * w

        # Feasible cases (optim_case > 0)
        # Case 3 or 4: ignore constraint
        lam_unconstrained = jnp.sqrt(q / (2.0 * target_kl))
        nu_unconstrained = 0.0

        # Case 1 or 2: solve for optimal lam, nu
        # Compute projection bounds
        LA_0, LA_1 = (
            jnp.where(c < 0.0, 0.0, r / (c + 1e-8)),
            jnp.where(c < 0.0, r / (c - 1e-8), jnp.inf),
        )
        LB_0, LB_1 = (
            jnp.where(c < 0.0, r / (c - 1e-8), 0.0),
            jnp.where(c < 0.0, jnp.inf, r / (c + 1e-8)),
        )

        proj = lambda x, L0, L1: jnp.maximum(L0, jnp.minimum(L1, x))
        lam_a = proj(jnp.sqrt(A / (B + 1e-8)), LA_0, LA_1)
        lam_b = proj(jnp.sqrt(q / (2.0 * target_kl)), LB_0, LB_1)

        f_a = -0.5 * (A / (lam_a + 1e-8) + B * lam_a) - r * c / (s + 1e-8)
        f_b = -0.5 * (q / (lam_b + 1e-8) + 2.0 * target_kl * lam_b)

        lam_constrained = jnp.where(f_a >= f_b, lam_a, lam_b)
        nu_constrained = jnp.maximum(0.0, lam_constrained * c - r) / (s + 1e-8)

        # Select between constrained and unconstrained based on optim_case
        lam = jnp.where(optim_case > 2, lam_unconstrained, lam_constrained)
        nu = jnp.where(optim_case > 2, nu_unconstrained, nu_constrained)

        direction_feasible = (v + nu * w) / (lam + 1e-8)

        # Select between recovery and feasible
        direction = jnp.where(optim_case == 0, direction_recovery, direction_feasible)

        return direction, optim_case

    # Use jax.lax.cond for top-level branching
    return jax.lax.cond(
        (not use_constraint) | use_trpo,
        lambda _: trpo_case(),
        lambda _: cpo_case(),
        None,
    )


def calculate_gae(
    traj_batch: Transition,
    last_val: jax.Array,
    gamma: float,
    lambda_: float,
) -> tuple[jax.Array, jax.Array]:
    def _get_advantages(carry, x):
        gae, next_value = carry
        done, value, reward = x
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * lambda_ * (1 - done) * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        (traj_batch.done, traj_batch.value, traj_batch.reward),
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + traj_batch.value


def make_train(
    env: Environment,
    num_steps: int,
    num_envs: int,
    train_freq: int,
    critic_epochs: int = 80,
    state_model_epochs: int = 80,
    num_taus: int = 8,
    embedding_dim: int = 64,
    rnn_hidden_dim: int = 128,
    activation: Callable = jax.nn.tanh,
    anneal_critic_lr: bool = True,
    anneal_state_lr: bool = True,
    model_objective: ModelObjective = ModelObjective.REWARD,
    use_constraint: bool = True,
):
    num_updates = num_steps // train_freq
    env = LogWrapper(env)
    use_model_policy = model_objective in (ModelObjective.REWARD, ModelObjective.BOTH)
    use_model_cost = model_objective in (ModelObjective.COST, ModelObjective.BOTH)

    def train(config: DynamicConfig) -> tuple[TrainState, dict]:
        obs_dim = env.observation_space(config.env_params).shape[0]
        action_dim = env.action_space(config.env_params).shape[0]
        rng, actor_rng, critic_rng, state_model_rng = jax.random.split(config.rng, 4)

        _actor = Actor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            activation=activation,
            rngs=nnx.Rngs(actor_rng),
        )
        actor_graph, _actor_params = nnx.split(_actor)

        _critic = Critic(
            obs_dim=obs_dim,
            activation=activation,
            rngs=nnx.Rngs(critic_rng),
        )
        critic_graph, _critic_params = nnx.split(_critic)

        _state_model = StateModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            embedding_dim=embedding_dim,
            rnn_hidden_dim=rnn_hidden_dim,
            rngs=nnx.Rngs(state_model_rng),
        )
        state_graph, _state_model_params = nnx.split(_state_model)

        if anneal_critic_lr:
            schedule = optax.linear_schedule(
                init_value=config.critic_lr,
                end_value=0.0,
                transition_steps=num_updates * critic_epochs,
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adam(learning_rate=schedule, eps=config.adam_eps),
            )
        else:
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adam(config.critic_lr, eps=config.adam_eps),
            )
        _critic_opt_params = critic_tx.init(_critic_params)

        if anneal_state_lr:
            schedule = optax.linear_schedule(
                init_value=config.state_model_lr,
                end_value=0.0,
                transition_steps=num_updates * state_model_epochs,
            )
            state_model_tx = optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adam(learning_rate=schedule, eps=config.adam_eps),
            )
        else:
            state_model_tx = optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adam(config.state_model_lr, eps=config.adam_eps),
            )
        _state_opt_params = state_model_tx.init(_state_model_params)

        train_state = TrainState(
            actor_params=_actor_params,
            critic_params=_critic_params,
            state_model_params=_state_model_params,
            critic_opt_params=_critic_opt_params,
            state_opt_params=_state_opt_params,
            margin=0.0,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, num_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(
            reset_rng, config.env_params
        )
        obs_norm_state = norm.init(env.observation_space(config.env_params).shape)
        obs_norm_state = norm.welford_update(obs_norm_state, obsv)
        norm_obsv = jax.vmap(norm.normalize, in_axes=(None, 0))(obs_norm_state, obsv)
        running_cost = jnp.zeros((num_envs,))
        cost_discount = jnp.ones((num_envs,))
        h = jnp.zeros((num_envs, rnn_hidden_dim))

        runner_state = RunnerState(
            train_state=train_state,
            env_state=env_state,
            obs_norm_state=obs_norm_state,
            norm_obs=norm_obsv,
            obs=obsv,
            running_cost=running_cost,
            cost_discount=cost_discount,
            h=h,
            rng=rng,
        )

        # TRAIN LOOP
        def _update_step(
            runner_state: RunnerState, update_idx: int
        ) -> tuple[RunnerState, dict]:
            actor_params = runner_state.train_state.actor_params
            actor: Actor = nnx.merge(actor_graph, actor_params)
            critic_params = runner_state.train_state.critic_params
            critic: Critic = nnx.merge(critic_graph, critic_params)
            state_model_params = runner_state.train_state.state_model_params
            state_model: StateModel = nnx.merge(state_graph, state_model_params)

            def _env_step(
                runner_state: RunnerState, _
            ) -> tuple[RunnerState, Transition]:
                train_state = runner_state.train_state
                env_state = runner_state.env_state
                obs_norm_state = runner_state.obs_norm_state
                obs = runner_state.obs
                norm_obs = runner_state.norm_obs
                running_cost = runner_state.running_cost
                cost_discount = runner_state.cost_discount
                h = runner_state.h
                rng = runner_state.rng

                # SELECT ACTION
                rng, action_rng = jax.random.split(rng)
                pi = actor(norm_obs)
                value, cost_value = critic(norm_obs)
                action = pi.sample(seed=action_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, step_rng = jax.random.split(rng)
                rng_step = jax.random.split(step_rng, num_envs)
                next_obs, next_env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, config.env_params)
                next_norm_obs = jax.vmap(norm.normalize, in_axes=(None, 0))(
                    obs_norm_state, next_obs
                )

                # Extract cost from info
                # For environments without explicit costs, cost = 0
                cost = info.get("cost", jnp.zeros_like(reward))

                transition = Transition(
                    done,
                    action,
                    value,
                    cost_value,
                    reward,
                    cost,
                    running_cost,
                    cost_discount,
                    log_prob,
                    obs,
                    norm_obs,
                    next_norm_obs,
                    next_obs,
                    h,
                    info,
                )

                h_next = state_model.get_h(h, norm_obs, action)
                h_next = jnp.where(done[:, None], jnp.zeros_like(h_next), h_next)
                next_running_cost = jax.lax.select(
                    done,
                    jnp.zeros_like(running_cost),
                    running_cost + cost_discount * cost,
                )
                next_cost_discount = jax.lax.select(
                    done,
                    jnp.ones_like(cost_discount),
                    cost_discount * config.cost_gamma,
                )

                runner_state = RunnerState(
                    train_state=train_state,
                    env_state=next_env_state,
                    obs_norm_state=obs_norm_state,
                    obs=next_obs,
                    norm_obs=next_norm_obs,
                    running_cost=next_running_cost,
                    cost_discount=next_cost_discount,
                    h=h_next,
                    rng=rng,
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, length=train_freq
            )
            train_state = runner_state.train_state
            rng = runner_state.rng
            obs_norm_state = runner_state.obs_norm_state

            # CALCULATE ADVANTAGE AND COST ADVANTAGE
            last_val, last_cost_val = critic(runner_state.norm_obs)

            # Flatten batch: (train_freq, num_envs, obs_dim) -> (train_freq * num_envs, obs_dim)
            batch_obs = traj_batch.next_obs.reshape(-1, *traj_batch.next_obs.shape[2:])
            obs_norm_state = norm.welford_update(obs_norm_state, batch_obs)

            # Reward advantages
            advantages, return_targets = calculate_gae(
                traj_batch, last_val, config.gae_gamma, config.gae_lambda
            )

            # Cost advantages
            cost_advantages, cost_targets = calculate_gae(
                traj_batch.replace(reward=traj_batch.cost, value=traj_batch.cost_value),
                last_cost_val,
                config.cost_gamma,
                config.gae_lambda,
            )

            # Normalize reward advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            # Center but do not rescale cost advantages
            cost_advantages = cost_advantages - cost_advantages.mean()

            # Get episode cost statistics
            is_terminal = traj_batch.done
            num_episodes = jnp.maximum(is_terminal.sum(), 1.0)
            terminal_costs = (
                traj_batch.running_cost + traj_batch.cost_discount * traj_batch.cost
            )
            sparse_costs = jnp.where(
                is_terminal, terminal_costs, 0.0
            )  # (train_freq, num_envs)

            traj_cost_return = sparse_costs.sum() / num_episodes
            td_cost_return = (traj_batch.running_cost + cost_targets).mean()

            # Compute constraint violation
            c_raw = td_cost_return - config.cost_limit
            new_margin = jnp.maximum(0.0, train_state.margin + config.margin_lr * c_raw)
            c = c_raw + new_margin
            # c = c / (train_freq + 1e-8)

            # UPDATE POLICY
            pi_old = actor(traj_batch.norm_obs)
            flat_params, unravel_fn = jax.flatten_util.ravel_pytree(actor_params)

            rng, tau_rng = jax.random.split(rng)
            tau = jax.random.uniform(tau_rng, traj_batch.norm_obs.shape)

            action_noise = (traj_batch.action - pi_old.loc) / (pi_old.scale_diag + 1e-8)

            _, pred_next_obs = state_model(
                traj_batch.h, traj_batch.norm_obs, traj_batch.action, tau
            )
            state_noise = traj_batch.norm_next_obs - pred_next_obs

            def policy_loss_model_based(params) -> jax.Array:
                """Compute surrogate policy loss using Model-Based Pathwise gradients (SVG)."""
                model: Actor = nnx.merge(actor_graph, params)
                pi = model(traj_batch.norm_obs)

                # A. Importance sampling weight (STRICTLY STOPPED GRADIENT)
                log_prob = pi.log_prob(traj_batch.action)
                ratio = jax.lax.stop_gradient(jnp.exp(log_prob - traj_batch.log_prob))

                # B. Continuous Action Reparameterization (Gaussian Trick)
                soft_action = pi.loc + pi.scale_diag * jax.lax.stop_gradient(
                    action_noise
                )

                # C. Continuous State Reparameterization (Applying inferred state noise)
                _, pred_next_obs_soft = state_model(
                    traj_batch.h, traj_batch.norm_obs, soft_action, tau
                )
                soft_next_norm_obs = pred_next_obs_soft + jax.lax.stop_gradient(
                    state_noise
                )

                # D. Pathwise Target Evaluation
                next_value, _ = critic(soft_next_norm_obs)

                # E. Pathwise Reward Evaluation
                mean = obs_norm_state.mean
                std = jnp.sqrt(obs_norm_state.var + 1e-8)
                soft_next_obs = (soft_next_norm_obs * std) + mean
                soft_reward = env.reward_fn(
                    traj_batch.obs, soft_action, soft_next_obs, config.env_params
                )

                # The fully differentiable continuous pathway
                value_path = soft_reward + config.gae_gamma * next_value * (
                    1.0 - traj_batch.done
                )

                # F. Final Objective
                policy_loss = -(ratio * value_path).mean() - (
                    config.entropy_coeff * pi.entropy().mean()
                )

                return policy_loss

            def policy_loss_model_free(params):
                """Compute surrogate model-free policy loss."""
                model: Actor = nnx.merge(actor_graph, params)
                pi = model(traj_batch.norm_obs)
                log_prob = pi.log_prob(traj_batch.action)
                ratio = jnp.exp(log_prob - traj_batch.log_prob)

                policy_loss = -(ratio * advantages).mean() - (
                    config.entropy_coeff * pi.entropy().mean()
                )

                return policy_loss

            def cost_loss_model_based(params) -> jax.Array:
                """Compute surrogate cost constraint loss using Model-Based Pathwise gradients (SVG)."""
                model: Actor = nnx.merge(actor_graph, params)
                pi = model(traj_batch.norm_obs)

                # A. Importance sampling weight (STRICTLY STOPPED GRADIENT)
                log_prob = pi.log_prob(traj_batch.action)
                ratio = jax.lax.stop_gradient(jnp.exp(log_prob - traj_batch.log_prob))

                # B. Continuous Action Reparameterization (Gaussian Trick)
                soft_action = pi.loc + pi.scale_diag * jax.lax.stop_gradient(
                    action_noise
                )

                # C. Continuous State Reparameterization (Applying inferred state noise)
                _, pred_next_obs_soft = state_model(
                    traj_batch.h, traj_batch.norm_obs, soft_action, tau
                )
                soft_next_norm_obs = pred_next_obs_soft + jax.lax.stop_gradient(
                    state_noise
                )

                # D. Pathwise Target Evaluation
                _, next_value = critic(soft_next_norm_obs)

                # E. Pathwise Reward Evaluation
                mean = obs_norm_state.mean
                std = jnp.sqrt(obs_norm_state.var + 1e-8)
                soft_next_obs = (soft_next_norm_obs * std) + mean
                soft_cost = env.cost_fn(
                    traj_batch.obs, soft_action, soft_next_obs, config.env_params
                )

                # The fully differentiable continuous pathway
                cost_value_path = soft_cost + config.cost_gamma * next_value * (
                    1.0 - traj_batch.done
                )

                # F. Final Objective
                cost_loss = (ratio * cost_value_path).mean()

                return cost_loss

            def cost_loss_model_free(params) -> jax.Array:
                """Compute surrogate model-free cost constraint loss."""
                model: Actor = nnx.merge(actor_graph, params)
                pi = model(traj_batch.norm_obs)
                log_prob = pi.log_prob(traj_batch.action)
                ratio = jnp.exp(log_prob - traj_batch.log_prob)

                cost_loss = (ratio * cost_advantages).mean()

                return cost_loss

            if use_model_policy:
                policy_loss_fn = policy_loss_model_based
                other_policy_loss_fn = policy_loss_model_free
            else:
                policy_loss_fn = policy_loss_model_free
                other_policy_loss_fn = policy_loss_model_based
            if use_model_cost:
                cost_loss_fn = cost_loss_model_based
                other_cost_loss_fn = cost_loss_model_free
            else:
                cost_loss_fn = cost_loss_model_free
                other_cost_loss_fn = cost_loss_model_based

            # Get gradients of policy and cost losses
            old_policy_loss, g_tree = jax.value_and_grad(policy_loss_fn)(actor_params)
            old_cost_loss, b_tree = jax.value_and_grad(cost_loss_fn)(actor_params)
            _, other_g_tree = jax.value_and_grad(other_policy_loss_fn)(actor_params)
            _, other_b_tree = jax.value_and_grad(other_cost_loss_fn)(actor_params)

            # Flatten gradients
            g, _ = jax.flatten_util.ravel_pytree(g_tree)
            b, _ = jax.flatten_util.ravel_pytree(b_tree)
            other_g, _ = jax.flatten_util.ravel_pytree(other_g_tree)
            other_b, _ = jax.flatten_util.ravel_pytree(other_b_tree)

            # Measure difference between model-based and model-free gradients
            policy_grad_cos_sim = cosine_similarity(g, other_g)
            cost_grad_cos_sim = cosine_similarity(b, other_b)
            g_norm = jnp.linalg.norm(g)
            other_g_norm = jnp.linalg.norm(other_g)
            b_norm = jnp.linalg.norm(b)
            other_b_norm = jnp.linalg.norm(other_b)
            policy_grad_norm_ratio = g_norm / (other_g_norm + 1e-8)
            cost_grad_norm_ratio = b_norm / (other_b_norm + 1e-8)

            def kl_fn(new_params_flat):
                """Compute KL divergence between old and new policy."""
                params_unflat = unravel_fn(new_params_flat)
                model_new: Actor = nnx.merge(actor_graph, params_unflat)
                pi_new = model_new(traj_batch.norm_obs)
                return pi_new.kl_divergence(pi_old).mean()

            hvp_fn = lambda v: hvp(kl_fn, (flat_params,), (v,))

            # Compute CPO step direction
            direction, optim_case = compute_cpo_step(
                g, b, c, hvp_fn, config.target_kl, use_constraint, config.damping_coeff
            )

            # Backtracking line search
            def line_search_body(search_state):
                i, current_params, accepted = search_state

                step_size = config.backtrack_coeff**i
                new_flat_params = flat_params - step_size * direction
                new_params = unravel_fn(new_flat_params)

                new_policy_loss = policy_loss_fn(new_params)
                new_cost_loss = cost_loss_fn(new_params)

                kl = kl_fn(new_flat_params)

                # Check acceptance criteria
                loss_improve = (optim_case <= 1) | (new_policy_loss <= old_policy_loss)
                cost_improve = (
                    (new_cost_loss - old_cost_loss <= jnp.maximum(-c, 0))
                    if use_constraint
                    else True
                )
                kl_ok = kl <= config.target_kl

                accept = loss_improve & cost_improve & kl_ok

                updated_params = jax.tree.map(
                    lambda new, old: jnp.where(accept, new, old),
                    new_params,
                    current_params,
                )

                return i + 1, updated_params, accept

            def line_search_cond(search_state):
                i, _, accepted = search_state
                return (i < config.backtrack_iters) & (~accepted)

            # Line search (start with critic-updated params)
            _, final_params, accepted = jax.lax.while_loop(
                line_search_cond,
                line_search_body,
                (0, actor_params, False),
            )
            final_params_flat, _ = jax.flatten_util.ravel_pytree(final_params)
            final_kl = kl_fn(final_params_flat)

            train_state = train_state.replace(
                actor_params=final_params,
                margin=new_margin,
            )

            # UPDATE VALUE CRITICS
            def critic_loss_fn(
                params: nnx.State,
            ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
                """Compute critic loss (for both value and cost-value heads)."""
                critic = nnx.merge(critic_graph, params)
                value, cost_value = critic(traj_batch.norm_obs)
                value_loss = ((value - return_targets) ** 2).mean()
                cost_value_loss = ((cost_value - cost_targets) ** 2).mean()
                total_loss = value_loss + cost_value_loss
                return total_loss, (value_loss, cost_value_loss)

            def _update_critic(train_state: TrainState, _):
                """Update value and cost critic parameters."""
                params = train_state.critic_params

                (_, (value_loss, cost_value_loss)), grads = jax.value_and_grad(
                    critic_loss_fn, has_aux=True
                )(params)

                updates, new_opt_state = critic_tx.update(
                    grads, train_state.critic_opt_params, params
                )
                new_params = optax.apply_updates(params, updates)

                train_state = train_state.replace(
                    critic_params=new_params, critic_opt_params=new_opt_state
                )
                return train_state, (value_loss, cost_value_loss)

            train_state, (value_losses, cost_value_losses) = jax.lax.scan(
                _update_critic, train_state, None, critic_epochs
            )

            # UPDATE STATE MODEL
            def state_loss_fn(params: nnx.State, tau: jax.Array) -> jax.Array:
                """Pinball loss averaged over multiple tau samples."""
                state_model = nnx.merge(state_graph, params)

                # 1. Vectorize the model over the tau dimension (axis 0)
                # in_axes: (h: None, obs: None, action: None, tau: 0)
                batched_forward = jax.vmap(state_model, in_axes=(None, None, None, 0))

                # pred_norm_next_obs shape: (num_taus, *batch_shape, obs_dim)
                _, pred_norm_next_obs = batched_forward(
                    traj_batch.h, traj_batch.norm_obs, traj_batch.action, tau
                )

                # 2. Broadcast the target to match the new num_taus dimension
                # target shape becomes: (1, *batch_shape, obs_dim)
                target = traj_batch.norm_next_obs[None, ...]

                # delta shape: (num_taus, *batch_shape, obs_dim)
                delta = target - pred_norm_next_obs
                indicator = (delta < 0.0).astype(delta.dtype)

                # 3. Calculate pinball loss
                pinball_loss = delta * (tau - indicator)

                # 4. Sum over independent features (axis=-1)
                # Then take the mean over both the batch AND the num_taus dimensions
                loss = pinball_loss.sum(axis=-1).mean()

                return loss

            def _update_state_model(train_state: TrainState, tau: jax.Array):
                params = train_state.state_model_params
                loss, grads = jax.value_and_grad(state_loss_fn)(params, tau)
                updates, new_opt_state = state_model_tx.update(
                    grads, train_state.state_opt_params, params
                )
                new_params = optax.apply_updates(params, updates)

                train_state = train_state.replace(
                    state_model_params=new_params, state_opt_params=new_opt_state
                )
                return train_state, loss

            rng, tau_rng = jax.random.split(rng)
            tau_shape = (state_model_epochs, num_taus, *traj_batch.norm_obs.shape)
            tau_epochs = jax.random.uniform(tau_rng, tau_shape)
            train_state, state_model_losses = jax.lax.scan(
                _update_state_model, train_state, xs=tau_epochs
            )

            sparse_battery_used = jnp.where(
                is_terminal, 100 - traj_batch.info["battery"], 0
            )
            episode_battery_used = sparse_battery_used.sum() / num_episodes
            metrics = {
                "kl": final_kl,
                "constraint_violation": c,
                "margin": new_margin,
                "traj_cost_return": traj_cost_return,
                "td_cost_return": td_cost_return,
                "optim_case": optim_case,
                "episode_return": traj_batch.info["returned_episode_returns"].mean(),
                "episode_cost_return": traj_batch.info[
                    "returned_episode_cost_returns"
                ].mean(),
                "episode_length": traj_batch.info["returned_episode_lengths"].mean(),
                "episode_budget_used": episode_battery_used,
                "accepted": accepted,
                "value_loss": value_losses.mean(),
                "cost_value_loss": cost_value_losses.mean(),
                "state_model_loss": state_model_losses.mean(),
                "policy_grad_cos_sim": policy_grad_cos_sim,
                "cost_grad_cos_sim": cost_grad_cos_sim,
                "policy_grad_norm_ratio": policy_grad_norm_ratio,
                "cost_grad_norm_ratio": cost_grad_norm_ratio,
            }
            runner_state = runner_state.replace(
                train_state=train_state, obs_norm_state=obs_norm_state, rng=rng
            )

            return runner_state, metrics

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, xs=jnp.arange(num_updates)
        )
        return runner_state.train_state, metrics

    return train


if __name__ == "__main__":
    SEED = 0
    NUM_RUNS = 1

    brax_env = EcoAntV2(battery_limit=100.0)
    env = BraxToGymnaxWrapper(env=brax_env, episode_length=1000)
    env_params = [env.default_params] * NUM_RUNS

    rng = jax.random.PRNGKey(SEED)
    rngs = jax.random.split(rng, NUM_RUNS)

    dynamic_config = DynamicConfig(
        rng=rngs,
        env_params=jax.tree.map(lambda *xs: jnp.stack(xs), *env_params),
        cost_limit=jnp.ones(NUM_RUNS) * 50.0,
        critic_lr=jnp.ones(NUM_RUNS) * 3e-4,
        state_model_lr=jnp.ones(NUM_RUNS) * 5e-4,
        gae_gamma=jnp.ones(NUM_RUNS) * 0.99,
        gae_lambda=jnp.ones(NUM_RUNS) * 0.95,
        cost_gamma=jnp.ones(NUM_RUNS) * 0.999,
        max_grad_norm=jnp.ones(NUM_RUNS) * 0.5,
        target_kl=jnp.ones(NUM_RUNS) * 0.01,
        entropy_coeff=jnp.ones(NUM_RUNS) * 0.0,
        backtrack_coeff=jnp.ones(NUM_RUNS) * 0.8,
        backtrack_iters=jnp.ones(NUM_RUNS) * 10,
        damping_coeff=jnp.ones(NUM_RUNS) * 0.1,
        margin_lr=jnp.ones(NUM_RUNS) * 0.0,
        adam_eps=jnp.ones(NUM_RUNS) * 1e-5,
    )

    train_fn = make_train(
        env,
        num_steps=int(200000),
        num_envs=128,
        train_freq=100,
        critic_epochs=10,
        state_model_epochs=10,
        model_objective=ModelObjective.REWARD,
    )
    train_vjit = jax.jit(jax.vmap(train_fn))
    start_time = time.perf_counter()
    runner_states, all_metrics = jax.block_until_ready(train_vjit(dynamic_config))
    runtime = time.perf_counter() - start_time
    print(f"Runtime: {runtime:.2f}s")

    log.save_local(
        algo_name="acpo_reward",
        env_name=brax_env.name,
        metrics=all_metrics,
    )

    log.save_wandb(
        project="EcoAnt",
        algo_name="acpo_reward",
        env_name=brax_env.name,
        metrics=all_metrics,
    )
