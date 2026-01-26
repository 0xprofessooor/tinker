"""
Constrained Policy Optimization (CPO)

Based on the paper: https://arxiv.org/abs/1705.10528
Implements CPO for safe reinforcement learning with cost constraints.
"""

from typing import Tuple, Callable
import chex
import distrax
from flax import nnx, struct
import jax
import jax.numpy as jnp
import optax
from gymnax.environments.environment import Environment, EnvParams, EnvState
import time

from safenax.wrappers import BraxToGymnaxWrapper, LogWrapper
from safenax import EcoAntV1
from tinker import norm, log


class ActorCritic(nnx.Module):
    """Combined actor-critic network with separate value and cost-value heads."""

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

        # Actor network
        self.actor_fc1 = nnx.Linear(obs_dim, 256, rngs=rngs)
        self.actor_fc2 = nnx.Linear(256, 256, rngs=rngs)
        self.actor_mean = nnx.Linear(256, action_dim, rngs=rngs)
        self.log_std = nnx.Param(jnp.zeros(action_dim))

        # Value critic (for rewards)
        self.value_fc1 = nnx.Linear(obs_dim, 256, rngs=rngs)
        self.value_fc2 = nnx.Linear(256, 256, rngs=rngs)
        self.value_out = nnx.Linear(256, 1, rngs=rngs)

        # Cost-value critic (for costs)
        self.cost_value_fc1 = nnx.Linear(obs_dim, 256, rngs=rngs)
        self.cost_value_fc2 = nnx.Linear(256, 256, rngs=rngs)
        self.cost_value_out = nnx.Linear(256, 1, rngs=rngs)

    def __call__(self, x):
        # Actor network
        actor_x = self.activation(self.actor_fc1(x))
        actor_x = self.activation(self.actor_fc2(actor_x))
        actor_mean = self.actor_mean(actor_x)
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(self.log_std[...]))

        # Value critic
        value_x = self.activation(self.value_fc1(x))
        value_x = self.activation(self.value_fc2(value_x))
        value = jnp.squeeze(self.value_out(value_x), axis=-1)

        # Cost-value critic
        cost_value_x = self.activation(self.cost_value_fc1(x))
        cost_value_x = self.activation(self.cost_value_fc2(cost_value_x))
        cost_value = jnp.squeeze(self.cost_value_out(cost_value_x), axis=-1)

        return pi, value, cost_value


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
    norm_obs: jax.Array
    next_obs: jax.Array
    info: dict


@struct.dataclass
class CPOState:
    """Extended state for CPO including constraint tracking."""

    model_state: nnx.State
    opt_state: optax.OptState
    margin: float
    num_updates: int


@struct.dataclass
class DynamicConfig:
    """Holds dynamic configuration parameters for CPO training.

    :param rng: Random number generator key.
    :param env_params: Environment parameters.
    :param cost_limit: Maximum expected cost.
    :param lr: Learning rate.
    :param gae_gamma: Discount factor for GAE.
    :param gae_lambda: Lambda parameter for GAE.
    :param max_grad_norm: Maximum gradient norm for clipping.
    :param target_kl: Target KL divergence for policy updates.
    :param entropy_coeff: Coefficient for entropy regularization.
    :param backtrack_coeff: Backtracking line search coefficient.
    :param backtrack_iters: Maximum backtracking line search iterations.
    :param damping_coeff: Damping coefficient for Hessian-vector product.
    :param margin_lr: Learning rate for constraint margin updates.
    """

    rng: jax.Array
    env_params: EnvParams
    cost_limit: float
    lr: float = (3e-4,)
    gae_gamma: float = (0.99,)
    cost_gamma: float = (0.999,)
    gae_lambda: float = (0.95,)
    max_grad_norm: float = (0.5,)
    target_kl: float = (0.01,)
    entropy_coeff: float = (0.0,)
    backtrack_coeff: float = (0.8,)
    backtrack_iters: int = (10,)
    damping_coeff: float = (0.1,)
    margin_lr: float = (0.05,)


def hvp(f: Callable, primals: Tuple, tangents: Tuple) -> jnp.ndarray:
    """Hessian-vector product using JAX autodiff."""
    return jax.jvp(jax.grad(f), primals, tangents)[1]


def cg_solve(
    hvp_fn: Callable,
    b: jnp.ndarray,
    max_iter: int = 10,
    residual_tol: float = 1e-10,
) -> jnp.ndarray:
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
    g: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,  # scalar array
    hvp_fn: Callable,
    target_kl: float,
    use_constraint: bool,
    damping_coeff: float,
) -> Tuple[jnp.ndarray, int]:
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


def make_train(
    env: Environment,
    num_steps: int,
    num_envs: int,
    train_freq: int,
    critic_epochs: int = 80,
    activation: Callable = jax.nn.tanh,
    anneal_lr: bool = True,
    use_constraint: bool = True,
):
    """Generate a jitted JAX CPO train function.

    :param env: Gymnax environment (must provide cost signals).
    :param num_steps: Number of steps to train per environment.
    :param num_envs: Number of parallel environments to run.
    :param train_freq: Number of steps to run between training updates.
    :param critic_epochs: Number of critic update iterations per rollout.
    :param activation: Activation function for the network hidden layers.
    :param anneal_lr: Whether to anneal the learning rate over time.
    :param use_constraint: Whether to use safety constraints.
    """

    num_updates = num_steps // train_freq
    env = LogWrapper(env)

    def train(config: DynamicConfig) -> Tuple[CPOState, dict]:
        # INIT NETWORK
        rng, model_rng = jax.random.split(config.rng)
        model = ActorCritic(
            obs_dim=env.observation_space(config.env_params).shape[0],
            action_dim=env.action_space(config.env_params).shape[0],
            activation=activation,
            rngs=nnx.Rngs(model_rng),
        )
        graphdef, params = nnx.split(model)

        # INIT OPTIMIZER
        if anneal_lr:
            schedule = optax.linear_schedule(
                init_value=config.lr,
                end_value=0.0,
                transition_steps=num_updates * critic_epochs,
            )
            tx = optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adam(learning_rate=schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adam(config.lr, eps=1e-5),
            )
        opt_state = tx.init(params)

        cpo_state = CPOState(
            model_state=params, opt_state=opt_state, margin=0.0, num_updates=0
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

        # TRAIN LOOP
        def _update_step(
            runner_state: Tuple[
                CPOState,
                EnvState,
                norm.RunningMeanStdState,
                jax.Array,
                jax.Array,
                jax.Array,
                chex.PRNGKey,
            ],
            _,
        ):
            params = runner_state[0].model_state
            model: ActorCritic = nnx.merge(graphdef, params)

            # COLLECT TRAJECTORIES
            def _env_step(
                runner_state: Tuple[
                    CPOState,
                    EnvState,
                    norm.RunningMeanStdState,
                    jax.Array,
                    jax.Array,
                    jax.Array,
                    chex.PRNGKey,
                ],
                _,
            ):
                (
                    cpo_state,
                    env_state,
                    obs_norm_state,
                    norm_obs,
                    running_cost,
                    cost_discount,
                    rng,
                ) = runner_state

                # SELECT ACTION
                rng, action_rng = jax.random.split(rng)
                pi, value, cost_value = model(norm_obs)
                action = pi.sample(seed=action_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, step_rng = jax.random.split(rng)
                rng_step = jax.random.split(step_rng, num_envs)
                next_obs, next_env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, config.env_params)

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
                    norm_obs,
                    next_obs,
                    info,
                )

                next_norm_obs = jax.vmap(norm.normalize, in_axes=(None, 0))(
                    obs_norm_state, next_obs
                )
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

                runner_state = (
                    cpo_state,
                    next_env_state,
                    obs_norm_state,
                    next_norm_obs,
                    next_running_cost,
                    next_cost_discount,
                    rng,
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, train_freq
            )
            (
                cpo_state,
                env_state,
                obs_norm_state,
                last_norm_obs,
                last_running_cost,
                last_cost_discount,
                rng,
            ) = runner_state
            _, last_val, last_cost_val = model(last_norm_obs)

            # Flatten batch: (train_freq, num_envs, obs_dim) -> (train_freq * num_envs, obs_dim)
            batch_obs = traj_batch.next_obs.reshape(-1, *traj_batch.next_obs.shape[2:])
            obs_norm_state = norm.welford_update(obs_norm_state, batch_obs)

            # CALCULATE ADVANTAGE AND COST ADVANTAGE

            def _calculate_gae(
                traj_batch: Transition,
                last_val: jnp.ndarray,
                gamma: float,
                lambda_: float,
            ) -> Tuple[jnp.ndarray, jnp.ndarray]:
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

            # Reward advantages
            advantages, return_targets = _calculate_gae(
                traj_batch, last_val, config.gae_gamma, config.gae_lambda
            )

            # Cost advantages
            cost_advantages, cost_targets = _calculate_gae(
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
            new_margin = jnp.maximum(0.0, cpo_state.margin + config.margin_lr * c_raw)
            c = c_raw + new_margin
            # c = c / (train_freq + 1e-8)

            # UPDATE POLICY (CPO STEP)
            # Clone current model for reference in KL and line search
            pi_old, _, _ = model(traj_batch.norm_obs)
            flat_params, unravel_fn = jax.flatten_util.ravel_pytree(params)

            def policy_loss_fn(params):
                """Compute surrogate policy loss."""
                model: ActorCritic = nnx.merge(graphdef, params)
                pi, _, _ = model(traj_batch.norm_obs)
                log_prob = pi.log_prob(traj_batch.action)
                ratio = jnp.exp(log_prob - traj_batch.log_prob)

                policy_loss = -(ratio * advantages).mean() - (
                    config.entropy_coeff * pi.entropy().mean()
                )

                return policy_loss

            def cost_loss_fn(params):
                """Compute surrogate cost loss."""
                model: ActorCritic = nnx.merge(graphdef, params)
                pi, _, _ = model(traj_batch.norm_obs)
                log_prob = pi.log_prob(traj_batch.action)
                ratio = jnp.exp(log_prob - traj_batch.log_prob)

                cost_loss = (ratio * cost_advantages).mean()

                return cost_loss

            def joint_loss_fn(params):
                model: ActorCritic = nnx.merge(graphdef, params)
                pi, _, _ = model(traj_batch.norm_obs)
                log_prob = pi.log_prob(traj_batch.action)
                ratio = jnp.exp(log_prob - traj_batch.log_prob)

                policy_loss = -(ratio * advantages).mean()
                cost_loss = (ratio * cost_advantages).mean()

                return policy_loss, cost_loss

            # Get gradients of policy and cost losses
            old_policy_loss, g_tree = jax.value_and_grad(policy_loss_fn)(params)
            old_cost_loss, b_tree = jax.value_and_grad(cost_loss_fn)(params)

            # Flatten gradients
            g, _ = jax.flatten_util.ravel_pytree(g_tree)
            b, _ = jax.flatten_util.ravel_pytree(b_tree)

            def kl_fn(new_params_flat):
                """Compute KL divergence between old and new policy."""
                params_unflat = unravel_fn(new_params_flat)
                model_new: ActorCritic = nnx.merge(graphdef, params_unflat)
                pi_new, _, _ = model_new(traj_batch.norm_obs)
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

                new_policy_loss, new_cost_loss = joint_loss_fn(new_params)
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
                (0, params, False),
            )
            final_params_flat, _ = jax.flatten_util.ravel_pytree(final_params)
            final_kl = kl_fn(final_params_flat)

            # Update model parameters in place with final params from line search
            nnx.update(model, final_params)
            cpo_state = cpo_state.replace(
                model_state=final_params,
                margin=new_margin,
                num_updates=cpo_state.num_updates + 1,
            )

            # UPDATE VALUE CRITICS
            def critic_loss_fn(params):
                """Compute critic loss (for both value and cost-value heads)."""
                model = nnx.merge(graphdef, params)
                _, value, cost_value = model(traj_batch.norm_obs)
                value_loss = ((value - return_targets) ** 2).mean()
                cost_value_loss = ((cost_value - cost_targets) ** 2).mean()
                total_loss = value_loss + cost_value_loss
                return total_loss, (value_loss, cost_value_loss)

            def _update_critic(cpo_state: CPOState, _):
                """Update value and cost critic parameters."""
                params = cpo_state.model_state

                (_, (value_loss, cost_value_loss)), grads = jax.value_and_grad(
                    critic_loss_fn, has_aux=True
                )(params)

                updates, new_opt_state = tx.update(grads, cpo_state.opt_state, params)
                new_params = optax.apply_updates(params, updates)

                cpo_state = cpo_state.replace(
                    model_state=new_params, opt_state=new_opt_state
                )
                return cpo_state, (value_loss, cost_value_loss)

            cpo_state, (value_losses, cost_value_losses) = jax.lax.scan(
                _update_critic, cpo_state, None, critic_epochs
            )

            sparse_battery_used = jnp.where(
                is_terminal, 500 - traj_batch.info["battery"], 0
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
                "episode_battery_used": episode_battery_used,
                "accepted": accepted,
            }

            runner_state = (
                cpo_state,
                env_state,
                obs_norm_state,
                last_norm_obs,
                last_running_cost,
                last_cost_discount,
                rng,
            )

            return runner_state, metrics

        # RUN TRAINING LOOP
        rng, _rng = jax.random.split(rng)
        runner_state = (
            cpo_state,
            env_state,
            obs_norm_state,
            norm_obsv,
            running_cost,
            cost_discount,
            _rng,
        )

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, num_updates
        )

        return runner_state[0], metrics

    return train


if __name__ == "__main__":
    SEED = 0
    NUM_RUNS = 5

    brax_env = EcoAntV1(battery_limit=500.0)
    env = BraxToGymnaxWrapper(env=brax_env, episode_length=1000)
    env_params = [env.default_params] * NUM_RUNS

    rng = jax.random.PRNGKey(SEED)
    rngs = jax.random.split(rng, NUM_RUNS)

    dynamic_config = DynamicConfig(
        rng=rngs,
        env_params=jax.tree.map(lambda *xs: jnp.stack(xs), *env_params),
        cost_limit=jnp.ones(NUM_RUNS) * 0.1,
        lr=jnp.ones(NUM_RUNS) * 3e-4,
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
    )

    train_fn = make_train(
        env,
        num_steps=int(2e6),
        num_envs=5,
        train_freq=500,
    )
    train_vjit = jax.jit(jax.vmap(train_fn))
    start_time = time.perf_counter()
    runner_states, all_metrics = jax.block_until_ready(train_vjit(dynamic_config))
    runtime = time.perf_counter() - start_time
    print(f"Runtime: {runtime:.2f}s")

    log.save_local(
        algo_name="cpo",
        env_name=brax_env.name,
        metrics=all_metrics,
    )

    log.save_wandb(
        project="EcoAnt",
        algo_name="cpo",
        env_name=brax_env.name,
        metrics=all_metrics,
    )
