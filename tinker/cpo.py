"""Constrained Policy Optimization (CPO)

Based on the paper: https://arxiv.org/abs/1705.10528
Implements CPO for safe reinforcement learning with cost constraints.
"""

from typing import NamedTuple, Tuple, Callable

import chex
import distrax
from flax import nnx
import jax
import jax.numpy as jnp
import optax
from gymnax.environments.environment import Environment, EnvParams
from gymnax.wrappers import LogWrapper

import wandb


class ActorCritic(nnx.Module):
    """Combined actor-critic network with separate value and cost-value heads."""

    def __init__(
        self, action_dim: int, activation: Callable = jax.nn.tanh, rngs: nnx.Rngs = None
    ):
        super().__init__()
        self.action_dim = action_dim
        self.activation = activation

        # Actor network
        self.actor_fc1 = nnx.Linear(256, rngs=rngs)
        self.actor_fc2 = nnx.Linear(256, rngs=rngs)
        self.actor_mean = nnx.Linear(action_dim, rngs=rngs)
        self.log_std = nnx.Param(jnp.zeros(action_dim))

        # Value critic (for rewards)
        self.value_fc1 = nnx.Linear(256, rngs=rngs)
        self.value_fc2 = nnx.Linear(256, rngs=rngs)
        self.value_out = nnx.Linear(1, rngs=rngs)

        # Cost-value critic (for costs)
        self.cost_value_fc1 = nnx.Linear(256, rngs=rngs)
        self.cost_value_fc2 = nnx.Linear(256, rngs=rngs)
        self.cost_value_out = nnx.Linear(1, rngs=rngs)

    def __call__(self, x):
        # Actor network
        actor_x = self.activation(self.actor_fc1(x))
        actor_x = self.activation(self.actor_fc2(actor_x))
        actor_mean = self.actor_mean(actor_x)
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(self.log_std.value))

        # Value critic
        value_x = self.activation(self.value_fc1(x))
        value_x = self.activation(self.value_fc2(value_x))
        value = jnp.squeeze(self.value_out(value_x), axis=-1)

        # Cost-value critic
        cost_value_x = self.activation(self.cost_value_fc1(x))
        cost_value_x = self.activation(self.cost_value_fc2(cost_value_x))
        cost_value = jnp.squeeze(self.cost_value_out(cost_value_x), axis=-1)

        return pi, value, cost_value


class Transition(NamedTuple):
    """Transition tuple including cost information."""

    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    cost_value: jnp.ndarray
    reward: jnp.ndarray
    cost: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


class CPOState(NamedTuple):
    """Extended state for CPO including constraint tracking."""

    optimizer: nnx.Optimizer  # Holds model AND optimizer state (momentum, etc.)
    margin: float  # Constraint margin for safety


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
    c: float,
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
    env_params: EnvParams,
    num_steps: int,
    num_envs: int,
    train_freq: int,
    critic_epochs: int = 80,
    activation: Callable = jax.nn.tanh,
    lr: float = 3e-4,
    anneal_lr: bool = False,
    gae_gamma: float = 0.99,
    cost_gamma: float = 0.99,
    gae_lambda: float = 0.95,
    max_grad_norm: float = 0.5,
    target_kl: float = 0.01,
    cost_limit: float = 25.0,
    backtrack_coeff: float = 0.8,
    backtrack_iters: int = 10,
    damping_coeff: float = 0.1,
    margin_lr: float = 0.05,
    use_constraint: bool = True,
):
    """Generate a jitted JAX CPO train function.

    :param env: Gymnax environment (must provide cost signals).
    :param env_params: Environment parameters.
    :param num_steps: Number of steps to train per environment.
    :param num_envs: Number of parallel environments to run.
    :param train_freq: Number of steps to run between training updates.
    :param critic_epochs: Number of critic update iterations per rollout.
    :param activation: Activation function for the network hidden layers.
    :param lr: Learning rate for the critic optimizer.
    :param anneal_lr: Whether to anneal the learning rate over time.
    :param gae_gamma: Discount factor for rewards.
    :param cost_gamma: Discount factor for costs.
    :param gae_lambda: Lambda for the Generalized Advantage Estimation.
    :param max_grad_norm: Maximum gradient norm for clipping.
    :param target_kl: Target KL divergence threshold.
    :param cost_limit: Constraint threshold for cumulative cost.
    :param backtrack_coeff: Coefficient for line search backtracking.
    :param backtrack_iters: Maximum number of backtracking iterations.
    :param damping_coeff: Damping coefficient for conjugate gradient.
    :param margin_lr: Learning rate for the constraint margin.
    :param use_constraint: Whether to use safety constraints.
    """

    num_updates = num_steps // train_freq
    env = LogWrapper(env)

    def train(rng: chex.PRNGKey) -> Tuple[CPOState, dict]:
        # INIT NETWORK
        rng, model_rng = jax.random.split(rng)
        model = ActorCritic(
            action_dim=env.action_space(env_params).shape[0],
            activation=activation,
            rngs=nnx.Rngs(model_rng),
        )

        # INIT OPTIMIZER
        if anneal_lr:
            schedule = optax.linear_schedule(
                init_value=lr,
                end_value=0.0,
                transition_steps=num_updates * critic_epochs,
            )
            tx = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate=schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(lr, eps=1e-5),
            )

        optimizer = nnx.Optimizer(model, tx)
        cpo_state = CPOState(optimizer=optimizer, margin=0.0)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, num_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, _):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, _):
                cpo_state, env_state, obs, rng = runner_state

                # SELECT ACTION (use optimizer.model to access the current model)
                rng, action_rng = jax.random.split(rng)
                pi, value, cost_value = cpo_state.optimizer.model(obs)
                action = pi.sample(seed=action_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, step_rng = jax.random.split(rng)
                rng_step = jax.random.split(step_rng, num_envs)
                next_obs, next_env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)

                # Extract cost from info (environment-dependent)
                # For environments without explicit costs, cost = 0
                cost = info.get("cost", jnp.zeros_like(reward))

                transition = Transition(
                    done,
                    action,
                    value,
                    cost_value,
                    reward,
                    cost,
                    log_prob,
                    obs,
                    info,
                )
                runner_state = (cpo_state, next_env_state, next_obs, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, train_freq
            )

            # CALCULATE ADVANTAGE AND COST ADVANTAGE
            cpo_state, env_state, last_obs, rng = runner_state
            _, last_val, last_cost_val = cpo_state.optimizer.model(last_obs)

            def _calculate_gae(traj_batch, last_val, gamma, lambda_):
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
                traj_batch, last_val, gae_gamma, gae_lambda
            )

            # Cost advantages
            cost_advantages, cost_targets = _calculate_gae(
                traj_batch._replace(
                    reward=traj_batch.cost, value=traj_batch.cost_value
                ),
                last_cost_val,
                cost_gamma,
                gae_lambda,
            )

            # Normalize reward advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Compute constraint violation
            avg_ep_cost = cost_targets.mean()
            c = avg_ep_cost - cost_limit
            new_margin = jnp.maximum(0.0, cpo_state.margin + margin_lr * c)
            c = c + new_margin

            # UPDATE VALUE CRITICS FIRST
            def critic_loss_fn(model: ActorCritic):
                """Compute critic loss (for both value and cost-value heads)."""
                _, value, cost_value = model(traj_batch.obs)
                value_loss = ((value - return_targets) ** 2).mean()
                cost_value_loss = ((cost_value - cost_targets) ** 2).mean()
                return value_loss + cost_value_loss

            # Update critics for critic_epochs iterations using the optimizer
            def _update_critic(optimizer: nnx.Optimizer, _):
                """Update critic using Adam optimizer (preserves momentum)."""
                loss, grads = nnx.value_and_grad(critic_loss_fn)(optimizer.model)
                optimizer.update(grads)
                return optimizer, loss

            # Start from current optimizer
            optimizer, critic_losses = jax.lax.scan(
                _update_critic, cpo_state.optimizer, None, critic_epochs
            )

            # UPDATE POLICY (CPO STEP) - using the critic-updated model
            # Split ORIGINAL model (before critic update) for KL computation
            graphdef_orig, params_orig = nnx.split(cpo_state.optimizer.model, nnx.Param)

            # Split CRITIC-UPDATED model for policy gradients
            graphdef, params = nnx.split(optimizer.model, nnx.Param)

            def compute_policy_loss(params_arg):
                """Compute surrogate policy loss and cost loss."""
                model = nnx.merge(graphdef, params_arg)
                pi, _, _ = model(traj_batch.obs)
                log_prob = pi.log_prob(traj_batch.action)
                ratio = jnp.exp(log_prob - traj_batch.log_prob)

                policy_loss = -(ratio * advantages).mean()
                cost_loss = (ratio * cost_advantages).mean()

                return policy_loss, cost_loss

            def compute_kl(params_new):
                """Compute KL divergence between old and new policy."""
                model_new = nnx.merge(graphdef, params_new)
                pi_new, _, _ = model_new(traj_batch.obs)

                model_orig = nnx.merge(graphdef_orig, params_orig)
                pi_old, _, _ = model_orig(traj_batch.obs)
                return distrax.kl_divergence(pi_old, pi_new).mean()

            # Get gradient of REWARD loss (g)
            (old_policy_loss, old_cost_loss), g_tree = jax.value_and_grad(
                compute_policy_loss, has_aux=True
            )(params)

            # Get gradient of COST loss (b)
            def cost_loss_fn(p):
                return compute_policy_loss(p)[1]

            b_tree = jax.grad(cost_loss_fn)(params)

            # Flatten gradients
            g, unravel_fn = jax.flatten_util.ravel_pytree(g_tree)
            b, _ = jax.flatten_util.ravel_pytree(b_tree)

            # Define Hessian-vector product for KL divergence
            def kl_fn(params_flat):
                params_unflat = unravel_fn(params_flat)
                return compute_kl(params_unflat)

            flat_params, _ = jax.flatten_util.ravel_pytree(params)
            hvp_fn = lambda v: hvp(kl_fn, (flat_params,), (v,))

            # Compute CPO step direction
            direction, optim_case = compute_cpo_step(
                g, b, float(c), hvp_fn, target_kl, use_constraint, damping_coeff
            )

            # Backtracking line search
            def line_search_body(search_state):
                i, current_params, accepted = search_state

                step_size = backtrack_coeff**i
                new_flat_params = flat_params - step_size * direction
                new_params = unravel_fn(new_flat_params)

                new_policy_loss, new_cost_loss = compute_policy_loss(new_params)
                kl = compute_kl(new_params)

                # Check acceptance criteria
                loss_improve = (optim_case > 1) or (new_policy_loss <= old_policy_loss)
                cost_improve = (
                    (new_cost_loss - old_cost_loss <= jnp.maximum(-c, 0))
                    if use_constraint
                    else True
                )
                kl_ok = kl <= target_kl

                accept = loss_improve & cost_improve & kl_ok

                updated_params = jax.tree.map(
                    lambda new, old: jnp.where(accept, new, old),
                    new_params,
                    current_params,
                )

                return i + 1, updated_params, accept

            def line_search_cond(search_state):
                i, _, accepted = search_state
                return (i < backtrack_iters) & (~accepted)

            # Line search (start with critic-updated params)
            _, final_params, accepted = jax.lax.while_loop(
                line_search_cond,
                line_search_body,
                (0, params, False),
            )

            # Merge final params back into model
            final_model = nnx.merge(graphdef, final_params)
            optimizer = optimizer.replace(model=final_model)

            # Update CPO state with preserved optimizer
            cpo_state = CPOState(optimizer=optimizer, margin=new_margin)

            # Compute final KL for logging
            final_kl = compute_kl(final_params)

            metrics = {
                "policy_loss": old_policy_loss,
                "cost_loss": old_cost_loss,
                "critic_loss": critic_losses.mean(),
                "kl": final_kl,
                "constraint_violation": c,
                "margin": new_margin,
                "avg_ep_cost": avg_ep_cost,
                "optim_case": optim_case,
                "total_reward": traj_batch.reward.sum(),
                "accepted": accepted,
            }

            runner_state = (cpo_state, env_state, last_obs, rng)

            return runner_state, metrics

        # RUN TRAINING LOOP
        rng, _rng = jax.random.split(rng)
        runner_state = (cpo_state, env_state, obsv, _rng)

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, num_updates
        )

        return runner_state[0], metrics

    return train


if __name__ == "__main__":
    import time

    # Example usage with a simple environment
    rng = jax.random.PRNGKey(0)

    # Example: Create a simple safe environment wrapper
    # You would need to modify your environment to provide cost signals in the info dict

    """
    Example integration:
    
    from tinker.cpo_refactored import make_train
    from your_safe_env import SafeEnvironment
    
    env = SafeEnvironment()
    env_params = env.default_params
    
    # Create train function
    train_fn = make_train(
        env=env,
        env_params=env_params,
        num_steps=100000,
        num_envs=4,
        train_freq=1000,
        vf_iters=80,
        lr=3e-4,
        target_kl=0.01,
        cost_limit=25.0,  # Constraint on expected cumulative cost
        use_constraint=True,
    )
    
    # Train the agent
    rng = jax.random.PRNGKey(42)
    final_state, metrics = jax.jit(train_fn)(rng)
    
    # Access trained model
    trained_model = final_state.model
    
    # Use the trained policy
    def get_action(obs, rng):
        pi, value, cost_value = trained_model(obs)
        action = pi.sample(seed=rng)
        return action
    """

    print("CPO implementation ready (NNX version).")
    print("This implementation uses Flax NNX for a more Pythonic API.")
    print(
        "Your environment must provide 'cost' values in the info dict for constraint tracking."
    )
