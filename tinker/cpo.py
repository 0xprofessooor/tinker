"""Constrained Policy Optimization (CPO) - JAX Implementation.

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
    cost: jnp.ndarray  # Cost signal from environment
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
    x = jnp.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rdotr = jnp.dot(r, r)

    for _ in range(max_iter):
        Ap = hvp_fn(p)
        alpha = rdotr / (jnp.dot(p, Ap) + 1e-8)
        x = x + alpha * p
        r = r - alpha * Ap
        new_rdotr = jnp.dot(r, r)

        if new_rdotr < residual_tol:
            break

        beta = new_rdotr / (rdotr + 1e-8)
        p = r + beta * p
        rdotr = new_rdotr

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
    use_trpo = (jnp.dot(b, b) <= 1e-8) and (c < 0.0)

    if not use_constraint or use_trpo:
        # TRPO case: no constraint
        optim_case = 4
        lam = jnp.sqrt(q / (2.0 * target_kl))
        direction = v / (lam + 1e-8)
        return direction, optim_case

    # CPO case: solve with constraint
    w = cg_solve(damped_hvp, b, max_iter=10)
    r = jnp.dot(w, approx_g)
    s = jnp.dot(w, damped_hvp(w))

    A = q - r**2 / (s + 1e-8)
    B = 2.0 * target_kl - c**2 / (s + 1e-8)

    # Determine optimization case
    if c < 0.0 and B < 0.0:
        optim_case = 3
    elif c < 0.0 and B >= 0.0:
        optim_case = 2
    elif c >= 0.0 and B >= 0.0:
        optim_case = 1
    else:
        optim_case = 0

    # Compute step based on optimization case
    if optim_case == 0:
        # Recovery case
        nu = jnp.sqrt(2.0 * target_kl / (s + 1e-8))
        direction = nu * w
    else:
        # Feasible cases
        if optim_case > 2:
            # Ignore constraint
            lam = jnp.sqrt(q / (2.0 * target_kl))
            nu = 0.0
        else:
            # Solve for optimal lam, nu
            if c < 0.0:
                LA, LB = [0.0, r / c], [r / c, jnp.inf]
            else:
                LA, LB = [r / c, jnp.inf], [0.0, r / c]

            proj = lambda x, L: jnp.maximum(L[0], jnp.minimum(L[1], x))
            lam_a = proj(jnp.sqrt(A / (B + 1e-8)), LA)
            lam_b = proj(jnp.sqrt(q / (2 * target_kl)), LB)

            f_a = -0.5 * (A / (lam_a + 1e-8) + B * lam_a) - r * c / (s + 1e-8)
            f_b = -0.5 * (q / (lam_b + 1e-8) + 2.0 * target_kl * lam_b)

            lam = jnp.where(f_a >= f_b, lam_a, lam_b)
            nu = jnp.maximum(0.0, lam * c - r) / (s + 1e-8)

        direction = (v + nu * w) / (lam + 1e-8)

    return direction, optim_case


def make_train(
    env: Environment,
    env_params: EnvParams,
    num_steps: int,
    num_envs: int,
    train_freq: int,
    vf_iters: int = 80,
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
    :param vf_iters: Number of value function update iterations per rollout.
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

    def linear_schedule(count):
        frac = 1.0 - (count // vf_iters) / num_updates
        return lr * frac

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
                transition_steps=num_updates * vf_iters,
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
                cpo_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION (use optimizer.model to access the current model)
                rng, _rng = jax.random.split(rng)
                pi, value, cost_value = cpo_state.optimizer.model(last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, num_envs)
                obsv, env_state, reward, done, info = jax.vmap(
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
                    last_obs,
                    info,
                )
                runner_state = (cpo_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, train_freq
            )

            # CALCULATE ADVANTAGE AND COST ADVANTAGE
            cpo_state, env_state, last_obs, rng = runner_state
            _, last_val, last_cost_val = cpo_state.optimizer.model(last_obs)

            def _calculate_gae(traj_batch, last_val, gamma, lambda_):
                def _get_advantages(gae_and_next_value, transition_value_reward):
                    gae, next_value = gae_and_next_value
                    done, value, reward = transition_value_reward
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
            advantages, targets = _calculate_gae(
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

            # Center cost advantages (don't normalize)
            cost_advantages = cost_advantages - cost_advantages.mean()

            # Compute constraint violation
            ep_cost = traj_batch.cost.sum()
            c = ep_cost - cost_limit
            new_margin = jnp.maximum(0.0, cpo_state.margin + margin_lr * c)
            c = c + new_margin
            c = c / (train_freq * num_envs + 1e-8)

            # ====================================================================
            # ACTOR-CRITIC UPDATE (CORRECT ORDER)
            # ====================================================================
            # 1. Update value critics FIRST (multiple iterations with Adam)
            #    This gives us V_k+1 that better estimates the returns
            # 2. Update policy SECOND (one CPO step)
            #    The policy becomes greedy w.r.t. the updated value function
            #    while respecting trust region and cost constraints
            #
            # This order ensures:
            # - Critics are trained on collected data
            # - Policy update uses the BEST available value estimates
            # - CPO step is not overwritten by subsequent critic updates
            # - Optimizer state (Adam momentum) is preserved across updates
            # ====================================================================

            # UPDATE VALUE CRITICS FIRST (using Adam optimizer properly)
            def critic_loss_fn(model: ActorCritic):
                """Compute critic loss (for both value and cost-value heads)."""
                _, value, cost_value = model(traj_batch.obs)
                value_loss = ((value - targets) ** 2).mean()
                cost_value_loss = ((cost_value - cost_targets) ** 2).mean()
                return value_loss + cost_value_loss

            # Update critics for vf_iters iterations using the optimizer
            def _update_critic(optimizer: nnx.Optimizer, _):
                """Update critic using Adam optimizer (preserves momentum)."""
                loss, grads = nnx.value_and_grad(critic_loss_fn)(optimizer.model)
                optimizer.update(grads)  # ✅ Uses Adam correctly, updates in-place
                return optimizer, loss

            # Start from current optimizer (preserves Adam state)
            optimizer_after_critic_update, critic_losses = jax.lax.scan(
                _update_critic, cpo_state.optimizer, None, vf_iters
            )

            # UPDATE POLICY (CPO STEP) - using the critic-updated model
            # Split ORIGINAL model (before critic update) for KL computation
            graphdef_orig, state_orig = nnx.split(cpo_state.optimizer.model)

            # Split CRITIC-UPDATED model for policy gradients
            graphdef, state_after_critic_update = nnx.split(
                optimizer_after_critic_update.model
            )

            def model_apply(state, obs):
                """Apply model with given state."""
                model = nnx.merge(graphdef, state)
                return model(obs)

            def compute_policy_loss(state):
                """Compute surrogate policy loss and cost loss."""
                pi, _, _ = model_apply(state, traj_batch.obs)
                log_prob = pi.log_prob(traj_batch.action)
                ratio = jnp.exp(log_prob - traj_batch.log_prob)

                policy_loss = -(ratio * advantages).mean()
                cost_loss = (ratio * cost_advantages).mean()

                return policy_loss, cost_loss

            def compute_kl(new_state):
                """Compute KL divergence between old and new policy."""
                pi_new, _, _ = model_apply(new_state, traj_batch.obs)
                # Compare against the ORIGINAL policy (before critic update)
                model_orig = nnx.merge(graphdef_orig, state_orig)
                pi_old, _, _ = model_orig(traj_batch.obs)
                return distrax.kl_divergence(pi_old, pi_new).mean()

            # Compute policy gradient and cost gradient using CRITIC-UPDATED state
            (old_policy_loss, old_cost_loss), (g_tree, b_tree) = jax.value_and_grad(
                compute_policy_loss, has_aux=True
            )(state_after_critic_update)

            # Flatten gradients
            g, unravel_fn = jax.flatten_util.ravel_pytree(g_tree)
            b, _ = jax.flatten_util.ravel_pytree(b_tree)

            # Define Hessian-vector product for KL divergence
            # HVP is computed at the critic-updated state
            def kl_fn(state_flat):
                state_unflat = unravel_fn(state_flat)
                return compute_kl(state_unflat)

            flat_state, _ = jax.flatten_util.ravel_pytree(state_after_critic_update)
            hvp_fn = lambda v: hvp(kl_fn, (flat_state,), (v,))

            # Compute CPO step direction
            direction, optim_case = compute_cpo_step(
                g, b, float(c), hvp_fn, target_kl, use_constraint, damping_coeff
            )

            # Backtracking line search starting from critic-updated state
            def line_search_body(search_state):
                i, current_state, accepted = search_state

                step_size = backtrack_coeff**i
                new_flat_state = flat_state - step_size * direction
                new_state = unravel_fn(new_flat_state)

                new_policy_loss, new_cost_loss = compute_policy_loss(new_state)
                kl = compute_kl(new_state)

                # Check acceptance criteria
                loss_improve = (optim_case > 1) or (new_policy_loss <= old_policy_loss)
                cost_improve = (
                    (new_cost_loss - old_cost_loss <= jnp.maximum(-c, 0))
                    if use_constraint
                    else True
                )
                kl_ok = kl <= target_kl

                accept = loss_improve & cost_improve & kl_ok

                updated_state = jax.tree.map(
                    lambda new, old: jnp.where(accept, new, old),
                    new_state,
                    current_state,
                )

                return i + 1, updated_state, accept

            def line_search_cond(search_state):
                i, _, accepted = search_state
                return (i < backtrack_iters) & (~accepted)

            # Line search starts from critic-updated state
            _, final_state, accepted = jax.lax.while_loop(
                line_search_cond,
                line_search_body,
                (0, state_after_critic_update, False),
            )

            # Merge final state (both actor and critic updates applied)
            final_model = nnx.merge(graphdef, final_state)

            # ✅ PRESERVE OPTIMIZER STATE - Don't create new optimizer!
            # Replace the model in the critic-updated optimizer with CPO-updated model
            # This keeps Adam's momentum/statistics intact
            final_optimizer = optimizer_after_critic_update.replace(model=final_model)

            # Update CPO state with preserved optimizer
            cpo_state = CPOState(optimizer=final_optimizer, margin=new_margin)

            # Compute final KL for logging
            final_kl = compute_kl(final_state)

            metrics = {
                "policy_loss": old_policy_loss,
                "cost_loss": old_cost_loss,
                "critic_loss": critic_losses.mean(),
                "kl": final_kl,
                "constraint_violation": c,
                "margin": new_margin,
                "episode_cost": ep_cost,
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
