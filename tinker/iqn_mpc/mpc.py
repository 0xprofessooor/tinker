"""Model Predictive Control with IQN uncertainty quantification.

Implements MPC planning using the learned IQN state transition model,
with support for chance constraints and CVaR-based risk measures.
Key reference: Lobo, Fazel, Boyd (2002) for convex optimization with uncertainty.
"""

from typing import Callable, Optional, Tuple
import chex
import jax
import jax.numpy as jnp
from flax import nnx, struct

from tinker.iqn_mpc.iqn import IQNTransitionModel


@struct.dataclass
class MPCConfig:
    """Configuration for MPC planner.
    
    :param horizon: Planning horizon (number of steps to look ahead).
    :param n_samples: Number of action sequence samples for CEM/random shooting.
    :param n_elite: Number of elite samples to keep in CEM.
    :param n_iterations: Number of CEM iterations.
    :param n_quantile_samples: Number of quantile samples per state transition.
    :param action_dim: Dimension of action space.
    :param action_low: Lower bound on actions.
    :param action_high: Upper bound on actions.
    :param gamma: Discount factor for rewards.
    :param risk_level: CVaR/VaR risk level α ∈ (0, 1]. α=1 is risk-neutral (expectation).
    :param constraint_threshold: Threshold for chance constraints (if any).
    :param constraint_probability: Required probability of satisfying constraints.
    """
    horizon: int = 10
    n_samples: int = 500
    n_elite: int = 50
    n_iterations: int = 5
    n_quantile_samples: int = 16
    action_dim: int = 1
    action_low: float = -1.0
    action_high: float = 1.0
    gamma: float = 0.99
    risk_level: float = 1.0  # 1.0 = risk-neutral, <1.0 = risk-averse (CVaR)
    constraint_threshold: Optional[float] = None
    constraint_probability: float = 0.95


def sample_trajectories(
    model: IQNTransitionModel,
    initial_state: jax.Array,
    action_sequences: jax.Array,
    key: chex.PRNGKey,
    n_quantile_samples: int = 16,
) -> jax.Array:
    """Sample state trajectories using IQN model.
    
    For each action sequence, samples multiple possible trajectories
    by drawing different quantile levels at each step.
    
    :param model: Trained IQN transition model.
    :param initial_state: Starting state, shape (state_dim,).
    :param action_sequences: Action sequences, shape (n_samples, horizon, action_dim).
    :param key: Random key for sampling.
    :param n_quantile_samples: Number of trajectory samples per action sequence.
    :return: Sampled trajectories, shape (n_samples, n_quantile_samples, horizon+1, state_dim).
    """
    network = nnx.merge(model.graphdef, model.params)
    
    n_samples, horizon, action_dim = action_sequences.shape
    state_dim = initial_state.shape[0]
    
    def rollout_single_action_seq(carry, action):
        """Roll out one step for all quantile samples."""
        states, key = carry  # states: (n_quantile_samples, state_dim)
        key, tau_key = jax.random.split(key)
        
        # Sample random quantiles for this step
        tau = jax.random.uniform(tau_key, (n_quantile_samples, 1))
        
        # Broadcast action to all quantile samples
        action_broadcast = jnp.broadcast_to(action, (n_quantile_samples, action_dim))
        
        # Get next state predictions at sampled quantiles
        # IQN returns (batch, n_quantiles, state_dim), but we sample 1 quantile per trajectory
        next_states = network(states, action_broadcast, tau)  # (n_quantile_samples, 1, state_dim)
        next_states = next_states.squeeze(1)  # (n_quantile_samples, state_dim)
        
        return (next_states, key), next_states
    
    def rollout_action_sequence(key, actions):
        """Roll out a single action sequence for all quantile samples."""
        # Initialize all quantile samples at the same initial state
        init_states = jnp.broadcast_to(initial_state, (n_quantile_samples, state_dim))
        
        (final_states, _), trajectory = jax.lax.scan(
            rollout_single_action_seq,
            (init_states, key),
            actions,  # (horizon, action_dim)
        )
        
        # Prepend initial state to trajectory
        init_states_expanded = init_states[None, :, :]  # (1, n_quantile_samples, state_dim)
        trajectory = jnp.concatenate(
            [init_states_expanded, trajectory],  # trajectory: (horizon, n_quantile_samples, state_dim)
            axis=0,
        )  # (horizon+1, n_quantile_samples, state_dim)
        
        # Transpose to (n_quantile_samples, horizon+1, state_dim)
        return trajectory.transpose(1, 0, 2)
    
    # Vectorize over action sequences
    keys = jax.random.split(key, n_samples)
    trajectories = jax.vmap(rollout_action_sequence)(keys, action_sequences)
    
    return trajectories  # (n_samples, n_quantile_samples, horizon+1, state_dim)


def compute_trajectory_returns(
    trajectories: jax.Array,
    action_sequences: jax.Array,
    reward_fn: Callable[[jax.Array, jax.Array], jax.Array],
    gamma: float = 0.99,
    risk_level: float = 1.0,
) -> jax.Array:
    """Compute returns for sampled trajectories with optional CVaR.
    
    :param trajectories: State trajectories, shape (n_samples, n_quantile, horizon+1, state_dim).
    :param action_sequences: Actions, shape (n_samples, horizon, action_dim).
    :param reward_fn: Function (state, action) -> reward.
    :param gamma: Discount factor.
    :param risk_level: CVaR level (1.0 = expectation, <1.0 = risk-averse).
    :return: Returns for each action sequence, shape (n_samples,).
    """
    n_samples, n_quantile, horizon_plus_1, state_dim = trajectories.shape
    horizon = horizon_plus_1 - 1
    
    # States for computing rewards (exclude final state for rewards)
    states = trajectories[:, :, :-1, :]  # (n_samples, n_quantile, horizon, state_dim)
    
    # Expand actions for broadcasting with quantile samples
    actions_expanded = action_sequences[:, None, :, :]  # (n_samples, 1, horizon, action_dim)
    actions_expanded = jnp.broadcast_to(
        actions_expanded, (n_samples, n_quantile, horizon, action_sequences.shape[-1])
    )
    
    # Compute rewards for all (state, action) pairs
    rewards = jax.vmap(jax.vmap(jax.vmap(reward_fn)))(states, actions_expanded)
    # rewards shape: (n_samples, n_quantile, horizon)
    
    # Compute discounted returns
    discount_factors = gamma ** jnp.arange(horizon)  # (horizon,)
    discounted_rewards = rewards * discount_factors  # (n_samples, n_quantile, horizon)
    total_returns = discounted_rewards.sum(axis=-1)  # (n_samples, n_quantile)
    
    if risk_level >= 1.0:
        # Risk-neutral: take mean over quantile samples
        return total_returns.mean(axis=-1)  # (n_samples,)
    else:
        # CVaR: average over worst α fraction of outcomes
        sorted_returns = jnp.sort(total_returns, axis=-1)  # Sort ascending (worst first)
        n_cvar = int(jnp.ceil(n_quantile * risk_level))
        n_cvar = max(1, n_cvar)
        cvar_returns = sorted_returns[:, :n_cvar].mean(axis=-1)  # (n_samples,)
        return cvar_returns


def check_chance_constraints(
    trajectories: jax.Array,
    constraint_fn: Callable[[jax.Array], jax.Array],
    threshold: float,
    required_probability: float,
) -> jax.Array:
    """Check if action sequences satisfy chance constraints.
    
    Constraint is satisfied if P(g(s) <= threshold) >= required_probability.
    
    :param trajectories: State trajectories, shape (n_samples, n_quantile, horizon+1, state_dim).
    :param constraint_fn: Function state -> constraint value (should be <= threshold).
    :param threshold: Constraint threshold.
    :param required_probability: Required probability of satisfaction.
    :return: Boolean mask of feasible action sequences, shape (n_samples,).
    """
    # Compute constraint values for all states
    constraint_values = jax.vmap(jax.vmap(jax.vmap(constraint_fn)))(trajectories)
    # shape: (n_samples, n_quantile, horizon+1)
    
    # For each trajectory, check if constraint is satisfied at all timesteps
    satisfied_per_trajectory = (constraint_values <= threshold).all(axis=-1)
    # shape: (n_samples, n_quantile)
    
    # Empirical probability of satisfaction
    empirical_prob = satisfied_per_trajectory.mean(axis=-1)  # (n_samples,)
    
    # Check if required probability is met
    return empirical_prob >= required_probability


class MPCPlanner:
    """Cross-Entropy Method (CEM) planner with IQN uncertainty.
    
    Uses CEM to optimize action sequences, evaluating candidates
    via IQN-sampled trajectories with optional risk measures.
    """

    def __init__(
        self,
        model: IQNTransitionModel,
        reward_fn: Callable[[jax.Array, jax.Array], jax.Array],
        config: MPCConfig,
        constraint_fn: Optional[Callable[[jax.Array], jax.Array]] = None,
    ):
        """
        :param model: Trained IQN transition model.
        :param reward_fn: Reward function (state, action) -> scalar.
        :param config: MPC configuration.
        :param constraint_fn: Optional constraint function state -> value (should be <= threshold).
        """
        self.model = model
        self.reward_fn = reward_fn
        self.config = config
        self.constraint_fn = constraint_fn

    def plan(self, state: jax.Array, key: chex.PRNGKey) -> Tuple[jax.Array, dict]:
        """Plan optimal action sequence from current state.
        
        :param state: Current state, shape (state_dim,).
        :param key: Random key.
        :return: Tuple of (first action, info dict with full planned sequence).
        """
        config = self.config
        
        # Initialize action distribution (Gaussian)
        mean = jnp.zeros((config.horizon, config.action_dim))
        std = jnp.ones((config.horizon, config.action_dim)) * (
            (config.action_high - config.action_low) / 4
        )

        def cem_iteration(carry, _):
            mean, std, key = carry
            key, sample_key, traj_key = jax.random.split(key, 3)
            
            # Sample action sequences from current distribution
            noise = jax.random.normal(sample_key, (config.n_samples, config.horizon, config.action_dim))
            action_sequences = mean + std * noise
            action_sequences = jnp.clip(action_sequences, config.action_low, config.action_high)
            
            # Sample trajectories using IQN
            trajectories = sample_trajectories(
                self.model,
                state,
                action_sequences,
                traj_key,
                config.n_quantile_samples,
            )
            
            # Compute returns (with risk measure)
            returns = compute_trajectory_returns(
                trajectories,
                action_sequences,
                self.reward_fn,
                config.gamma,
                config.risk_level,
            )
            
            # Apply chance constraints if specified
            if self.constraint_fn is not None and config.constraint_threshold is not None:
                feasible = check_chance_constraints(
                    trajectories,
                    self.constraint_fn,
                    config.constraint_threshold,
                    config.constraint_probability,
                )
                # Set infeasible returns to -inf
                returns = jnp.where(feasible, returns, -jnp.inf)
            
            # Select elite samples
            elite_indices = jnp.argsort(returns)[-config.n_elite:]
            elite_actions = action_sequences[elite_indices]
            
            # Update distribution from elites
            new_mean = elite_actions.mean(axis=0)
            new_std = elite_actions.std(axis=0) + 1e-6  # Prevent collapse
            
            best_return = returns[elite_indices[-1]]
            
            return (new_mean, new_std, key), {"best_return": best_return, "mean_return": returns.mean()}

        # Run CEM iterations
        key, cem_key = jax.random.split(key)
        (final_mean, final_std, _), metrics = jax.lax.scan(
            cem_iteration,
            (mean, std, cem_key),
            None,
            length=config.n_iterations,
        )
        
        # Return first action of the planned sequence
        first_action = jnp.clip(final_mean[0], config.action_low, config.action_high)
        
        info = {
            "planned_sequence": final_mean,
            "planned_std": final_std,
            "best_return_history": metrics["best_return"],
            "mean_return_history": metrics["mean_return"],
        }
        
        return first_action, info


def create_mpc_policy(
    model: IQNTransitionModel,
    reward_fn: Callable[[jax.Array, jax.Array], jax.Array],
    config: MPCConfig,
    constraint_fn: Optional[Callable[[jax.Array], jax.Array]] = None,
) -> Callable[[jax.Array, chex.PRNGKey], jax.Array]:
    """Create a stateless MPC policy function.
    
    :param model: Trained IQN model.
    :param reward_fn: Reward function.
    :param config: MPC configuration.
    :param constraint_fn: Optional constraint function.
    :return: Policy function (state, key) -> action.
    """
    planner = MPCPlanner(model, reward_fn, config, constraint_fn)
    
    def policy(state: jax.Array, key: chex.PRNGKey) -> jax.Array:
        action, _ = planner.plan(state, key)
        return action
    
    return policy
