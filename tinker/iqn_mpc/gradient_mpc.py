"""Gradient-based MPC with IQN dynamics.

Based on Lobo-Fazel-Boyd (2002) portfolio optimization formulation,
but using learned IQN dynamics and gradient descent optimization.

Key differences from CEM:
- Direct gradient descent on action sequence
- More sample-efficient
- Can incorporate soft constraints via penalties
"""

import jax
import jax.numpy as jnp
from flax import nnx
from typing import Callable, Tuple
import optax


def create_gradient_mpc(
    model: nnx.Module,
    state_dim: int,
    action_dim: int,
    horizon: int = 5,
    n_quantile_samples: int = 16,
    linear_cost_rate: float = 0.001,  # 0.1% linear transaction cost
    variance_penalty: float = 1.0,    # Risk aversion parameter (λ)
    cvar_alpha: float = 0.2,          # CVaR level (optimize worst α%)
    cvar_penalty: float = 0.0,        # CVaR penalty weight (0 = off)
    lr: float = 0.1,
    n_iters: int = 50,
) -> Callable:
    """
    Create a gradient-based MPC policy.
    
    Objective (from Lobo-Fazel-Boyd + CVaR extension):
        max  E[return] - linear_costs - λ*Var[return] - β*CVaR_penalty
        
    where:
        - E[return] = Σ w_i * μ_i  (expected portfolio return)
        - linear_costs = c * Σ |Δw_i|  (transaction costs)
        - Var[return] = Σ w_i² * σ_i²  (portfolio variance, diagonal approx)
        - CVaR_penalty = max(0, threshold - CVaR_α)  (tail risk penalty)
    
    :param model: Trained IQN model.
    :param state_dim: Dimension of state (e.g., 4 for [vol_a, vol_b, mu_a, mu_b]).
    :param action_dim: Dimension of action (e.g., 3 for [cash, asset_a, asset_b]).
    :param horizon: Planning horizon.
    :param n_quantile_samples: Number of quantile samples for CVaR estimation.
    :param linear_cost_rate: Linear transaction cost (fraction of trade value).
    :param variance_penalty: Risk aversion parameter λ.
    :param cvar_alpha: CVaR level (e.g., 0.2 for worst 20%).
    :param cvar_penalty: Weight on CVaR violation penalty.
    :param lr: Learning rate for gradient descent.
    :param n_iters: Number of optimization iterations.
    :return: Policy function (obs, prev_weights) -> action.
    """
    
    n_assets = action_dim - 1  # Exclude cash
    
    @jax.jit
    def compute_trajectory_value(
        actions: jnp.ndarray,  # (horizon, action_dim)
        obs: jnp.ndarray,      # (state_dim,)
        prev_weights: jnp.ndarray,  # (action_dim,) - previous portfolio weights
        key: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, dict]:
        """
        Compute expected value of action sequence with costs and risk penalties.
        
        Uses multiple quantile samples to estimate:
        - Expected return (mean over quantiles)
        - CVaR (mean of worst α fraction)
        """
        
        # Sample multiple trajectories using different quantiles
        def single_trajectory(tau_key):
            """Roll out one trajectory with sampled quantiles."""
            curr_obs = obs
            curr_weights = prev_weights
            total_return = 0.0
            total_cost = 0.0
            total_variance = 0.0
            
            for t in range(horizon):
                # Get weights from action logits
                new_weights = jax.nn.softmax(actions[t])
                
                # Transaction costs (linear)
                weight_changes = jnp.abs(new_weights - curr_weights)
                transaction_cost = linear_cost_rate * jnp.sum(weight_changes[1:])  # Exclude cash
                total_cost += transaction_cost
                
                # Extract volatilities and expected returns from observation
                # Assumes obs = [vol_1, vol_2, ..., mu_1, mu_2, ...]
                vols = curr_obs[:n_assets]
                mus = curr_obs[n_assets:]
                
                # Expected return (excluding cash which has 0 return)
                asset_weights = new_weights[1:]
                expected_return = jnp.sum(asset_weights * mus)
                
                # Portfolio variance (diagonal approximation)
                portfolio_variance = jnp.sum((asset_weights ** 2) * (vols ** 2))
                total_variance += portfolio_variance
                
                # Accumulate discounted return
                gamma = 0.99
                total_return += (gamma ** t) * expected_return
                
                # Predict next state using IQN with sampled quantile
                tau = jax.random.uniform(tau_key, (1,))
                tau_key, _ = jax.random.split(tau_key)
                next_obs = model(curr_obs, actions[t], tau).squeeze()
                
                curr_obs = next_obs
                curr_weights = new_weights
            
            return total_return, total_cost, total_variance
        
        # Sample multiple trajectories
        keys = jax.random.split(key, n_quantile_samples)
        results = jax.vmap(single_trajectory)(keys)
        returns, costs, variances = results
        
        # Expected values
        mean_return = jnp.mean(returns)
        mean_cost = jnp.mean(costs)
        mean_variance = jnp.mean(variances)
        
        # CVaR: average of worst α fraction of returns
        sorted_returns = jnp.sort(returns)
        n_cvar = max(1, int(n_quantile_samples * cvar_alpha))
        cvar_value = jnp.mean(sorted_returns[:n_cvar])
        
        # Objective: maximize return - costs - variance_penalty*variance - cvar_penalty*(cvar shortfall)
        # We minimize the negative
        objective = -(
            mean_return 
            - mean_cost 
            - variance_penalty * mean_variance
            - cvar_penalty * jax.nn.relu(-cvar_value)  # Penalize negative CVaR
        )
        
        metrics = {
            "return": mean_return,
            "cost": mean_cost,
            "variance": mean_variance,
            "cvar": cvar_value,
            "objective": -objective,
        }
        
        return objective, metrics
    
    @jax.jit
    def optimize_actions(
        obs: jnp.ndarray,
        prev_weights: jnp.ndarray,
        key: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, dict]:
        """
        Optimize action sequence via gradient descent.
        """
        # Initialize actions (zeros = equal weight after softmax)
        actions = jnp.zeros((horizon, action_dim))
        
        # Optimizer
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(actions)
        
        def step(carry, _):
            actions, opt_state, key = carry
            key, subkey = jax.random.split(key)
            
            # Compute gradient of objective w.r.t. actions
            (loss, metrics), grads = jax.value_and_grad(
                lambda a: compute_trajectory_value(a, obs, prev_weights, subkey),
                has_aux=True
            )(actions)
            
            # Update actions
            updates, new_opt_state = optimizer.update(grads, opt_state, actions)
            new_actions = optax.apply_updates(actions, updates)
            
            return (new_actions, new_opt_state, key), (loss, metrics)
        
        # Run optimization
        (final_actions, _, _), (losses, all_metrics) = jax.lax.scan(
            step, (actions, opt_state, key), None, length=n_iters
        )
        
        return final_actions, all_metrics
    
    def policy(
        obs: jnp.ndarray,
        prev_weights: jnp.ndarray = None,
        key: jax.random.PRNGKey = None,
    ) -> jnp.ndarray:
        """
        MPC policy: optimize and return first action.
        
        :param obs: Current observation.
        :param prev_weights: Previous portfolio weights (for transaction cost calc).
        :param key: Random key.
        :return: Action (logits for softmax weights).
        """
        if prev_weights is None:
            # Default: all cash
            prev_weights = jnp.zeros(action_dim)
            prev_weights = prev_weights.at[0].set(1.0)
        
        if key is None:
            key = jax.random.PRNGKey(0)
        
        optimal_actions, metrics = optimize_actions(obs, prev_weights, key)
        
        # Return first action of the optimized sequence
        return optimal_actions[0]
    
    return policy


def create_gradient_mpc_with_constraints(
    model: nnx.Module,
    state_dim: int,
    action_dim: int,
    horizon: int = 5,
    linear_cost_rate: float = 0.001,
    max_position: float = 0.5,        # Max allocation to any single asset
    min_cash: float = 0.1,            # Minimum cash allocation
    variance_limit: float = 0.01,     # Maximum portfolio variance
    lr: float = 0.1,
    n_iters: int = 50,
) -> Callable:
    """
    Gradient MPC with hard constraints via penalty method.
    
    Constraints (from Lobo-Fazel-Boyd):
    - Position limits: w_i ≤ max_position
    - Minimum cash: w_0 ≥ min_cash
    - Variance limit: Var[r] ≤ variance_limit
    """
    
    n_assets = action_dim - 1
    penalty_weight = 100.0  # Large penalty for constraint violations
    
    @jax.jit
    def compute_penalized_objective(
        actions: jnp.ndarray,
        obs: jnp.ndarray,
        prev_weights: jnp.ndarray,
    ) -> jnp.ndarray:
        """Objective with constraint penalties."""
        
        total_return = 0.0
        total_cost = 0.0
        total_penalty = 0.0
        curr_weights = prev_weights
        curr_obs = obs
        
        vols = obs[:n_assets]
        mus = obs[n_assets:]
        
        for t in range(horizon):
            new_weights = jax.nn.softmax(actions[t])
            
            # Transaction costs
            total_cost += linear_cost_rate * jnp.sum(jnp.abs(new_weights - curr_weights)[1:])
            
            # Expected return
            asset_weights = new_weights[1:]
            total_return += (0.99 ** t) * jnp.sum(asset_weights * mus)
            
            # Constraint penalties
            # Position limit violation
            position_violation = jnp.sum(jax.nn.relu(asset_weights - max_position))
            
            # Minimum cash violation
            cash_violation = jax.nn.relu(min_cash - new_weights[0])
            
            # Variance limit violation
            portfolio_var = jnp.sum((asset_weights ** 2) * (vols ** 2))
            variance_violation = jax.nn.relu(portfolio_var - variance_limit)
            
            total_penalty += penalty_weight * (
                position_violation + cash_violation + variance_violation
            )
            
            # Use median quantile for deterministic gradient
            tau = jnp.array([0.5])
            next_obs = model(curr_obs, actions[t], tau).squeeze()
            curr_obs = next_obs
            curr_weights = new_weights
        
        # Minimize negative objective + penalties
        return -(total_return - total_cost) + total_penalty
    
    @jax.jit
    def optimize(obs, prev_weights):
        actions = jnp.zeros((horizon, action_dim))
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(actions)
        
        def step(carry, _):
            actions, opt_state = carry
            loss, grads = jax.value_and_grad(
                lambda a: compute_penalized_objective(a, obs, prev_weights)
            )(actions)
            updates, new_opt_state = optimizer.update(grads, opt_state, actions)
            return (optax.apply_updates(actions, updates), new_opt_state), loss
        
        (final_actions, _), losses = jax.lax.scan(
            step, (actions, opt_state), None, length=n_iters
        )
        return final_actions
    
    def policy(obs, prev_weights=None):
        if prev_weights is None:
            prev_weights = jnp.array([1.0] + [0.0] * n_assets)
        return optimize(obs, prev_weights)[0]
    
    return policy
