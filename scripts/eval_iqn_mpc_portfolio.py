"""Evaluation: IQN-MPC vs Markowitz on GARCH Portfolio Environment.

Compares:
1. IQN-MPC: Model-based planning with distributional state model
2. Markowitz: Classic mean-variance optimization baseline

Outputs portfolio value curves for comparison.
"""

import sys
sys.path.insert(0, "/Users/morty/.openclaw/workspace/tinker")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import nnx, struct

# Direct import to avoid brax dependency from safenax.__init__
sys.path.insert(0, "/Users/morty/.openclaw/workspace/safenax/safenax/portfolio_optimization")
from po_garch import (
    PortfolioOptimizationGARCH,
    GARCHParams,
    EnvState,
)
from tinker.iqn_mpc.iqn import (
    IQNStateNetwork,
    IQNTransitionModel,
    Transition,
    pinball_loss,
)
from tinker.iqn_mpc.mpc import MPCConfig, sample_trajectories
import optax


# ============================================================================
# Environment Setup
# ============================================================================

def create_garch_env(key: jax.Array, num_steps: int = 10000):
    """Create GARCH portfolio environment with 2 assets."""
    # BTC-like parameters (high vol, positive drift)
    btc_params = GARCHParams(
        omega=0.00001,
        alpha=jnp.array([0.1]),
        beta=jnp.array([0.85]),
        mu=0.0003,  # ~10% annual
        initial_price=100.0,
    )
    
    # ETH-like parameters (higher vol, higher drift)
    eth_params = GARCHParams(
        omega=0.000015,
        alpha=jnp.array([0.12]),
        beta=jnp.array([0.83]),
        mu=0.0004,  # ~15% annual
        initial_price=50.0,
    )
    
    garch_params = {"BTC": btc_params, "ETH": eth_params}
    
    env = PortfolioOptimizationGARCH(
        rng=key,
        garch_params=garch_params,
        step_size=1,
        num_samples=num_steps,
        num_trajectories=5,
    )
    
    return env


# ============================================================================
# Data Collection
# ============================================================================

@struct.dataclass
class PortfolioTransition:
    """Extended transition for portfolio optimization."""
    state: jax.Array        # [vol_btc, vol_eth, mu_btc, mu_eth]
    action: jax.Array       # [w_cash, w_btc, w_eth] (logits)
    next_state: jax.Array   # [vol_btc, vol_eth, mu_btc, mu_eth]
    reward: jax.Array       # log return
    portfolio_value: jax.Array


def collect_transitions(env, env_params, key, num_episodes=50, max_steps=200):
    """Collect transitions using random policy."""
    
    def run_episode(key):
        key, reset_key = jax.random.split(key)
        obs, state = env.reset_env(reset_key, env_params)
        
        def step_fn(carry, _):
            state, obs, key, portfolio_values = carry
            key, action_key, step_key = jax.random.split(key, 3)
            
            # Random action (will be softmaxed in env)
            action = jax.random.normal(action_key, (env.num_assets + 1,))
            
            next_obs, next_state, reward, done, info = env.step_env(
                step_key, state, action, env_params
            )
            
            transition = PortfolioTransition(
                state=obs,
                action=action,
                next_state=next_obs,
                reward=reward,
                portfolio_value=next_state.total_value,
            )
            
            return (next_state, next_obs, key, portfolio_values), transition
        
        _, transitions = jax.lax.scan(
            step_fn,
            (state, obs, key, jnp.zeros(max_steps)),
            None,
            length=max_steps,
        )
        return transitions
    
    keys = jax.random.split(key, num_episodes)
    all_transitions = jax.vmap(run_episode)(keys)
    
    # Flatten episodes: (num_episodes, max_steps, ...) -> (num_episodes * max_steps, ...)
    flat_transitions = jax.tree.map(
        lambda x: x.reshape(-1, *x.shape[2:]), all_transitions
    )
    
    return flat_transitions


# ============================================================================
# IQN Training
# ============================================================================

def train_iqn_model(
    transitions: PortfolioTransition,
    key: jax.Array,
    state_dim: int,
    action_dim: int,
    num_updates: int = 5000,
    batch_size: int = 256,
    n_quantiles: int = 32,
    lr: float = 1e-3,
):
    """Train IQN state transition model."""
    
    # Initialize network
    key, model_key = jax.random.split(key)
    model = IQNStateNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        embed_dim=64,
        n_cos=64,
        rngs=nnx.Rngs(model_key),
    )
    graphdef, params = nnx.split(model)
    
    # Optimizer
    tx = optax.adam(learning_rate=lr)
    opt_state = tx.init(params)
    
    n_samples = transitions.state.shape[0]
    
    def update_step(carry, _):
        params, opt_state, key = carry
        key, batch_key, tau_key = jax.random.split(key, 3)
        
        # Sample batch
        indices = jax.random.randint(batch_key, (batch_size,), 0, n_samples)
        batch_state = transitions.state[indices]
        batch_action = transitions.action[indices]
        batch_next_state = transitions.next_state[indices]
        
        # Sample quantiles
        tau = jax.random.uniform(tau_key, (batch_size, n_quantiles))
        
        def loss_fn(params):
            model = nnx.merge(graphdef, params)
            pred = model(batch_state, batch_action, tau)
            return pinball_loss(pred, batch_next_state, tau)
        
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        return (new_params, new_opt_state, key), loss
    
    key, train_key = jax.random.split(key)
    (final_params, _, _), losses = jax.lax.scan(
        update_step, (params, opt_state, train_key), None, length=num_updates
    )
    
    trained_model = IQNTransitionModel(graphdef=graphdef, params=final_params)
    return trained_model, losses


# ============================================================================
# Markowitz Baseline
# ============================================================================

def markowitz_weights(mu: jax.Array, sigma: jax.Array, risk_aversion: float = 1.0):
    """
    Compute Markowitz mean-variance optimal weights.
    
    For simplicity, assume diagonal covariance (no correlations).
    w* = (1/gamma) * Sigma^{-1} @ mu
    
    Then normalize to sum to 1 (long-only, cash gets remainder).
    """
    # Inverse variance weighting
    var = sigma ** 2 + 1e-8
    raw_weights = mu / (risk_aversion * var)
    
    # Clip negative weights (long-only constraint)
    raw_weights = jnp.maximum(raw_weights, 0.0)
    
    # Normalize risky assets to not exceed 100%
    risky_sum = jnp.sum(raw_weights)
    risky_weights = jnp.where(
        risky_sum > 1.0,
        raw_weights / risky_sum,
        raw_weights,
    )
    
    # Cash weight is remainder
    cash_weight = 1.0 - jnp.sum(risky_weights)
    weights = jnp.concatenate([jnp.array([cash_weight]), risky_weights])
    
    return weights


def markowitz_policy(obs: jax.Array, risk_aversion: float = 2.0):
    """
    Markowitz policy based on observation.
    
    Obs format: [vol_btc, vol_eth, mu_btc, mu_eth]
    """
    num_assets = obs.shape[0] // 2
    sigma = obs[:num_assets]
    mu = obs[num_assets:]
    
    weights = markowitz_weights(mu, sigma, risk_aversion)
    
    # Convert to logits (inverse softmax approximation)
    # Since env does softmax, we use log(weights) as action
    logits = jnp.log(weights + 1e-8)
    
    return logits


# ============================================================================
# IQN-MPC Policy
# ============================================================================

def iqn_mpc_policy(
    model: IQNTransitionModel,
    obs: jax.Array,
    key: jax.Array,
    horizon: int = 5,
    n_samples: int = 100,
    n_elite: int = 10,
    n_iterations: int = 3,
    risk_aversion: float = 2.0,
):
    """
    IQN-MPC policy using CEM planning.
    
    Objective: maximize expected portfolio return over horizon,
    with penalty for variance (risk aversion).
    """
    network = nnx.merge(model.graphdef, model.params)
    num_assets = obs.shape[0] // 2
    action_dim = num_assets + 1  # cash + assets
    
    # Initialize action distribution
    mean = jnp.zeros((horizon, action_dim))
    std = jnp.ones((horizon, action_dim)) * 0.5
    
    def evaluate_sequence(key, actions, initial_obs):
        """Roll out action sequence and compute return."""
        
        def step(carry, action):
            obs, key = carry
            key, tau_key = jax.random.split(key)
            
            # Sample quantile for next state prediction
            tau = jax.random.uniform(tau_key, (1,))
            next_obs = network(obs, action, tau).squeeze(0)
            
            # Compute reward (simplified: based on expected return)
            weights = jax.nn.softmax(action)
            sigma = obs[:num_assets]
            mu = obs[num_assets:]
            
            # Expected return minus variance penalty
            expected_return = jnp.sum(weights[1:] * mu)
            variance = jnp.sum((weights[1:] ** 2) * (sigma ** 2))
            reward = expected_return - 0.5 * risk_aversion * variance
            
            return (next_obs, key), reward
        
        (_, _), rewards = jax.lax.scan(step, (initial_obs, key), actions)
        
        # Discounted return
        gamma = 0.99
        discounts = gamma ** jnp.arange(horizon)
        return jnp.sum(rewards * discounts)
    
    def cem_iteration(carry, _):
        mean, std, key = carry
        key, sample_key, eval_key = jax.random.split(key, 3)
        
        # Sample action sequences
        noise = jax.random.normal(sample_key, (n_samples, horizon, action_dim))
        action_sequences = mean + std * noise
        
        # Evaluate all sequences
        eval_keys = jax.random.split(eval_key, n_samples)
        returns = jax.vmap(lambda k, a: evaluate_sequence(k, a, obs))(
            eval_keys, action_sequences
        )
        
        # Select elite
        elite_idx = jnp.argsort(returns)[-n_elite:]
        elite_actions = action_sequences[elite_idx]
        
        # Update distribution
        new_mean = elite_actions.mean(axis=0)
        new_std = elite_actions.std(axis=0) + 0.01
        
        return (new_mean, new_std, key), returns.max()
    
    key, cem_key = jax.random.split(key)
    (final_mean, _, _), _ = jax.lax.scan(
        cem_iteration, (mean, std, cem_key), None, length=n_iterations
    )
    
    # Return first action
    return final_mean[0]


# ============================================================================
# Evaluation Loop
# ============================================================================

def evaluate_policy(env, env_params, policy_fn, key, num_episodes=10, max_steps=200):
    """Evaluate a policy and return portfolio value curves."""
    
    def run_episode(key):
        key, reset_key = jax.random.split(key)
        obs, state = env.reset_env(reset_key, env_params)
        
        initial_value = state.total_value
        
        def step_fn(carry, _):
            state, obs, key = carry
            key, policy_key, step_key = jax.random.split(key, 3)
            
            action = policy_fn(obs, policy_key)
            next_obs, next_state, reward, done, info = env.step_env(
                step_key, state, action, env_params
            )
            
            return (next_state, next_obs, key), next_state.total_value
        
        _, values = jax.lax.scan(
            step_fn, (state, obs, key), None, length=max_steps
        )
        
        # Prepend initial value
        values = jnp.concatenate([jnp.array([initial_value]), values])
        return values
    
    keys = jax.random.split(key, num_episodes)
    all_values = jax.vmap(run_episode)(keys)
    
    return all_values  # (num_episodes, max_steps + 1)


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("IQN-MPC vs Markowitz Portfolio Optimization")
    print("=" * 60)
    
    # Setup
    key = jax.random.PRNGKey(42)
    key, env_key, collect_key, train_key, eval_key = jax.random.split(key, 5)
    
    print("\n[1/5] Creating GARCH environment...")
    env = create_garch_env(env_key, num_steps=50000)
    env_params = env.default_params.replace(max_steps=200)
    
    state_dim = env.num_assets * 2  # vol + mu for each asset
    action_dim = env.num_assets + 1  # cash + assets
    
    print(f"  Assets: {env.asset_names}")
    print(f"  State dim: {state_dim}, Action dim: {action_dim}")
    
    # Collect data
    print("\n[2/5] Collecting transitions with random policy...")
    transitions = collect_transitions(
        env, env_params, collect_key, num_episodes=100, max_steps=200
    )
    print(f"  Collected {transitions.state.shape[0]} transitions")
    
    # Train IQN
    print("\n[3/5] Training IQN model...")
    iqn_model, losses = train_iqn_model(
        transitions, train_key,
        state_dim=state_dim,
        action_dim=action_dim,
        num_updates=3000,
        batch_size=256,
    )
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Loss reduction: {losses[0]:.6f} -> {losses[-1]:.6f}")
    
    # Create policies
    print("\n[4/5] Creating policies...")
    
    def markowitz_policy_fn(obs, key):
        return markowitz_policy(obs, risk_aversion=2.0)
    
    def iqn_mpc_policy_fn(obs, key):
        return iqn_mpc_policy(
            iqn_model, obs, key,
            horizon=5, n_samples=50, n_elite=5, n_iterations=3,
        )
    
    # Buy-and-hold baseline (equal weight, hold forever)
    def buy_hold_policy_fn(obs, key):
        # Equal weight: 1/3 each
        return jnp.zeros(action_dim)  # softmax makes this equal weights
    
    # Evaluate
    print("\n[5/5] Evaluating policies...")
    
    eval_key1, eval_key2, eval_key3 = jax.random.split(eval_key, 3)
    
    markowitz_values = evaluate_policy(
        env, env_params, markowitz_policy_fn, eval_key1, num_episodes=20, max_steps=200
    )
    print(f"  Markowitz evaluated")
    
    iqn_mpc_values = evaluate_policy(
        env, env_params, iqn_mpc_policy_fn, eval_key2, num_episodes=20, max_steps=200
    )
    print(f"  IQN-MPC evaluated")
    
    buy_hold_values = evaluate_policy(
        env, env_params, buy_hold_policy_fn, eval_key3, num_episodes=20, max_steps=200
    )
    print(f"  Buy & Hold evaluated")
    
    # Plot results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    
    # Compute statistics
    def compute_stats(values):
        final_values = values[:, -1]
        returns = (final_values - values[:, 0]) / values[:, 0]
        return {
            "mean_return": float(returns.mean()),
            "std_return": float(returns.std()),
            "sharpe": float(returns.mean() / (returns.std() + 1e-8)),
            "mean_final": float(final_values.mean()),
        }
    
    markowitz_stats = compute_stats(markowitz_values)
    iqn_mpc_stats = compute_stats(iqn_mpc_values)
    buy_hold_stats = compute_stats(buy_hold_values)
    
    print(f"\nMarkowitz:")
    print(f"  Mean return: {markowitz_stats['mean_return']*100:.2f}%")
    print(f"  Sharpe ratio: {markowitz_stats['sharpe']:.3f}")
    
    print(f"\nIQN-MPC:")
    print(f"  Mean return: {iqn_mpc_stats['mean_return']*100:.2f}%")
    print(f"  Sharpe ratio: {iqn_mpc_stats['sharpe']:.3f}")
    
    print(f"\nBuy & Hold (Equal Weight):")
    print(f"  Mean return: {buy_hold_stats['mean_return']*100:.2f}%")
    print(f"  Sharpe ratio: {buy_hold_stats['sharpe']:.3f}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Individual trajectories
    ax1 = axes[0]
    for i in range(min(5, markowitz_values.shape[0])):
        ax1.plot(markowitz_values[i], 'b-', alpha=0.3, label='Markowitz' if i == 0 else '')
        ax1.plot(iqn_mpc_values[i], 'r-', alpha=0.3, label='IQN-MPC' if i == 0 else '')
        ax1.plot(buy_hold_values[i], 'g-', alpha=0.3, label='Buy & Hold' if i == 0 else '')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Portfolio Value')
    ax1.set_title('Portfolio Value Trajectories (Sample)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Mean with std bands
    ax2 = axes[1]
    steps = jnp.arange(markowitz_values.shape[1])
    
    for values, label, color in [
        (markowitz_values, 'Markowitz', 'blue'),
        (iqn_mpc_values, 'IQN-MPC', 'red'),
        (buy_hold_values, 'Buy & Hold', 'green'),
    ]:
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        ax2.plot(steps, mean, color=color, label=label, linewidth=2)
        ax2.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Portfolio Value')
    ax2.set_title('Portfolio Value (Mean ± Std)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/morty/.openclaw/workspace/portfolio_comparison.png', dpi=150)
    print(f"\n✓ Plot saved to: /Users/morty/.openclaw/workspace/portfolio_comparison.png")
    plt.close()


if __name__ == "__main__":
    main()
