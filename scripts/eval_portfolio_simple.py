"""Simplified evaluation: IQN-MPC vs Markowitz on GARCH Portfolio.

Non-JIT version for faster iteration.
"""

import sys
sys.path.insert(0, "/Users/morty/.openclaw/workspace/tinker")
sys.path.insert(0, "/Users/morty/.openclaw/workspace/safenax/safenax/portfolio_optimization")

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from flax import nnx
from po_garch import PortfolioOptimizationGARCH, GARCHParams
from tinker.iqn_mpc.iqn import IQNStateNetwork, pinball_loss
import optax

print("Imports OK")

# ============================================================================
# Environment Setup  
# ============================================================================

def create_env(key, num_steps=20000):
    btc = GARCHParams(
        omega=0.00001,
        alpha=jnp.array([0.1]),
        beta=jnp.array([0.85]),
        mu=0.0003,
        initial_price=100.0,
    )
    eth = GARCHParams(
        omega=0.000015,
        alpha=jnp.array([0.12]),
        beta=jnp.array([0.83]),
        mu=0.0004,
        initial_price=50.0,
    )
    
    env = PortfolioOptimizationGARCH(
        rng=key,
        garch_params={"BTC": btc, "ETH": eth},
        step_size=1,
        num_samples=num_steps,
        num_trajectories=3,
    )
    return env


# ============================================================================
# Data Collection (Python loop, no JIT)
# ============================================================================

def collect_data(env, env_params, key, n_episodes=30, max_steps=100):
    """Collect transitions using random policy."""
    states, actions, next_states, rewards = [], [], [], []
    
    for ep in range(n_episodes):
        key, reset_key = jax.random.split(key)
        obs, state = env.reset_env(reset_key, env_params)
        
        for t in range(max_steps):
            key, action_key, step_key = jax.random.split(key, 3)
            action = jax.random.normal(action_key, (env.num_assets + 1,))
            
            next_obs, next_state, reward, done, _ = env.step_env(
                step_key, state, action, env_params
            )
            
            states.append(np.array(obs))
            actions.append(np.array(action))
            next_states.append(np.array(next_obs))
            rewards.append(float(reward))
            
            state, obs = next_state, next_obs
    
    return {
        "state": jnp.array(states),
        "action": jnp.array(actions),
        "next_state": jnp.array(next_states),
        "reward": jnp.array(rewards),
    }


# ============================================================================
# IQN Training (simplified)
# ============================================================================

def train_iqn(data, key, state_dim, action_dim, n_updates=2000, batch_size=128):
    """Train IQN with simple loop."""
    key, model_key = jax.random.split(key)
    
    model = IQNStateNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=64,
        embed_dim=32,
        n_cos=32,
        rngs=nnx.Rngs(model_key),
    )
    
    tx = optax.adam(1e-3)
    opt_state = tx.init(nnx.state(model))
    
    n_samples = data["state"].shape[0]
    losses = []
    
    @jax.jit
    def update_step(model_state, opt_state, batch_state, batch_action, batch_next, tau):
        def loss_fn(params):
            nnx.update(model, params)
            pred = model(batch_state, batch_action, tau)
            return pinball_loss(pred, batch_next, tau)
        
        loss, grads = jax.value_and_grad(loss_fn)(model_state)
        updates, new_opt_state = tx.update(grads, opt_state, model_state)
        new_params = optax.apply_updates(model_state, updates)
        return new_params, new_opt_state, loss
    
    model_state = nnx.state(model)
    
    for i in range(n_updates):
        key, batch_key, tau_key = jax.random.split(key, 3)
        
        idx = jax.random.randint(batch_key, (batch_size,), 0, n_samples)
        batch_state = data["state"][idx]
        batch_action = data["action"][idx]
        batch_next = data["next_state"][idx]
        tau = jax.random.uniform(tau_key, (batch_size, 16))
        
        model_state, opt_state, loss = update_step(
            model_state, opt_state, batch_state, batch_action, batch_next, tau
        )
        losses.append(float(loss))
        
        if (i + 1) % 500 == 0:
            print(f"  Step {i+1}/{n_updates}, Loss: {loss:.6f}")
    
    nnx.update(model, model_state)
    return model, losses


# ============================================================================
# Policies
# ============================================================================

def markowitz_policy(obs, risk_aversion=2.0):
    """Mean-variance optimal weights."""
    num_assets = obs.shape[0] // 2
    sigma = obs[:num_assets]
    mu = obs[num_assets:]
    
    var = sigma ** 2 + 1e-8
    raw_weights = mu / (risk_aversion * var)
    raw_weights = jnp.maximum(raw_weights, 0.0)
    
    risky_sum = jnp.sum(raw_weights)
    risky_weights = jnp.where(risky_sum > 1.0, raw_weights / risky_sum, raw_weights)
    cash_weight = 1.0 - jnp.sum(risky_weights)
    
    weights = jnp.concatenate([jnp.array([cash_weight]), risky_weights])
    return jnp.log(weights + 1e-8)  # logits


def iqn_mpc_policy(model, obs, key, horizon=3, n_samples=30, n_elite=5, n_iters=2):
    """Simple CEM-based MPC."""
    num_assets = obs.shape[0] // 2
    action_dim = num_assets + 1
    
    mean = jnp.zeros((horizon, action_dim))
    std = jnp.ones((horizon, action_dim)) * 0.3
    
    for _ in range(n_iters):
        key, sample_key = jax.random.split(key)
        noise = jax.random.normal(sample_key, (n_samples, horizon, action_dim))
        actions = mean + std * noise
        
        # Evaluate each sequence
        returns = []
        for j in range(n_samples):
            curr_obs = obs
            total_return = 0.0
            for t in range(horizon):
                key, tau_key = jax.random.split(key)
                tau = jax.random.uniform(tau_key, (1,))
                
                # Predict next state
                next_obs = model(curr_obs, actions[j, t], tau).squeeze()
                
                # Simple reward: expected return from weights
                weights = jax.nn.softmax(actions[j, t])
                sigma = curr_obs[:num_assets]
                mu = curr_obs[num_assets:]
                reward = float(jnp.sum(weights[1:] * mu))
                
                total_return += (0.99 ** t) * reward
                curr_obs = next_obs
            
            returns.append(total_return)
        
        returns = jnp.array(returns)
        elite_idx = jnp.argsort(returns)[-n_elite:]
        elite = actions[elite_idx]
        
        mean = elite.mean(axis=0)
        std = elite.std(axis=0) + 0.01
    
    return mean[0]


def equal_weight_policy(obs):
    """Equal weight baseline."""
    return jnp.zeros(obs.shape[0] // 2 + 1)


# ============================================================================
# Evaluation
# ============================================================================

def run_policy(env, env_params, policy_fn, key, n_episodes=10, max_steps=100):
    """Run policy and collect portfolio values."""
    all_values = []
    
    for ep in range(n_episodes):
        key, reset_key = jax.random.split(key)
        obs, state = env.reset_env(reset_key, env_params)
        
        values = [float(state.total_value)]
        
        for t in range(max_steps):
            key, policy_key, step_key = jax.random.split(key, 3)
            action = policy_fn(obs, policy_key)
            
            next_obs, next_state, reward, done, _ = env.step_env(
                step_key, state, action, env_params
            )
            
            values.append(float(next_state.total_value))
            state, obs = next_state, next_obs
        
        all_values.append(values)
    
    return np.array(all_values)


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("IQN-MPC vs Markowitz Portfolio Optimization")
    print("=" * 60)
    
    key = jax.random.PRNGKey(42)
    key, env_key, data_key, train_key, eval_key = jax.random.split(key, 5)
    
    # Create environment
    print("\n[1/5] Creating environment...")
    env = create_env(env_key, num_steps=30000)
    env_params = env.default_params.replace(max_steps=100)
    
    state_dim = env.num_assets * 2
    action_dim = env.num_assets + 1
    print(f"  Assets: {env.asset_names}")
    print(f"  State dim: {state_dim}, Action dim: {action_dim}")
    
    # Collect data
    print("\n[2/5] Collecting data...")
    data = collect_data(env, env_params, data_key, n_episodes=50, max_steps=100)
    print(f"  Collected {data['state'].shape[0]} transitions")
    
    # Train IQN
    print("\n[3/5] Training IQN...")
    iqn_model, losses = train_iqn(
        data, train_key, state_dim, action_dim, n_updates=2000, batch_size=128
    )
    print(f"  Final loss: {losses[-1]:.6f}")
    
    # Evaluate policies
    print("\n[4/5] Evaluating policies...")
    
    eval_key1, eval_key2, eval_key3 = jax.random.split(eval_key, 3)
    
    print("  Running Markowitz...")
    markowitz_values = run_policy(
        env, env_params,
        lambda obs, key: markowitz_policy(obs),
        eval_key1, n_episodes=15, max_steps=100
    )
    
    print("  Running IQN-MPC...")
    iqn_values = run_policy(
        env, env_params,
        lambda obs, key: iqn_mpc_policy(iqn_model, obs, key),
        eval_key2, n_episodes=15, max_steps=100
    )
    
    print("  Running Equal Weight...")
    equal_values = run_policy(
        env, env_params,
        lambda obs, key: equal_weight_policy(obs),
        eval_key3, n_episodes=15, max_steps=100
    )
    
    # Results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    
    def stats(values):
        final = values[:, -1]
        init = values[:, 0]
        returns = (final - init) / init
        return {
            "mean_return": returns.mean() * 100,
            "std_return": returns.std() * 100,
            "sharpe": returns.mean() / (returns.std() + 1e-8),
        }
    
    m_stats = stats(markowitz_values)
    i_stats = stats(iqn_values)
    e_stats = stats(equal_values)
    
    print(f"\nMarkowitz Mean-Variance:")
    print(f"  Return: {m_stats['mean_return']:.2f}% ± {m_stats['std_return']:.2f}%")
    print(f"  Sharpe: {m_stats['sharpe']:.3f}")
    
    print(f"\nIQN-MPC:")
    print(f"  Return: {i_stats['mean_return']:.2f}% ± {i_stats['std_return']:.2f}%")
    print(f"  Sharpe: {i_stats['sharpe']:.3f}")
    
    print(f"\nEqual Weight (Baseline):")
    print(f"  Return: {e_stats['mean_return']:.2f}% ± {e_stats['std_return']:.2f}%")
    print(f"  Sharpe: {e_stats['sharpe']:.3f}")
    
    # Plot
    print("\n[5/5] Generating plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sample trajectories
    ax1 = axes[0]
    for i in range(min(5, len(markowitz_values))):
        ax1.plot(markowitz_values[i], 'b-', alpha=0.4, label='Markowitz' if i == 0 else '')
        ax1.plot(iqn_values[i], 'r-', alpha=0.4, label='IQN-MPC' if i == 0 else '')
        ax1.plot(equal_values[i], 'g-', alpha=0.4, label='Equal Weight' if i == 0 else '')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_title('Portfolio Value Trajectories')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Mean ± std
    ax2 = axes[1]
    steps = np.arange(markowitz_values.shape[1])
    
    for values, label, color in [
        (markowitz_values, 'Markowitz', 'blue'),
        (iqn_values, 'IQN-MPC', 'red'),
        (equal_values, 'Equal Weight', 'green'),
    ]:
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        ax2.plot(steps, mean, color=color, label=label, linewidth=2)
        ax2.fill_between(steps, mean - std, mean + std, color=color, alpha=0.15)
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Portfolio Value ($)')
    ax2.set_title('Portfolio Value (Mean ± Std)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = '/Users/morty/.openclaw/workspace/portfolio_comparison.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
