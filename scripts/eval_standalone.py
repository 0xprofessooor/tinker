"""Standalone evaluation: IQN-MPC vs Markowitz on synthetic GARCH data.

No safenax dependency - generates GARCH data directly.
"""

import sys
sys.path.insert(0, "/Users/morty/.openclaw/workspace/tinker")

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from flax import nnx
from tinker.iqn_mpc.iqn import IQNStateNetwork, pinball_loss
import optax

print("Starting evaluation...")

# ============================================================================
# GARCH Data Generation (simple, no JIT)
# ============================================================================

def generate_garch_returns(
    n_steps: int,
    omega: float = 0.00001,
    alpha: float = 0.1,
    beta: float = 0.85,
    mu: float = 0.0003,
    seed: int = 42,
):
    """Generate GARCH(1,1) returns using numpy."""
    np.random.seed(seed)
    
    # Unconditional variance
    uncond_var = omega / (1 - alpha - beta)
    
    returns = np.zeros(n_steps)
    volatilities = np.zeros(n_steps)
    volatilities[0] = np.sqrt(uncond_var)
    
    for t in range(1, n_steps):
        # GARCH variance update
        variance = omega + alpha * (returns[t-1] - mu)**2 + beta * volatilities[t-1]**2
        volatilities[t] = np.sqrt(max(variance, 1e-10))
        
        # Return
        returns[t] = mu + volatilities[t] * np.random.randn()
    
    return returns, volatilities


def generate_portfolio_data(n_steps=10000, n_assets=2, seed=42):
    """Generate multi-asset GARCH data."""
    np.random.seed(seed)
    
    # Parameters for different assets (BTC-like, ETH-like)
    params = [
        {"omega": 0.00001, "alpha": 0.10, "beta": 0.85, "mu": 0.0003},  # BTC
        {"omega": 0.000015, "alpha": 0.12, "beta": 0.83, "mu": 0.0004}, # ETH
    ]
    
    all_returns = []
    all_vols = []
    
    for i, p in enumerate(params[:n_assets]):
        ret, vol = generate_garch_returns(n_steps, seed=seed + i, **p)
        all_returns.append(ret)
        all_vols.append(vol)
    
    returns = np.stack(all_returns, axis=1)  # (n_steps, n_assets)
    vols = np.stack(all_vols, axis=1)  # (n_steps, n_assets)
    mus = np.array([p["mu"] for p in params[:n_assets]])  # (n_assets,)
    
    return returns, vols, mus


# ============================================================================
# Portfolio Simulation
# ============================================================================

class PortfolioEnv:
    """Simple portfolio environment."""
    
    def __init__(self, returns, vols, mus, initial_value=1000.0):
        self.returns = returns  # (T, n_assets)
        self.vols = vols        # (T, n_assets)
        self.mus = mus          # (n_assets,)
        self.n_assets = returns.shape[1]
        self.initial_value = initial_value
        self.T = returns.shape[0]
    
    def get_obs(self, t):
        """Observation: current volatilities + mean returns."""
        return np.concatenate([self.vols[t], self.mus])
    
    def step(self, t, weights, portfolio_value):
        """
        Execute one step.
        weights: (n_assets + 1,) with first being cash
        Returns: new_portfolio_value, reward
        """
        # Normalize weights
        weights = np.array(weights)
        weights = np.exp(weights) / np.exp(weights).sum()  # softmax
        
        # Cash earns nothing, assets earn their returns
        asset_returns = self.returns[t]  # (n_assets,)
        
        # Portfolio return
        portfolio_return = np.sum(weights[1:] * asset_returns)
        
        new_value = portfolio_value * (1 + portfolio_return)
        reward = np.log(new_value / portfolio_value)
        
        return new_value, reward


# ============================================================================
# IQN Training
# ============================================================================

def collect_transitions(env, n_episodes=30, episode_length=100, seed=0):
    """Collect transitions with random policy."""
    np.random.seed(seed)
    
    states, actions, next_states, rewards = [], [], [], []
    
    for ep in range(n_episodes):
        start_t = np.random.randint(0, env.T - episode_length - 1)
        
        for t in range(start_t, start_t + episode_length):
            obs = env.get_obs(t)
            action = np.random.randn(env.n_assets + 1)
            next_obs = env.get_obs(t + 1)
            
            states.append(obs)
            actions.append(action)
            next_states.append(next_obs)
            rewards.append(0.0)  # dummy
    
    return {
        "state": jnp.array(np.stack(states)),
        "action": jnp.array(np.stack(actions)),
        "next_state": jnp.array(np.stack(next_states)),
    }


def train_iqn(data, state_dim, action_dim, n_updates=1500, batch_size=64):
    """Train IQN model."""
    key = jax.random.PRNGKey(0)
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
    model_state = nnx.state(model)
    opt_state = tx.init(model_state)
    
    n_samples = data["state"].shape[0]
    losses = []
    
    for i in range(n_updates):
        key, batch_key, tau_key = jax.random.split(key, 3)
        
        idx = jax.random.randint(batch_key, (batch_size,), 0, n_samples)
        batch_s = data["state"][idx]
        batch_a = data["action"][idx]
        batch_ns = data["next_state"][idx]
        tau = jax.random.uniform(tau_key, (batch_size, 8))
        
        def loss_fn(params):
            nnx.update(model, params)
            pred = model(batch_s, batch_a, tau)
            return pinball_loss(pred, batch_ns, tau)
        
        loss, grads = jax.value_and_grad(loss_fn)(model_state)
        updates, opt_state = tx.update(grads, opt_state, model_state)
        model_state = optax.apply_updates(model_state, updates)
        losses.append(float(loss))
        
        if (i + 1) % 300 == 0:
            print(f"  IQN Training: {i+1}/{n_updates}, Loss: {loss:.6f}")
    
    nnx.update(model, model_state)
    return model, losses


# ============================================================================
# Policies
# ============================================================================

def markowitz_policy(obs, risk_aversion=2.0):
    """Mean-variance optimal weights (returns logits)."""
    n_assets = len(obs) // 2
    sigma = obs[:n_assets]
    mu = obs[n_assets:]
    
    # Inverse variance weighting
    var = sigma ** 2 + 1e-8
    raw_w = mu / (risk_aversion * var)
    raw_w = np.maximum(raw_w, 0.0)
    
    # Normalize
    if raw_w.sum() > 1.0:
        raw_w = raw_w / raw_w.sum()
    
    cash_w = max(0.0, 1.0 - raw_w.sum())
    weights = np.concatenate([[cash_w], raw_w])
    
    return np.log(weights + 1e-8)


def iqn_mpc_policy(model, obs, n_samples=20, n_elite=3, horizon=3, n_iters=2):
    """Simple CEM-based MPC with IQN."""
    key = jax.random.PRNGKey(np.random.randint(10000))
    obs = jnp.array(obs)
    n_assets = len(obs) // 2
    action_dim = n_assets + 1
    
    mean = np.zeros((horizon, action_dim))
    std = np.ones((horizon, action_dim)) * 0.3
    
    for _ in range(n_iters):
        key, sample_key = jax.random.split(key)
        actions = mean + std * np.random.randn(n_samples, horizon, action_dim)
        
        # Evaluate
        returns = []
        for j in range(n_samples):
            curr_obs = obs
            total_r = 0.0
            for t in range(horizon):
                key, tau_key = jax.random.split(key)
                tau = jax.random.uniform(tau_key, (1,))
                action = jnp.array(actions[j, t])
                
                next_obs = model(curr_obs, action, tau).squeeze()
                
                # Reward based on expected return
                weights = jax.nn.softmax(action)
                sigma = curr_obs[:n_assets]
                mu = curr_obs[n_assets:]
                r = float(jnp.sum(weights[1:] * mu))
                total_r += (0.95 ** t) * r
                
                curr_obs = next_obs
            
            returns.append(total_r)
        
        returns = np.array(returns)
        elite_idx = np.argsort(returns)[-n_elite:]
        elite = actions[elite_idx]
        
        mean = elite.mean(axis=0)
        std = elite.std(axis=0) + 0.01
    
    return mean[0]


def equal_weight_policy(obs):
    """Equal weight baseline."""
    n_assets = len(obs) // 2
    return np.zeros(n_assets + 1)


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_policy(env, policy_fn, n_episodes=15, episode_length=100, seed=0):
    """Run policy and collect portfolio values."""
    np.random.seed(seed)
    all_values = []
    
    for ep in range(n_episodes):
        start_t = np.random.randint(0, env.T - episode_length - 1)
        
        values = [env.initial_value]
        pv = env.initial_value
        
        for t in range(start_t, start_t + episode_length):
            obs = env.get_obs(t)
            action = policy_fn(obs)
            pv, _ = env.step(t, action, pv)
            values.append(pv)
        
        all_values.append(values)
    
    return np.array(all_values)


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("IQN-MPC vs Markowitz Portfolio Optimization")
    print("=" * 60)
    
    # Generate data
    print("\n[1/5] Generating GARCH data...")
    returns, vols, mus = generate_portfolio_data(n_steps=15000, n_assets=2, seed=42)
    print(f"  Generated {returns.shape[0]} time steps, {returns.shape[1]} assets")
    print(f"  Mean returns: {mus}")
    
    env = PortfolioEnv(returns, vols, mus)
    state_dim = env.n_assets * 2  # vol + mu
    action_dim = env.n_assets + 1  # cash + assets
    
    # Collect transitions
    print("\n[2/5] Collecting transitions...")
    data = collect_transitions(env, n_episodes=40, episode_length=100, seed=0)
    print(f"  Collected {data['state'].shape[0]} transitions")
    
    # Train IQN
    print("\n[3/5] Training IQN...")
    iqn_model, losses = train_iqn(data, state_dim, action_dim, n_updates=1500)
    print(f"  Final loss: {losses[-1]:.6f}")
    
    # Evaluate
    print("\n[4/5] Evaluating policies...")
    
    print("  Running Markowitz...")
    markowitz_values = evaluate_policy(
        env, lambda obs: markowitz_policy(obs, risk_aversion=2.0),
        n_episodes=20, episode_length=100, seed=100
    )
    
    print("  Running IQN-MPC...")
    iqn_values = evaluate_policy(
        env, lambda obs: iqn_mpc_policy(iqn_model, obs),
        n_episodes=20, episode_length=100, seed=100
    )
    
    print("  Running Equal Weight...")
    equal_values = evaluate_policy(
        env, equal_weight_policy,
        n_episodes=20, episode_length=100, seed=100
    )
    
    # Results
    print("\n" + "=" * 60)
    print("Results (100-step episodes)")
    print("=" * 60)
    
    def stats(values):
        final = values[:, -1]
        init = values[:, 0]
        returns = (final - init) / init
        return {
            "mean_return": returns.mean() * 100,
            "std_return": returns.std() * 100,
            "sharpe": returns.mean() / (returns.std() + 1e-8),
            "mean_final": final.mean(),
        }
    
    m_stats = stats(markowitz_values)
    i_stats = stats(iqn_values)
    e_stats = stats(equal_values)
    
    print(f"\nMarkowitz Mean-Variance:")
    print(f"  Return: {m_stats['mean_return']:.2f}% ± {m_stats['std_return']:.2f}%")
    print(f"  Sharpe: {m_stats['sharpe']:.3f}")
    print(f"  Final Value: ${m_stats['mean_final']:.2f}")
    
    print(f"\nIQN-MPC (Distributional Model + Planning):")
    print(f"  Return: {i_stats['mean_return']:.2f}% ± {i_stats['std_return']:.2f}%")
    print(f"  Sharpe: {i_stats['sharpe']:.3f}")
    print(f"  Final Value: ${i_stats['mean_final']:.2f}")
    
    print(f"\nEqual Weight Baseline:")
    print(f"  Return: {e_stats['mean_return']:.2f}% ± {e_stats['std_return']:.2f}%")
    print(f"  Sharpe: {e_stats['sharpe']:.3f}")
    print(f"  Final Value: ${e_stats['mean_final']:.2f}")
    
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
    ax1.set_title('Portfolio Value Trajectories (Sample)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1000, color='k', linestyle='--', alpha=0.3)
    
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
    ax2.axhline(y=1000, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    out_path = '/Users/morty/.openclaw/workspace/portfolio_comparison.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved: {out_path}")
    plt.close()
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
