"""2-Asset evaluation: IQN-MPC (CVaR) vs Markowitz (vol penalty).

Assets:
- AAPL-like: lower vol, moderate return (blue chip)
- BTC-like: higher vol, higher return (crypto)
- Cash: risk-free
"""

import sys
sys.path.insert(0, "/Users/morty/.openclaw/workspace/tinker")

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flax import nnx
from tinker.iqn_mpc.iqn import IQNStateNetwork, pinball_loss
import optax

print("="*60)
print("2-Asset Portfolio: IQN-MPC (CVaR) vs Markowitz (Vol Penalty)")
print("="*60)

# ============================================================================
# GARCH Data Generation
# ============================================================================

def gen_garch(n, omega, alpha, beta, mu, seed):
    """Generate GARCH(1,1) returns."""
    np.random.seed(seed)
    vol = np.zeros(n)
    ret = np.zeros(n)
    
    # Unconditional vol
    uncond_var = omega / max(1 - alpha - beta, 0.01)
    vol[0] = np.sqrt(uncond_var)
    
    for t in range(1, n):
        vol[t] = np.sqrt(omega + alpha * (ret[t-1] - mu)**2 + beta * vol[t-1]**2)
        ret[t] = mu + vol[t] * np.random.randn()
    
    return ret, vol

print("\n[1/6] Generating 2-asset GARCH data...")

# AAPL-like: ~15% annual vol, ~10% annual return
# Daily: vol ~0.01, mu ~0.0004
aapl_ret, aapl_vol = gen_garch(
    n=8000, omega=0.000002, alpha=0.08, beta=0.90, mu=0.0004, seed=42
)

# BTC-like: ~60% annual vol, ~30% annual return  
# Daily: vol ~0.04, mu ~0.0012
btc_ret, btc_vol = gen_garch(
    n=8000, omega=0.00004, alpha=0.12, beta=0.85, mu=0.0012, seed=43
)

print(f"  AAPL: mean ret={aapl_ret.mean()*100:.4f}%, mean vol={aapl_vol.mean()*100:.2f}%")
print(f"  BTC:  mean ret={btc_ret.mean()*100:.4f}%, mean vol={btc_vol.mean()*100:.2f}%")

# ============================================================================
# Environment
# ============================================================================

class TwoAssetEnv:
    def __init__(self, aapl_ret, aapl_vol, btc_ret, btc_vol):
        self.aapl_ret = aapl_ret
        self.aapl_vol = aapl_vol
        self.btc_ret = btc_ret
        self.btc_vol = btc_vol
        self.aapl_mu = 0.0004
        self.btc_mu = 0.0012
        self.T = len(aapl_ret)
    
    def obs(self, t):
        """State: [aapl_vol, btc_vol, aapl_mu, btc_mu]"""
        return np.array([
            self.aapl_vol[t], self.btc_vol[t],
            self.aapl_mu, self.btc_mu
        ])
    
    def step(self, t, action, pv):
        """
        action: [cash_logit, aapl_logit, btc_logit]
        Returns: new_pv, reward, portfolio_return
        """
        w = np.exp(action) / np.exp(action).sum()  # softmax weights
        
        # Portfolio return: w_cash*0 + w_aapl*r_aapl + w_btc*r_btc
        port_ret = w[1] * self.aapl_ret[t] + w[2] * self.btc_ret[t]
        
        new_pv = pv * (1 + port_ret)
        reward = np.log(max(new_pv / pv, 1e-10))
        
        return new_pv, reward, port_ret

env = TwoAssetEnv(aapl_ret, aapl_vol, btc_ret, btc_vol)
state_dim = 4  # [aapl_vol, btc_vol, aapl_mu, btc_mu]
action_dim = 3  # [cash, aapl, btc]

# ============================================================================
# Data Collection
# ============================================================================

print("\n[2/6] Collecting transitions...")

states, actions, next_states, rewards = [], [], [], []
np.random.seed(0)

for _ in range(50):
    start = np.random.randint(0, 7000)
    for t in range(start, start + 80):
        states.append(env.obs(t))
        a = np.random.randn(3)
        actions.append(a)
        next_states.append(env.obs(t+1))
        _, r, _ = env.step(t, a, 1000)
        rewards.append(r)

data = {
    "state": jnp.array(states),
    "action": jnp.array(actions),
    "next_state": jnp.array(next_states),
}
print(f"  Collected {len(states)} transitions")

# ============================================================================
# Train IQN
# ============================================================================

print("\n[3/6] Training IQN state model...")

key = jax.random.PRNGKey(0)
key, mk = jax.random.split(key)

model = IQNStateNetwork(
    state_dim=state_dim, action_dim=action_dim,
    hidden_dim=64, embed_dim=32, n_cos=32,
    rngs=nnx.Rngs(mk)
)

tx = optax.adam(1e-3)
ms = nnx.state(model)
opt = tx.init(ms)

losses = []
n_data = len(states)

for i in range(800):
    key, bk, tk = jax.random.split(key, 3)
    idx = jax.random.randint(bk, (64,), 0, n_data)
    tau = jax.random.uniform(tk, (64, 12))
    
    def loss_fn(p):
        nnx.update(model, p)
        pred = model(data["state"][idx], data["action"][idx], tau)
        return pinball_loss(pred, data["next_state"][idx], tau)
    
    l, g = jax.value_and_grad(loss_fn)(ms)
    u, opt = tx.update(g, opt, ms)
    ms = optax.apply_updates(ms, u)
    losses.append(float(l))
    
    if (i+1) % 200 == 0:
        print(f"  Step {i+1}/800, Loss: {l:.6f}")

nnx.update(model, ms)
print(f"  Final loss: {losses[-1]:.6f}")

# ============================================================================
# Policies
# ============================================================================

def markowitz_policy(obs, risk_aversion=2.0):
    """
    Mean-variance optimization with volatility penalty.
    
    max  w'μ - (λ/2) * w'Σw
    
    For diagonal Σ (uncorrelated): w_i ∝ μ_i / (λ * σ_i²)
    """
    aapl_vol, btc_vol, aapl_mu, btc_mu = obs
    
    # Inverse variance weighting with risk aversion
    aapl_var = aapl_vol**2 + 1e-8
    btc_var = btc_vol**2 + 1e-8
    
    w_aapl = aapl_mu / (risk_aversion * aapl_var)
    w_btc = btc_mu / (risk_aversion * btc_var)
    
    # Long-only constraint
    w_aapl = max(0, w_aapl)
    w_btc = max(0, w_btc)
    
    # Cap total risky allocation at 100%
    total = w_aapl + w_btc
    if total > 1.0:
        w_aapl /= total
        w_btc /= total
    
    w_cash = 1.0 - w_aapl - w_btc
    
    # Convert to logits
    weights = np.array([w_cash, w_aapl, w_btc])
    return np.log(weights + 1e-8)


def iqn_mpc_cvar_policy(obs, key, cvar_alpha=0.2, n_samples=30, n_elite=5, horizon=3):
    """
    IQN-MPC with CVaR constraint.
    
    CVaR_α = average of worst α fraction of outcomes.
    α=0.2 means we optimize for the worst 20% of trajectories.
    """
    obs = jnp.array(obs)
    
    # CEM optimization
    mean = np.zeros((horizon, action_dim))
    std = np.ones((horizon, action_dim)) * 0.4
    
    for cem_iter in range(3):
        key, sk = jax.random.split(key)
        
        # Sample action sequences
        noise = np.random.randn(n_samples, horizon, action_dim)
        action_seqs = mean + std * noise
        
        # Evaluate each sequence with multiple trajectory samples
        n_traj_samples = 8  # Sample multiple trajectories per action sequence
        
        all_returns = []
        for j in range(n_samples):
            traj_returns = []
            
            for traj_idx in range(n_traj_samples):
                key, tk = jax.random.split(key)
                curr_obs = obs
                total_ret = 0.0
                
                for t in range(horizon):
                    # Sample quantile for stochastic rollout
                    tau = jax.random.uniform(tk, (1,))
                    key, tk = jax.random.split(key)
                    
                    action = jnp.array(action_seqs[j, t])
                    next_obs = model(curr_obs, action, tau).squeeze()
                    
                    # Compute step reward (expected return)
                    w = jax.nn.softmax(action)
                    aapl_mu, btc_mu = curr_obs[2], curr_obs[3]
                    step_ret = float(w[1] * aapl_mu + w[2] * btc_mu)
                    
                    total_ret += (0.95 ** t) * step_ret
                    curr_obs = next_obs
                
                traj_returns.append(total_ret)
            
            # CVaR: average of worst α fraction
            traj_returns = np.array(traj_returns)
            traj_returns_sorted = np.sort(traj_returns)
            n_cvar = max(1, int(np.ceil(n_traj_samples * cvar_alpha)))
            cvar_return = traj_returns_sorted[:n_cvar].mean()
            
            all_returns.append(cvar_return)
        
        all_returns = np.array(all_returns)
        
        # Select elite (best CVaR returns)
        elite_idx = np.argsort(all_returns)[-n_elite:]
        elite = action_seqs[elite_idx]
        
        mean = elite.mean(axis=0)
        std = elite.std(axis=0) + 0.02
    
    return mean[0]


def equal_weight_policy(obs):
    """1/3 each in cash, AAPL, BTC."""
    return np.array([0.0, 0.0, 0.0])

# ============================================================================
# Evaluation
# ============================================================================

print("\n[4/6] Evaluating policies...")

def run_policy(policy_fn, name, seed, n_ep=20, ep_len=60):
    np.random.seed(seed)
    all_vals = []
    all_rets = []
    
    for ep in range(n_ep):
        start = np.random.randint(100, 7000)
        pv = 1000.0
        vals = [pv]
        rets = []
        
        for t in range(start, start + ep_len):
            obs = env.obs(t)
            
            if 'iqn' in name.lower():
                key = jax.random.PRNGKey(t + ep * 10000)
                a = policy_fn(obs, key)
            else:
                a = policy_fn(obs)
            
            pv, _, port_ret = env.step(t, a, pv)
            vals.append(pv)
            rets.append(port_ret)
        
        all_vals.append(vals)
        all_rets.append(rets)
    
    return np.array(all_vals), np.array(all_rets)

# Run evaluations
print("  Running Markowitz (λ=1.0, low risk aversion)...")
mark_low_v, mark_low_r = run_policy(
    lambda obs: markowitz_policy(obs, risk_aversion=1.0),
    "markowitz_low", seed=200
)

print("  Running Markowitz (λ=3.0, high risk aversion)...")
mark_high_v, mark_high_r = run_policy(
    lambda obs: markowitz_policy(obs, risk_aversion=3.0),
    "markowitz_high", seed=200
)

print("  Running IQN-MPC (CVaR α=0.2, risk-averse)...")
iqn_cvar_v, iqn_cvar_r = run_policy(
    lambda obs, key: iqn_mpc_cvar_policy(obs, key, cvar_alpha=0.2),
    "iqn_cvar", seed=200
)

print("  Running Equal Weight baseline...")
equal_v, equal_r = run_policy(equal_weight_policy, "equal", seed=200)

# ============================================================================
# Results
# ============================================================================

print("\n" + "="*60)
print("Results (60-step episodes, 20 runs)")
print("="*60)

def compute_stats(vals, rets):
    final = vals[:, -1]
    init = vals[:, 0]
    total_ret = (final - init) / init
    
    # Compute per-episode volatility
    ep_vols = [np.std(r) for r in rets]
    
    # Worst 20% of episodes (CVaR proxy)
    sorted_rets = np.sort(total_ret)
    n_worst = max(1, int(len(sorted_rets) * 0.2))
    cvar_20 = sorted_rets[:n_worst].mean()
    
    return {
        "mean_ret": total_ret.mean() * 100,
        "std_ret": total_ret.std() * 100,
        "sharpe": total_ret.mean() / (total_ret.std() + 1e-8),
        "mean_vol": np.mean(ep_vols) * 100,
        "cvar_20": cvar_20 * 100,
        "min_ret": total_ret.min() * 100,
    }

stats = {
    "Markowitz (λ=1)": compute_stats(mark_low_v, mark_low_r),
    "Markowitz (λ=3)": compute_stats(mark_high_v, mark_high_r),
    "IQN-MPC (CVaR 20%)": compute_stats(iqn_cvar_v, iqn_cvar_r),
    "Equal Weight": compute_stats(equal_v, equal_r),
}

print(f"\n{'Strategy':<22} {'Return':<14} {'Sharpe':<8} {'CVaR 20%':<10} {'Worst':<10}")
print("-"*64)
for name, s in stats.items():
    print(f"{name:<22} {s['mean_ret']:>5.2f}% ± {s['std_ret']:<5.2f}% {s['sharpe']:<8.3f} {s['cvar_20']:<10.2f}% {s['min_ret']:<10.2f}%")

# ============================================================================
# Plot
# ============================================================================

print("\n[5/6] Generating plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top left: Sample trajectories
ax1 = axes[0, 0]
colors = {'mark_low': 'blue', 'mark_high': 'cyan', 'iqn': 'red', 'equal': 'green'}
for i in range(min(5, len(mark_low_v))):
    ax1.plot(mark_low_v[i], 'b-', alpha=0.3, label='Markowitz λ=1' if i==0 else '')
    ax1.plot(mark_high_v[i], 'c-', alpha=0.3, label='Markowitz λ=3' if i==0 else '')
    ax1.plot(iqn_cvar_v[i], 'r-', alpha=0.3, label='IQN-MPC CVaR' if i==0 else '')
    ax1.plot(equal_v[i], 'g-', alpha=0.3, label='Equal Wt' if i==0 else '')
ax1.axhline(y=1000, color='k', linestyle='--', alpha=0.3)
ax1.legend(loc='upper left')
ax1.set_xlabel('Step')
ax1.set_ylabel('Portfolio Value ($)')
ax1.set_title('Sample Trajectories')
ax1.grid(True, alpha=0.3)

# Top right: Mean ± Std
ax2 = axes[0, 1]
for v, label, color in [
    (mark_low_v, 'Markowitz λ=1', 'blue'),
    (mark_high_v, 'Markowitz λ=3', 'cyan'),
    (iqn_cvar_v, 'IQN-MPC CVaR', 'red'),
    (equal_v, 'Equal Wt', 'green'),
]:
    m, s = v.mean(0), v.std(0)
    ax2.plot(m, color=color, label=label, lw=2)
    ax2.fill_between(range(len(m)), m-s, m+s, color=color, alpha=0.15)
ax2.axhline(y=1000, color='k', linestyle='--', alpha=0.3)
ax2.legend(loc='upper left')
ax2.set_xlabel('Step')
ax2.set_ylabel('Portfolio Value ($)')
ax2.set_title('Mean ± Std')
ax2.grid(True, alpha=0.3)

# Bottom left: Return distribution
ax3 = axes[1, 0]
data_hist = [
    ((mark_low_v[:,-1] - 1000) / 10, 'Markowitz λ=1', 'blue'),
    ((mark_high_v[:,-1] - 1000) / 10, 'Markowitz λ=3', 'cyan'),
    ((iqn_cvar_v[:,-1] - 1000) / 10, 'IQN-MPC CVaR', 'red'),
    ((equal_v[:,-1] - 1000) / 10, 'Equal Wt', 'green'),
]
for vals, label, color in data_hist:
    ax3.hist(vals, bins=10, alpha=0.5, label=label, color=color)
ax3.axvline(x=0, color='k', linestyle='--', alpha=0.5)
ax3.legend()
ax3.set_xlabel('Total Return (%)')
ax3.set_ylabel('Frequency')
ax3.set_title('Return Distribution')
ax3.grid(True, alpha=0.3)

# Bottom right: Risk-Return tradeoff
ax4 = axes[1, 1]
for name, s in stats.items():
    color = {'Markowitz (λ=1)': 'blue', 'Markowitz (λ=3)': 'cyan', 
             'IQN-MPC (CVaR 20%)': 'red', 'Equal Weight': 'green'}[name]
    ax4.scatter(s['std_ret'], s['mean_ret'], s=100, c=color, label=name, zorder=5)
    ax4.annotate(name.split('(')[0].strip(), (s['std_ret'], s['mean_ret']), 
                 textcoords="offset points", xytext=(5,5), fontsize=8)
ax4.set_xlabel('Volatility (Std of Returns) %')
ax4.set_ylabel('Mean Return %')
ax4.set_title('Risk-Return Tradeoff')
ax4.grid(True, alpha=0.3)
ax4.legend(loc='lower right')

plt.tight_layout()
out_path = '/Users/morty/.openclaw/workspace/portfolio_2asset_cvar.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: {out_path}")

print("\n[6/6] Done!")
print("="*60)
