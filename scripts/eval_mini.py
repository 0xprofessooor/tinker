"""Minimal evaluation: IQN-MPC vs Markowitz."""

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

print("Starting...")

# Generate simple GARCH-like data
def gen_data(n=5000, seed=42):
    np.random.seed(seed)
    vol = np.zeros(n)
    ret = np.zeros(n)
    vol[0] = 0.02
    
    omega, alpha, beta, mu = 0.00001, 0.1, 0.85, 0.0003
    
    for t in range(1, n):
        vol[t] = np.sqrt(omega + alpha * (ret[t-1] - mu)**2 + beta * vol[t-1]**2)
        ret[t] = mu + vol[t] * np.random.randn()
    
    return ret, vol, np.array([mu])

print("[1/5] Generating data...")
ret, vol, mu = gen_data(5000)
print(f"  Generated {len(ret)} steps")

# Simple env
class SimpleEnv:
    def __init__(self, ret, vol, mu):
        self.ret, self.vol, self.mu = ret, vol, mu
    
    def obs(self, t):
        return np.array([self.vol[t], self.mu[0]])
    
    def step(self, t, w, pv):
        w = np.exp(w) / np.exp(w).sum()  # softmax
        r = w[1] * self.ret[t]  # asset return
        return pv * (1 + r)

env = SimpleEnv(ret, vol, mu)
state_dim, action_dim = 2, 2  # [vol, mu], [cash, asset]

# Collect data
print("[2/5] Collecting transitions...")
states, actions, next_states = [], [], []
np.random.seed(0)
for _ in range(30):
    start = np.random.randint(0, 4500)
    for t in range(start, start + 100):
        states.append(env.obs(t))
        actions.append(np.random.randn(2))
        next_states.append(env.obs(t+1))

data = {
    "state": jnp.array(states),
    "action": jnp.array(actions),
    "next_state": jnp.array(next_states),
}
print(f"  Collected {len(states)} transitions")

# Train IQN
print("[3/5] Training IQN...")
key = jax.random.PRNGKey(0)
key, mk = jax.random.split(key)

model = IQNStateNetwork(
    state_dim=state_dim, action_dim=action_dim,
    hidden_dim=32, embed_dim=16, n_cos=16,
    rngs=nnx.Rngs(mk)
)

tx = optax.adam(1e-3)
ms = nnx.state(model)
opt = tx.init(ms)

losses = []
for i in range(500):
    key, bk, tk = jax.random.split(key, 3)
    idx = jax.random.randint(bk, (64,), 0, len(states))
    tau = jax.random.uniform(tk, (64, 8))
    
    def loss_fn(p):
        nnx.update(model, p)
        pred = model(data["state"][idx], data["action"][idx], tau)
        return pinball_loss(pred, data["next_state"][idx], tau)
    
    l, g = jax.value_and_grad(loss_fn)(ms)
    u, opt = tx.update(g, opt, ms)
    ms = optax.apply_updates(ms, u)
    losses.append(float(l))
    if (i+1) % 100 == 0:
        print(f"  Step {i+1}/500, Loss: {l:.6f}")

nnx.update(model, ms)
print(f"  Final loss: {losses[-1]:.6f}")

# Policies
def markowitz(obs):
    vol, mu = obs[0], obs[1]
    w = mu / (2.0 * vol**2 + 1e-8)
    w = max(0, min(1, w))
    return np.array([np.log(1-w + 1e-8), np.log(w + 1e-8)])

def iqn_mpc(obs, key):
    obs = jnp.array(obs)
    best_a = np.zeros(2)
    best_r = -1e9
    
    for _ in range(20):
        key, sk = jax.random.split(key)
        a = jax.random.normal(sk, (2,))
        
        # Simple 1-step lookahead
        tau = jax.random.uniform(sk, (1,))
        next_obs = model(obs, a, tau).squeeze()
        
        w = jax.nn.softmax(a)
        r = float(w[1] * obs[1])  # expected return
        
        if r > best_r:
            best_r = r
            best_a = np.array(a)
    
    return best_a

def equal_weight(obs):
    return np.array([0.0, 0.0])

# Evaluate
print("[4/5] Evaluating...")

def run_policy(policy_fn, seed, n_ep=15, ep_len=50):
    np.random.seed(seed)
    all_vals = []
    
    for ep in range(n_ep):
        start = np.random.randint(0, 4000)
        pv = 1000.0
        vals = [pv]
        
        for t in range(start, start + ep_len):
            obs = env.obs(t)
            if policy_fn.__name__ == 'iqn_mpc':
                key = jax.random.PRNGKey(t + ep * 1000)
                a = policy_fn(obs, key)
            else:
                a = policy_fn(obs)
            pv = env.step(t, a, pv)
            vals.append(pv)
        
        all_vals.append(vals)
    
    return np.array(all_vals)

mark_v = run_policy(markowitz, 100)
print("  Markowitz done")
iqn_v = run_policy(iqn_mpc, 100)
print("  IQN-MPC done")
eq_v = run_policy(equal_weight, 100)
print("  Equal weight done")

# Results
print("\n" + "="*50)
print("Results (50-step episodes)")
print("="*50)

def stats(v):
    r = (v[:,-1] - v[:,0]) / v[:,0]
    return r.mean()*100, r.std()*100, r.mean()/(r.std()+1e-8)

m_r, m_s, m_sh = stats(mark_v)
i_r, i_s, i_sh = stats(iqn_v)
e_r, e_s, e_sh = stats(eq_v)

print(f"\nMarkowitz: {m_r:.2f}% ± {m_s:.2f}%, Sharpe: {m_sh:.3f}")
print(f"IQN-MPC:   {i_r:.2f}% ± {i_s:.2f}%, Sharpe: {i_sh:.3f}")
print(f"Equal Wt:  {e_r:.2f}% ± {e_s:.2f}%, Sharpe: {e_sh:.3f}")

# Plot
print("\n[5/5] Plotting...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax1 = axes[0]
for i in range(min(5, len(mark_v))):
    ax1.plot(mark_v[i], 'b-', alpha=0.3, label='Markowitz' if i==0 else '')
    ax1.plot(iqn_v[i], 'r-', alpha=0.3, label='IQN-MPC' if i==0 else '')
    ax1.plot(eq_v[i], 'g-', alpha=0.3, label='Equal Wt' if i==0 else '')
ax1.legend()
ax1.set_xlabel('Step')
ax1.set_ylabel('Portfolio Value ($)')
ax1.set_title('Sample Trajectories')
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
for v, l, c in [(mark_v, 'Markowitz', 'blue'), (iqn_v, 'IQN-MPC', 'red'), (eq_v, 'Equal Wt', 'green')]:
    m, s = v.mean(0), v.std(0)
    ax2.plot(m, c, label=l, lw=2)
    ax2.fill_between(range(len(m)), m-s, m+s, color=c, alpha=0.15)
ax2.legend()
ax2.set_xlabel('Step')
ax2.set_ylabel('Portfolio Value ($)')
ax2.set_title('Mean ± Std')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/morty/.openclaw/workspace/portfolio_comparison.png', dpi=150)
print(f"\n✓ Saved: /Users/morty/.openclaw/workspace/portfolio_comparison.png")
