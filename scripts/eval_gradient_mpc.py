"""Evaluation: Gradient-based MPC vs Markowitz.

Uses direct gradient descent through IQN model instead of CEM.
Includes linear transaction costs as in Lobo-Fazel-Boyd (2002).
"""
import sys; sys.path.insert(0, "/Users/morty/.openclaw/workspace/tinker")
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flax import nnx
from tinker.iqn_mpc.iqn import IQNStateNetwork, pinball_loss
from tinker.iqn_mpc.gradient_mpc import create_gradient_mpc
import optax

print("="*60, flush=True)
print("Gradient-based MPC vs Markowitz (with transaction costs)", flush=True)
print("="*60, flush=True)

# GARCH
def garch(n, omega, alpha, beta, mu, seed):
    np.random.seed(seed)
    v, r = np.zeros(n), np.zeros(n)
    v[0] = np.sqrt(omega / max(1-alpha-beta, 0.01))
    for t in range(1, n):
        v[t] = np.sqrt(omega + alpha*(r[t-1]-mu)**2 + beta*v[t-1]**2)
        r[t] = mu + v[t] * np.random.randn()
    return r, v

print("\n[1/5] Generating data...", flush=True)
aapl_r, aapl_v = garch(5000, 2e-6, 0.08, 0.90, 0.0004, 42)
btc_r, btc_v = garch(5000, 4e-5, 0.12, 0.85, 0.0012, 43)
print(f"  AAPL: μ={aapl_r.mean()*100:.4f}%, σ={aapl_v.mean()*100:.2f}%", flush=True)
print(f"  BTC:  μ={btc_r.mean()*100:.4f}%, σ={btc_v.mean()*100:.2f}%", flush=True)

LINEAR_COST = 0.001  # 0.1% transaction cost

class Env:
    def __init__(s): 
        s.ar, s.av, s.br, s.bv = aapl_r, aapl_v, btc_r, btc_v
        s.am, s.bm = 0.0004, 0.0012
    def obs(s, t): 
        return np.array([s.av[t], s.bv[t], s.am, s.bm])
    def step(s, t, action, pv, prev_w):
        """Step with transaction costs."""
        w = np.exp(action)/np.exp(action).sum()
        
        # Transaction cost
        trade_cost = LINEAR_COST * np.sum(np.abs(w - prev_w)[1:]) * pv
        
        # Portfolio return
        port_ret = w[1]*s.ar[t] + w[2]*s.br[t]
        new_pv = (pv - trade_cost) * (1 + port_ret)
        
        return new_pv, w
env = Env()

print("\n[2/5] Collecting transitions...", flush=True)
S, A, NS = [], [], []
np.random.seed(0)
for _ in range(30):
    st = np.random.randint(0, 4500)
    for t in range(st, st+50):
        S.append(env.obs(t)); A.append(np.random.randn(3)); NS.append(env.obs(t+1))
S, A, NS = jnp.array(S), jnp.array(A), jnp.array(NS)
print(f"  {len(S)} transitions", flush=True)

print("\n[3/5] Training IQN...", flush=True)
model = IQNStateNetwork(4, 3, 64, 32, 32, rngs=nnx.Rngs(jax.random.PRNGKey(0)))
tx = optax.adam(1e-3)
ms = nnx.state(model)
opt = tx.init(ms)

@jax.jit
def train_step(ms, opt, idx, tau):
    def loss_fn(p):
        nnx.update(model, p)
        return pinball_loss(model(S[idx], A[idx], tau), NS[idx], tau)
    l, g = jax.value_and_grad(loss_fn)(ms)
    u, new_opt = tx.update(g, opt, ms)
    return optax.apply_updates(ms, u), new_opt, l

# Warmup
idx = jnp.array(np.random.randint(0, len(S), 64))
tau = jnp.array(np.random.rand(64, 12).astype(np.float32))
ms, opt, _ = train_step(ms, opt, idx, tau)
print("  Compiled training", flush=True)

for i in range(400):
    idx = jnp.array(np.random.randint(0, len(S), 64))
    tau = jnp.array(np.random.rand(64, 12).astype(np.float32))
    ms, opt, l = train_step(ms, opt, idx, tau)
    if (i+1) % 100 == 0: 
        print(f"  {i+1}/400, loss={float(l):.5f}", flush=True)
nnx.update(model, ms)

# Create gradient MPC policy
print("\n  Creating gradient MPC policy...", flush=True)
gradient_mpc_policy = create_gradient_mpc(
    model=model,
    state_dim=4,
    action_dim=3,
    horizon=3,
    n_quantile_samples=8,
    linear_cost_rate=LINEAR_COST,
    variance_penalty=2.0,  # Risk aversion
    cvar_alpha=0.2,
    cvar_penalty=0.5,  # Penalize bad tail outcomes
    lr=0.1,
    n_iters=30,
)

# Warmup compilation
print("  Compiling gradient MPC...", flush=True)
_ = gradient_mpc_policy(jnp.array([0.01, 0.03, 0.0004, 0.0012]), None, jax.random.PRNGKey(0))
print("  Done!", flush=True)

# Markowitz with transaction costs
def markowitz(obs, prev_w, lam=2.0):
    av, bv, am, bm = obs
    
    # Optimal weights without costs
    wa_opt = max(0, am/(lam*av**2+1e-8))
    wb_opt = max(0, bm/(lam*bv**2+1e-8))
    if wa_opt+wb_opt > 1: 
        wa_opt, wb_opt = wa_opt/(wa_opt+wb_opt), wb_opt/(wa_opt+wb_opt)
    
    # Only rebalance if benefit exceeds cost
    # Simple heuristic: don't trade if change < threshold
    threshold = 0.05
    wa = wa_opt if abs(wa_opt - prev_w[1]) > threshold else prev_w[1]
    wb = wb_opt if abs(wb_opt - prev_w[2]) > threshold else prev_w[2]
    
    # Ensure weights sum to 1
    total = wa + wb
    if total > 1:
        wa, wb = wa/total, wb/total
    
    return np.log(np.array([1-wa-wb, wa, wb])+1e-8)

print("\n[4/5] Evaluating...", flush=True)

def run(pol_fn, name, n=12, L=50):
    np.random.seed(200)
    vals, costs = [], []
    
    for ep in range(n):
        st = np.random.randint(100, 4500)
        pv, total_cost = 1000.0, 0.0
        vs = [pv]
        prev_w = np.array([1.0, 0.0, 0.0])  # Start all cash
        
        for t in range(st, st+L):
            obs = env.obs(t)
            action = pol_fn(obs, prev_w, t + ep*1000)
            
            old_pv = pv
            pv, prev_w = env.step(t, action, pv, prev_w)
            total_cost += max(0, old_pv - (pv / (1 + prev_w[1]*env.ar[t] + prev_w[2]*env.br[t])))
            vs.append(pv)
        
        vals.append(vs)
        costs.append(total_cost)
        if (ep+1) % 4 == 0: 
            print(f"  {name}: {ep+1}/{n}", flush=True)
    
    return np.array(vals), np.array(costs)

# Markowitz policies
m1_vals, m1_costs = run(
    lambda o, pw, seed: markowitz(o, pw, lam=1.0), 
    "Mark-1"
)
m3_vals, m3_costs = run(
    lambda o, pw, seed: markowitz(o, pw, lam=3.0), 
    "Mark-3"
)

# Gradient MPC
gm_vals, gm_costs = run(
    lambda o, pw, seed: np.array(gradient_mpc_policy(jnp.array(o), jnp.array(pw), jax.random.PRNGKey(seed))),
    "Grad-MPC"
)

# Equal weight (rebalance each step)
eq_vals, eq_costs = run(
    lambda o, pw, seed: np.zeros(3),
    "Equal"
)

print("\n" + "="*60, flush=True)
print("Results (with 0.1% linear transaction costs)", flush=True)
print("="*60, flush=True)

def st(v, c):
    r = (v[:,-1]-v[:,0])/v[:,0]
    cv = np.mean(sorted(r)[:max(1,int(len(r)*0.2))])
    return r.mean()*100, r.std()*100, r.mean()/(r.std()+1e-8), cv*100, c.mean()

print(f"\n{'Strategy':<18} {'Return':<12} {'Sharpe':<8} {'CVaR20%':<10} {'Avg Cost':<10}", flush=True)
print("-"*60, flush=True)
for nm, v, c in [("Markowitz λ=1", m1_vals, m1_costs), 
                  ("Markowitz λ=3", m3_vals, m3_costs), 
                  ("Gradient MPC", gm_vals, gm_costs), 
                  ("Equal Wt", eq_vals, eq_costs)]:
    m, s, sh, cv, cost = st(v, c)
    print(f"{nm:<18} {m:>5.2f}% ± {s:<4.2f}% {sh:<8.3f} {cv:<10.2f}% ${cost:<10.2f}", flush=True)

print("\n[5/5] Plotting...", flush=True)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for i in range(min(5, len(m1_vals))):
    ax[0].plot(m1_vals[i], 'b-', alpha=.3, label='Mark λ=1' if i==0 else '')
    ax[0].plot(m3_vals[i], 'c-', alpha=.3, label='Mark λ=3' if i==0 else '')
    ax[0].plot(gm_vals[i], 'r-', alpha=.3, label='Grad MPC' if i==0 else '')
    ax[0].plot(eq_vals[i], 'g-', alpha=.3, label='Equal' if i==0 else '')
ax[0].axhline(y=1000, color='k', ls='--', alpha=.3)
ax[0].legend(); ax[0].set_xlabel('Step'); ax[0].set_ylabel('$'); ax[0].set_title('Trajectories'); ax[0].grid(True, alpha=.3)

for v, l, c in [(m1_vals, 'Mark λ=1', 'blue'), (m3_vals, 'Mark λ=3', 'cyan'), 
                (gm_vals, 'Grad MPC', 'red'), (eq_vals, 'Equal', 'green')]:
    ax[1].plot(v.mean(0), c, lw=2, label=l)
    ax[1].fill_between(range(len(v[0])), v.mean(0)-v.std(0), v.mean(0)+v.std(0), color=c, alpha=.12)
ax[1].axhline(y=1000, color='k', ls='--', alpha=.3)
ax[1].legend(); ax[1].set_xlabel('Step'); ax[1].set_ylabel('$'); ax[1].set_title('Mean±Std'); ax[1].grid(True, alpha=.3)

plt.tight_layout()
plt.savefig('/Users/morty/.openclaw/workspace/portfolio_gradient_mpc.png', dpi=150)
print("\n✓ Saved: portfolio_gradient_mpc.png", flush=True)
