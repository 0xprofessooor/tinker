"""Improved 2-asset eval with better CVaR planning."""
import sys; sys.path.insert(0, "/Users/morty/.openclaw/workspace/tinker")
import os; os.environ['JAX_PLATFORM_NAME'] = 'cpu'
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flax import nnx
from tinker.iqn_mpc.iqn import IQNStateNetwork, pinball_loss
import optax

print("="*60)
print("2-Asset: IMPROVED IQN-MPC (CVaR) vs Markowitz")
print("="*60)

# GARCH data
def garch(n, omega, alpha, beta, mu, seed):
    np.random.seed(seed)
    v, r = np.zeros(n), np.zeros(n)
    v[0] = np.sqrt(omega / max(1-alpha-beta, 0.01))
    for t in range(1, n):
        v[t] = np.sqrt(omega + alpha*(r[t-1]-mu)**2 + beta*v[t-1]**2)
        r[t] = mu + v[t] * np.random.randn()
    return r, v

print("\n[1/5] Generating data...")
aapl_r, aapl_v = garch(5000, 2e-6, 0.08, 0.90, 0.0004, 42)
btc_r, btc_v = garch(5000, 4e-5, 0.12, 0.85, 0.0012, 43)
print(f"  AAPL: μ={aapl_r.mean()*100:.4f}%, σ={aapl_v.mean()*100:.2f}%")
print(f"  BTC:  μ={btc_r.mean()*100:.4f}%, σ={btc_v.mean()*100:.2f}%")

# Env
class Env:
    def __init__(s): 
        s.ar, s.av, s.br, s.bv = aapl_r, aapl_v, btc_r, btc_v
        s.am, s.bm = 0.0004, 0.0012
    def obs(s, t): return np.array([s.av[t], s.bv[t], s.am, s.bm])
    def step(s, t, a, pv):
        w = np.exp(a)/np.exp(a).sum()
        ret = w[1]*s.ar[t] + w[2]*s.br[t]
        return pv*(1+ret), ret
env = Env()

# Collect data
print("\n[2/5] Collecting...")
S, A, NS = [], [], []
np.random.seed(0)
for _ in range(40):
    st = np.random.randint(0, 4500)
    for t in range(st, st+60):
        S.append(env.obs(t)); A.append(np.random.randn(3)); NS.append(env.obs(t+1))
S, A, NS = jnp.array(S), jnp.array(A), jnp.array(NS)
print(f"  {len(S)} transitions")

# Train IQN (bigger model)
print("\n[3/5] Training IQN...")
model = IQNStateNetwork(4, 3, 64, 32, 32, rngs=nnx.Rngs(jax.random.PRNGKey(0)))
tx = optax.adam(1e-3)
ms, opt = nnx.state(model), tx.init(nnx.state(model))

for i in range(500):
    idx = np.random.randint(0, len(S), 64)
    tau = jnp.array(np.random.rand(64, 16).astype(np.float32))
    def loss(p): 
        nnx.update(model, p)
        return pinball_loss(model(S[idx], A[idx], tau), NS[idx], tau)
    l, g = jax.value_and_grad(loss)(ms)
    u, opt = tx.update(g, opt, ms); ms = optax.apply_updates(ms, u)
    if (i+1) % 100 == 0: print(f"  {i+1}/500, loss={float(l):.5f}", flush=True)
nnx.update(model, ms)

# JIT-compiled forward pass for speed
@jax.jit
def model_forward(obs, action, tau):
    return model(obs, action, tau)

# Policies
def markowitz(obs, lam=2.0):
    av, bv, am, bm = obs
    wa = max(0, am/(lam*av**2+1e-8))
    wb = max(0, bm/(lam*bv**2+1e-8))
    if wa+wb > 1: wa, wb = wa/(wa+wb), wb/(wa+wb)
    return np.log(np.array([1-wa-wb, wa, wb])+1e-8)

def iqn_cvar_improved(obs, cvar_alpha=0.2, n_samples=50, n_elite=8, 
                       n_traj=16, horizon=3, n_iters=4):
    """
    Improved CEM with:
    - More samples (50 vs 15)
    - More trajectory samples (16 vs 6)
    - More CEM iterations (4 vs 3)
    - Variance penalty to avoid extreme allocations
    """
    obs_j = jnp.array(obs)
    am, bm = obs[2], obs[3]
    av, bv = obs[0], obs[1]
    
    mean = np.zeros((horizon, 3))
    std = np.ones((horizon, 3)) * 0.3
    
    for cem_it in range(n_iters):
        # Sample action sequences
        noise = np.random.randn(n_samples, horizon, 3)
        actions = mean + std * noise
        
        seq_cvars = []
        for j in range(n_samples):
            traj_rets = []
            for _ in range(n_traj):
                curr = obs_j
                total_ret = 0.0
                
                for t in range(horizon):
                    tau = jnp.array(np.random.rand(1).astype(np.float32))
                    a_j = jnp.array(actions[j, t])
                    
                    # Predict next state
                    next_obs = model_forward(curr, a_j, tau).squeeze()
                    
                    # Compute reward with variance penalty
                    w = np.exp(actions[j, t])
                    w = w / w.sum()
                    
                    # Expected return
                    exp_ret = w[1]*am + w[2]*bm
                    
                    # Variance penalty (penalize BTC overweight)
                    var_penalty = 0.5 * (w[1]**2 * av**2 + w[2]**2 * bv**2)
                    
                    reward = exp_ret - var_penalty
                    total_ret += (0.95 ** t) * reward
                    curr = next_obs
                
                traj_rets.append(total_ret)
            
            # CVaR: average of worst alpha fraction
            traj_rets = np.sort(traj_rets)
            n_worst = max(1, int(np.ceil(n_traj * cvar_alpha)))
            cvar = np.mean(traj_rets[:n_worst])
            seq_cvars.append(cvar)
        
        # Select elites
        seq_cvars = np.array(seq_cvars)
        elite_idx = np.argsort(seq_cvars)[-n_elite:]
        elites = actions[elite_idx]
        
        # Update distribution
        mean = elites.mean(axis=0)
        std = elites.std(axis=0) + 0.05  # Prevent collapse
    
    return mean[0]

def iqn_mean_var(obs, lam=2.0, n_samples=30, n_elite=5, horizon=2):
    """IQN-MPC with mean-variance objective (same as Markowitz but with learned model)."""
    obs_j = jnp.array(obs)
    am, bm = obs[2], obs[3]
    av, bv = obs[0], obs[1]
    
    mean = np.zeros((horizon, 3))
    std = np.ones((horizon, 3)) * 0.3
    
    for _ in range(3):
        actions = mean + std * np.random.randn(n_samples, horizon, 3)
        
        returns = []
        for j in range(n_samples):
            # Use median quantile (τ=0.5) for mean prediction
            tau = jnp.array([0.5])
            curr = obs_j
            total = 0.0
            
            for t in range(horizon):
                a_j = jnp.array(actions[j, t])
                next_obs = model_forward(curr, a_j, tau).squeeze()
                
                w = np.exp(actions[j, t])
                w = w / w.sum()
                
                # Mean-variance objective
                exp_ret = w[1]*am + w[2]*bm
                var = w[1]**2 * av**2 + w[2]**2 * bv**2
                reward = exp_ret - lam * var
                
                total += (0.95 ** t) * reward
                curr = next_obs
            
            returns.append(total)
        
        elite_idx = np.argsort(returns)[-n_elite:]
        elites = actions[elite_idx]
        mean, std = elites.mean(0), elites.std(0) + 0.05
    
    return mean[0]

# Evaluate
print("\n[4/5] Evaluating...", flush=True)
def run(pol, nm, n=15, L=50):
    np.random.seed(200); vals = []
    for ep in range(n):
        st, pv, vs = np.random.randint(100, 4500), 1000.0, [1000.0]
        for t in range(st, st+L):
            a = pol(env.obs(t))
            pv, _ = env.step(t, a, pv)
            vs.append(pv)
        vals.append(vs)
        if (ep+1) % 5 == 0: print(f"  {nm}: {ep+1}/{n} episodes", flush=True)
    return np.array(vals)

m1 = run(lambda o: markowitz(o, 1.0), "Markowitz λ=1")
m3 = run(lambda o: markowitz(o, 3.0), "Markowitz λ=3")
print("  Running IQN-MPC CVaR (improved)...", flush=True)
iq_cvar = run(lambda o: iqn_cvar_improved(o, cvar_alpha=0.2), "IQN-CVaR")
print("  Running IQN-MPC Mean-Var...", flush=True)
iq_mv = run(lambda o: iqn_mean_var(o, lam=2.0), "IQN-MV")
eq = run(lambda o: np.zeros(3), "Equal")

print("\n" + "="*60)
print("Results")
print("="*60)
def st(v):
    r = (v[:,-1]-v[:,0])/v[:,0]
    cv = np.mean(sorted(r)[:max(1,int(len(r)*0.2))])
    return r.mean()*100, r.std()*100, r.mean()/(r.std()+1e-8), cv*100, r.min()*100

print(f"\n{'Strategy':<22} {'Return':<14} {'Sharpe':<8} {'CVaR20%':<10} {'Worst':<10}")
print("-"*64)
for nm, v in [("Markowitz λ=1", m1), ("Markowitz λ=3", m3), 
              ("IQN-MPC CVaR", iq_cvar), ("IQN-MPC Mean-Var", iq_mv), ("Equal Wt", eq)]:
    m, s, sh, cv, mn = st(v)
    print(f"{nm:<22} {m:>5.2f}% ± {s:<5.2f}% {sh:<8.3f} {cv:<10.2f}% {mn:<10.2f}%")

# Plot
print("\n[5/5] Plotting...", flush=True)
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Trajectories
for i in range(min(5, len(m1))):
    ax[0].plot(m1[i], 'b-', alpha=.3, label='Markowitz λ=1' if i==0 else '')
    ax[0].plot(m3[i], 'c-', alpha=.3, label='Markowitz λ=3' if i==0 else '')
    ax[0].plot(iq_cvar[i], 'r-', alpha=.3, label='IQN CVaR' if i==0 else '')
    ax[0].plot(iq_mv[i], 'm-', alpha=.3, label='IQN Mean-Var' if i==0 else '')
    ax[0].plot(eq[i], 'g-', alpha=.3, label='Equal Wt' if i==0 else '')
ax[0].axhline(y=1000, color='k', linestyle='--', alpha=.3)
ax[0].legend(loc='upper left')
ax[0].set_xlabel('Step'); ax[0].set_ylabel('Portfolio ($)')
ax[0].set_title('Sample Trajectories'); ax[0].grid(True, alpha=.3)

# Mean ± Std
for v, l, c in [(m1, 'Mark λ=1', 'blue'), (m3, 'Mark λ=3', 'cyan'), 
                (iq_cvar, 'IQN CVaR', 'red'), (iq_mv, 'IQN M-V', 'magenta'), (eq, 'Equal', 'green')]:
    ax[1].plot(v.mean(0), c, lw=2, label=l)
    ax[1].fill_between(range(len(v[0])), v.mean(0)-v.std(0), v.mean(0)+v.std(0), color=c, alpha=.12)
ax[1].axhline(y=1000, color='k', linestyle='--', alpha=.3)
ax[1].legend(loc='upper left')
ax[1].set_xlabel('Step'); ax[1].set_ylabel('Portfolio ($)')
ax[1].set_title('Mean ± Std'); ax[1].grid(True, alpha=.3)

plt.tight_layout()
plt.savefig('/Users/morty/.openclaw/workspace/portfolio_2asset_improved.png', dpi=150)
print("\n✓ Saved: portfolio_2asset_improved.png")
