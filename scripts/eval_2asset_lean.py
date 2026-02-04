"""Lean 2-asset eval - minimal memory footprint."""
import sys; sys.path.insert(0, "/Users/morty/.openclaw/workspace/tinker")
import os; os.environ['JAX_PLATFORM_NAME'] = 'cpu'
import jax; jax.config.update('jax_disable_jit', True)  # Disable JIT to save memory
import jax.numpy as jnp
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flax import nnx
from tinker.iqn_mpc.iqn import IQNStateNetwork, pinball_loss
import optax

print("2-Asset: IQN-MPC (CVaR) vs Markowitz")
print("="*50)

# GARCH data
def garch(n, omega, alpha, beta, mu, seed):
    np.random.seed(seed)
    v, r = np.zeros(n), np.zeros(n)
    v[0] = np.sqrt(omega / max(1-alpha-beta, 0.01))
    for t in range(1, n):
        v[t] = np.sqrt(omega + alpha*(r[t-1]-mu)**2 + beta*v[t-1]**2)
        r[t] = mu + v[t] * np.random.randn()
    return r, v

print("[1/5] Generating data...")
aapl_r, aapl_v = garch(5000, 2e-6, 0.08, 0.90, 0.0004, 42)  # AAPL-like
btc_r, btc_v = garch(5000, 4e-5, 0.12, 0.85, 0.0012, 43)    # BTC-like
print(f"  AAPL: μ={aapl_r.mean()*100:.4f}%, σ={aapl_v.mean()*100:.2f}%")
print(f"  BTC:  μ={btc_r.mean()*100:.4f}%, σ={btc_v.mean()*100:.2f}%")

# Env
class Env:
    def __init__(s): s.ar, s.av, s.br, s.bv = aapl_r, aapl_v, btc_r, btc_v
    def obs(s, t): return np.array([s.av[t], s.bv[t], 0.0004, 0.0012])
    def step(s, t, a, pv):
        w = np.exp(a)/np.exp(a).sum()
        ret = w[1]*s.ar[t] + w[2]*s.br[t]
        return pv*(1+ret), ret
env = Env()

# Collect data
print("[2/5] Collecting...")
S, A, NS = [], [], []
np.random.seed(0)
for _ in range(30):
    st = np.random.randint(0, 4500)
    for t in range(st, st+60):
        S.append(env.obs(t)); A.append(np.random.randn(3)); NS.append(env.obs(t+1))
S, A, NS = jnp.array(S), jnp.array(A), jnp.array(NS)
print(f"  {len(S)} transitions")

# Train IQN (small)
print("[3/5] Training IQN...")
model = IQNStateNetwork(4, 3, 32, 16, 16, rngs=nnx.Rngs(jax.random.PRNGKey(0)))
tx = optax.adam(1e-3)
ms, opt = nnx.state(model), tx.init(nnx.state(model))
for i in range(300):
    idx = np.random.randint(0, len(S), 32)
    tau = np.random.rand(32, 8).astype(np.float32)
    def loss(p): nnx.update(model, p); return pinball_loss(model(S[idx], A[idx], tau), NS[idx], tau)
    l, g = jax.value_and_grad(loss)(ms)
    u, opt = tx.update(g, opt, ms); ms = optax.apply_updates(ms, u)
    if (i+1) % 100 == 0: print(f"  {i+1}/300, loss={float(l):.5f}")
nnx.update(model, ms)

# Policies
def markowitz(obs, lam=2.0):
    av, bv, am, bm = obs
    wa = max(0, am/(lam*av**2+1e-8))
    wb = max(0, bm/(lam*bv**2+1e-8))
    if wa+wb > 1: wa, wb = wa/(wa+wb), wb/(wa+wb)
    return np.log(np.array([1-wa-wb, wa, wb])+1e-8)

def iqn_cvar(obs, cvar_a=0.2):
    obs = jnp.array(obs)
    best_a, best_r = np.zeros(3), -1e9
    for _ in range(15):
        a = np.random.randn(3)
        rets = []
        for _ in range(6):
            tau = np.random.rand(1).astype(np.float32)
            nobs = model(obs, jnp.array(a), tau).squeeze()
            w = np.exp(a)/np.exp(a).sum()
            rets.append(float(w[1]*obs[2] + w[2]*obs[3]))
        cv = np.mean(sorted(rets)[:max(1,int(len(rets)*cvar_a))])
        if cv > best_r: best_r, best_a = cv, a
    return best_a

# Evaluate
print("[4/5] Evaluating...")
def run(pol, nm, n=15, L=50):
    np.random.seed(100); vals = []
    for _ in range(n):
        st, pv, vs = np.random.randint(100, 4500), 1000.0, [1000.0]
        for t in range(st, st+L):
            a = pol(env.obs(t))
            pv, _ = env.step(t, a, pv)
            vs.append(pv)
        vals.append(vs)
    return np.array(vals)

m1 = run(lambda o: markowitz(o, 1.0), "Markowitz λ=1")
m3 = run(lambda o: markowitz(o, 3.0), "Markowitz λ=3")
iq = run(lambda o: iqn_cvar(o, 0.2), "IQN-MPC CVaR")
eq = run(lambda o: np.zeros(3), "Equal")

print("\n" + "="*50)
print("Results")
print("="*50)
def st(v):
    r = (v[:,-1]-v[:,0])/v[:,0]
    cv = np.mean(sorted(r)[:max(1,int(len(r)*0.2))])
    return r.mean()*100, r.std()*100, r.mean()/(r.std()+1e-8), cv*100

for nm, v in [("Markowitz λ=1", m1), ("Markowitz λ=3", m3), ("IQN-MPC CVaR", iq), ("Equal Wt", eq)]:
    m, s, sh, cv = st(v)
    print(f"{nm:<18} {m:>6.2f}% ± {s:<5.2f}%  Sharpe:{sh:<6.3f}  CVaR20%:{cv:>6.2f}%")

# Plot
print("\n[5/5] Plotting...")
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for i in range(5):
    ax[0].plot(m1[i], 'b-', alpha=.3, label='Markowitz λ=1' if i==0 else '')
    ax[0].plot(m3[i], 'c-', alpha=.3, label='Markowitz λ=3' if i==0 else '')
    ax[0].plot(iq[i], 'r-', alpha=.3, label='IQN-MPC CVaR' if i==0 else '')
    ax[0].plot(eq[i], 'g-', alpha=.3, label='Equal Wt' if i==0 else '')
ax[0].legend(); ax[0].set_xlabel('Step'); ax[0].set_ylabel('Portfolio ($)'); ax[0].set_title('Trajectories'); ax[0].grid(True, alpha=.3)
for v, l, c in [(m1, 'Mark λ=1', 'blue'), (m3, 'Mark λ=3', 'cyan'), (iq, 'IQN CVaR', 'red'), (eq, 'Equal', 'green')]:
    ax[1].plot(v.mean(0), c, lw=2, label=l); ax[1].fill_between(range(len(v[0])), v.mean(0)-v.std(0), v.mean(0)+v.std(0), color=c, alpha=.15)
ax[1].legend(); ax[1].set_xlabel('Step'); ax[1].set_ylabel('Portfolio ($)'); ax[1].set_title('Mean ± Std'); ax[1].grid(True, alpha=.3)
plt.tight_layout(); plt.savefig('/Users/morty/.openclaw/workspace/portfolio_2asset_cvar.png', dpi=150)
print("✓ Saved: portfolio_2asset_cvar.png")
