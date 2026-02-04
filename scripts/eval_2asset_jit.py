"""2-asset eval WITH JIT enabled."""
import sys; sys.path.insert(0, "/Users/morty/.openclaw/workspace/tinker")
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flax import nnx
from tinker.iqn_mpc.iqn import IQNStateNetwork, pinball_loss
import optax

print("="*60, flush=True)
print("2-Asset: IQN-MPC vs Markowitz (JIT ENABLED)", flush=True)
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

class Env:
    def __init__(s): 
        s.ar, s.av, s.br, s.bv = aapl_r, aapl_v, btc_r, btc_v
        s.am, s.bm = 0.0004, 0.0012
    def obs(s, t): return np.array([s.av[t], s.bv[t], s.am, s.bm])
    def step(s, t, a, pv):
        w = np.exp(a)/np.exp(a).sum()
        return pv*(1 + w[1]*s.ar[t] + w[2]*s.br[t])
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

# JIT compile the training step
@jax.jit
def train_step(ms, opt, idx, tau):
    def loss_fn(p):
        nnx.update(model, p)
        pred = model(S[idx], A[idx], tau)
        return pinball_loss(pred, NS[idx], tau)
    l, g = jax.value_and_grad(loss_fn)(ms)
    u, new_opt = tx.update(g, opt, ms)
    return optax.apply_updates(ms, u), new_opt, l

print("  Compiling training step...", flush=True)
# Warmup compilation
idx = jnp.array(np.random.randint(0, len(S), 64))
tau = jnp.array(np.random.rand(64, 12).astype(np.float32))
ms, opt, _ = train_step(ms, opt, idx, tau)
print("  Compiled! Training...", flush=True)

for i in range(400):
    idx = jnp.array(np.random.randint(0, len(S), 64))
    tau = jnp.array(np.random.rand(64, 12).astype(np.float32))
    ms, opt, l = train_step(ms, opt, idx, tau)
    if (i+1) % 100 == 0: 
        print(f"  {i+1}/400, loss={float(l):.5f}", flush=True)
nnx.update(model, ms)

# JIT compile forward pass
@jax.jit
def model_forward(obs, action, tau):
    return model(obs, action, tau)

print("  Compiling forward pass...", flush=True)
_ = model_forward(jnp.ones(4), jnp.ones(3), jnp.array([0.5]))
print("  Done!", flush=True)

# Policies
def markowitz(obs, lam=2.0):
    av, bv, am, bm = obs
    wa = max(0, am/(lam*av**2+1e-8))
    wb = max(0, bm/(lam*bv**2+1e-8))
    if wa+wb > 1: wa, wb = wa/(wa+wb), wb/(wa+wb)
    return np.log(np.array([1-wa-wb, wa, wb])+1e-8)

def iqn_cvar(obs, cvar_alpha=0.2):
    obs_j = jnp.array(obs)
    am, bm, av, bv = obs[2], obs[3], obs[0], obs[1]
    
    mean = np.zeros((2, 3))  # horizon=2
    std = np.ones((2, 3)) * 0.3
    
    for _ in range(3):  # CEM iterations
        actions = mean + std * np.random.randn(40, 2, 3)
        cvars = []
        
        for j in range(40):
            rets = []
            for _ in range(12):  # trajectory samples
                curr = obs_j
                total = 0.0
                for t in range(2):
                    tau = jnp.array([np.random.rand()])
                    nxt = model_forward(curr, jnp.array(actions[j,t]), tau).squeeze()
                    w = np.exp(actions[j,t]); w = w/w.sum()
                    total += (0.95**t) * (w[1]*am + w[2]*bm - 0.3*(w[1]**2*av**2 + w[2]**2*bv**2))
                    curr = nxt
                rets.append(total)
            rets = np.sort(rets)
            cvars.append(np.mean(rets[:max(1,int(len(rets)*cvar_alpha))]))
        
        elite_idx = np.argsort(cvars)[-6:]
        elites = actions[elite_idx]
        mean, std = elites.mean(0), elites.std(0) + 0.05
    
    return mean[0]

print("\n[4/5] Evaluating...", flush=True)
def run(pol, nm, n=12, L=50):
    np.random.seed(200); vals = []
    for ep in range(n):
        st, pv, vs = np.random.randint(100, 4500), 1000.0, [1000.0]
        for t in range(st, st+L):
            pv = env.step(t, pol(env.obs(t)), pv)
            vs.append(pv)
        vals.append(vs)
        if (ep+1) % 4 == 0: print(f"  {nm}: {ep+1}/{n}", flush=True)
    return np.array(vals)

m1 = run(lambda o: markowitz(o, 1.0), "Mark-1")
m3 = run(lambda o: markowitz(o, 3.0), "Mark-3")
iq = run(lambda o: iqn_cvar(o), "IQN-CVaR")
eq = run(lambda o: np.zeros(3), "Equal")

print("\n" + "="*60, flush=True)
print("Results", flush=True)
print("="*60, flush=True)

def st(v):
    r = (v[:,-1]-v[:,0])/v[:,0]
    cv = np.mean(sorted(r)[:max(1,int(len(r)*0.2))])
    return r.mean()*100, r.std()*100, r.mean()/(r.std()+1e-8), cv*100

print(f"\n{'Strategy':<18} {'Return':<12} {'Sharpe':<8} {'CVaR20%':<10}", flush=True)
print("-"*50, flush=True)
for nm, v in [("Markowitz λ=1", m1), ("Markowitz λ=3", m3), ("IQN-MPC CVaR", iq), ("Equal Wt", eq)]:
    m, s, sh, cv = st(v)
    print(f"{nm:<18} {m:>5.2f}% ± {s:<4.2f}% {sh:<8.3f} {cv:<10.2f}%", flush=True)

print("\n[5/5] Plotting...", flush=True)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for i in range(min(5, len(m1))):
    ax[0].plot(m1[i], 'b-', alpha=.3, label='Mark λ=1' if i==0 else '')
    ax[0].plot(m3[i], 'c-', alpha=.3, label='Mark λ=3' if i==0 else '')
    ax[0].plot(iq[i], 'r-', alpha=.3, label='IQN CVaR' if i==0 else '')
    ax[0].plot(eq[i], 'g-', alpha=.3, label='Equal' if i==0 else '')
ax[0].legend(); ax[0].set_xlabel('Step'); ax[0].set_ylabel('$'); ax[0].set_title('Trajectories'); ax[0].grid(True, alpha=.3)
for v, l, c in [(m1, 'Mark λ=1', 'blue'), (m3, 'Mark λ=3', 'cyan'), (iq, 'IQN CVaR', 'red'), (eq, 'Equal', 'green')]:
    ax[1].plot(v.mean(0), c, lw=2, label=l); ax[1].fill_between(range(len(v[0])), v.mean(0)-v.std(0), v.mean(0)+v.std(0), color=c, alpha=.12)
ax[1].legend(); ax[1].set_xlabel('Step'); ax[1].set_ylabel('$'); ax[1].set_title('Mean±Std'); ax[1].grid(True, alpha=.3)
plt.tight_layout(); plt.savefig('/Users/morty/.openclaw/workspace/portfolio_jit.png', dpi=150)
print("\n✓ Saved: portfolio_jit.png", flush=True)
