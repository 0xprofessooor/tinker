# IQN-MPC: Implicit Quantile Networks for Model Predictive Control

Distributional state transition modeling combined with gradient-based planning for risk-aware portfolio optimization.

## Overview

This module implements a **model-based reinforcement learning** approach to portfolio optimization:

1. **Learn** the distribution of state transitions P(s'|s,a) using Implicit Quantile Networks
2. **Plan** optimal actions via gradient descent through the learned model
3. **Control risk** using CVaR (Conditional Value-at-Risk) constraints

The approach combines:
- **Dabney et al. (2018)** — IQN for distributional modeling
- **Lobo, Fazel, Boyd (2002)** — Convex portfolio optimization with transaction costs
- **Rockafellar & Uryasev (2000)** — CVaR for tail risk management

## Algorithm

### Part 1: IQN State Transition Model

**Goal:** Learn the full distribution P(s'|s,a), not just the mean E[s'|s,a].

**Why distributional?** Financial returns have heavy tails. A mean-only model misses:
- Volatility clustering (GARCH effects)
- Tail risk (rare but severe losses)
- Asymmetric distributions (skewness)

**Architecture:**

```
Input: (state s, action a, quantile level τ ∈ [0,1])
Output: τ-quantile of next state distribution

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   (s, a) ──► Encoder ──► embedding e                        │
│                              │                              │
│      τ ──► Cosine Embed ──► τ_embed                         │
│                              │                              │
│                         e ⊙ τ_embed  (Hadamard product)     │
│                              │                              │
│                          Decoder                            │
│                              │                              │
│                           s'_τ  (predicted τ-quantile)      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Cosine Embedding** (maps scalar τ to vector):
```
φ(τ) = ReLU(W · [cos(0·π·τ), cos(1·π·τ), ..., cos((n-1)·π·τ)]ᵀ + b)
```

**Training Loss** — Quantile Regression (Pinball Loss):
```
ρ_τ(u) = u · (τ - 𝟙{u < 0})

L = E_{τ~U(0,1)} [ ρ_τ(s'_true - s'_predicted) ]
```

This loss ensures the network output at quantile τ has exactly τ fraction of true values below it.

### Part 2: Gradient-Based MPC Planning

**Goal:** Find the optimal action sequence by directly optimizing through the learned model.

Unlike sampling-based methods (CEM, random shooting), we use **gradient descent** on the action sequence. This is possible because:
1. IQN forward pass is differentiable (neural network)
2. Portfolio weights use softmax (differentiable)
3. Reward function is differentiable

**Optimization Problem** (extends Lobo-Fazel-Boyd 2002):

```
maximize    E[return] - transaction_costs - λ·Var[return] - β·CVaR_penalty
   a_0:H

subject to  w_i ≥ 0           (long-only)
            Σ w_i = 1         (fully invested)
            w_i ≤ w_max       (position limits)
```

Where:
- `E[return] = Σ_t γ^t · (w_t · μ)` — discounted expected returns
- `transaction_costs = c · Σ_t |w_t - w_{t-1}|` — linear costs (Lobo-Fazel-Boyd)
- `Var[return] = Σ w_i² σ_i²` — portfolio variance (diagonal approx)
- `CVaR_penalty = max(0, threshold - CVaR_α)` — tail risk (our extension)

**Algorithm:**

```python
def gradient_mpc(obs, prev_weights, horizon=5, n_iters=50):
    # Initialize action sequence
    actions = zeros((horizon, n_assets + 1))
    
    for iteration in range(n_iters):
        # Sample multiple trajectories for CVaR estimation
        trajectory_returns = []
        for tau_sample in uniform_samples(n_quantile_samples):
            
            # Roll out trajectory through IQN model
            total_return = 0
            state = obs
            weights = prev_weights
            
            for t in range(horizon):
                new_weights = softmax(actions[t])
                
                # Transaction cost
                cost = linear_rate * sum(|new_weights - weights|)
                
                # Expected return this step
                step_return = new_weights @ expected_returns(state)
                
                # Variance penalty
                variance = sum(new_weights² * volatilities(state)²)
                
                # Accumulate
                total_return += γ^t * (step_return - cost - λ*variance)
                
                # Predict next state using IQN
                state = IQN(state, actions[t], tau_sample)
                weights = new_weights
            
            trajectory_returns.append(total_return)
        
        # Compute CVaR (average of worst α fraction)
        sorted_returns = sort(trajectory_returns)
        cvar = mean(sorted_returns[:n_worst])
        
        # Total objective
        objective = mean(trajectory_returns) - β * relu(-cvar)
        
        # Gradient descent step
        grad = ∇_actions objective
        actions = actions + lr * grad
    
    return actions[0]  # Return first action (receding horizon)
```

### Part 3: CVaR Risk Constraint

**CVaR (Conditional Value-at-Risk)** = Expected loss given that loss exceeds VaR.

```
CVaR_α = E[X | X ≤ VaR_α]
```

For α = 0.2, this is the average of the worst 20% of outcomes.

**Why CVaR over Variance?**
- Variance penalizes upside and downside equally
- CVaR focuses specifically on tail losses
- More meaningful for risk management ("average loss in bad scenarios")

**How IQN enables CVaR:**

Standard models predict E[s'|s,a] — can't estimate tails.

IQN predicts quantiles — we can:
1. Sample τ ∈ {0.05, 0.1, 0.15, ...} to get distribution
2. Compute empirical CVaR from samples
3. Backpropagate through the CVaR computation

This is the key advantage of distributional RL for risk-sensitive control.

## Comparison with Baselines

| Aspect | Markowitz | CEM-MPC | Gradient MPC (ours) |
|--------|-----------|---------|---------------------|
| Model | None | Learned (IQN) | Learned (IQN) |
| Optimization | Closed-form | Sampling | Gradient descent |
| Transaction costs | Post-hoc | In objective | In objective |
| Risk measure | Variance | CVaR (noisy) | CVaR (stable) |
| Samples needed | 0 | ~1000s | ~10s (quantiles) |
| Differentiable | N/A | No | Yes |

## Experimental Results

**Setup:** 2 assets (AAPL-like + BTC-like) + cash, GARCH(1,1) dynamics, 0.1% linear transaction costs.

| Strategy | Return | Std | Sharpe | CVaR 20% | Avg Cost |
|----------|--------|-----|--------|----------|----------|
| Markowitz λ=1 | 6.26% | 8.79% | 0.711 | -3.48% | $1.91 |
| Markowitz λ=3 | 6.20% | 8.81% | 0.704 | -3.48% | $1.91 |
| **Gradient MPC** | 2.96% | **5.54%** | 0.534 | **-1.89%** | $1.44 |
| Equal Weight | 2.44% | 9.99% | 0.245 | -9.72% | $0.67 |

**Key findings:**
- Gradient MPC achieves **best CVaR** (-1.89%) — proper tail risk control
- **Lowest variance** (5.54%) — most stable returns  
- Lower return is the cost of risk aversion
- Lower transaction costs — not overtrading

## Files

```
tinker/iqn_mpc/
├── __init__.py          # Module exports
├── iqn.py               # IQN network, quantile embedding, pinball loss
├── mpc.py               # CEM-based planner (baseline)
├── gradient_mpc.py      # Gradient-based planner with CVaR
└── README.md            # This file

scripts/
├── eval_mini.py         # Basic 1-asset test
├── eval_2asset_jit.py   # 2-asset CEM comparison
└── eval_gradient_mpc.py # 2-asset gradient MPC with transaction costs
```

## Usage

```python
from tinker.iqn_mpc import IQNStateNetwork, make_iqn_train
from tinker.iqn_mpc.gradient_mpc import create_gradient_mpc

# 1. Train IQN on collected transitions
train_fn = make_iqn_train(state_dim=4, action_dim=3, num_updates=5000)
model, metrics = train_fn(key, transitions)

# 2. Create gradient MPC policy
policy = create_gradient_mpc(
    model=model,
    state_dim=4,
    action_dim=3,
    horizon=5,
    linear_cost_rate=0.001,    # 0.1% transaction cost
    variance_penalty=2.0,       # Risk aversion λ
    cvar_alpha=0.2,            # CVaR level
    cvar_penalty=0.5,          # CVaR constraint weight
    lr=0.1,
    n_iters=50,
)

# 3. Use in environment
action = policy(obs, prev_weights, key)
new_weights = jax.nn.softmax(action)
```

## References

1. **Dabney et al. (2018)** "Implicit Quantile Networks for Distributional Reinforcement Learning" — IQN architecture
2. **Lobo, Fazel, Boyd (2002)** "Portfolio Optimization with Linear and Fixed Transaction Costs" — Convex formulation
3. **Rockafellar & Uryasev (2000)** "Optimization of Conditional Value-at-Risk" — CVaR theory
4. **Chua et al. (2018)** "Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models" — Model-based RL with uncertainty

## Future Work

- [ ] Fixed transaction costs (requires mixed-integer extension from Lobo-Fazel-Boyd)
- [ ] Multi-period CVaR constraints
- [ ] Correlation modeling (non-diagonal covariance)
- [ ] Integration with safenax GARCH environment
