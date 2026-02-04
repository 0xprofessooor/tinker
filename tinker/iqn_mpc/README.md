# IQN-MPC: Implicit Quantile Networks for Model Predictive Control

Distributional state transition modeling + uncertainty-aware planning for portfolio optimization.

## Algorithm Overview

### 1. IQN State Transition Model

**Goal:** Learn the full distribution P(s'|s,a), not just the mean.

**Architecture:**
```
Input: (state s, action a, quantile level τ ∈ [0,1])
Output: τ-quantile of next state distribution

┌─────────────┐     ┌──────────────────┐
│  (s, a)     │────▶│  Encoder (MLP)   │────▶ embedding e
└─────────────┘     └──────────────────┘
                                              │
┌─────────────┐     ┌──────────────────┐      │
│     τ       │────▶│ Cosine Embedding │────▶ τ_embed
└─────────────┘     └──────────────────┘      │
                                              ▼
                    ┌──────────────────┐   ┌─────────┐
                    │     Decoder      │◀──│ e ⊙ τ_e │ (Hadamard product)
                    └──────────────────┘   └─────────┘
                             │
                             ▼
                         s'_τ (predicted τ-quantile of next state)
```

**Cosine Embedding** (from Dabney et al. 2018):
```
φ(τ) = ReLU(W · [cos(0·π·τ), cos(1·π·τ), ..., cos((n-1)·π·τ)]ᵀ + b)
```

**Training Loss** - Quantile Regression (Pinball Loss):
```
ρ_τ(u) = u · (τ - 𝟙{u < 0})

L = E_τ~U(0,1) [ ρ_τ(s'_true - s'_predicted) ]
```

This loss penalizes:
- Under-predictions (s'_pred < s'_true) by factor τ
- Over-predictions (s'_pred > s'_true) by factor (1-τ)

Result: The network learns to output the τ-quantile of the true distribution.

### 2. MPC Planning with CVaR

**Goal:** Find action sequence that maximizes risk-adjusted returns.

**CVaR (Conditional Value-at-Risk):**
```
CVaR_α = E[X | X ≤ VaR_α]  (average of worst α fraction of outcomes)
```

For α=0.2, we're optimizing for the worst 20% of scenarios — a risk-averse objective.

**Cross-Entropy Method (CEM) Planning:**
```
1. Initialize: μ = 0, σ = 0.5 for action sequence distribution

2. For each CEM iteration:
   a. Sample N action sequences from N(μ, σ)
   
   b. For each sequence, simulate K trajectories:
      - At each step, sample τ ~ U(0,1)
      - Predict next state: s' = IQN(s, a, τ)
      - Accumulate discounted rewards
   
   c. Compute CVaR for each sequence:
      - Sort trajectory returns
      - Average bottom α fraction
   
   d. Select top-E elite sequences (highest CVaR)
   
   e. Update: μ = mean(elites), σ = std(elites)

3. Return first action of μ
```

### 3. Portfolio Application

**State:** [σ_AAPL, σ_BTC, μ_AAPL, μ_BTC] (volatilities + expected returns)

**Action:** [w_cash, w_AAPL, w_BTC] (portfolio weights as logits, softmax-normalized)

**Dynamics:** GARCH(1,1) process for each asset
```
σ²_t = ω + α·(r_{t-1} - μ)² + β·σ²_{t-1}
r_t = μ + σ_t · ε_t,  ε_t ~ N(0,1)
```

**Why IQN helps:**
- GARCH has heavy tails — mean prediction misses tail risk
- IQN captures the full return distribution
- CVaR planning explicitly optimizes for worst-case scenarios

## Comparison with Markowitz

| Aspect | Markowitz | IQN-MPC |
|--------|-----------|---------|
| Model | None (uses realized μ, σ) | Learned P(s'\|s,a) |
| Objective | E[r] - λ·Var[r] | CVaR_α (worst α% outcomes) |
| Planning | Single-step (myopic) | Multi-step lookahead |
| Uncertainty | Assumes Gaussian | Learns true distribution |

## Current Limitations

1. **CEM is sample-inefficient** — needs many samples for good CVaR estimates
2. **No JIT compilation** — slow evaluation (memory issues with JIT)
3. **Simple reward** — just expected return, could add transaction costs

## Files

- `iqn.py` — IQN network, quantile embedding, pinball loss, training
- `mpc.py` — CEM planner, trajectory sampling, CVaR computation
- `../scripts/eval_2asset_lean.py` — 2-asset portfolio evaluation

## References

- Dabney et al. (2018) "Implicit Quantile Networks for Distributional RL"
- Rockafellar & Uryasev (2000) "Optimization of CVaR"
- Chua et al. (2018) "PETS: Deep RL with Probabilistic Ensembles"
- Markowitz (1952) "Portfolio Selection"
