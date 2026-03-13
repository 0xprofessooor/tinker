# VaR-CPO

A single-file, end-to-end JIT-compilable JAX implementation of Value-at-Risk Constrained Policy Optimization (VaR-CPO).

VaR-CPO extends CPO to handle **chance constraints** — bounding the probability that cumulative cost exceeds a threshold, rather than just bounding the mean. It uses Chebyshev's inequality to convert the variance of cost returns into a tractable constraint for CPO's trust region solver.

For continuous action space environments, run:

```bash
uv run -m tinker.var_cpo.continuous
```

For discrete action space environments, run:

```bash
uv run -m tinker.var_cpo.discrete
```

## Architecture

```
Observation + [cost_discount, running_cost/threshold]
                        │
                        v
              ┌───────────────────────────┐
              │  ActorCritic               │
              │  ┌───────────────────────┐ │
              │  │ Actor Head            │ │──> π(a|s)
              │  │ aug_dim→256→256→a_dim │ │
              │  └───────────────────────┘ │
              │  ┌───────────────────────┐ │
              │  │ Value Critic          │ │──> V(s)   (reward)
              │  └───────────────────────┘ │
              │  ┌───────────────────────┐ │
              │  │ Cost-Value Critic     │ │──> C(s)   (mean cost)
              │  └───────────────────────┘ │
              │  ┌───────────────────────┐ │
              │  │ Aug Cost-Value Critic │ │──> A(s)   (augmented cost)
              │  └───────────────────────┘ │
              └───────────────────────────┘
                        │
          ┌─────────────┼─────────────┐
          v             v             v
    Reward GAE    Cost GAE      Aug-Cost GAE
          │             │             │
          v             v             v
        ┌───────────────────────────────┐
        │  CPO Solver                    │
        │  if mean_unsafe:               │
        │    linear constraint on C(s)   │
        │  else:                         │
        │    Chebyshev constraint on A(s)│
        └───────────────────────────────┘
```

## Key Techniques

- **Augmented state**: Observations are extended with `[cost_discount, running_cost / var_threshold]`, letting the policy condition on its remaining safety budget.
- **Augmented cost**: `aug_cost = β·γ_c·cost² + 2(β·running_cost + d)·cost`, where β = (1/p) − 1, to track variance-related quantities needed by Chebyshev's bound.
- **Dual constraint modes**:
  - *Mean-unsafe* (episode cost > threshold): Falls back to a linear mean-cost constraint, like standard CPO.
  - *Mean-safe*: Applies Chebyshev constraint — bounds P(cost > threshold) ≤ p by constraining the augmented cost value.
- **CPO trust region**: Same conjugate gradient + 4-case solver + backtracking line search as CPO.
- **Three value heads**: Separate critics for reward, mean cost, and augmented (variance-aware) cost.
- **Asymmetric discounting**: Cost γ=0.999 vs reward γ=0.99.
- **Adaptive constraint margin**: Accumulated penalty on repeated violations.

## Continuous vs Discrete

| | Continuous | Discrete |
|---|---|---|
| Actor output | mean + learned log_std | categorical logits |
| Distribution | `distrax.MultivariateNormalDiag` | `distrax.Categorical` |
| Observation handling | Welford running normalization | One-hot encoding |
| Default environment | EcoAntV2 | FrozenLakeV2 |
| `var_threshold` | 500.0 | 15.0 |
| `var_probability` | 0.1 | 0.05 |

## Default Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `lr` | 3e-4 | Critic optimizer (Adam, ε=1e-5) |
| `gae_gamma` | 0.99 | Reward discount |
| `cost_gamma` | 0.999 | Cost discount |
| `gae_lambda` | 0.95 | GAE lambda |
| `target_kl` | 0.01 | Trust region size |
| `damping_coeff` | 0.1 | Hessian damping |
| `backtrack_coeff` | 0.8 | Line search step decay |
| `backtrack_iters` | 10 | Max line search iterations |
| `margin_lr` | 0.0 | Constraint margin learning rate |
| `max_grad_norm` | 0.5 | Gradient clip norm |
| `entropy_coeff` | 0.0 | Entropy regularization |
| `var_threshold` | 500.0 / 15.0 | Safety threshold (env-dependent) |
| `var_probability` | 0.1 / 0.05 | Acceptable exceedance probability |
