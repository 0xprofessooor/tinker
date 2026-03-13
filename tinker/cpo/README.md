# CPO

A single-file, end-to-end JIT-compilable JAX implementation of Constrained Policy Optimization (CPO).

CPO is a model-free safe RL algorithm that optimizes a policy subject to hard cost constraints using second-order trust region methods. It solves a constrained optimization problem at each step: maximize expected reward while keeping expected cost below a threshold, subject to a KL divergence trust region.

For continuous action space environments, run:

```bash
uv run -m tinker.cpo.continuous
```

For discrete action space environments, run:

```bash
uv run -m tinker.cpo.discrete
```

## Architecture

```
Observation ──> [Obs Normalization (continuous only)]
                        │
                        v
              ┌───────────────────────┐
              │  ActorCritic           │
              │  ┌───────────────────┐ │
              │  │ Actor Head        │ │──> π(a|s)
              │  │ 256→256→a_dim     │ │    (Normal or Categorical)
              │  └───────────────────┘ │
              │  ┌───────────────────┐ │
              │  │ Value Critic      │ │──> V(s)   (reward)
              │  │ 256→256→1         │ │
              │  └───────────────────┘ │
              │  ┌───────────────────┐ │
              │  │ Cost-Value Critic │ │──> C(s)   (cost)
              │  │ 256→256→1         │ │
              │  └───────────────────┘ │
              └───────────────────────┘
                        │
              ┌─────────┴──────────┐
              v                    v
        Reward GAE            Cost GAE
        (γ=0.99)              (γ=0.999)
              │                    │
              v                    v
        ┌──────────────────────────────┐
        │  CPO Solver                   │
        │  1. Conjugate Gradient (Hg,Hb)│
        │  2. 4-case optimization       │
        │  3. Backtracking line search  │
        └──────────────────────────────┘
```

## Key Techniques

- **Conjugate gradient solver**: Efficiently solves `Hx = g` (reward direction) and `Hx = b` (cost direction) using Hessian-vector products via JAX autodiff, avoiding explicit Hessian construction.
- **4-case constrained optimization**:
  - *Case 0*: Recovery mode — constraint violated, step toward feasibility.
  - *Case 1–2*: Feasible constrained — project reward direction onto constraint-satisfying half-space.
  - *Case 3–4*: Unconstrained — behaves like TRPO when constraint is satisfied.
- **Backtracking line search**: Accepts steps that improve policy loss, reduce cost violation, and stay within the KL trust region (α decays by 0.8 per iteration, max 10 steps).
- **Asymmetric discounting**: Cost γ=0.999 vs reward γ=0.99 for long-horizon safety awareness.
- **Adaptive constraint margin**: `margin = max(0, margin + margin_lr * c_raw)` accumulates penalty on repeated violations.
- **Separate advantage normalization**: Reward advantages normalized to mean=0, std=1; cost advantages centered but not rescaled (preserving magnitude for the constraint).
- **Hessian damping**: 0.1 added to diagonal for numerical stability in conjugate gradient.

## Continuous vs Discrete

| | Continuous | Discrete |
|---|---|---|
| Actor output | mean + learned log_std | categorical logits |
| Distribution | `distrax.MultivariateNormalDiag` | `distrax.Categorical` |
| Observation handling | Welford running normalization | One-hot encoding |
| Default environment | EcoAntV2 (battery constraint) | FrozenLakeV2 (thin ice hazards) |
| `num_envs` | 25 | 5 |
| `train_freq` | 100 | 200 |
| `critic_epochs` | 10 | 80 |

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
| `margin_lr` | 0.05 | Constraint margin learning rate |
| `max_grad_norm` | 0.5 | Gradient clip norm |
| `entropy_coeff` | 0.0 | Entropy regularization |

## Links

- CPO paper: https://arxiv.org/abs/1705.10528
