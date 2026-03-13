# CPPO

A single-file, end-to-end JIT-compilable JAX implementation of CVaR Proximal Policy Optimization (CPPO).

CPPO extends PPO with a Conditional Value-at-Risk (CVaR) cost constraint via Lagrangian relaxation. Instead of constraining the mean cost (like CPO), it constrains the expected cost in the worst-case tail of the distribution — focusing safety enforcement on the most dangerous episodes.

For continuous action space environments, run:

```bash
uv run -m tinker.cppo.continuous
```

For discrete action space environments, run:

```bash
uv run -m tinker.cppo.discrete
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
              │  │ 256→256→a_dim     │ │
              │  └───────────────────┘ │
              │  ┌───────────────────┐ │
              │  │ Value Critic      │ │──> V(s)
              │  │ 256→256→1         │ │
              │  └───────────────────┘ │
              │  ┌───────────────────┐ │
              │  │ Cost-Value Critic │ │──> C(s)
              │  │ 256→256→1         │ │
              │  └───────────────────┘ │
              └───────────────────────┘
                        │
              ┌─────────┴──────────┐
              v                    v
        Reward GAE            Cost GAE + CVaR
              │                    │
              v                    v
        PPO clipped loss    Lagrangian penalty
        ─────────────────────────────
        total_loss = actor + vf + cvf - ent + λ·penalty
```

## Key Techniques

- **CVaR constraint**: Sorts terminal episode costs, computes the mean of the top-k worst (controlled by `cvar_probability`). Constrains this tail expectation rather than the mean.
- **Lagrangian dual update**: Multiplier λ adapts online: `λ += lam_lr * (CVaR_estimate - cvar_limit)`, clamped to λ ≥ 0.
- **Advantage penalty**: Cost-value advantages are penalized proportionally to λ and inversely to the CVaR probability level: `penalty = (λ/cvar_probability) * (cost_returns - ν)`.
- **PPO clipping**: Standard surrogate clipping at ε=0.2 for the reward objective.
- **Multi-head loss**: Combined loss of actor, value critic, cost-value critic, and entropy, each with configurable coefficients.
- **Observation normalization**: Welford running mean/variance (continuous only).
- **Linear LR annealing**: Learning rate decays to zero over training.

## Continuous vs Discrete

| | Continuous | Discrete |
|---|---|---|
| Actor output | mean + learned log_std | categorical logits |
| Distribution | `distrax.MultivariateNormalDiag` | `distrax.Categorical` |
| Observation handling | Welford running normalization | One-hot encoding |
| Network hidden size | 256 | 64 |
| Default environment | EcoAnt | FrozenLake |
| `cvar_probability` | 0.1 (10% tail) | 0.05 (5% tail) |
| `cvar_limit` | 50.0 | 15.0 |
| `lam_start` | 10.0 | 100.0 |

## Default Hyperparameters

| Parameter | Continuous | Discrete | Notes |
|---|---|---|---|
| `lr` | 3e-4 | 3e-4 | Adam with ε=1e-5 |
| `num_envs` | 5 | 5 | Parallel environments |
| `train_freq` | 1000 | 200 | Rollout length |
| `total_timesteps` | 1M | 200K | Total env steps |
| `num_epochs` | 10 | 4 | Epochs per update |
| `batch_size` | 500 | 40 | Minibatch size |
| `gamma` | 0.99 | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | 0.95 | GAE lambda |
| `clip_eps` | 0.2 | 0.2 | PPO clip range |
| `ent_coef` | 0.0075 | 0.01 | Entropy coefficient |
| `vf_coef` | 0.5 | 0.5 | Value loss coefficient |
| `cvar_probability` | 0.1 | 0.05 | CVaR tail fraction |
| `cvar_limit` | 50.0 | 15.0 | CVaR threshold |
| `lam_start` | 10.0 | 100.0 | Initial Lagrange multiplier |
| `lam_lr` | 1e-2 | 1e-2 | Multiplier learning rate |
| `max_grad_norm` | 0.5 | 0.5 | Gradient clip norm |
