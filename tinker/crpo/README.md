# CRPO

A single-file, end-to-end JIT-compilable JAX implementation of Constraint Rectified Policy Optimization (CRPO).

CRPO is a model-free safe RL algorithm that handles constrained MDPs by switching between two objectives: when the cost constraint is satisfied, it optimizes for reward using a standard PPO objective; when the constraint is violated, it instead optimizes to reduce cost. This binary switching avoids the need for a Lagrange multiplier while still enforcing safety.

For continuous action space environments, run:

```bash
uv run -m tinker.crpo.continuous
```

## Architecture

```
Observation ──> [Obs Normalization]
                        │
                        v
              ┌───────────────────────┐
              │  ActorCritic           │
              │  ┌───────────────────┐ │
              │  │ Actor Head        │ │──> π(a|s)
              │  │ 256→256→a_dim     │ │    (MultivariateNormalDiag)
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
              └──────────┬─────────┘
                         v
              ┌─────────────────────┐
              │  CRPO Switch        │
              │  cost_return > limit│
              │  ? cost PPO loss    │
              │  : reward PPO loss  │
              └─────────────────────┘
```

## Key Techniques

- **Constraint rectification**: At each minibatch update, checks if the average episode cost return from the current rollout exceeds `cost_limit`. If so, runs a PPO update on the cost objective (minimizing cost advantages); otherwise runs a standard PPO reward update. The switch uses `jax.lax.cond` to remain JIT-compatible.
- **No Lagrange multiplier**: Unlike Lagrangian methods (ACPO, CPPO), CRPO has no adaptive penalty weight — the switching logic itself enforces the constraint without tuning an extra parameter.
- **Separate GAE for reward and cost**: Both reward and cost advantages are computed via GAE with independent discount factors. Reward advantages are normalized to mean=0, std=1; cost advantages are only centered (preserving magnitude for the constraint signal).
- **PPO clipping**: Both reward and cost actor losses use standard surrogate clipping at ε=0.2. The cost loss maximizes the clipped negative ratio (i.e., minimizes cost-weighted probability).
- **Clipped value targets**: Both value critics use clipped value loss targets (same ε as actor clipping) to stabilize training.
- **Observation normalization**: Welford running mean/variance updated each rollout before the optimizer step.
- **Linear LR annealing**: Learning rate decays to zero over training (optional).

## Default Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `lr` | 3e-4 | Adam optimizer (ε=1e-5) |
| `num_envs` | 10 | Parallel environments |
| `train_freq` | 1024 | Rollout steps per update |
| `total_timesteps` | 1M | Total environment steps |
| `num_epochs` | 10 | Epochs per update |
| `batch_size` | 512 | Minibatch size |
| `gamma` | 0.99 | Reward discount factor |
| `cost_gamma` | 0.999 | Cost discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `clip_eps` | 0.2 | PPO clip range |
| `ent_coef` | 0.0075 | Entropy coefficient (reward update only) |
| `vf_coef` | 0.5 | Value loss coefficient |
| `max_grad_norm` | 0.5 | Gradient clip norm |
| `cost_limit` | 200.0 | Episode cost threshold for switching |
| `anneal_lr` | True | Linear learning rate decay |

## Links

- CRPO paper: https://arxiv.org/abs/2011.05869
