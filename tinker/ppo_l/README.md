# PPO-L

A single-file, end-to-end JIT-compilable JAX implementation of PPO with Lagrangian cost constraints (PPO-L) for continuous action spaces.

PPO-L extends PPO with a safety constraint on the mean episode cost via Lagrangian relaxation. A dual variable λ is updated online to penalize policy updates that violate the cost budget, trading off reward maximization against constraint satisfaction.

For continuous action space environments, run:

```bash
uv run -m tinker.ppo_l.continuous
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
        Reward GAE            Cost GAE
              │                    │
              v                    v
        PPO clipped loss    Lagrangian penalty
        ─────────────────────────────
        total_loss = actor + λ·cost_surrogate + vf·(vf_loss + cvf_loss) - ent·entropy
```

## Key Techniques

- **Lagrangian dual update**: Multiplier λ adapts online using projected gradient ascent: `λ = max(0, λ + lagrange_lr * (mean_episode_cost - cost_limit))`. Uses mean cost over completed episodes in each rollout.
- **Cost surrogate (pessimistic clipping)**: Cost policy gradient uses `max` over clipped ratios (vs. `min` for reward), making the constraint conservative.
- **Separate cost GAE**: Cost returns use a dedicated discount factor `cost_gae_gamma` (default 0.999), allowing different time horizons for reward and cost.
- **Advantage normalization**: Reward advantages are fully normalized (mean 0, std 1); cost advantages are only mean-centered, preserving scale for the Lagrangian penalty.
- **PPO clipping**: Standard surrogate clipping at ε=0.2 for the reward objective.
- **Observation normalization**: Welford running mean/variance, updated each rollout.
- **Orthogonal initialization**: Hidden layers use orthogonal(√2), actor output orthogonal(0.01), critic outputs orthogonal(1.0).
- **Linear LR annealing**: Learning rate decays linearly to zero over training.
- **Gradient clipping**: Global norm clipping at 0.5.

## Default Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `lr` | 3e-4 | Adam with ε=1e-5 |
| `num_envs` | 10 | Parallel environments |
| `train_freq` | 1024 | Rollout length |
| `total_timesteps` | 1M | Total env steps |
| `update_epochs` | 10 | Epochs per update |
| `batch_size` | 512 | Minibatch size |
| `gamma` | 0.99 | Reward discount factor |
| `cost_gamma` | 0.999 | Cost discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `clip_eps` | 0.2 | PPO clip range |
| `ent_coef` | 0.0075 | Entropy coefficient |
| `vf_coef` | 0.5 | Value loss coefficient (applied to both critics) |
| `max_grad_norm` | 0.5 | Gradient clip norm |
| `cost_limit` | 400.0 | Constraint threshold on mean episode cost |
| `lagrange_lr` | 1e-2 | Lagrange multiplier learning rate |
| `init_lagrange_lambda` | 0.0 | Initial Lagrange multiplier |
| `anneal_lr` | True | Linear LR decay |
| `activation` | tanh | Hidden layer activation |
| `default_env` | EcoAntV2 | Default environment |

## Links

- PPO paper: https://arxiv.org/abs/1707.06347
- Lagrangian methods for constrained RL: https://arxiv.org/abs/2001.06782
