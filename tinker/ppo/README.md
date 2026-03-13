# PPO

A single-file, end-to-end JIT-compilable JAX implementation of Proximal Policy Optimization (PPO), the most widely-used on-policy model-free actor-critic method for deep RL.

PPO stabilizes policy gradient updates by clipping the likelihood ratio, preventing destructively large steps while remaining simple to implement and tune.

For continuous action space environments, run:

```bash
uv run -m tinker.ppo.continuous
```

For discrete action space environments, run:

```bash
uv run -m tinker.ppo.discrete
```

## Architecture

```
Observation ──> [Obs Normalization (continuous only)]
                        │
                        v
              ┌──────────────────┐
              │  ActorCritic      │
              │  ┌──────────────┐ │
              │  │ Actor Head   │ │──> Policy π(a|s)
              │  │ 256→256→a_dim│ │    (Normal or Categorical)
              │  └──────────────┘ │
              │  ┌──────────────┐ │
              │  │ Critic Head  │ │──> V(s)
              │  │ 256→256→1    │ │
              │  └──────────────┘ │
              └──────────────────┘
```

## Key Techniques

- **PPO clipping**: Surrogate objective clipped at ratio ε=0.2, preventing large policy updates.
- **Generalized Advantage Estimation (GAE)**: TD(λ) advantage estimates with γ=0.99, λ=0.95 via reverse `lax.scan`.
- **Observation normalization**: Running mean/variance via Welford's algorithm (continuous only), clipped to [-10, 10].
- **Entropy regularization**: Entropy bonus encourages exploration (default 0.0075 continuous, 0.01 discrete).
- **Value clipping**: Critic loss also clipped for stability.
- **Orthogonal initialization**: Hidden layers use orthogonal(√2), actor output orthogonal(0.01), critic output orthogonal(1.0).
- **Linear LR annealing**: Learning rate decays linearly to zero over training.
- **Gradient clipping**: Global norm clipping at 0.5.

## Continuous vs Discrete

| | Continuous | Discrete |
|---|---|---|
| Actor output | mean + learned log_std | categorical logits |
| Distribution | `distrax.MultivariateNormalDiag` | `distrax.Categorical` |
| Observation handling | Welford running normalization | One-hot encoding |
| Network hidden size | 256 | 64 |
| Default environment | EcoAnt | FrozenLake |

## Default Hyperparameters

| Parameter | Continuous | Discrete | Notes |
|---|---|---|---|
| `lr` | 3e-4 | 3e-4 | Adam with ε=1e-5 |
| `num_envs` | 5 | 5 | Parallel environments |
| `train_freq` | 500 | 200 | Rollout length |
| `total_timesteps` | 2M | 200K | Total env steps |
| `update_epochs` | 10 | 4 | Epochs per update |
| `batch_size` | 500 | 40 | Minibatch size |
| `gamma` | 0.99 | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | 0.95 | GAE lambda |
| `clip_eps` | 0.2 | 0.2 | PPO clip range |
| `ent_coef` | 0.0075 | 0.01 | Entropy coefficient |
| `vf_coef` | 0.5 | 0.5 | Value loss coefficient |
| `max_grad_norm` | 0.5 | 0.5 | Gradient clip norm |

## Links

- PPO paper: https://arxiv.org/abs/1707.06347
