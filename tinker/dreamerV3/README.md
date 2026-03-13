# DreamerV3

A single-file, end-to-end JIT-compilable JAX implementation of DreamerV3, ported from the [r2dreamer](https://github.com/zhixuan-lin/r2dreamer) PyTorch codebase.

DreamerV3 is a model-based RL algorithm that learns a world model from experience and trains an actor-critic entirely inside imagined trajectories. It works across diverse domains without hyperparameter tuning.

For continuous action space environments, run:

```bash
uv run -m tinker.dreamer.continuous
```

For discrete action space environments, run:

```bash
uv run -m tinker.dreamer.discrete
```

## Architecture

```
Observation ──> Encoder (MLP + symlog) ──> Embedding
                                              │
                                              v
              ┌─────────────────────────────────────────────┐
              │  RSSM World Model                           │
              │  ┌───────────┐    ┌──────────┐              │
              │  │ DeterNet  │───>│ ObsNet   │──> Posterior │
              │  │ (GRU)     │    │          │   (stoch)    │
              │  └───────────┘    └──────────┘              │
              │       │           ┌──────────┐              │
              │       └──────────>│ ImgNet   │──> Prior     │
              │                   │          │   (stoch)    │
              │                   └──────────┘              │
              └─────────────────────────────────────────────┘
                          │
                          v
                   Feature = [stoch_flat, deter]
                          │
           ┌──────────────┼──────────────┬──────────────┐
           v              v              v              v
       Decoder        Reward         Cont          Actor/Critic
    (symlog MSE)     (TwoHot)     (Bernoulli)    (Normal/Categorical)
```

## Key DreamerV3 Techniques

- **Symlog/Symexp transforms**: Compress large value ranges for stable learning. Applied to encoder inputs, decoder targets, and reward/value bin spacing.
- **TwoHot categorical regression**: Reward and value heads predict over 255 symexp-spaced bins instead of scalar regression, giving better gradient signal.
- **Discrete stochastic state**: RSSM uses categorical distributions (32 groups x 16 categories) with Gumbel-Softmax straight-through gradients, not Gaussian.
- **Unimix**: 1% uniform mixture added to stochastic state and actor distributions to prevent mode collapse.
- **Free-bits KL**: KL loss clipped to a minimum of 1.0 nat per group, preventing posterior collapse early in training.
- **Separate dyn/rep KL losses**: Dynamics loss (trains prior toward posterior) scaled at 1.0, representation loss (trains posterior toward prior) scaled at 0.1.
- **Imagination rollouts**: Actor-critic trained entirely in latent space over 15-step imagined trajectories.
- **Lambda returns**: GAE-style returns with `lambda=0.95` and `discount = 1 - 1/333`.
- **Return EMA normalization**: Advantages normalized by running 5th/95th percentile EMA of returns.
- **Slow value target**: EMA copy of critic (tau=0.02) provides stable value targets.
- **Replay-based value learning (repval)**: Additional value loss on real transitions (scale 0.3) with gradients flowing through the world model.
- **LR warmup**: Linear warmup over 1000 optimizer steps.

## Continuous vs Discrete

The two files differ only in the actor and action handling:

| | Continuous | Discrete |
|---|---|---|
| Actor output | mean + std | logits with unimix |
| Distribution | `distrax.Normal` | `distrax.Categorical` |
| Imagination sampling | Normal sample, clipped to [-1, 1] | Gumbel-Softmax (straight-through) |
| Action in RSSM | float, normalized by max(abs, 1) | one-hot vector |
| Default network size | deter=512, hidden=256 | deter=256, hidden=128 |

## Default Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `stoch` | 32 | Stochastic state groups |
| `discrete` | 16 | Categories per group |
| `num_bins` | 255 | TwoHot bins for reward/value |
| `imag_horizon` | 15 | Imagination rollout length |
| `kl_free` | 1.0 | Free nats for KL loss |
| `kl_dyn_scale` | 1.0 | Dynamics KL loss scale |
| `kl_rep_scale` | 0.1 | Representation KL loss scale |
| `lr` | 4e-5 | Learning rate |
| `batch_size` | 16 | Trajectory slices per batch |
| `batch_length` | 64 | Length of each slice |

## Links

- DreamerV3 paper: https://arxiv.org/abs/2301.04104
- Reference PyTorch implementation (r2dreamer): https://github.com/zhixuan-lin/r2dreamer
