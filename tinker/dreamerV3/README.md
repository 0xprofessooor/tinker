# DreamerV3

A single-file, end-to-end JIT-compilable JAX implementation of DreamerV3, ported from the [r2dreamer](https://github.com/zhixuan-lin/r2dreamer) PyTorch codebase.

DreamerV3 is a model-based RL algorithm that learns a world model from experience and trains an actor-critic entirely inside imagined trajectories. It works across diverse domains without hyperparameter tuning.

For continuous action space environments, run:

```bash
uv run -m tinker.dreamerV3.continuous
```

For discrete action space environments, run:

```bash
uv run -m tinker.dreamerV3.discrete
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

## Training Loop

Each training iteration consists of three phases:

```
for each update step:
  ┌─────────────────────────────────────────────────────────────┐
  │ 1. COLLECT: Run num_envs environments for train_freq steps  │
  │    Actor samples actions from current policy                │
  │    Transitions stored in replay buffer                      │
  └──────────────────────────┬──────────────────────────────────┘
                             │
  ┌──────────────────────────▼──────────────────────────────────┐
  │ 2. TRAIN: Repeat num_epochs times:                          │
  │    Sample batch of trajectory slices from replay buffer     │
  │                                                             │
  │    ┌─ WM Loss ─────────────────────────────────────────┐    │
  │    │  Posterior rollout on real observations             │    │
  │    │  KL(posterior || prior) with free bits              │    │
  │    │  Reconstruct obs (symlog MSE)                      │    │
  │    │  Predict rewards (TwoHot) + continuation (BCE)     │    │
  │    │  → updates: encoder, RSSM, decoder, reward, cont   │    │
  │    └───────────────────────────────────────────────────┘    │
  │                                                             │
  │    ┌─ Actor Loss ──────────────────────────────────────┐    │
  │    │  Imagine 15-step trajectories in latent space      │    │
  │    │  Predict rewards, continuation, values (with grads)│    │
  │    │  Compute λ-returns through differentiable dynamics │    │
  │    │  Maximize returns / max(1, S) + entropy bonus      │    │
  │    │  → updates: actor only (WM + critic frozen)        │    │
  │    └───────────────────────────────────────────────────┘    │
  │                                                             │
  │    ┌─ Critic Loss ─────────────────────────────────────┐    │
  │    │  Imagination: TwoHot regression on λ-returns       │    │
  │    │  Repval: TwoHot regression on real-data returns    │    │
  │    │  Both use stopped features (WM protected)          │    │
  │    │  Batched forward pass (imag + repval concatenated) │    │
  │    │  → updates: critic only (WM + actor frozen)        │    │
  │    └───────────────────────────────────────────────────┘    │
  │                                                             │
  │    Combine gradients and apply single optimizer step         │
  │    Update slow value target (EMA, τ=0.02)                   │
  └─────────────────────────────────────────────────────────────┘
```

The three loss functions compute gradients independently with strict isolation: each component's `stop_gradient` barriers ensure the world model stays grounded in reality, the actor can't hallucinate rewards, and the critic learns from both imagined and real data without corrupting representations.

## Key DreamerV3 Techniques

- **Symlog/Symexp transforms**: Compress large value ranges for stable learning. Applied to encoder inputs, decoder targets, and reward/value bin spacing.
- **TwoHot categorical regression**: Reward and value heads predict over 255 symexp-spaced bins instead of scalar regression, giving better gradient signal.
- **Discrete stochastic state**: RSSM uses categorical distributions (32 groups x 16 categories) with Gumbel-Softmax straight-through gradients, not Gaussian.
- **Unimix**: 1% uniform mixture added to stochastic state and actor distributions to prevent mode collapse.
- **Free-bits KL**: KL loss clipped to a minimum of 1.0 nat per group, preventing posterior collapse early in training.
- **Separate dyn/rep KL losses**: Dynamics loss (trains prior toward posterior) scaled at 1.0, representation loss (trains posterior toward prior) scaled at 0.1.
- **Imagination rollouts**: Actor-critic trained entirely in latent space over 15-step imagined trajectories.
- **Lambda returns**: GAE-style returns with `lambda=0.95` and `discount = 1 - 1/333`.
- **Return EMA normalization**: Returns normalized by running 5th/95th percentile EMA range (eq. 12 in paper).
- **Slow value target**: EMA copy of critic (tau=0.02) provides stable value targets.
- **Replay-based value learning (repval)**: Additional critic loss on real transitions (scale 0.3), anchoring the value function to observed data.
- **LR warmup**: Linear warmup over 1000 optimizer steps.
- **Gradient isolation**: Three separate gradient computations ensure the world model, actor, and critic are "trained concurrently without sharing gradients" (paper). Selective `stop_gradient` via NNX state partitioning prevents adversarial hallucination and representation collapse.

## Continuous vs Discrete

The two files differ in the actor, action handling, and policy gradient estimator:

| | Continuous | Discrete |
|---|---|---|
| Actor output | mean + std | logits with unimix |
| Distribution | `distrax.Normal` | `distrax.Categorical` |
| Policy gradient | Stochastic backpropagation (pathwise) | REINFORCE (score function) |
| Imagination sampling | Reparameterized Normal, clipped to [-1, 1] | Gumbel-Softmax (straight-through) |
| Action in RSSM | float, normalized by max(abs, 1) | one-hot vector |
| Default network size | deter=512, hidden=256 | deter=256, hidden=128 |

Following the paper (eq. 11), the **continuous** actor uses **stochastic backpropagation**: actions are sampled via the reparameterization trick (`action = mean + std * noise`), and gradients flow through the differentiable world model dynamics back to the actor — including through the critic's value predictions, giving the actor foresight beyond the imagination horizon. The continuous implementation also supports a `use_model_grads=False` flag to fall back to REINFORCE. The **discrete** actor always uses **REINFORCE** since discrete sampling is not differentiable.

## Default Hyperparameters

Structural parameters are set via `make_train` (must be concrete at JIT trace time):

| Parameter | Value | Notes |
|---|---|---|
| `num_epochs` | 1 | Gradient updates per collection phase |
| `batch_size` | 16 | Trajectory slices per batch |
| `batch_length` | 64 | Length of each slice |
| `imag_horizon` | 15 | Imagination rollout length |
| `stoch` | 32 | Stochastic state groups |
| `discrete` | 16 | Categories per group |
| `num_bins` | 255 | TwoHot bins for reward/value |
| `use_model_grads` | True | Pathwise (True) or REINFORCE (False) |

Training hyperparameters are set via `DynamicConfig` (can be swept across parallel runs with `vmap`):

| Parameter | Default | Notes |
|---|---|---|
| `lr` | 4e-5 | Learning rate |
| `kl_free` | 1.0 | Free nats for KL loss |
| `kl_dyn_scale` | 1.0 | Dynamics KL loss scale |
| `kl_rep_scale` | 0.1 | Representation KL loss scale |
| `horizon` | 333 | Planning horizon (discount = 1 - 1/horizon) |
| `gae_lambda` | 0.95 | Lambda for GAE-style returns |
| `entropy_coeff` | 3e-4 | Policy entropy regularization |
| `slow_target_frac` | 0.02 | EMA rate for slow value target |
| `repval_scale` | 0.3 | Replay-based value loss weight |
| `warmup_steps` | 1000 | Linear LR warmup steps |

## Links

- DreamerV3 paper: https://arxiv.org/abs/2301.04104
- Reference PyTorch implementation (r2dreamer): https://github.com/zhixuan-lin/r2dreamer
