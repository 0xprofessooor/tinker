# ACPO

A single-file, end-to-end JIT-compilable JAX implementation of Asynchronous Constrained Policy Optimization (ACPO).

ACPO is a hybrid safe RL algorithm that combines a **model-based policy objective** (Stochastic Value Gradients through a learned world model) with a **model-free cost constraint** (CPO-style trust region). This asymmetry lets the policy exploit sample-efficient model-based gradients for reward maximization while relying on conservative model-free estimates for safety.

For continuous action space environments, run:

```bash
uv run -m tinker.acpo.continuous
```

For discrete action space environments, run:

```bash
uv run -m tinker.acpo.discrete
```

## Architecture

```
Observation ──> [Obs Normalization] ──> obs
                                         │
                  ┌──────────────────────┼──────────────────────┐
                  v                      v                      v
          ┌──────────────┐     ┌──────────────────┐    ┌──────────────┐
          │  Actor        │     │  StateModel       │    │  Critic       │
          │  256→256→a_dim│     │  (GRU World Model) │    │  ┌──────────┐│
          │  → Normal/Cat │     │  obs,a → GRU → h   │    │  │ V(s)    ││
          └──────────────┘     │  h → Encoder → ψ   │    │  │ reward  ││
                  │             │  τ → CosineNet → φ │    │  └──────────┘│
                  │             │  ψ·φ → Decoder     │    │  ┌──────────┐│
                  │             │  → quantile pred   │    │  │ C(s)    ││
                  │             └──────────────────┘    │  │ cost    ││
                  │                      │              │  └──────────┘│
                  v                      v              └──────────────┘
          Policy gradients      Pathwise SVG gradients
          (model-free cost)     (model-based reward)
```

## Key Techniques

- **Stochastic Value Gradients (SVG)**: Policy loss differentiates through the world model via reparameterized actions and predicted next states, creating a fully differentiable reward pathway.
- **Model-free cost constraint**: Cost advantages use standard TD estimation with importance sampling — no world model in the constraint path, keeping safety conservative.
- **Quantile world model**: GRU-based recurrent model predicts observation quantiles via cosine basis functions and pinball loss, capturing aleatoric uncertainty.
- **CPO trust region**: Conjugate gradient solver with Hessian-vector products, backtracking line search, and 4-case optimization (recovery, feasible constrained, feasible unconstrained, TRPO).
- **Adaptive constraint margin**: Learnable margin accumulates penalty on repeated violations: `margin = max(0, margin + margin_lr * c_raw)`.
- **Asymmetric discounting**: Reward γ=0.99, cost γ=0.999 — costs are discounted less aggressively to encourage long-horizon safety awareness.
- **Gradient divergence tracking**: Logs cosine similarity and norm ratio between model-based and model-free gradients for analysis.
- **Gumbel-Softmax reparameterization** (discrete): Conditional Gumbel noise inference enables pathwise gradients through discrete actions.

## Continuous vs Discrete

| | Continuous | Discrete |
|---|---|---|
| Actor output | mean + learned log_std | categorical logits |
| Distribution | `distrax.MultivariateNormalDiag` | `distrax.Categorical` |
| SVG reparameterization | Gaussian noise conditioning | Conditional Gumbel-Softmax |
| World model loss | Pinball (quantile regression) | Cross-entropy |
| Observation handling | Welford running normalization | One-hot encoding |

## Default Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `critic_lr` | 3e-4 | Adam with ε=1e-5 |
| `state_model_lr` | 3e-4 | World model learning rate |
| `gae_gamma` | 0.99 | Reward discount |
| `cost_gamma` | 0.999 | Cost discount (higher for long-term) |
| `gae_lambda` | 0.95 | GAE lambda |
| `target_kl` | 0.01 | Trust region size |
| `damping_coeff` | 0.1 | Hessian damping |
| `backtrack_coeff` | 0.8 | Line search decay |
| `backtrack_iters` | 10 | Max line search steps |
| `margin_lr` | 0.05 | Constraint margin learning rate |
| `critic_epochs` | 10–80 | Critic updates per rollout |
| `state_model_epochs` | 10–80 | World model updates per rollout |
| `num_taus` | 8 | Quantile samples for pinball loss |
| `embedding_dim` | 64 | World model embedding size |
| `rnn_hidden_dim` | 128 | GRU hidden state size |
