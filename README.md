# tinker
Implementations of reinforcement learning algorithms

## Quick Start

To setup your environment, simply run:

```bash
scripts/setup_dev.sh
```

This will install `uv` for environment management, any package dependencies from `uv.lock`, as well as pre-commit hooks for code formatting defined in `.pre-commit-config.yaml`.

You'll also need to add your Weights and Biases key in a `.env` file as shown in the `.env.example`:

```bash
WANDB_KEY="YOUR_WANDB_API_KEY"
```

## Algorithms

| Directory | Algorithm | Description |
|---|---|---|
| `tinker/ppo/` | [PPO](tinker/ppo/README.md) | On-policy model-free actor-critic with clipped surrogate objective |
| `tinker/cpo/` | [CPO](tinker/cpo/README.md) | Constrained policy optimization with trust region and cost constraints |
| `tinker/cppo/` | [CPPO](tinker/cppo/README.md) | PPO with CVaR (tail-risk) cost constraints via Lagrangian relaxation |
| `tinker/acpo/` | [ACPO](tinker/acpo/README.md) | Model-based reward objective + model-free cost constraint (hybrid safe RL) |
| `tinker/var_cpo/` | [VaR-CPO](tinker/var_cpo/README.md) | CPO with chance constraints via Chebyshev's inequality on cost variance |
| `tinker/dreamerV3/` | [DreamerV3](tinker/dreamerV3/README.md) | Model-based RL with learned world model and imagined rollouts |
| `tinker/stock_pred/` | [Stock Prediction](tinker/stock_pred/README.md) | Quantile-based financial return and volatility prediction on GARCH environments |

All implementations are single-file, end-to-end JIT-compilable JAX with parallel seed execution via `jax.vmap`. Each directory has a `README.md` with architecture details, hyperparameters, and run instructions.
