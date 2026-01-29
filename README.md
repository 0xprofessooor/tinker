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

There is a `README.md` file in each algorithm folder with more details on how to run.
