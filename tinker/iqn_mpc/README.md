# IQN-MPC: Implicit Quantile Networks for Model Predictive Control

Distributional model learning combined with uncertainty-aware planning.

## Overview

This module implements:

1. **IQN State Network**: Learns the full distribution of state transitions P(s'|s,a) via quantile regression
2. **MPC Planner**: Uses the learned model for planning with support for:
   - Risk-sensitive objectives (CVaR)
   - Chance constraints

## Key Components

### IQN (Implicit Quantile Network)

Unlike categorical approaches (C51), IQN uses implicit quantile functions:
- Input: (state, action, τ) where τ ~ U(0,1)
- Output: τ-quantile of next state distribution
- Trained with pinball loss (quantile regression)

**Architecture:**
```
(s, a) → Encoder → embedding
    τ  → CosineEmbed → τ_embedding
         embedding ⊙ τ_embedding → Decoder → s'_τ
```

### MPC Planning

Uses Cross-Entropy Method (CEM) to optimize action sequences:
1. Sample action sequences from current distribution
2. Roll out trajectories using IQN (sampling quantiles)
3. Evaluate with risk measure (expectation or CVaR)
4. Optionally filter by chance constraints
5. Update distribution from elite samples

## Usage

```python
from tinker.iqn_mpc import (
    make_iqn_train,
    MPCConfig,
    MPCPlanner,
    Transition,
)

# Collect transitions from environment
transitions = Transition(
    state=states,      # (n_transitions, state_dim)
    action=actions,    # (n_transitions, action_dim)
    next_state=next_states,
    reward=rewards,
    done=dones,
)

# Train IQN model
train_fn = make_iqn_train(
    state_dim=4,
    action_dim=1,
    num_updates=10000,
)
model, metrics = train_fn(key, transitions)

# Create MPC planner
config = MPCConfig(
    horizon=20,
    action_dim=1,
    action_low=-1.0,
    action_high=1.0,
    risk_level=0.2,  # CVaR_0.2 (risk-averse)
)

def reward_fn(state, action):
    return -state[0]**2  # Example: minimize first state component

planner = MPCPlanner(model, reward_fn, config)
action, info = planner.plan(current_state, key)
```

## Evaluating the IQN Model

### Learning Quality

Use `evaluate_iqn_calibration` to check if the model is well-calibrated:

```python
from tinker.iqn_mpc import evaluate_iqn_calibration

cal_metrics = evaluate_iqn_calibration(model, test_transitions)
print(f"Mean calibration error: {cal_metrics['calibration_error_mean']:.4f}")
```

A well-calibrated model should have empirical coverage ≈ τ for all quantile levels.

### MPC Solver Quality

Evaluate by comparing:
1. **Open-loop performance**: How well does the optimized sequence perform?
2. **Closed-loop performance**: How well does receding horizon control work?
3. **Constraint satisfaction**: Empirical violation rate vs. specified probability

## References

- Dabney et al. (2018) "Implicit Quantile Networks for Distributional RL"
- Lobo, Fazel, Boyd (2002) "Convex Optimization under Uncertainty"
- Chua et al. (2018) "PETS: Model-Based RL via Ensembles"
