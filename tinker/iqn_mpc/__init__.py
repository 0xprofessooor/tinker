"""IQN-MPC: Implicit Quantile Networks for Model Predictive Control.

Distributional state transition model using quantile regression,
combined with gradient-based MPC planning under uncertainty.

Key components:
- IQN: Learns P(s'|s,a) via quantile regression
- Gradient MPC: Direct optimization through learned dynamics
- CVaR: Tail risk control using distributional quantiles

References:
- Dabney et al. (2018) - IQN architecture
- Lobo, Fazel, Boyd (2002) - Portfolio optimization with transaction costs
- Rockafellar & Uryasev (2000) - CVaR optimization
"""

from tinker.iqn_mpc.iqn import (
    IQNStateNetwork,
    IQNTransitionModel,
    IQNTrainState,
    pinball_loss,
    make_iqn_train,
    evaluate_iqn_calibration,
)
from tinker.iqn_mpc.mpc import (
    MPCConfig,
    MPCPlanner,
    sample_trajectories,
)
from tinker.iqn_mpc.gradient_mpc import (
    create_gradient_mpc,
    create_gradient_mpc_with_constraints,
)

__all__ = [
    # IQN model
    "IQNStateNetwork",
    "IQNTransitionModel",
    "IQNTrainState",
    "pinball_loss",
    "make_iqn_train",
    "evaluate_iqn_calibration",
    # CEM-based MPC (baseline)
    "MPCConfig",
    "MPCPlanner",
    "sample_trajectories",
    # Gradient-based MPC (recommended)
    "create_gradient_mpc",
    "create_gradient_mpc_with_constraints",
]
