"""IQN-MPC: Implicit Quantile Networks for Model Predictive Control.

Distributional state transition model using quantile regression,
combined with MPC planning under uncertainty.
"""

from tinker.iqn_mpc.iqn import (
    IQNStateNetwork,
    IQNTransitionModel,
    IQNTrainState,
    pinball_loss,
    make_iqn_train,
)
from tinker.iqn_mpc.mpc import (
    MPCConfig,
    MPCPlanner,
    sample_trajectories,
)

__all__ = [
    "IQNStateNetwork",
    "IQNTransitionModel",
    "IQNTrainState",
    "pinball_loss",
    "make_iqn_train",
    "MPCConfig",
    "MPCPlanner",
    "sample_trajectories",
]
