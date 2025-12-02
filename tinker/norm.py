import functools
import jax
import jax.numpy as jnp
from typing import NamedTuple


class RunningMeanStdState(NamedTuple):
    mean: jax.Array
    var: jax.Array
    count: float


@functools.partial(jax.jit, static_argnames=["shape"])
def init(shape: tuple):
    """Initialize running mean/std state."""
    return RunningMeanStdState(
        mean=jnp.zeros(shape),
        var=jnp.ones(shape),
        count=1e-4,  # Small epsilon to prevent div by zero on first step
    )


@jax.jit
def welford_update(state: RunningMeanStdState, batch_x: jax.Array):
    """
    Updates the running statistics using the parallel Welford algorithm.

    This provides numerically stable cumulative mean and variance computation
    where all historical observations have equal weight. Suitable for stationary
    distributions where statistics should converge to true population values.

    Args:
        state: Current running statistics
        batch_x: New batch of observations, shape (batch_size, *feature_dims)

    Returns:
        Updated RunningMeanStdState with new mean, variance, and count

    Note: As count grows, new observations have decreasing influence.
    For non-stationary data, consider using ema_update() instead.
    """
    batch_mean = jnp.mean(batch_x, axis=0)
    batch_var = jnp.var(batch_x, axis=0)
    batch_count = batch_x.shape[0]

    delta = batch_mean - state.mean
    tot_count = state.count + batch_count

    new_mean = state.mean + delta * batch_count / tot_count

    m_a = state.var * state.count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
    new_var = M2 / tot_count

    return RunningMeanStdState(mean=new_mean, var=new_var, count=tot_count)


@jax.jit
def ema_update(state: RunningMeanStdState, batch_x: jax.Array, momentum: float = 0.95):
    """
    Updates the running statistics using exponential moving average.
    Suitable for non-stationary data.

    Args:
        state: Current running statistics
        batch_x: New batch of observations, shape (batch_size, *feature_dims)
        momentum: Decay factor (default 0.95)
                  - 0.99: Slow adaptation (~100 update memory)
                  - 0.95: Moderate adaptation (~20 update memory)
                  - 0.9: Fast adaptation (~10 update memory)

    For portfolio/trading environments with regime changes, 0.95 is recommended.
    """
    batch_mean = jnp.mean(batch_x, axis=0)
    batch_var = jnp.var(batch_x, axis=0)
    batch_count = batch_x.shape[0]

    new_mean = momentum * state.mean + (1 - momentum) * batch_mean
    new_var = momentum * state.var + (1 - momentum) * batch_var
    new_count = state.count + batch_count

    return RunningMeanStdState(mean=new_mean, var=new_var, count=new_count)


@jax.jit
def normalize(state: RunningMeanStdState, x: jax.Array):
    """
    Normalizes x to be roughly N(0, 1) and clips huge outliers.
    """
    return jnp.clip((x - state.mean) / jnp.sqrt(state.var + 1e-8), -10.0, 10.0)
