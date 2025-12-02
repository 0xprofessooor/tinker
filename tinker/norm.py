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
def update(state: RunningMeanStdState, batch_x: jax.Array):
    """
    Updates the running statistics using a batch of data.
    Uses the parallel Welford algorithm for numerical stability.
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
def normalize(state: RunningMeanStdState, x: jax.Array):
    """
    Normalizes x to be roughly N(0, 1) and clips huge outliers.
    """
    return jnp.clip((x - state.mean) / jnp.sqrt(state.var + 1e-8), -10.0, 10.0)
