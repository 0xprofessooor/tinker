"""DreamerV3 Model-Based Deep Reinforcement Learning Algorithm."""

from typing import Sequence, Tuple

import chex
import distrax
import flax.linen as nn
import jax
from gymnax.environments.environment import Environment
from jax import numpy as jnp
from jax import scipy


@jax.jit
def symlog(x: jnp.ndarray) -> jnp.ndarray:
    """Symmetric log transform."""
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


@jax.jit
def symexp(x: jnp.ndarray) -> jnp.ndarray:
    """Symmetric exponential transform."""
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))


@jax.jit
def _uniform_mix(logits: jnp.ndarray, mix_coeff: float = 0.01) -> jnp.ndarray:
    probs = nn.softmax(logits, axis=-1)
    uniform = jnp.ones_like(probs) / logits.shape[-1]
    probs = (1 - mix_coeff) * probs + mix_coeff * uniform
    logits = scipy.special.logit(probs)
    return logits


@jax.jit
def _one_hot_state(logits: jnp.ndarray, rng: chex.PRNGKey) -> jnp.ndarray:
    stoch_latent_values = jax.random.categorical(rng, logits, axis=-1)
    z = nn.one_hot(stoch_latent_values, logits.shape[-1])
    return z


class WorldModel(nn.Module):
    """Learn latent state representation"""

    obs_size: int
    recurrent_size: int = 256
    stoch_size: int = 32
    one_hot_size: int = 32
    mlp_layers: int = 1
    mlp_size: int = 256

    def setup(self) -> None:
        self.sequence_model = nn.recurrent.GRUCell(self.recurrent_size)
        self.encoder = [nn.Dense(self.mlp_size) for _ in range(self.mlp_layers)] + [
            nn.Dense(self.stoch_size * self.one_hot_size)
        ]
        self.dynamics_model = [
            nn.Dense(self.mlp_size) for _ in range(self.mlp_layers)
        ] + [nn.Dense(self.stoch_size * self.one_hot_size)]
        self.decoder = [nn.Dense(self.mlp_size) for _ in range(self.mlp_layers)] + [
            nn.Dense(self.obs_size)
        ]
        self.reward_model = [
            nn.Dense(self.mlp_size) for _ in range(self.mlp_layers)
        ] + [nn.Dense(1)]
        self.continue_model = [
            nn.Dense(self.mlp_size) for _ in range(self.mlp_layers)
        ] + [nn.Dense(1)]

    def forward_sequence(
        self, h: jnp.ndarray, z: jnp.ndarray, action: jnp.ndarray
    ) -> jnp.ndarray:
        """Calculate the deterministic part of the latent state."""
        x = jnp.concatenate(
            [z.reshape(-1, self.one_hot_size * self.stoch_size), action], axis=-1
        )
        h, _ = self.sequence_model(inputs=x, carry=h)
        h = nn.silu(h)
        return h

    def forward_encoder(self, h: jnp.ndarray, obs: jnp.ndarray) -> jnp.ndarray:
        """Calculate the stochastic part of the latent state with the observation."""
        x = jnp.concatenate([h, obs], axis=-1)
        for layer in self.encoder:
            x = layer(x)
            x = nn.silu(x)
        stoch_logits = x.reshape((-1, self.stoch_size, self.one_hot_size))
        return stoch_logits

    def forward_dynamics(self, h: jnp.ndarray) -> jnp.ndarray:
        """Calculate the stochastic part of the latent state without the observation."""
        x = h
        for layer in self.dynamics_model:
            x = layer(x)
            x = nn.silu(x)
        stoch_logits = x.reshape((-1, self.stoch_size, self.one_hot_size))
        return stoch_logits

    def forward_reward(self, h: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        """Predict the reward from the latent state."""
        x = jnp.concatenate(
            [h, z.reshape(-1, self.one_hot_size * self.stoch_size)], axis=-1
        )
        for layer in self.reward_model:
            x = layer(x)
            x = nn.silu(x)
        return x

    def forward_continue(self, h: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        """Predict the continue signal from the latent state."""
        x = jnp.concatenate(
            [h, z.reshape(-1, self.one_hot_size * self.stoch_size)], axis=-1
        )
        for layer in self.continue_model:
            x = layer(x)
            x = nn.silu(x)
        return x

    def forward_decoder(self, h: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        """Predict the observation from the latent state."""
        x = jnp.concatenate(
            [h, z.reshape(-1, self.one_hot_size * self.stoch_size)], axis=-1
        )
        for layer in self.decoder:
            x = layer(x)
            x = nn.silu(x)
        return x

    def __call__(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        h: jnp.ndarray,
        z: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Dummy method only required for parameter initialization."""

        # Get latent state
        h = self.forward_sequence(h, z, action)
        z = self.forward_encoder(h, obs)

        # Predict next observation, reward, and continue
        obs_pred = self.forward_decoder(h, z)
        reward_pred = self.forward_reward(h, z)
        continue_pred = self.forward_continue(h, z)

        return obs_pred, reward_pred, continue_pred, h, z


class ActorCriticDiscrete(nn.Module):
    """Use latent state from WorldModel to predict actions and values."""

    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(
        self, latent_state: jnp.ndarray
    ) -> Tuple[distrax.Distribution, jnp.ndarray]:
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            64,
            kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
            bias_init=nn.initializers.constant(0.0),
        )(latent_state)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64,
            kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
            bias_init=nn.initializers.constant(0.0),
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.constant(0.0),
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64,
            kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
            bias_init=nn.initializers.constant(0.0),
        )(latent_state)
        critic = activation(critic)
        critic = nn.Dense(
            64,
            kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
            bias_init=nn.initializers.constant(0.0),
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.constant(0.0),
        )(critic)

        return pi, jnp.squeeze(critic, axis=-1)


def make_train(
    env: Environment,
    num_steps: int,
):
    """Generate a jittable JAX DreamerV3 train function."""

    def train(rng: chex.PRNGKey):
        # Initialize networks
        world_model = WorldModel(env.observation_space().shape[0])
        actor_critic = ActorCriticDiscrete(env.action_space().n)
        params_rng, forward_rng = jax.random.split(rng)

    return train
