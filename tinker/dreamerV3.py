"""DreamerV3 Model-Based Deep Reinforcement Learning Algorithm."""

from typing import Sequence, Tuple

import chex
import distrax
import flax.linen as nn
import jax
from gymnax.environments.environment import Environment
from jax import numpy as jnp
from jax import scipy


class WorldModel(nn.Module):
    """Learn latent state representation"""

    obs_size: int
    hidden_size: int = 512
    encoder_size: int = 512
    encoder_layers: int = 4
    decoder_size: int = 512
    decoder_layers: int = 4
    stoch_size: int = 32
    one_hot_size: int = 32
    unimix: float = 0.01

    def setup(self) -> None:
        self.sequence_model = nn.recurrent.GRUCell(self.hidden_size)
        self.encoder = nn.Dense(self.stoch_size * self.one_hot_size)
        self.dynamics_model = nn.Dense(self.stoch_size * self.one_hot_size)
        self.decoder = [
            nn.Dense(self.decoder_size) for _ in range(self.decoder_layers)
        ] + [nn.Dense(self.obs_size)]
        self.reward_model = nn.Dense(1)
        self.continue_model = nn.Dense(1)

    def _uniform_mix(self, stoch_logits: jnp.ndarray) -> jnp.ndarray:
        if self.unimix > 0:
            probs = nn.softmax(stoch_logits, axis=-1)
            uniform = jnp.ones_like(probs) / self.one_hot_size
            probs = (1 - self.unimix) * probs + self.unimix * uniform
            stoch_logits = scipy.special.logit(probs)
        return stoch_logits

    def forward_sequence(
        self, h: jnp.ndarray, z: jnp.ndarray, action: jnp.ndarray
    ) -> jnp.ndarray:
        """Calculate the deterministic part of the latent state."""
        h, _ = self.sequence_model(
            inputs=jnp.concatenate([z, action], axis=-1), carry=h
        )
        return h

    def forward_encoder(
        self, h: jnp.ndarray, obs: jnp.ndarray, rng: chex.PRNGKey
    ) -> jnp.ndarray:
        """Calculate the stochastic part of the latent state with the observation."""
        stoch_logits = self.encoder(jnp.concatenate([h, obs], axis=-1)).reshape(
            (-1, self.stoch_size, self.one_hot_size)
        )
        stoch_logits = self._uniform_mix(stoch_logits)
        stoch_latent_values = jax.random.categorical(rng, stoch_logits, axis=-1)
        z = nn.one_hot(stoch_latent_values, self.one_hot_size)
        return z

    def forward_dynamics(self, h: jnp.ndarray, rng: chex.PRNGKey) -> jnp.ndarray:
        """Calculate the stochastic part of the latent state without the observation."""
        stoch_logits = self.dynamics_model(h).reshape(
            (-1, self.stoch_size, self.one_hot_size)
        )
        stoch_logits = self._uniform_mix(stoch_logits)
        stoch_latent_values = jax.random.categorical(rng, stoch_logits, axis=-1)
        z = nn.one_hot(stoch_latent_values, self.one_hot_size)
        return z

    def forward_reward(self, latent_state: jnp.ndarray) -> jnp.ndarray:
        """Predict the reward from the latent state."""
        return self.reward_model(latent_state)

    def forward_continue(self, latent_state: jnp.ndarray) -> jnp.ndarray:
        """Predict the continue signal from the latent state."""
        return self.continue_model(latent_state)

    def forward_decoder(self, latent_state: jnp.ndarray) -> jnp.ndarray:
        """Predict the observation from the latent state."""
        obs_pred = latent_state
        for layer in self.decoder:
            obs_pred = layer(obs_pred)
        return obs_pred

    def __call__(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        latent_state: jnp.ndarray,
        rng: chex.PRNGKey,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Predict the next latent state and reward."""
        h = latent_state[:, : self.hidden_size]
        z = latent_state[:, self.hidden_size :]

        # Get latent state
        h = self.forward_sequence(h, z, action)
        z = self.forward_encoder(h, obs, rng)
        z = z.reshape(-1, self.one_hot_size * self.stoch_size)
        next_latent_state = jnp.concatenate([h, z], axis=-1)

        # Predict next observation, reward, and continue
        obs_pred = self.forward_decoder(next_latent_state)
        reward_pred = self.forward_reward(next_latent_state)
        continue_pred = self.forward_continue(next_latent_state)

        return obs_pred, reward_pred, continue_pred, next_latent_state


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


@jax.jit
def symlog(x: jnp.ndarray) -> jnp.ndarray:
    """Symmetric log transform."""
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


@jax.jit
def symexp(x: jnp.ndarray) -> jnp.ndarray:
    """Symmetric exponential transform."""
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))


def make_train(
    env: Environment,
    num_steps: int,
    num_envs: int,
    train_freq: int,
    num_minibatches: int,
    num_epochs: int,
    activation: str,
    lr: float,
    anneal_lr: bool,
    gamma: float,
    gae_lambda: float,
    entropy_coef: float,
    value_coef: float,
    max_grad_norm: float,
    clip_eps: float,
    log_wandb: bool = False,
):
    """Generate a jittable JAX DreamerV3 train function."""
