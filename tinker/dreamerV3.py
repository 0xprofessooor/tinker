"""DreamerV3 Model-Based Deep Reinforcement Learning Algorithm."""

from typing import Sequence, Tuple

import chex
import distrax
import flashbax as fbx
import flax.linen as nn
import jax
import optax
from flashbax.buffers.trajectory_buffer import BufferState
from flax.training.train_state import TrainState
from gymnax.environments.environment import Environment
from gymnax.wrappers import LogWrapper
from jax import numpy as jnp
from jax import scipy

import wandb


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


DEFAULT_WORLD_MODEL_CONFIG = {
    "recurrent_size": 256,
    "stoch_size": 32,
    "one_hot_size": 32,
    "mlp_layers": 1,
    "mlp_size": 256,
    "reward_bins": 255,
}


DEFAULT_ACTOR_CRITIC_CONFIG = {"activation": "tanh"}


class WorldModel(nn.Module):
    """Learn latent state representation"""

    obs_size: int
    recurrent_size: int = 256
    stoch_size: int = 32
    one_hot_size: int = 32
    mlp_layers: int = 1
    mlp_size: int = 256
    reward_bins: int = 255

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
        ] + [nn.Dense(self.reward_bins)]
        self.continue_model = [
            nn.Dense(self.mlp_size) for _ in range(self.mlp_layers)
        ] + [nn.Dense(1)]

    def forward_sequence(
        self, h: jnp.ndarray, z: jnp.ndarray, action: jnp.ndarray
    ) -> jnp.ndarray:
        """Calculate the deterministic part of the latent state."""
        x = jnp.concatenate([z, action], axis=-1)
        h, _ = self.sequence_model(inputs=x, carry=h)
        h = nn.silu(h)
        return h

    def forward_encoder(self, h: jnp.ndarray, obs: jnp.ndarray) -> jnp.ndarray:
        """Calculate the stochastic part of the latent state with the observation."""
        z_logits = jnp.concatenate([h, obs], axis=-1)
        for layer in self.encoder:
            z_logits = layer(z_logits)
            z_logits = nn.silu(z_logits)
        return z_logits

    def forward_dynamics(self, h: jnp.ndarray) -> jnp.ndarray:
        """Calculate the stochastic part of the latent state without the observation."""
        z_logits = h
        for layer in self.dynamics_model:
            z_logits = layer(z_logits)
            z_logits = nn.silu(z_logits)
        return z_logits

    def forward_reward(self, latent_state: jnp.ndarray) -> jnp.ndarray:
        """Predict the reward from the latent state."""
        for layer in self.reward_model:
            latent_state = layer(latent_state)
            latent_state = nn.silu(latent_state)
        return latent_state

    def forward_continue(self, latent_state: jnp.ndarray) -> jnp.ndarray:
        """Predict the continue signal from the latent state."""
        for layer in self.continue_model:
            latent_state = layer(latent_state)
            latent_state = nn.silu(latent_state)
        return latent_state

    def forward_decoder(self, latent_state: jnp.ndarray) -> jnp.ndarray:
        """Predict the observation from the latent state."""
        for layer in self.decoder:
            latent_state = layer(latent_state)
            latent_state = nn.silu(latent_state)
        return latent_state

    def __call__(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        h: jnp.ndarray,
        z: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Dummy method for parameter initialization."""
        h = self.forward_sequence(h, z, action)
        _ = self.forward_dynamics(h)
        z = self.forward_encoder(h, obs)
        latent_state = jnp.concatenate([h, z], axis=-1)
        obs_pred = self.forward_decoder(latent_state)
        reward_pred = self.forward_reward(latent_state)
        continue_pred = self.forward_continue(latent_state)
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


@chex.dataclass(frozen=True)
class Transition:
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    next_obs: jnp.ndarray
    done: jnp.ndarray


def make_train(
    env: Environment,
    num_steps: int,
    buffer_size: int,
    epochs_world_model: int,
    sequence_length: int = 50,
    imagination_horizon: int = 15,
    world_model_update_freq: int = 1,
    uniform_mix_coeff: float = 0.01,
    lr_world_model: float = 1e-4,
    lr_actor_critic: float = 8e-5,
    world_model_config: dict = DEFAULT_WORLD_MODEL_CONFIG,
    actor_critic_config: dict = DEFAULT_ACTOR_CRITIC_CONFIG,
    log_freq: int = 100,
):
    """Generate a jittable JAX DreamerV3 train function."""

    env = LogWrapper(env)

    def train(rng: chex.PRNGKey):
        # INIT DUMMY VARIABLES
        dummy_rng = jax.random.PRNGKey(0)
        _, dummy_state = env.reset(dummy_rng)
        dummy_action = env.action_space().sample(dummy_rng)
        dummy_obs, _, dummy_reward, dummy_done, _ = env.step(
            dummy_rng, dummy_state, dummy_action
        )
        if dummy_action.ndim == 0:
            dummy_action = jnp.expand_dims(dummy_action, axis=0)

        # INIT NETWORKS
        world_model_rng, actor_critic_rng = jax.random.split(rng)
        world_model = WorldModel(
            env.observation_space(env.default_params).shape[0], **world_model_config
        )
        h = jnp.zeros((world_model.recurrent_size))  # Initial hidden state
        z = jnp.zeros(
            (world_model.stoch_size * world_model.one_hot_size)
        )  # Initial stochastic state
        latent_state = jnp.concatenate([h, z], axis=-1)
        actor_critic = ActorCriticDiscrete(env.action_space().n, **actor_critic_config)
        world_model_params = world_model.init(
            world_model_rng,
            dummy_obs,
            dummy_action,
            h,
            z,
        )
        actor_critic_params = actor_critic.init(actor_critic_rng, latent_state)

        # INIT OPTIMIZER
        tx_world_model = optax.adam(learning_rate=lr_world_model)
        tx_actor_critic = optax.adam(learning_rate=lr_actor_critic)

        # INIT BUFFER
        # shape = (add_batch_size, time, element_dim)
        buffer = fbx.make_trajectory_buffer(
            add_batch_size=1,
            sample_batch_size=epochs_world_model,
            sample_sequence_length=sequence_length,
            period=1,
            min_length_time_axis=sequence_length,
            max_length_time_axis=buffer_size,
        )
        buffer: fbx.trajectory_buffer.TrajectoryBuffer = buffer.replace(
            init=jax.jit(buffer.init),
            add=jax.jit(buffer.add, donate_argnums=0),
            sample=jax.jit(buffer.sample),
            can_sample=jax.jit(buffer.can_sample),
        )
        dummy_transition = Transition(
            obs=dummy_obs,
            action=dummy_action,
            reward=dummy_reward,
            next_obs=dummy_obs,
            done=dummy_done,
        )
        buffer_state = buffer.init(dummy_transition)

        # INIT TRAIN STATE
        world_model_state = TrainState.create(
            apply_fn=world_model.apply,
            params=world_model_params,
            tx=tx_world_model,
        )
        actor_critic_state = TrainState.create(
            apply_fn=actor_critic.apply,
            params=actor_critic_params,
            tx=tx_actor_critic,
        )

        # INIT ENV
        _, env_rng = jax.random.split(rng)
        obs, env_state = env.reset(env_rng)

        def eval_latent_state(carry: tuple, x: jnp.ndarray):
            """Evaluate the world model latent state from an observation and action."""
            h, world_model_state, rng = carry
            obs = x[: world_model.obs_size]
            action = x[world_model.obs_size : -1]
            done = x[-1]

            z_logits = world_model.apply(
                world_model_state.params,
                h,
                symlog(obs),
                method=world_model.forward_encoder,
            ).reshape((world_model.stoch_size, world_model.one_hot_size))
            z_logits = _uniform_mix(z_logits, mix_coeff=uniform_mix_coeff)
            _, rng_one_hot = jax.random.split(rng)
            z = _one_hot_state(z_logits, rng_one_hot)
            z_model_in = z.reshape((world_model.stoch_size * world_model.one_hot_size))
            h = jax.lax.cond(  # reset deterministic state if done
                done == 1,
                lambda _: jnp.zeros_like(h),
                lambda _: world_model.apply(
                    world_model_state.params,
                    h,
                    z_model_in,
                    action,
                    method=world_model.forward_sequence,
                ),
                None,
            )
            latent_state = jnp.concatenate([h, z_model_in], axis=-1)
            return (h, world_model_state), latent_state

        def update_world_model(
            world_model_state: TrainState, buffer_state: BufferState, rng: chex.PRNGKey
        ):
            """Run an update step on the world model."""
            rng_epoch, rng_sample = jax.random.split(rng)
            buffer_sample = buffer.sample(buffer_state, rng_sample).experience
            obs = buffer_sample.obs
            action = buffer_sample.action
            done = jnp.expand_dims(buffer_sample.done, axis=-1)
            data = jnp.concatenate(
                [obs, action, done], axis=-1
            )  # (num_epochs, time, obs_dim + action_dim + 1)

            def epoch_update(carry: tuple, sequence: jnp.ndarray):
                """Update the world model for a single epoch."""
                world_model_state, rng = carry
                h = jnp.zeros((world_model.recurrent_size,))
                _, rng_ = jax.random.split(rng)
                _, latent_states = jax.lax.scan(
                    eval_latent_state, (h, world_model_state, rng_), sequence
                )

            _, _ = jax.lax.scan(
                epoch_update, init=(world_model_state, rng_epoch), xs=data
            )

            return world_model_state, 0.0, 0.0, 0.0, 0.0

        def update_actor_critic():
            pass

        def train_loop(carry: tuple, env_step: int):
            (
                env_state,
                obs,
                h,
                world_model_state,
                actor_critic_state,
                buffer_state,
                rng,
            ) = carry
            rng_action, rng_step, world_model_update_rng, next_rng = jax.random.split(
                rng, 4
            )

            # STEP ENV
            z_logits = world_model.apply(
                world_model_state.params,
                h,
                symlog(obs),
                method=world_model.forward_encoder,
            ).reshape((world_model.stoch_size, world_model.one_hot_size))
            z_logits = _uniform_mix(z_logits, mix_coeff=uniform_mix_coeff)
            _, rng_one_hot = jax.random.split(rng_step)
            z = _one_hot_state(z_logits, rng_one_hot)
            z_model_in = z.reshape((world_model.stoch_size * world_model.one_hot_size))
            latent_state = jnp.concatenate([h, z_model_in], axis=-1)
            pi, _ = actor_critic.apply(actor_critic_state.params, latent_state)
            action = pi.sample(seed=rng_action)
            next_obs, env_state, reward, done, info = env.step(
                rng_step, env_state, action
            )
            if action.ndim == 0:
                action = jnp.expand_dims(action, axis=0)
            h = jax.lax.cond(  # reset deterministic state if done
                done == 1,
                lambda _: jnp.zeros_like(h),
                lambda _: world_model.apply(
                    world_model_state.params,
                    h,
                    z_model_in,
                    action,
                    method=world_model.forward_sequence,
                ),
                None,
            )

            # ADD TO BUFFER
            transition = Transition(
                obs=obs[None][None],  # (1, 1, obs_dim)
                action=action[None][None],  # (1, 1, action_dim)
                reward=reward[None][None],  # (1, 1)
                next_obs=next_obs[None][None],  # (1, 1, obs_dim)
                done=done[None][None],  # (1, 1)
            )
            buffer_state = buffer.add(buffer_state, transition)

            # UPDATE WORLD MODEL
            is_world_model_update_step = (buffer.can_sample(buffer_state)) & (
                env_step % world_model_update_freq == 0
            )

            world_model_state, total_world_loss, pred_loss, dyn_loss, rep_loss = (
                jax.lax.cond(
                    is_world_model_update_step,
                    lambda world_model_state,
                    buffer_state,
                    world_model_update_rng: update_world_model(
                        world_model_state, buffer_state, world_model_update_rng
                    ),
                    lambda world_model_state, buffer_state, world_model_update_rng: (
                        world_model_state,
                        jnp.array(0.0),
                        jnp.array(0.0),
                        jnp.array(0.0),
                        jnp.array(0.0),
                    ),
                    world_model_state,
                    buffer_state,
                    world_model_update_rng,
                )
            )

            # LOGGING METRICS
            metrics = {
                "timesteps": env_step,
                "world_model_updates": world_model_state.step,
                "world_model_loss": total_world_loss,
                "pred_loss": pred_loss,
                "dyn_loss": dyn_loss,
                "rep_loss": rep_loss,
                "returns": info["returned_episode_returns"],
                "episode_lengths": info["returned_episode_lengths"],
            }

            if log_freq > 0 and metrics["timesteps"] % log_freq == 0:
                jax.debug.callback(wandb.log, metrics)

            runner_state = (
                env_state,
                next_obs,
                h,
                world_model_state,
                actor_critic_state,
                buffer_state,
                next_rng,
            )

            return runner_state, metrics

        runner_state = (
            env_state,
            obs,
            h,
            world_model_state,
            actor_critic_state,
            buffer_state,
            rng,
        )
        final_state, metrics = jax.lax.scan(
            train_loop,
            init=runner_state,
            xs=jnp.arange(1, num_steps + 1),
            length=num_steps,
        )
        return final_state, metrics

    return train


if __name__ == "__main__":
    from gymnax.environments import CartPole

    env = CartPole()
    train = make_train(
        env=env,
        num_steps=100,
        buffer_size=1000,
        batch_size=2,
        sequence_length=10,
        log_freq=0,
    )
    train_jit = jax.jit(train)

    key = jax.random.PRNGKey(0)
    out = train(key)

    # wandb.login(os.environ.get("WANDB_KEY"))
    # wandb.init(
    #    project="Tinker",
    #    tags=["DreamerV3", f"{env.name.upper()}", f"jax_{jax.__version__}"],
    #    name=f"dreamer_{env.name}",
    #    mode="online",
    # )
