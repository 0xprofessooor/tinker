"""Deep Deterministic Policy Gradient (DDPG)."""

import os
from typing import NamedTuple, Tuple

import chex
import flashbax as fbx
from flashbax.buffers.trajectory_buffer import BufferState
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import orthogonal, uniform
from flax.training.train_state import TrainState
from gymnax.environments.environment import Environment, EnvParams, EnvState
from gymnax.wrappers import LogWrapper

import wandb


class Actor(nn.Module):
    """Deterministic policy network for DDPG."""

    action_dim: int
    activation: callable = nn.tanh

    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(400, kernel_init=orthogonal(np.sqrt(2)))(obs)
        x = self.activation(x)
        x = nn.Dense(300, kernel_init=orthogonal(np.sqrt(2)))(x)
        x = self.activation(x)

        x = nn.Dense(self.action_dim, kernel_init=uniform(3e-3))(x)
        actions = nn.tanh(x)

        return actions


class Critic(nn.Module):
    """Q-function network for DDPG (takes both obs and action as input)."""

    activation: callable = nn.relu

    @nn.compact
    def __call__(self, obs, action):
        x = jnp.concatenate([obs, action], axis=-1)

        x = nn.Dense(400, kernel_init=orthogonal(np.sqrt(2)))(x)
        x = self.activation(x)
        x = nn.Dense(300, kernel_init=orthogonal(np.sqrt(2)))(x)
        x = self.activation(x)

        q_value = nn.Dense(1, kernel_init=uniform(3e-3))(x)

        return jnp.squeeze(q_value, axis=-1)


class DDPGTrainState(TrainState):
    target_params: chex.ArrayTree


class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    next_obs: jnp.ndarray
    done: jnp.ndarray


RunnerState = Tuple[
    DDPGTrainState, DDPGTrainState, EnvState, jnp.ndarray, BufferState, chex.PRNGKey
]


def add_noise(
    action: jnp.ndarray, noise_scale: float, rng: chex.PRNGKey
) -> jnp.ndarray:
    """Add Gaussian noise to actions for exploration."""
    noise = jax.random.normal(rng, action.shape) * noise_scale
    return jnp.clip(action + noise, -1.0, 1.0)


def make_train(
    env: Environment,
    env_params: EnvParams,
    num_steps: int,
    num_envs: int,
    batch_size: int,
    buffer_size: int,
    actor_epochs: int = 1,
    critic_epochs: int = 1,
    actor_activation: callable = nn.relu,
    critic_activation: callable = nn.relu,
    train_freq: int = 1,
    start_steps: int = 1000,
    explorer_noise_scale: float = 0.1,
    td_gamma: float = 0.99,
    lr_actor: float = 1e-3,
    lr_critic: float = 1e-3,
    anneal_lr: bool = True,
    polyak_coeff: float = 0.005,
    max_grad_norm: float = 1.0,
):
    env = LogWrapper(env)

    num_gradient_updates = (num_steps - start_steps) // train_freq

    def linear_schedule(initial_lr: float):
        def schedule(count):
            if not anneal_lr:
                return initial_lr
            frac = 1.0 - (count / num_gradient_updates)
            return initial_lr * frac

        return schedule

    def train(rng: chex.PRNGKey):
        # SETUP DUMMY PARAMS
        rng, dummy_key = jax.random.split(rng)
        action_space = env.action_space(env_params)
        action_scale = (action_space.high - action_space.low) / 2.0
        _, dummy_state = env.reset(dummy_key, env_params)
        dummy_action = action_space.sample(dummy_key)
        dummy_obs, _, dummy_reward, dummy_done, _ = env.step(
            dummy_key, dummy_state, dummy_action, env_params
        )

        # INIT NETWORKS
        actor = Actor(action_dim=action_space.shape[0], activation=actor_activation)
        critic = Critic(activation=critic_activation)
        rng, actor_key, critic_key = jax.random.split(rng, 3)
        actor_params = actor.init(actor_key, dummy_obs)
        critic_params = critic.init(critic_key, dummy_obs, dummy_action)

        # CREATE OPTIMIZERS
        if anneal_lr:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate=linear_schedule(lr_actor), eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate=linear_schedule(lr_critic), eps=1e-5),
            )
        else:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(lr_actor, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(lr_critic, eps=1e-5),
            )

        # CREATE TRAIN STATES
        actor_state = DDPGTrainState.create(
            apply_fn=actor.apply,
            params=actor_params,
            tx=actor_tx,
            target_params=jax.tree.map(lambda x: jnp.copy(x), actor_params),
        )
        critic_state = DDPGTrainState.create(
            apply_fn=critic.apply,
            params=critic_params,
            tx=critic_tx,
            target_params=jax.tree.map(lambda x: jnp.copy(x), critic_params),
        )

        # INIT BUFFER
        buffer = fbx.make_flat_buffer(
            max_length=buffer_size,
            min_length=batch_size,
            sample_batch_size=batch_size,
            add_sequences=False,
            add_batch_size=num_envs,
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

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, num_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        def ddpg_action(
            actor_state: DDPGTrainState, obs: jnp.ndarray, rng: chex.PRNGKey, step: int
        ) -> Tuple[jnp.ndarray, chex.PRNGKey]:
            rng, noise_rng = jax.random.split(rng)

            def random_action():
                random_actions = jax.vmap(action_space.sample)(
                    jax.random.split(noise_rng, num_envs)
                )
                return random_actions

            def policy_action():
                actor_actions = actor.apply(actor_state.params, obs)
                return action_scale * add_noise(
                    actor_actions,
                    noise_scale=explorer_noise_scale,
                    rng=noise_rng,
                )

            action = jax.lax.cond(
                step < start_steps,
                random_action,
                policy_action,
            )
            return action, rng

        def update_critic(carry, _):
            actor_state, critic_state, trajectories = carry

            next_actions = actor.apply(
                actor_state.target_params, trajectories.second.obs
            )
            next_q_values = critic.apply(
                critic_state.target_params, trajectories.second.obs, next_actions
            )
            target_q_values = (
                trajectories.first.reward
                + td_gamma * (1.0 - trajectories.first.done) * next_q_values
            )

            def loss_fn(params):
                q_values = critic.apply(
                    params, trajectories.first.obs, trajectories.first.action
                )
                td_errors = q_values - target_q_values
                loss = jnp.mean(td_errors**2)
                return loss

            loss, grads = jax.value_and_grad(loss_fn)(critic_state.params)
            critic_state = critic_state.apply_gradients(grads=grads)
            carry = (actor_state, critic_state, trajectories)
            return carry, loss

        def update_actor(carry, _):
            actor_state, critic_state, trajectories = carry

            def loss_fn(params):
                actions = actor.apply(params, trajectories.first.obs)
                q_values = critic.apply(
                    critic_state.params, trajectories.first.obs, actions
                )
                loss = -jnp.mean(q_values)
                return loss

            loss, grads = jax.value_and_grad(loss_fn)(actor_state.params)
            actor_state = actor_state.apply_gradients(grads=grads)
            carry = (actor_state, critic_state, trajectories)
            return carry, loss

        def update(
            actor_state: DDPGTrainState,
            critic_state: DDPGTrainState,
            buffer_state: BufferState,
            rng: chex.PRNGKey,
        ) -> Tuple[DDPGTrainState, DDPGTrainState, jnp.ndarray]:
            rng, buffer_rng = jax.random.split(rng)
            sample = buffer.sample(buffer_state, buffer_rng)
            trajectories = sample.experience
            (actor_state, critic_state, trajectories), critic_losses = jax.lax.scan(
                update_critic,
                (actor_state, critic_state, trajectories),
                None,
                critic_epochs,
            )
            (actor_state, critic_state, trajectories), actor_losses = jax.lax.scan(
                update_actor,
                (actor_state, critic_state, trajectories),
                None,
                actor_epochs,
            )

            # UPDATE TARGET NETWORKS
            actor_state = actor_state.replace(
                target_params=optax.incremental_update(
                    actor_state.params,
                    actor_state.target_params,
                    polyak_coeff,
                )
            )
            critic_state = critic_state.replace(
                target_params=optax.incremental_update(
                    critic_state.params,
                    critic_state.target_params,
                    polyak_coeff,
                )
            )
            return actor_state, critic_state, actor_losses, critic_losses

        def train_loop(carry: RunnerState, xs: int):
            env_step = xs
            actor_state, critic_state, env_state, obs, buffer_state, rng = carry

            # STEP ENV
            rng, rng_action, rng_step = jax.random.split(rng, 3)
            action, rng = ddpg_action(actor_state, obs, rng_action, env_step)
            rng_step = jax.random.split(rng, num_envs)
            next_obs, next_env_state, reward, done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(rng_step, env_state, action, env_params)

            # ADD TO BUFFER
            transition = Transition(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
            )
            buffer_state = buffer.add(buffer_state, transition)

            # UPDATE
            is_learning_step = (
                (buffer.can_sample(buffer_state))
                & (env_step >= start_steps)
                & (env_step % train_freq == 0)
            )
            rng, update_rng = jax.random.split(rng)
            actor_state, critic_state, actor_losses, critic_losses = jax.lax.cond(
                is_learning_step,
                lambda args: update(*args),
                lambda args: (
                    args[0],
                    args[1],
                    jnp.zeros(actor_epochs),
                    jnp.zeros(critic_epochs),
                ),
                (actor_state, critic_state, buffer_state, update_rng),
            )

            metrics = {
                "step": env_step,
                "actor_loss": actor_losses.mean(),
                "critic_loss": critic_losses.mean(),
                "buffer_size": buffer_state.current_index,
                "returns": info["returned_episode_returns"].mean(),
                "episode_lengths": info["returned_episode_lengths"].mean(),
                "is_learning": is_learning_step,
            }

            carry = (
                actor_state,
                critic_state,
                next_env_state,
                next_obs,
                buffer_state,
                rng,
            )
            return carry, metrics

        rng, _rng = jax.random.split(rng)
        runner_state = (actor_state, critic_state, env_state, obsv, buffer_state, _rng)

        step_counter = jnp.arange(num_steps)
        runner_state, metrics = jax.lax.scan(train_loop, runner_state, step_counter)
        return runner_state, metrics

    return train


if __name__ == "__main__":
    import gymnax

    SEED = 0
    NUM_SEEDS = 1
    WANDB = "online"

    env, env_params = gymnax.make("Pendulum-v1")

    wandb.login(os.environ.get("WANDB_KEY"))
    wandb.init(
        project="Tinker",
        tags=["DDPG", f"{env.name.upper()}", f"jax_{jax.__version__}"],
        name=f"ddpg_{env.name}",
        mode=WANDB,
    )

    rng = jax.random.PRNGKey(SEED)
    rngs = jax.random.split(rng, NUM_SEEDS)

    train_fn = make_train(
        env,
        env_params,
        num_steps=int(1e5),
        num_envs=1,
        batch_size=64,
        buffer_size=int(1e5),
        actor_epochs=1,
        critic_epochs=1,
        train_freq=1,
        anneal_lr=False,
    )
    train_vjit = jax.jit(jax.vmap(train_fn))
    runner_states, all_metrics = jax.block_until_ready(train_vjit(rngs))

    if WANDB == "online":
        num_steps = len(all_metrics["step"][0])
        for update_idx in range(num_steps):
            log_dict = {}

            for run_idx in range(NUM_SEEDS):
                run_prefix = f"run_{run_idx}"
                log_dict.update(
                    {
                        f"{run_prefix}/step": all_metrics["step"][run_idx][update_idx],
                        f"{run_prefix}/returns": all_metrics["returns"][run_idx][
                            update_idx
                        ],
                        f"{run_prefix}/actor_loss": all_metrics["actor_loss"][run_idx][
                            update_idx
                        ],
                        f"{run_prefix}/critic_loss": all_metrics["critic_loss"][
                            run_idx
                        ][update_idx],
                        f"{run_prefix}/buffer_size": all_metrics["buffer_size"][
                            run_idx
                        ][update_idx],
                        f"{run_prefix}/is_learning": all_metrics["is_learning"][
                            run_idx
                        ][update_idx],
                        f"{run_prefix}/episode_lengths": all_metrics["episode_lengths"][
                            run_idx
                        ][update_idx],
                    }
                )

            wandb.log(log_dict)
