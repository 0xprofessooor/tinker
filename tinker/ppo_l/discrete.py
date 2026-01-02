"""Proximal Policy Optimization Lagrangian (PPO-L) with Categorical Policy."""

import os
from typing import NamedTuple, Tuple, Any

import chex
import distrax
import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from gymnax.environments.environment import Environment, EnvParams
from gymnax.environments import spaces
from safenax.wrappers import LogWrapper
from safenax import FrozenLakeV2

from tinker import log


class ActorCritic(nn.Module):
    """Actor critic neural network with disjoint architectures."""

    action_dim: int
    activation: callable = nn.tanh

    @nn.compact
    def __call__(self, x):
        # --- Actor ---
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = self.activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = self.activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        # --- Reward Critic ---
        critic_r = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic_r = self.activation(critic_r)
        critic_r = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic_r)
        critic_r = self.activation(critic_r)
        critic_r = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic_r
        )

        # --- Cost Critic (New) ---
        critic_c = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic_c = self.activation(critic_c)
        critic_c = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic_c)
        critic_c = self.activation(critic_c)
        critic_c = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic_c
        )

        return pi, jnp.squeeze(critic_r, axis=-1), jnp.squeeze(critic_c, axis=-1)


class PPOLTrainState(TrainState):
    lagrange_param: jnp.ndarray


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray  # Reward Value
    cost_value: jnp.ndarray  # Cost Value
    reward: jnp.ndarray
    cost: jnp.ndarray  # Immediate Cost
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(
    env: Environment,
    env_params: EnvParams,
    num_steps: int,
    num_envs: int,
    train_freq: int,
    batch_size: int,
    num_epochs: int,
    activation: callable = nn.tanh,
    lr: float = 3e-4,
    anneal_lr: bool = False,
    gae_gamma: float = 0.99,
    gae_lambda: float = 0.95,
    entropy_coeff: float = 0.01,
    value_coeff: float = 0.5,
    max_grad_norm: float = 0.5,
    ratio_clip: float = 0.2,
    # PPO-L Arguments
    cost_limit: float = 5.0,  # Max allowed cost per episode
    lagrange_lr: float = 0.05,
    initial_lambda: float = 1.0,
):
    num_updates = num_steps // train_freq
    num_minibatches = (num_envs * train_freq) // batch_size
    env = LogWrapper(env)

    def linear_schedule(count):
        frac = 1.0 - (count // (num_minibatches * num_epochs)) / num_updates
        return lr * frac

    def train(rng: chex.PRNGKey) -> Tuple[TrainState, dict]:
        # INIT NETWORK
        network = ActorCritic(env.action_space(env_params).n, activation=activation)
        rng, _rng = jax.random.split(rng)

        # PREPROCESSING (One-Hot for Discrete)
        obs_space = env.observation_space(env_params)
        if isinstance(obs_space, spaces.Discrete):
            init_x = jnp.zeros((obs_space.n,))

            def preprocess_obs(obs: jax.Array) -> jax.Array:
                return jax.nn.one_hot(obs, obs_space.n)
        else:
            init_x = jnp.zeros(obs_space.shape)

            def preprocess_obs(obs: jax.Array) -> jax.Array:
                return obs

        network_params = network.init(_rng, init_x)

        if anneal_lr:
            tx = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(lr, eps=1e-5),
            )

        train_state = PPOLTrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
            lagrange_param=jnp.array(initial_lambda),
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, num_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        obsv = preprocess_obs(obsv)

        # TRAIN LOOP
        def _update_step(runner_state, _):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, _):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, val_r, val_c = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, num_envs)
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)

                # Extract Cost from Info (SafeNax standard)
                # If info["cost"] is not present, define your cost logic here.
                cost = info.get("cost", jnp.zeros_like(reward))

                obsv = preprocess_obs(obsv)
                transition = Transition(
                    done, action, val_r, val_c, reward, cost, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, train_freq
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            _, last_val_r, last_val_c = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val, is_cost=False):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done = transition.done

                    # Select reward stream or cost stream
                    val = transition.cost_value if is_cost else transition.value
                    rew = transition.cost if is_cost else transition.reward

                    delta = rew + gae_gamma * next_value * (1 - done) - val
                    gae = delta + gae_gamma * gae_lambda * (1 - done) * gae
                    return (gae, val), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                # Return (Advantages, Targets)
                return advantages, advantages + (
                    traj_batch.cost_value if is_cost else traj_batch.value
                )

            adv_r, targ_r = _calculate_gae(traj_batch, last_val_r, is_cost=False)
            adv_c, targ_c = _calculate_gae(traj_batch, last_val_c, is_cost=True)

            # UPDATE NETWORK
            def _update_epoch(update_state, _):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, adv_r, adv_c, targ_r, targ_c = batch_info

                    def _loss_fn(
                        params, traj_batch, adv_r, adv_c, targ_r, targ_c, lagrange_val
                    ):
                        # RERUN NETWORK
                        pi, val_r, val_c = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # COMBINE ADVANTAGES: A = A_reward - lambda * A_cost
                        combined_advantage = adv_r - lagrange_val * adv_c

                        # VALUE LOSSES
                        # Reward Value Loss
                        val_r_pred_clipped = traj_batch.value + (
                            val_r - traj_batch.value
                        ).clip(-ratio_clip, ratio_clip)
                        val_r_loss = (
                            0.5
                            * jnp.maximum(
                                jnp.square(val_r - targ_r),
                                jnp.square(val_r_pred_clipped - targ_r),
                            ).mean()
                        )

                        # Cost Value Loss
                        val_c_pred_clipped = traj_batch.cost_value + (
                            val_c - traj_batch.cost_value
                        ).clip(-ratio_clip, ratio_clip)
                        val_c_loss = (
                            0.5
                            * jnp.maximum(
                                jnp.square(val_c - targ_c),
                                jnp.square(val_c_pred_clipped - targ_c),
                            ).mean()
                        )

                        total_value_loss = val_r_loss + val_c_loss

                        # ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        # Normalize combined advantage
                        combined_advantage = (
                            combined_advantage - combined_advantage.mean()
                        ) / (combined_advantage.std() + 1e-8)

                        loss_actor1 = ratio * combined_advantage
                        loss_actor2 = (
                            jnp.clip(ratio, 1.0 - ratio_clip, 1.0 + ratio_clip)
                            * combined_advantage
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + value_coeff * total_value_loss
                            - entropy_coeff * entropy
                        )
                        return total_loss, (val_r_loss, val_c_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params,
                        traj_batch,
                        adv_r,
                        adv_c,
                        targ_r,
                        targ_c,
                        train_state.lagrange_param,
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, adv_r, adv_c, targ_r, targ_c, rng = (
                    update_state
                )
                rng, _rng = jax.random.split(rng)

                # Flatten and Shuffle Batch
                total_batch_size = batch_size * num_minibatches
                permutation = jax.random.permutation(_rng, total_batch_size)

                batch = (traj_batch, adv_r, adv_c, targ_r, targ_c)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((total_batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [num_minibatches, batch_size] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    traj_batch,
                    adv_r,
                    adv_c,
                    targ_r,
                    targ_c,
                    rng,
                )
                return update_state, total_loss

            # Updating Training State and Metrics:
            update_state = (train_state, traj_batch, adv_r, adv_c, targ_r, targ_c, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, num_epochs
            )
            train_state = update_state[0]
            rng = update_state[-1]

            # --- LAGRANGE UPDATE ---
            # Update Lambda based on Average Episodic Cost vs Limit
            # Note: "returned_episode_cost_returns" is populated by Gymnax/Safenax LogWrapper
            # when episodes finish.
            avg_ep_cost = traj_batch.info["returned_episode_cost_returns"].mean()

            new_lambda = train_state.lagrange_param + lagrange_lr * (
                avg_ep_cost - cost_limit
            )
            new_lambda = jnp.maximum(0.0, new_lambda)
            train_state = train_state.replace(lagrange_param=new_lambda)

            metrics = {
                "updates": train_state.step,
                "actor_loss": loss_info[1][2].mean(),
                "critic_loss_r": loss_info[1][0].mean(),
                "critic_loss_c": loss_info[1][1].mean(),
                "entropy": loss_info[1][3].mean(),
                "lagrange_param": train_state.lagrange_param,
                "episode_cost_return": avg_ep_cost,
                "episode_return": traj_batch.info["returned_episode_returns"].mean(),
                "episode_length": traj_batch.info["returned_episode_lengths"].mean(),
            }

            return runner_state, metrics

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, num_updates
        )
        return runner_state, metrics

    return train


if __name__ == "__main__":
    config = {
        "ENV_NAME": "FrozenLake-v2",
        "LR": 3e-4,
        "NUM_ENVS": 5,
        "TRAIN_FREQ": 200,
        "TOTAL_TIMESTEPS": 2e5,
        "UPDATE_EPOCHS": 4,
        "BATCH_SIZE": 40,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": True,
        "WANDB_MODE": "online",
        "NUM_SEEDS": 5,
        "SEED": 0,
        # PPO-L Specifics
        "COST_LIMIT": 15.0,
        "LAG_LR": 0.0,
        "INIT_LAMBDA": 0.0,
    }

    env = FrozenLakeV2(
        map_name="4x4",
        is_slippery=False,
        safe_cost_std=0.0,
        thin_shock_prob=0.1,
        thin_shock_val=10.0,
    )
    env_params = env.default_params

    train_fn = make_train(
        env=env,
        env_params=env_params,
        num_steps=config["TOTAL_TIMESTEPS"],
        num_envs=config["NUM_ENVS"],
        train_freq=config["TRAIN_FREQ"],
        batch_size=config["BATCH_SIZE"],
        num_epochs=config["UPDATE_EPOCHS"],
        activation=nn.tanh if config["ACTIVATION"] == "tanh" else nn.relu,
        lr=config["LR"],
        anneal_lr=config["ANNEAL_LR"],
        gae_gamma=config["GAMMA"],
        gae_lambda=config["GAE_LAMBDA"],
        entropy_coeff=config["ENT_COEF"],
        value_coeff=config["VF_COEF"],
        max_grad_norm=config["MAX_GRAD_NORM"],
        ratio_clip=config["CLIP_EPS"],
        cost_limit=config["COST_LIMIT"],
        lagrange_lr=config["LAG_LR"],
        initial_lambda=config["INIT_LAMBDA"],
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(train_fn))
    runner_states, all_metrics = jax.block_until_ready(train_vjit(rngs))

    log.save_wandb(
        project="FrozenLake",
        algo_name="ppo-l",
        env_name=env.name,
        metrics=all_metrics,
    )

    log.save_local(
        algo_name="ppo-l",
        env_name=env.name,
        metrics=all_metrics,
    )
