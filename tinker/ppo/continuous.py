"""Proximal Policy Optimization Lagrange (PPO-L) with Continuous Action Space."""

from typing import Tuple

import distrax
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.linen.initializers import constant, orthogonal
from gymnax.environments.environment import Environment, EnvParams
import gymnax
from safenax import EcoAntV1
from safenax.wrappers import BraxToGymnaxWrapper, LogWrapper
from tinker import norm, log
import time


class ActorCritic(nnx.Module):
    def __init__(
        self, obs_dim: int, action_dim: int, activation: callable, rngs: nnx.Rngs
    ):
        self.activation = activation

        # Actor
        self.actor_dense1 = nnx.Linear(
            obs_dim,
            256,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            rngs=rngs,
        )
        self.actor_dense2 = nnx.Linear(
            256,
            256,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            rngs=rngs,
        )
        self.actor_out = nnx.Linear(
            256,
            action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            rngs=rngs,
        )
        self.log_std = nnx.Param(jnp.zeros(action_dim))

        # Critic
        self.critic_dense1 = nnx.Linear(
            obs_dim,
            256,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            rngs=rngs,
        )
        self.critic_dense2 = nnx.Linear(
            256,
            256,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            rngs=rngs,
        )
        self.critic_out = nnx.Linear(
            256, 1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), rngs=rngs
        )

    def __call__(self, x):
        # Actor
        actor_mean = self.actor_dense1(x)
        actor_mean = self.activation(actor_mean)
        actor_mean = self.actor_dense2(actor_mean)
        actor_mean = self.activation(actor_mean)
        actor_mean = self.actor_out(actor_mean)
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(self.log_std[...]))

        # Critic
        critic = self.critic_dense1(x)
        critic = self.activation(critic)
        critic = self.critic_dense2(critic)
        critic = self.activation(critic)
        critic = self.critic_out(critic)

        return pi, jnp.squeeze(critic, axis=-1)


@struct.dataclass
class Transition:
    done: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    obs: jax.Array
    next_obs: jax.Array
    info: dict


@struct.dataclass
class DynamicConfig:
    """Holds dynamic configuration parameters for PPO training.

    :param rng: Random number generator key.
    :param env_params: Environment parameters.
    :param lr: Learning rate.
    :param gae_gamma: Discount factor for GAE.
    :param gae_lambda: Lambda parameter for GAE.
    :param entropy_coeff: Coefficient for entropy bonus.
    :param value_coeff: Coefficient for value loss.
    :param max_grad_norm: Maximum gradient norm for clipping.
    :param ratio_clip: Clipping factor for PPO objective.
    """

    rng: jax.Array
    env_params: EnvParams
    lr: jax.Array
    gae_gamma: jax.Array
    gae_lambda: jax.Array
    entropy_coeff: jax.Array
    value_coeff: jax.Array
    max_grad_norm: jax.Array
    ratio_clip: jax.Array


@struct.dataclass
class RunnerState:
    params: nnx.Param
    opt_state: optax.OptState
    env_state: jax.Array
    obs_norm_state: norm.RunningMeanStdState
    last_obs: jax.Array
    rng: jax.Array


def make_train(
    env: Environment,
    num_steps: int,
    num_envs: int,
    train_freq: int,
    batch_size: int,
    num_epochs: int,
    activation: callable = jax.nn.tanh,
    anneal_lr: bool = True,
):
    """Generate a jitted JAX PPO train function.

    :param env: Gymnax environment.
    :param num_steps: Number of steps to train per environment.
    :param num_envs: Number of parallel environments to run.
    :param train_freq: Number of steps to run between training updates.
    :param batch_size: Minibatch size to make a single gradient descent step on.
    :param num_epochs: Number of epochs to train per update step.
    :param activation: Activation function for the network hidden layers.
    :param anneal_lr: Whether to anneal the learning rate over time.
    """

    num_updates = num_steps // train_freq
    num_minibatches = (num_envs * train_freq) // batch_size
    env = LogWrapper(env)

    def train(config: DynamicConfig) -> Tuple[tuple, dict]:
        def linear_schedule(count):
            frac = 1.0 - (count // (num_minibatches * num_epochs)) / num_updates
            return config.lr * frac

        # INIT NETWORK
        rng, _rng = jax.random.split(config.rng)
        obs_dim = env.observation_space(config.env_params).shape[0]
        action_dim = env.action_space(config.env_params).shape[0]
        network = ActorCritic(
            obs_dim, action_dim, activation=activation, rngs=nnx.Rngs(params=_rng)
        )
        graphdef, params = nnx.split(network)

        if anneal_lr:
            tx = optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adam(config.lr, eps=1e-5),
            )
        opt_state = tx.init(params)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, num_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(
            reset_rng, config.env_params
        )

        # INIT OBSERVATION NORMALIZATION
        obs_norm_state = norm.init(env.observation_space(config.env_params).shape)
        obs_norm_state = norm.welford_update(obs_norm_state, obsv)

        # TRAIN LOOP
        def _update_step(runner_state: RunnerState, _):
            model = nnx.merge(graphdef, runner_state.params)

            # COLLECT TRAJECTORIES
            def _env_step(runner_state: RunnerState, _):
                env_state = runner_state.env_state
                obs_norm_state = runner_state.obs_norm_state
                last_obs = runner_state.last_obs
                rng = runner_state.rng

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                normalized_obs = jax.vmap(norm.normalize, in_axes=(None, 0))(
                    obs_norm_state, last_obs
                )
                pi, value = model(normalized_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, num_envs)
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, config.env_params)
                transition = Transition(
                    done, action, value, reward, log_prob, normalized_obs, obsv, info
                )
                runner_state = RunnerState(
                    params=runner_state.params,
                    opt_state=runner_state.opt_state,
                    env_state=env_state,
                    obs_norm_state=obs_norm_state,
                    last_obs=obsv,
                    rng=rng,
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, train_freq
            )

            # UPDATE OBSERVATION NORMALIZATION and normalize all collected observations
            params = runner_state.params
            opt_state = runner_state.opt_state
            env_state = runner_state.env_state
            obs_norm_state = runner_state.obs_norm_state
            last_obs = runner_state.last_obs
            rng = runner_state.rng

            normalized_last_obs = jax.vmap(norm.normalize, in_axes=(None, 0))(
                obs_norm_state, last_obs
            )
            _, last_val = model(normalized_last_obs)

            # Flatten batch: (train_freq, num_envs, obs_dim) -> (train_freq * num_envs, obs_dim)
            batch_raw_obs = traj_batch.next_obs.reshape(
                -1, *traj_batch.next_obs.shape[2:]
            )
            obs_norm_state = norm.welford_update(obs_norm_state, batch_raw_obs)

            # CALCULATE ADVANTAGE
            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config.gae_gamma * next_value * (1 - done) - value
                    gae = (
                        delta + config.gae_gamma * config.gae_lambda * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, _):
                def _update_minbatch(carry, batch_info):
                    params, opt_state = carry
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = nnx.merge(graphdef, params)(traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config.ratio_clip, config.ratio_clip)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config.ratio_clip,
                                1.0 + config.ratio_clip,
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config.value_coeff * value_loss
                            - config.entropy_coeff * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (total_loss, aux), grads = grad_fn(
                        params, traj_batch, advantages, targets
                    )
                    updates, new_opt_state = tx.update(grads, opt_state)
                    new_params = optax.apply_updates(params, updates)
                    return (new_params, new_opt_state), (total_loss, aux)

                params, opt_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                # Batching and Shuffling
                total_batch_size = batch_size * num_minibatches
                assert total_batch_size == train_freq * num_envs, (
                    "total batch size must be equal to number of steps * number of envs"
                )
                permutation = jax.random.permutation(_rng, total_batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((total_batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                # Mini-batch Updates
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [num_minibatches, batch_size] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                (params, opt_state), total_loss = jax.lax.scan(
                    _update_minbatch, (params, opt_state), minibatches
                )
                update_state = (params, opt_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            # Updating Training State and Metrics:
            update_state = (params, opt_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, num_epochs
            )
            params, opt_state = update_state[0], update_state[1]
            rng = update_state[-1]

            runner_state = RunnerState(
                params=params,
                opt_state=opt_state,
                env_state=env_state,
                obs_norm_state=obs_norm_state,
                last_obs=last_obs,
                rng=rng,
            )

            metrics = {
                "actor_loss": loss_info[1][1].mean(),
                "critic_loss": loss_info[1][0].mean(),
                "entropy": loss_info[1][2].mean(),
                "episode_return": traj_batch.info["returned_episode_returns"].mean(),
                "episode_cost_return": traj_batch.info[
                    "returned_episode_cost_returns"
                ].mean(),
                "episode_length": traj_batch.info["returned_episode_lengths"].mean(),
                "dones": traj_batch.info["returned_episode"],
                "return_dist": traj_batch.info["returned_episode_returns"],
                "cost_return_dist": traj_batch.info["returned_episode_cost_returns"],
            }

            return runner_state, metrics

        rng, _rng = jax.random.split(rng)
        runner_state = RunnerState(
            params=params,
            opt_state=opt_state,
            env_state=env_state,
            obs_norm_state=obs_norm_state,
            last_obs=obsv,
            rng=_rng,
        )
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, num_updates
        )
        return runner_state, metrics

    return train


if __name__ == "__main__":
    pendulum_config = {
        "ENV_NAME": "Pendulum-v1",
        "LR": 3e-4,
        "NUM_ENVS": 512,
        "TRAIN_FREQ": 20,
        "TOTAL_TIMESTEPS": 10000,
        "UPDATE_EPOCHS": 10,
        "BATCH_SIZE": 128,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": False,
        "WANDB_MODE": "online",
        "NUM_RUNS": 1,
        "SEED": 0,
    }
    po_garch_config = {
        "ENV_NAME": "PO-GARCH",
        "LR": 3e-4,
        "NUM_ENVS": 10,
        "TRAIN_FREQ": 1000,
        "TOTAL_TIMESTEPS": 1e6,
        "UPDATE_EPOCHS": 10,
        "BATCH_SIZE": 100,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": False,
        "WANDB_MODE": "online",
        "NUM_RUNS": 1,
        "SEED": 30,
    }
    config = {
        "ENV_NAME": EcoAntV1().name,
        "LR": 3e-4,
        "NUM_ENVS": 5,
        "TRAIN_FREQ": 500,
        "TOTAL_TIMESTEPS": int(2e6),
        "UPDATE_EPOCHS": 10,
        "BATCH_SIZE": 500,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.0075,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": True,
        "WANDB_MODE": "online",
        "NUM_RUNS": 5,
        "SEED": 0,
    }

    rng = jax.random.PRNGKey(pendulum_config["SEED"])
    train_rngs = jax.random.split(rng, pendulum_config["NUM_RUNS"])
    env, default_params = gymnax.make(pendulum_config["ENV_NAME"])
    env_params = [default_params] * pendulum_config["NUM_RUNS"]

    dynamic_config = DynamicConfig(
        rng=train_rngs,
        env_params=jax.tree.map(lambda *xs: jnp.stack(xs), *env_params),
        lr=jnp.ones(pendulum_config["NUM_RUNS"]) * pendulum_config["LR"],
        gae_gamma=jnp.ones(pendulum_config["NUM_RUNS"]) * pendulum_config["GAMMA"],
        gae_lambda=jnp.ones(pendulum_config["NUM_RUNS"])
        * pendulum_config["GAE_LAMBDA"],
        entropy_coeff=jnp.ones(pendulum_config["NUM_RUNS"])
        * pendulum_config["ENT_COEF"],
        value_coeff=jnp.ones(pendulum_config["NUM_RUNS"]) * pendulum_config["VF_COEF"],
        max_grad_norm=jnp.ones(pendulum_config["NUM_RUNS"])
        * pendulum_config["MAX_GRAD_NORM"],
        ratio_clip=jnp.ones(pendulum_config["NUM_RUNS"]) * pendulum_config["CLIP_EPS"],
    )

    train_fn = make_train(
        env=env,
        num_steps=pendulum_config["TOTAL_TIMESTEPS"],
        num_envs=pendulum_config["NUM_ENVS"],
        train_freq=pendulum_config["TRAIN_FREQ"],
        batch_size=pendulum_config["BATCH_SIZE"],
        num_epochs=pendulum_config["UPDATE_EPOCHS"],
        activation=jax.nn.tanh
        if pendulum_config["ACTIVATION"] == "tanh"
        else jax.nn.relu,
        anneal_lr=pendulum_config["ANNEAL_LR"],
    )

    train_vjit = jax.jit(jax.vmap(train_fn))
    start_time = time.perf_counter()
    runner_states, all_metrics = jax.block_until_ready(train_vjit(dynamic_config))
    runtime = time.perf_counter() - start_time
    print(f"Runtime: {runtime:.2f}s")

    log.save_local(
        algo_name="ppo",
        env_name=env.name,
        metrics=all_metrics,
    )

    log.save_wandb(
        project="EcoAnt",
        algo_name="ppo",
        env_name=env.name,
        metrics=all_metrics,
    )
