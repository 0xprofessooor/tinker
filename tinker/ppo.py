"""Proximal Policy Optimization (PPO) with Categorical Policy."""

import os
from typing import NamedTuple, Sequence, Tuple

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
from gymnax.environments.environment import Environment
from gymnax.wrappers import LogWrapper

import wandb


class ActorCritic(nn.Module):
    """Actor critic neural network with disjoint architectures.

    :param action_dim: Discrete number of actions the agent can take.
    :param activation: Activation function for the network.
    """

    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


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
    """Generate a jitted JAX PPO train function.

    :param env: Gymnax environment.
    :param num_steps: Total number of steps to train for.
    :param num_envs: Number of parallel environments to run.
    :param train_freq: Number of steps to run between training updates.
    :param num_minibatches: Number of minibatches to split the data into.
    :param num_epochs: Number of epochs to train for.
    :param activation: Activation function for the network.
    :param lr: Learning rate for the optimizer.
    :param anneal_lr: Whether to anneal the learning rate over time.
    :param gamma: Discount factor for the returns.
    :param gae_lambda: Lambda for the Generalized Advantage Estimation.
    :param entropy_coef: Coefficient for the entropy loss.
    :param value_coef: Coefficient for the value loss.
    :param max_grad_norm: Maximum gradient norm for clipping.
    :param clip_eps: Epsilon for clipping the policy loss.
    :param log_wandb: Whether to log to wandb.
    """

    num_updates = num_steps // train_freq // num_envs
    minibatch_size = num_envs * train_freq // num_minibatches
    env_params = env.default_params
    env = LogWrapper(env)

    def linear_schedule(count):
        frac = 1.0 - (count // (num_minibatches * num_epochs)) / num_updates
        return lr * frac

    def train(rng: chex.PRNGKey) -> Tuple[TrainState, dict]:
        # INIT NETWORK
        network = ActorCritic(env.action_space(env_params).n, activation=activation)
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
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
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, num_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, num_envs)
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, train_freq
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + gamma * next_value * (1 - done) - value
                    gae = delta + gamma * gae_lambda * (1 - done) * gae
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
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-clip_eps, clip_eps)
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
                                1.0 - clip_eps,
                                1.0 + clip_eps,
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + value_coef * value_loss
                            - entropy_coef * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                # Batching and Shuffling
                batch_size = minibatch_size * num_minibatches
                assert batch_size == train_freq * num_envs, (
                    "batch size must be equal to number of steps * number of envs"
                )
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                # Mini-batch Updates
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [num_minibatches, -1] + list(x.shape[1:])),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            # Updating Training State and Metrics:
            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, num_epochs
            )
            train_state = update_state[0]
            rng = update_state[-1]

            runner_state = (train_state, env_state, last_obs, rng)
            metrics = {
                "updates": train_state.step,
                "actor_losses": loss_info[1][1],
                "critic_losses": loss_info[1][0],
                "entropy": loss_info[1][2],
                "batch_returns": traj_batch.info["returned_episode_returns"].mean(),
                "episode_lengths": traj_batch.info["returned_episode_lengths"].mean(),
                "dones": traj_batch.info["returned_episode"],
                "returns": traj_batch.info["returned_episode_returns"],
            }

            # report on wandb if required
            if log_wandb:

                def callback(metrics):
                    wandb.log(metrics)

                jax.debug.callback(callback, metrics)

            return runner_state, metrics

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, num_updates
        )
        return runner_state, metrics

    return train


def run(
    env: Environment,
    num_steps: int,
    network: nn.Module,
    network_params: dict,
    rng_input: chex.PRNGKey,
):
    """Rollout a jitted gymnax episode with lax.scan.

    :param env: Gymnax environment.
    :param num_steps: Number of steps to rollout.
    :param network: Flax network module.
    :param network_params: Network parameters.
    :param rng_input: Random number generator key.

    :return state: List of episode rollout states.
    :return obs: List of episode rollout observations.
    :return logits: List of episode rollout action distribution (Categorical) logits.
    :return actions: List of episode rollout actions.
    :return values: List of episode rollout value estimates.
    :return rewards: List of episode rollout rewards.
    :return next_state: List of episode rollout next states.
    :return next_obs: List of episode rollout next observations.
    :return dones: List of episode rollout done flags.
    """
    # Reset the environment
    rng_reset, rng_episode = jax.random.split(rng_input)
    obs, state = env.reset(rng_reset)

    def policy_step(state_input, tmp):
        """lax.scan compatible step transition in jax env."""
        obs, state, rng = state_input
        rng, rng_step, rng_net = jax.random.split(rng, 3)
        pi, value = network.apply(network_params, obs)
        action = pi.sample(seed=rng_net)
        next_obs, next_state, reward, done, _ = env.step(rng_step, state, action)
        carry = [next_obs, next_state, rng]
        return carry, [
            state,
            obs,
            pi.logits,
            action,
            value,
            reward,
            next_state,
            next_obs,
            done,
        ]

    # Scan over episode step loop
    carry, scan_out = jax.lax.scan(
        policy_step, [obs, state, rng_episode], (), num_steps
    )
    return scan_out


if __name__ == "__main__":
    config = {
        "ENV_NAME": "CartPole-v1",
        "LR": 3e-4,
        "NUM_ENVS": 1,
        "TRAIN_FREQ": 128,
        "TOTAL_TIMESTEPS": 5e5,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": False,
        "WANDB_MODE": "online",  # set to online to activate wandb
        "NUM_SEEDS": 1,
        "SEED": 0,
    }

    basic_env, env_params = gymnax.make(config["ENV_NAME"])

    wandb.login(os.environ.get("WANDB_KEY"))
    wandb.init(
        project="Tinker",
        tags=["PPO", config["ENV_NAME"].upper(), f"jax_{jax.__version__}"],
        name=f"purejaxrl_ppo_{config['ENV_NAME']}",
        config=config,
        mode=config["WANDB_MODE"],
    )

    train_fn = make_train(
        env=basic_env,
        num_steps=config["TOTAL_TIMESTEPS"],
        num_envs=config["NUM_ENVS"],
        train_freq=config["TRAIN_FREQ"],
        num_minibatches=config["NUM_MINIBATCHES"],
        num_epochs=config["UPDATE_EPOCHS"],
        activation=config["ACTIVATION"],
        lr=config["LR"],
        anneal_lr=config["ANNEAL_LR"],
        gamma=config["GAMMA"],
        gae_lambda=config["GAE_LAMBDA"],
        entropy_coef=config["ENT_COEF"],
        value_coef=config["VF_COEF"],
        max_grad_norm=config["MAX_GRAD_NORM"],
        clip_eps=config["CLIP_EPS"],
        log_wandb=True if config["WANDB_MODE"] == "online" else False,
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(train_fn))
    outs = jax.block_until_ready(train_vjit(rngs))
