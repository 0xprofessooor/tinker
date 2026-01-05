"""CVaR Proximal Policy Optimization (CPPO) with Categorical Policy."""

from typing import NamedTuple, Tuple

import chex
import distrax
import flax.linen as nn
import time
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
    """Actor critic neural network with disjoint architectures.

    :param action_dim: Discrete number of actions the agent can take.
    :param activation: Activation function for the network.
    """

    action_dim: int
    activation: callable = nn.tanh

    @nn.compact
    def __call__(self, x):
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

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = self.activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = self.activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        cost_critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        cost_critic = self.activation(cost_critic)
        cost_critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(cost_critic)
        cost_critic = self.activation(cost_critic)
        cost_critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            cost_critic
        )

        return pi, jnp.squeeze(critic, axis=-1), jnp.squeeze(cost_critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    cost_value: jnp.ndarray
    reward: jnp.ndarray
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
    cost_value_coeff: float = 0.5,
    max_grad_norm: float = 0.5,
    ratio_clip: float = 0.2,
):
    """Generate a jitted JAX PPO train function.

    :param env: Gymnax environment.
    :param env_params: Environment parameters.
    :param num_steps: Number of steps to train per environment.
    :param num_envs: Number of parallel environments to run.
    :param train_freq: Number of steps to run between training updates.
    :param batch_size: Minibatch size to make a single gradient descent step on.
    :param num_epochs: Number of epochs to train per update step.
    :param activation: Activation function for the network hidden layers.
    :param lr: Learning rate for the optimizer.
    :param anneal_lr: Whether to anneal the learning rate over time.
    :param gae_gamma: Discount factor for the returns.
    :param gae_lambda: Lambda for the Generalized Advantage Estimation.
    :param entropy_coeff: Coefficient for the entropy loss.
    :param value_coeff: Coefficient for the value loss.
    :param max_grad_norm: Maximum gradient norm for clipping.
    :param ratio_clip: The clipping factor for the clipped loss
    """

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
        obs_space = env.observation_space(env_params)
        if isinstance(obs_space, spaces.Discrete):
            init_x = jnp.zeros((obs_space.n,))

            def preprocess_obs(obs: jax.Array) -> jax.Array:
                """One-hot encode discrete observations."""
                return jax.nn.one_hot(obs, obs_space.n)
        else:
            init_x = jnp.zeros(obs_space.shape)

            def preprocess_obs(obs: jax.Array) -> jax.Array:
                """Pass-through for continuous observations."""
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
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
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
                pi, value, cost_value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, num_envs)
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)
                obsv = preprocess_obs(obsv)
                transition = Transition(
                    done, action, value, cost_value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, train_freq
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            _, last_val, last_cost_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + gae_gamma * next_value * (1 - done) - value
                    gae = delta + gae_gamma * gae_lambda * (1 - done) * gae
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

            cost_advantages, cost_targets = _calculate_gae(
                traj_batch._replace(
                    reward=traj_batch.info["cost"], value=traj_batch.cost_value
                ),
                last_cost_val,
            )

            # UPDATE NETWORK
            def _update_epoch(update_state, _):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets, cost_targets = batch_info

                    def _loss_fn(params, traj_batch, advantages, targets, cost_targets):
                        # RERUN NETWORK
                        pi, value, cost_value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-ratio_clip, ratio_clip)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE COST VALUE LOSS
                        cost_value_pred_clipped = traj_batch.cost_value + (
                            cost_value - traj_batch.cost_value
                        ).clip(-ratio_clip, ratio_clip)
                        cost_value_losses = jnp.square(cost_value - cost_targets)
                        cost_value_losses_clipped = jnp.square(
                            cost_value_pred_clipped - cost_targets
                        )
                        cost_value_loss = (
                            0.5
                            * jnp.maximum(
                                cost_value_losses, cost_value_losses_clipped
                            ).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (advantages - advantages.mean()) / (
                            advantages.std() + 1e-8
                        )
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - ratio_clip,
                                1.0 + ratio_clip,
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + value_coeff * value_loss
                            + cost_value_coeff * cost_value_loss
                            - entropy_coeff * entropy
                        )
                        return total_loss, (
                            value_loss,
                            cost_value_loss,
                            loss_actor,
                            entropy,
                        )

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params,
                        traj_batch,
                        advantages,
                        targets,
                        cost_targets,
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, rng = update_state
                rng, _rng = jax.random.split(rng)
                # Batching and Shuffling
                total_batch_size = batch_size * num_minibatches
                assert total_batch_size == train_freq * num_envs, (
                    "total batch size must be equal to number of steps * number of envs"
                )
                permutation = jax.random.permutation(_rng, total_batch_size)
                batch = (traj_batch, advantages, targets, cost_targets)
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
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, rng)
                return update_state, total_loss

            # Updating Training State and Metrics:
            update_state = (train_state, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, num_epochs
            )
            train_state = update_state[0]
            rng = update_state[-1]

            runner_state = (train_state, env_state, last_obs, rng)
            thin_tiles_visited = jnp.sum(
                jnp.where(traj_batch.info["tile_type"] == 84, 1, 0)
            )
            metrics = {
                "actor_loss": loss_info[1][2].mean(),
                "critic_loss": loss_info[1][0].mean(),
                "cost_critic_loss": loss_info[1][1].mean(),
                "entropy": loss_info[1][3].mean(),
                "cost_dist": traj_batch.info["returned_episode_cost_returns"],
                "episode_cost_return": traj_batch.info[
                    "returned_episode_cost_returns"
                ].mean(),
                "episode_return": traj_batch.info["returned_episode_returns"].mean(),
                "episode_length": traj_batch.info["returned_episode_lengths"].mean(),
                "dones": traj_batch.info["returned_episode"],
                "returns": traj_batch.info["returned_episode_returns"],
                "thin_tiles_visited": thin_tiles_visited,
            }

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
    env_params: EnvParams,
    num_steps: int,
    network: nn.Module,
    network_params: dict,
    rng_input: chex.PRNGKey,
):
    """Rollout a jitted gymnax episode with lax.scan.

    :param env: Gymnax environment.
    :param env_params: Environment parameters.
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
    obs, state = env.reset(rng_reset, env_params)

    def policy_step(state_input, tmp):
        """lax.scan compatible step transition in jax env."""
        obs, state, rng = state_input
        rng, rng_step, rng_net = jax.random.split(rng, 3)
        pi, value = network.apply(network_params, obs)
        action = pi.sample(seed=rng_net)
        next_obs, next_state, reward, done, _ = env.step(
            rng_step, state, action, env_params
        )
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
        "ENV_NAME": "FrozenLake-v2",
        "LR": 3e-4,
        "NUM_ENVS": 5,
        "TRAIN_FREQ": 200,
        "TOTAL_TIMESTEPS": 1e5,
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
        "NUM_SEEDS": 5,
        "SEED": 0,
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
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])

    # JIT and vmap the training function for parallel execution
    train_vjit = jax.jit(jax.vmap(train_fn))
    start_time = time.perf_counter()
    runner_states, all_metrics = jax.block_until_ready(train_vjit(rngs))
    runtime = time.perf_counter() - start_time
    print(f"Runtime: {runtime:.2f}s")

    log.save_wandb(
        project="FrozenLake",
        algo_name="cppo",
        env_name=env.name,
        metrics=all_metrics,
    )

    log.save_local(
        algo_name="cppo",
        env_name=env.name,
        metrics=all_metrics,
    )
