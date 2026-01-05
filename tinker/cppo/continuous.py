import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import NamedTuple, Tuple
from gymnax.environments import EnvParams
from gymnax.environments.environment import Environment
import chex
from flax.training.train_state import TrainState
import distrax
from safenax import EcoAntV2
from safenax.wrappers import BraxToGymnaxWrapper, LogWrapper
from tinker import log, norm
import time


class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # Actor
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
        actor_logtstd = self.param("log_std", constant(-0.5), (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        # Reward Critic
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

        # Cost Critic
        cost_critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        cost_critic = activation(cost_critic)
        cost_critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(cost_critic)
        cost_critic = activation(cost_critic)
        cost_critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            cost_critic
        )

        return pi, jnp.squeeze(critic, axis=-1), jnp.squeeze(cost_critic, axis=-1)


class Transition(NamedTuple):
    done: jax.Array
    action: jax.Array
    value: jax.Array
    cost_value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    running_cost: jax.Array
    norm_obs: jax.Array
    next_obs: jax.Array
    info: jax.Array


def make_train(
    env: Environment,
    env_params: EnvParams,
    num_steps: int,
    num_envs: int,
    train_freq: int,
    batch_size: int,
    num_epochs: int,
    cvar_limit: float,
    cvar_probability: float,
    lam_start: float = 0.5,
    lam_lr: float = 1e-3,
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
    """Generate a jitted JAX CPPO train function.

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
        network = ActorCritic(
            env.action_space(env_params).shape[0], activation=activation
        )
        rng, _rng = jax.random.split(rng)
        obs_space = env.observation_space(env_params)
        init_x = jnp.zeros(obs_space.shape)

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
        running_cost = jnp.zeros((num_envs,))

        # INIT OBSERVATION NORMALIZATION
        obs_norm_state = norm.init(env.observation_space(env_params).shape)
        obs_norm_state = norm.welford_update(obs_norm_state, obsv)

        # TRAIN LOOP
        def _update_step(runner_state, _):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, _):
                (
                    train_state,
                    env_state,
                    obs_norm_state,
                    last_obs,
                    running_cost,
                    lam,
                    rng,
                ) = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                normalized_obs = jax.vmap(norm.normalize, in_axes=(None, 0))(
                    obs_norm_state, last_obs
                )
                pi, value, cost_value = network.apply(
                    train_state.params, normalized_obs
                )
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, num_envs)
                next_obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)

                cost = info.get("cost", jnp.zeros_like(reward))

                transition = Transition(
                    done,
                    action,
                    value,
                    cost_value,
                    reward,
                    log_prob,
                    running_cost,
                    normalized_obs,
                    next_obsv,
                    info,
                )

                next_running_cost = (1 - done) * (running_cost + cost)

                runner_state = (
                    train_state,
                    env_state,
                    obs_norm_state,
                    next_obsv,
                    next_running_cost,
                    lam,
                    rng,
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, train_freq
            )
            train_state, env_state, obs_norm_state, last_obs, running_cost, lam, rng = (
                runner_state
            )

            normalized_last_obs = jax.vmap(norm.normalize, in_axes=(None, 0))(
                obs_norm_state, last_obs
            )
            _, last_val, last_cost_val = network.apply(
                train_state.params, normalized_last_obs
            )

            batch_raw_obs = traj_batch.next_obs.reshape(
                -1, *traj_batch.next_obs.shape[2:]
            )
            obs_norm_state = norm.welford_update(obs_norm_state, batch_raw_obs)

            # CALCULATE ADVANTAGE
            def _calculate_gae(traj_batch: Transition, last_val):
                def _get_advantages(gae_and_next_value, transition: Transition):
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

            cost_returns = traj_batch.running_cost + traj_batch.cost_value
            sorted_costs = jnp.sort(cost_returns.flatten())[::-1]
            k = int(num_envs * train_freq * cvar_probability)
            nu = jnp.mean(sorted_costs[:k])
            penalty_mask = jnp.where(cost_returns > nu, 1.0, 0.0)
            penalty = (lam / cvar_probability) * (cost_returns - nu) * penalty_mask
            advantages = advantages - penalty
            lam += lam_lr * (nu - cvar_limit)

            # UPDATE NETWORK
            def _update_epoch(update_state, _):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets, cost_targets = batch_info

                    def _loss_fn(
                        params,
                        traj_batch: Transition,
                        advantages,
                        targets,
                        cost_targets,
                    ):
                        # RERUN NETWORK
                        pi, value, cost_value = network.apply(
                            params, traj_batch.norm_obs
                        )
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

            runner_state = (
                train_state,
                env_state,
                obs_norm_state,
                last_obs,
                running_cost,
                lam,
                rng,
            )

            is_terminal = traj_batch.done
            num_episodes = jnp.maximum(is_terminal.sum(), 1.0)
            terminal_costs = traj_batch.running_cost + traj_batch.info["cost"]
            sparse_costs = jnp.where(is_terminal, terminal_costs, 0.0)
            exceeds_threshold = (sparse_costs > cvar_limit) & is_terminal
            num_exceedances = exceeds_threshold.sum()
            empirical_var_probability = num_exceedances / num_episodes
            metrics = {
                "actor_loss": loss_info[1][2].mean(),
                "critic_loss": loss_info[1][0].mean(),
                "cost_critic_loss": loss_info[1][1].mean(),
                "entropy": loss_info[1][3].mean(),
                "lam": lam,
                "nu": nu,
                "cost_dist": traj_batch.info["returned_episode_cost_returns"],
                "episode_cost_return": traj_batch.info[
                    "returned_episode_cost_returns"
                ].mean(),
                "episode_return": traj_batch.info["returned_episode_returns"].mean(),
                "episode_length": traj_batch.info["returned_episode_lengths"].mean(),
                "empirical_var_probability": empirical_var_probability,
            }

            return runner_state, metrics

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obs_norm_state,
            obsv,
            running_cost,
            lam_start,
            _rng,
        )
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, num_updates
        )
        return runner_state, metrics

    return train


if __name__ == "__main__":
    SEED = 0
    NUM_SEEDS = 5

    brax_env = EcoAntV2(battery_limit=50.0)
    env = BraxToGymnaxWrapper(env=brax_env, episode_length=1000)
    env_params = env.default_params

    rng = jax.random.PRNGKey(SEED)
    rngs = jax.random.split(rng, NUM_SEEDS)

    train_fn = make_train(
        env,
        env_params,
        num_steps=int(2e6),
        train_freq=1000,
        batch_size=500,
        num_epochs=10,
        num_envs=5,
        cvar_probability=0.1,
        cvar_limit=50.0,
        lam_start=10.0,
    )
    train_vjit = jax.jit(jax.vmap(train_fn))
    start_time = time.perf_counter()
    runner_states, all_metrics = jax.block_until_ready(train_vjit(rngs))
    runtime = time.perf_counter() - start_time
    print(f"Runtime: {runtime:.2f}s")

    log.save_local(
        algo_name="cppo50",
        env_name=brax_env.name,
        metrics=all_metrics,
    )

    log.save_wandb(
        project="EcoAnt",
        algo_name="cppo50",
        env_name=brax_env.name,
        metrics=all_metrics,
    )
