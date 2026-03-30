"""Proximal Policy Optimization Lagrange (PPO-L) with SPiDR Domain Randomization."""

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
from safenax import EcoAntV2
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

        # Reward Critic
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

        # Cost Critic
        self.cost_critic_dense1 = nnx.Linear(
            obs_dim,
            256,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            rngs=rngs,
        )
        self.cost_critic_dense2 = nnx.Linear(
            256,
            256,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            rngs=rngs,
        )
        self.cost_critic_out = nnx.Linear(
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

        # Reward Critic
        critic = self.critic_dense1(x)
        critic = self.activation(critic)
        critic = self.critic_dense2(critic)
        critic = self.activation(critic)
        critic = self.critic_out(critic)

        # Cost Critic
        cost_critic = self.cost_critic_dense1(x)
        cost_critic = self.activation(cost_critic)
        cost_critic = self.cost_critic_dense2(cost_critic)
        cost_critic = self.activation(cost_critic)
        cost_critic = self.cost_critic_out(cost_critic)

        return pi, jnp.squeeze(critic, axis=-1), jnp.squeeze(cost_critic, axis=-1)


@struct.dataclass
class Transition:
    done: jax.Array
    action: jax.Array
    value: jax.Array
    cost_value: jax.Array
    reward: jax.Array
    cost: jax.Array
    log_prob: jax.Array
    obs: jax.Array
    next_obs: jax.Array
    info: dict


@struct.dataclass
class DynamicConfig:
    """Holds dynamic configuration parameters for SPiDR PPO-Lagrange training.

    :param rng: Random number generator key.
    :param train_env_params: Per-env training params (num_envs, ...) for domain randomization.
    :param ensemble_env_params: Ensemble dynamics params (n_ensemble, ...) for SPiDR penalty.
    :param test_env_params: Prod/test environment parameters for post-training evaluation.
    :param pessimism: SPiDR λ scaling the variance penalty (Eq. 4).
    :param lr: Learning rate.
    :param gae_gamma: Discount factor for GAE.
    :param gae_lambda: Lambda parameter for GAE.
    :param entropy_coeff: Coefficient for entropy bonus.
    :param value_coeff: Coefficient for value loss.
    :param max_grad_norm: Maximum gradient norm for clipping.
    :param ratio_clip: Clipping factor for PPO objective.
    """

    rng: jax.Array
    train_env_params: EnvParams
    ensemble_env_params: EnvParams
    test_env_params: EnvParams
    pessimism: jax.Array
    lr: jax.Array
    gae_gamma: jax.Array
    gae_lambda: jax.Array
    entropy_coeff: jax.Array
    value_coeff: jax.Array
    max_grad_norm: jax.Array
    ratio_clip: jax.Array
    cost_limit: jax.Array
    lagrange_lr: jax.Array
    init_lagrange_lambda: jax.Array
    cost_gae_gamma: jax.Array


@struct.dataclass
class RunnerState:
    params: nnx.Param
    opt_state: optax.OptState
    env_state: jax.Array
    obs_norm_state: norm.RunningMeanStdState
    last_obs: jax.Array
    lagrange_lambda: jax.Array
    rng: jax.Array


def make_train(
    env: Environment,
    num_steps: int,
    num_envs: int,
    n_ensemble: int,
    train_freq: int,
    batch_size: int,
    num_epochs: int,
    num_test_steps: int = 1024,
    activation: callable = jax.nn.tanh,
    anneal_lr: bool = True,
):
    """Generate a jitted JAX PPO-Lagrange train function with SPiDR.

    :param env: Gymnax environment.
    :param num_steps: Number of steps to train per environment.
    :param num_envs: Number of parallel environments to run (N in SPiDR).
    :param n_ensemble: Number of ensemble dynamics for penalty (n in SPiDR Eq. 6).
    :param train_freq: Number of steps to run between training updates.
    :param batch_size: Minibatch size to make a single gradient descent step on.
    :param num_epochs: Number of epochs to train per update step.
    :param num_test_steps: Number of steps for post-training prod evaluation.
    :param activation: Activation function for the network hidden layers.
    :param anneal_lr: Whether to anneal the learning rate over time.
    """

    num_updates = num_steps // train_freq
    num_minibatches = (num_envs * train_freq) // batch_size
    base_env = env
    env = LogWrapper(env)

    def train(config: DynamicConfig) -> Tuple[tuple, dict]:
        def linear_schedule(count):
            frac = 1.0 - (count // (num_minibatches * num_epochs)) / num_updates
            return config.lr * frac

        # INIT NETWORK
        rng, _rng = jax.random.split(config.rng)
        ref_params = jax.tree.map(lambda x: x[0], config.train_env_params)
        obs_dim = env.observation_space(ref_params).shape[0]
        action_dim = env.action_space(ref_params).shape[0]
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

        # INIT ENV (per-env params for domain randomization)
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, num_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, 0))(
            reset_rng, config.train_env_params
        )

        # INIT OBSERVATION NORMALIZATION
        obs_norm_state = norm.init(env.observation_space(ref_params).shape)
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
                pi, value, cost_value = model(normalized_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV (per-env domain randomization params)
                pre_step_inner_state = env_state.env_state
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, num_envs)
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, 0)
                )(rng_step, env_state, action, config.train_env_params)

                # SPiDR ENSEMBLE PENALTY (Eq. 6: v(s,a) = ||Var(s_1,...,s_n)||_1)
                # Use same rng across ensemble so variance reflects dynamics, not noise
                rng, _rng_ens = jax.random.split(rng)
                ens_rng = jax.random.split(_rng_ens, num_envs)

                def _ensemble_step(ens_param):
                    obs_ens, _, _, _, _ = jax.vmap(
                        base_env.step, in_axes=(0, 0, 0, None)
                    )(ens_rng, pre_step_inner_state, action, ens_param)
                    return obs_ens

                ensemble_obs = jax.vmap(_ensemble_step)(
                    config.ensemble_env_params
                )  # (n_ensemble, num_envs, obs_dim)
                penalty = jnp.sum(jnp.var(ensemble_obs, axis=0), axis=-1)  # (num_envs,)

                # Penalized cost: c̃(s,a) = c(s,a) + λ·v(s,a)  (SPiDR Eq. 4)
                cost = info["cost"] + config.pessimism * penalty
                transition = Transition(
                    done,
                    action,
                    value,
                    cost_value,
                    reward,
                    cost,
                    log_prob,
                    normalized_obs,
                    obsv,
                    info,
                )
                runner_state = RunnerState(
                    params=runner_state.params,
                    opt_state=runner_state.opt_state,
                    env_state=env_state,
                    obs_norm_state=obs_norm_state,
                    last_obs=obsv,
                    lagrange_lambda=runner_state.lagrange_lambda,
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
            lagrange_lambda = runner_state.lagrange_lambda
            rng = runner_state.rng

            normalized_last_obs = jax.vmap(norm.normalize, in_axes=(None, 0))(
                obs_norm_state, last_obs
            )
            _, last_val, last_cost_val = model(normalized_last_obs)

            # Flatten batch: (train_freq, num_envs, obs_dim) -> (train_freq * num_envs, obs_dim)
            batch_raw_obs = traj_batch.next_obs.reshape(
                -1, *traj_batch.next_obs.shape[2:]
            )
            obs_norm_state = norm.welford_update(obs_norm_state, batch_raw_obs)

            # CALCULATE ADVANTAGE
            def _calculate_gae(traj_batch, last_val, gamma):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + gamma * next_value * (1 - done) - value
                    gae = delta + gamma * config.gae_lambda * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val, config.gae_gamma)
            cost_advantages, cost_targets = _calculate_gae(
                traj_batch.replace(reward=traj_batch.cost, value=traj_batch.cost_value),
                last_cost_val,
                config.cost_gae_gamma,
            )
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            cost_advantages = cost_advantages - cost_advantages.mean()

            # UPDATE NETWORK
            def _update_epoch(update_state, _):
                def _update_minbatch(carry, batch_info):
                    params, opt_state = carry
                    traj_batch, advantages, targets, cost_advantages, cost_targets = (
                        batch_info
                    )

                    def _loss_fn(
                        params, traj_batch, gae, targets, cost_gae, cost_targets
                    ):
                        # RERUN NETWORK
                        pi, value, cost_value = nnx.merge(graphdef, params)(
                            traj_batch.obs
                        )
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

                        # CALCULATE COST VALUE LOSS
                        cost_value_pred_clipped = traj_batch.cost_value + (
                            cost_value - traj_batch.cost_value
                        ).clip(-config.ratio_clip, config.ratio_clip)
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
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config.ratio_clip,
                                1.0 + config.ratio_clip,
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        entropy = pi.entropy().mean()

                        # CALCULATE COST SURROGATE (Lagrangian penalty)
                        cost_surrogate1 = ratio * cost_gae
                        cost_surrogate2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config.ratio_clip,
                                1.0 + config.ratio_clip,
                            )
                            * cost_gae
                        )
                        cost_surrogate = jnp.maximum(
                            cost_surrogate1, cost_surrogate2
                        ).mean()

                        total_loss = (
                            loss_actor
                            + lagrange_lambda * cost_surrogate
                            + config.value_coeff * (value_loss + cost_value_loss)
                            - config.entropy_coeff * entropy
                        )
                        return total_loss, (
                            value_loss,
                            cost_value_loss,
                            loss_actor,
                            cost_surrogate,
                            entropy,
                        )

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (total_loss, aux), grads = grad_fn(
                        params,
                        traj_batch,
                        advantages,
                        targets,
                        cost_advantages,
                        cost_targets,
                    )
                    updates, new_opt_state = tx.update(grads, opt_state)
                    new_params = optax.apply_updates(params, updates)
                    return (new_params, new_opt_state), (total_loss, aux)

                (
                    params,
                    opt_state,
                    traj_batch,
                    advantages,
                    targets,
                    cost_advantages,
                    cost_targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)
                # Batching and Shuffling
                total_batch_size = batch_size * num_minibatches
                assert total_batch_size == train_freq * num_envs, (
                    "total batch size must be equal to number of steps * number of envs"
                )
                permutation = jax.random.permutation(_rng, total_batch_size)
                batch = (traj_batch, advantages, targets, cost_advantages, cost_targets)
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
                update_state = (
                    params,
                    opt_state,
                    traj_batch,
                    advantages,
                    targets,
                    cost_advantages,
                    cost_targets,
                    rng,
                )
                return update_state, total_loss

            # Updating Training State and Metrics:
            update_state = (
                params,
                opt_state,
                traj_batch,
                advantages,
                targets,
                cost_advantages,
                cost_targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, num_epochs
            )
            params, opt_state = update_state[0], update_state[1]
            rng = update_state[-1]

            # DUAL UPDATE: projected gradient ascent on Lagrange multiplier
            # λ ← max(0, λ + η * (mean_episode_cost - cost_limit))
            done_mask = traj_batch.done
            num_episodes = done_mask.sum()
            traj_cost_return = (
                traj_batch.info["returned_episode_cost_returns"] * done_mask
            ).sum() / jnp.maximum(num_episodes, 1.0)
            effective_cost_return = jnp.where(
                num_episodes > 0, traj_cost_return, config.cost_limit
            )
            lagrange_lambda = jnp.maximum(
                0.0,
                lagrange_lambda
                + config.lagrange_lr * (effective_cost_return - config.cost_limit),
            )

            runner_state = RunnerState(
                params=params,
                opt_state=opt_state,
                env_state=env_state,
                obs_norm_state=obs_norm_state,
                last_obs=last_obs,
                lagrange_lambda=lagrange_lambda,
                rng=rng,
            )

            metrics = {
                "actor_loss": loss_info[1][2].mean(),
                "critic_loss": loss_info[1][0].mean(),
                "cost_critic_loss": loss_info[1][1].mean(),
                "cost_surrogate": loss_info[1][3].mean(),
                "lagrange_lambda": lagrange_lambda,
                "entropy": loss_info[1][4].mean(),
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
            lagrange_lambda=config.init_lagrange_lambda,
            rng=_rng,
        )
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, num_updates
        )

        # TEST EVALUATION on prod sim
        trained_model = nnx.merge(graphdef, runner_state.params)

        def _test_step(carry, _):
            test_obs, test_env_state, test_rng = carry
            normalized_obs = jax.vmap(norm.normalize, in_axes=(None, 0))(
                runner_state.obs_norm_state, test_obs
            )
            pi, _, _ = trained_model(normalized_obs)
            test_rng, _rng = jax.random.split(test_rng)
            action = pi.sample(seed=_rng)
            test_rng, _rng = jax.random.split(test_rng)
            test_rng_step = jax.random.split(_rng, num_envs)
            next_obs, test_env_state, reward, done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(test_rng_step, test_env_state, action, config.test_env_params)
            return (next_obs, test_env_state, test_rng), info

        rng_test, _rng_test = jax.random.split(runner_state.rng)
        test_reset_rngs = jax.random.split(_rng_test, num_envs)
        test_obs, test_env_state = jax.vmap(env.reset, in_axes=(0, None))(
            test_reset_rngs, config.test_env_params
        )
        _, test_infos = jax.lax.scan(
            _test_step, (test_obs, test_env_state, rng_test), None, num_test_steps
        )
        test_metrics = {
            "test_episode_return": test_infos["returned_episode_returns"],
            "test_episode_cost_return": test_infos["returned_episode_cost_returns"],
            "test_episode_length": test_infos["returned_episode_lengths"],
            "test_dones": test_infos["returned_episode"],
        }

        return runner_state, metrics, test_metrics

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
        "COST_LIMIT": 25.0,
        "LAGRANGE_LR": 1e-2,
        "COST_GAMMA": 0.99,
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
        "NUM_RUNS": 1,
        "SEED": 30,
    }
    config = {
        "ENV_NAME": EcoAntV2().name,
        "COST_LIMIT": 400.0,
        "LR": 3e-4,
        "NUM_ENVS": 10,
        "N_ENSEMBLE": 8,
        "NUM_TEST_STEPS": 1024,
        "TRAIN_FREQ": 1024,
        "TOTAL_TIMESTEPS": int(1e6),
        "UPDATE_EPOCHS": 10,
        "BATCH_SIZE": 512,
        "GAMMA": 0.99,
        "COST_GAMMA": 0.999,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.0075,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "LAGRANGE_LR": 1e-2,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": True,
        "NUM_RUNS": 5,
        "SEED": 0,
    }

    rng = jax.random.PRNGKey(config["SEED"])
    train_rngs = jax.random.split(rng, config["NUM_RUNS"])
    brax_env = EcoAntV2(battery_limit=500.0)
    env = BraxToGymnaxWrapper(brax_env)
    base_params = env.default_params

    def replicate_params(params, shape):
        """Tile env params to a given leading shape for vmap."""
        return jax.tree.map(lambda x: jnp.full(shape, x), params)

    # Domain randomization training params: (NUM_RUNS, NUM_ENVS, ...)
    # To randomize dynamics, replace individual env entries with varied params
    train_env_params = replicate_params(
        base_params, (config["NUM_RUNS"], config["NUM_ENVS"])
    )
    # Ensemble params for SPiDR penalty: (NUM_RUNS, N_ENSEMBLE, ...)
    ensemble_env_params = replicate_params(
        base_params, (config["NUM_RUNS"], config["N_ENSEMBLE"])
    )
    # Prod/test params: (NUM_RUNS, ...)
    test_env_params = replicate_params(base_params, (config["NUM_RUNS"],))

    dynamic_config = DynamicConfig(
        rng=train_rngs,
        train_env_params=train_env_params,
        ensemble_env_params=ensemble_env_params,
        test_env_params=test_env_params,
        pessimism=jnp.array([0.0, 0.5, 1.0, 1.5, 2.0]),
        lr=jnp.ones(config["NUM_RUNS"]) * config["LR"],
        gae_gamma=jnp.ones(config["NUM_RUNS"]) * config["GAMMA"],
        gae_lambda=jnp.ones(config["NUM_RUNS"]) * config["GAE_LAMBDA"],
        entropy_coeff=jnp.ones(config["NUM_RUNS"]) * config["ENT_COEF"],
        value_coeff=jnp.ones(config["NUM_RUNS"]) * config["VF_COEF"],
        max_grad_norm=jnp.ones(config["NUM_RUNS"]) * config["MAX_GRAD_NORM"],
        ratio_clip=jnp.ones(config["NUM_RUNS"]) * config["CLIP_EPS"],
        cost_limit=jnp.ones(config["NUM_RUNS"]) * config["COST_LIMIT"],
        lagrange_lr=jnp.array([5e-2, 1e-2, 5e-3, 1e-3, 5e-4]),
        init_lagrange_lambda=jnp.zeros(config["NUM_RUNS"]),
        cost_gae_gamma=jnp.ones(config["NUM_RUNS"]) * config["COST_GAMMA"],
    )

    train_fn = make_train(
        env=env,
        num_steps=config["TOTAL_TIMESTEPS"],
        num_envs=config["NUM_ENVS"],
        n_ensemble=config["N_ENSEMBLE"],
        train_freq=config["TRAIN_FREQ"],
        batch_size=config["BATCH_SIZE"],
        num_epochs=config["UPDATE_EPOCHS"],
        num_test_steps=config["NUM_TEST_STEPS"],
        activation=jax.nn.tanh if config["ACTIVATION"] == "tanh" else jax.nn.relu,
        anneal_lr=config["ANNEAL_LR"],
    )

    train_vjit = jax.jit(jax.vmap(train_fn))
    start_time = time.perf_counter()
    runner_states, all_metrics, test_metrics = jax.block_until_ready(
        train_vjit(dynamic_config)
    )
    runtime = time.perf_counter() - start_time
    print(f"Runtime: {runtime:.2f}s")

    log.save_local(
        algo_name="spidr-ppo-l",
        env_name=brax_env.name,
        metrics={**all_metrics, **test_metrics},
    )

    log.save_wandb(
        project="EcoAnt",
        algo_name="spidr-ppo-l",
        env_name=brax_env.name,
        metrics={**all_metrics, **test_metrics},
    )
