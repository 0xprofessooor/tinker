import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import NamedTuple
from gymnax.environments import EnvParams
from gymnax.environments.environment import Environment
import chex
from flax.training.train_state import TrainState
import distrax
from safenax import EcoAntV2
from safenax.wrappers import BraxToGymnaxWrapper, LogWrapper
from tinker import log


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

        # Cost Critic (Essential for estimating future cost violation)
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
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    cost_value: jnp.ndarray
    reward: jnp.ndarray
    cost: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    val_update: jnp.ndarray
    check_val: jnp.ndarray


def make_train(
    env: Environment,
    env_params: EnvParams,
    num_steps: int,  # Total steps per environment
    train_freq: int,  # Steps to collect per environment before updating
    num_envs: int,
    # CPPO Hyperparameters
    confidence: float,
    cvar_limit: float,
    lam_lr: float = 1e-2,
    nu_start: float = 0.0,
    lam_start: float = 0.5,
    nu_delay: float = 0.8,
    delay: float = 1.0,
    cvar_clip_ratio: float = 0.05,
    # PPO Hyperparameters
    learning_rate: float = 3e-4,
    num_minibatches: int = 4,
    update_epochs: int = 4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    max_grad_norm: float = 0.5,
):
    env = LogWrapper(env)
    num_updates = num_steps // train_freq
    minibatch_size = num_envs * train_freq // num_minibatches

    def train(rng: chex.PRNGKey):
        network = ActorCritic(env.action_space(env_params).shape[0])
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)

        tx = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate, eps=1e-5),
        )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, num_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        cppo_state = {
            "nu": jnp.array(nu_start, dtype=jnp.float32),
            "lam": jnp.array(lam_start, dtype=jnp.float32),
            "current_cost_ep_ret": jnp.zeros(num_envs, dtype=jnp.float32),
        }

        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, rng, cppo_state = runner_state

            # --- 1. COLLECT TRAJECTORIES ---
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng, cppo_state = runner_state

                rng, _rng = jax.random.split(rng)
                pi, value, cost_value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, num_envs)
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)

                # Handling Cost from Info
                cost = info["cost"]

                # Update Cost Episode Return
                new_cost_ep_ret = cppo_state["current_cost_ep_ret"] + cost

                # CPPO Estimation: Total Cost ~ Current Cost Return + Cost Value (Future) - Current Cost (Correction)
                check_val = new_cost_ep_ret - cost + cost_value

                # Violation: Cost > Nu (Upper Bound)
                is_violation = check_val > cppo_state["nu"]

                # Penalty: Positive if Cost is high
                raw_update = (
                    delay
                    * cppo_state["lam"]
                    / (1.0 - confidence)
                    * (check_val - cppo_state["nu"])
                )

                clip_bound = jnp.abs(cost_value) * cvar_clip_ratio
                val_update = jnp.where(
                    is_violation, jnp.minimum(raw_update, clip_bound), 0.0
                )

                final_cost_ep_ret = jnp.where(done, 0.0, new_cost_ep_ret)
                cppo_state["current_cost_ep_ret"] = final_cost_ep_ret

                transition = Transition(
                    done,
                    action,
                    value,
                    cost_value,
                    reward,
                    cost,
                    log_prob,
                    last_obs,
                    info,
                    val_update,
                    check_val,
                )
                runner_state = (train_state, env_state, obsv, rng, cppo_state)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, train_freq
            )

            # --- 2. CALCULATE ADVANTAGE ---
            train_state, env_state, last_obs, rng, cppo_state = runner_state
            _, last_val, last_cost_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val, last_cost_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value, cost_gae, next_cost_value = gae_and_next_value
                    done, value, cost_value, reward, cost, val_update = (
                        transition.done,
                        transition.value,
                        transition.cost_value,
                        transition.reward,
                        transition.cost,
                        transition.val_update,
                    )

                    # Reward GAE
                    delta = reward + gamma * next_value * (1 - done) - value
                    gae = delta + gamma * gae_lambda * (1 - done) * gae

                    # Cost GAE (Used only for Cost Critic Targets)
                    cost_delta = (
                        cost + gamma * next_cost_value * (1 - done) - cost_value
                    )
                    cost_gae = cost_delta + gamma * gae_lambda * (1 - done) * cost_gae

                    return (gae, value, cost_gae, cost_value), (
                        gae,
                        cost_gae,
                        val_update,
                    )

                _, (advantages, cost_advantages, val_updates) = jax.lax.scan(
                    _get_advantages,
                    (
                        jnp.zeros_like(last_val),
                        last_val,
                        jnp.zeros_like(last_cost_val),
                        last_cost_val,
                    ),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )

                # CPPO Policy Update: Advantage = Reward Adv - Cost Penalty
                modified_advantages = advantages - val_updates

                return (
                    modified_advantages,
                    advantages + traj_batch.value,
                    cost_advantages + traj_batch.cost_value,
                )

            advantages, targets, cost_targets = _calculate_gae(
                traj_batch, last_val, last_cost_val
            )

            # --- 3. UPDATE NETWORK ---
            def _update_epoch(update_state, unused):
                def _update_minibatch(train_state, batch_info):
                    traj_batch, advantages, targets, cost_targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets, cost_targets):
                        pi, value, cost_value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # Value Losses
                        value_loss = 0.5 * jnp.mean(jnp.square(value - targets))
                        cost_value_loss = 0.5 * jnp.mean(
                            jnp.square(cost_value - cost_targets)
                        )

                        # Policy Loss
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

                        entropy = pi.entropy().mean()

                        # Total Loss
                        total_loss = loss_actor + value_loss + cost_value_loss
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

                train_state, total_loss = jax.lax.scan(
                    _update_minibatch,
                    update_state,
                    minibatch,
                )
                return train_state, total_loss

            # Prepare batch
            minibatch = (traj_batch, advantages, targets, cost_targets)
            minibatch = jax.tree_util.tree_map(
                lambda x: x.reshape((train_freq * num_envs, -1)), minibatch
            )
            rng, _rng = jax.random.split(rng)
            permutation = jax.random.permutation(_rng, train_freq * num_envs)
            minibatch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), minibatch
            )
            minibatch = jax.tree_util.tree_map(
                lambda x: x.reshape(num_minibatches, minibatch_size, *x.shape[1:]),
                minibatch,
            )

            train_state, total_loss = jax.lax.scan(
                _update_epoch, train_state, None, update_epochs
            )

            # --- 4. UPDATE CPPO DUAL VARIABLES (COST) ---
            all_check_vals = traj_batch.check_val.reshape(-1)

            # CVaR (Upper Tail) Estimate
            cvar_loss = jnp.mean(jnp.maximum(0.0, all_check_vals - cppo_state["nu"]))
            cvar_est = cppo_state["nu"] + (1.0 / (1.0 - confidence)) * cvar_loss

            # Update Lambda: Based on Nu vs Limit
            new_lam = cppo_state["lam"] + lam_lr * (cppo_state["nu"] - cvar_limit)
            new_lam = jnp.maximum(0.0, new_lam)

            # Update Nu: Track Moving Average (Heuristic from original repo)
            avg_cost = jnp.mean(all_check_vals)
            new_nu = avg_cost * nu_delay

            cppo_state["nu"] = new_nu
            cppo_state["lam"] = new_lam

            # --- Calculate True Empirical Violation Rate ---
            is_terminal = traj_batch.done
            num_episodes = jnp.maximum(is_terminal.sum(), 1.0)

            # LogWrapper provides the episode returns at terminal steps
            terminal_costs = traj_batch.info["returned_episode_cost_returns"]

            # Check violations only at terminal steps
            exceeds_threshold = (terminal_costs > cvar_limit) & is_terminal
            num_exceedances = exceeds_threshold.sum()
            empirical_var_probability = num_exceedances / num_episodes

            metric = dict(
                entropy=total_loss[1][3],
                nu=cppo_state["nu"],
                lam=cppo_state["lam"],
                avg_cost_return=avg_cost,
                cvar_est=cvar_est,
                episode_return=traj_batch.info["returned_episode_returns"].mean(),
                episode_cost_return=traj_batch.info[
                    "returned_episode_cost_returns"
                ].mean(),
                empirical_var_probability=empirical_var_probability,
            )
            runner_state = (train_state, env_state, last_obs, rng, cppo_state)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng, cppo_state)

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, num_updates
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train


if __name__ == "__main__":
    SEED = 0
    NUM_SEEDS = 1

    brax_env = EcoAntV2(battery_limit=500.0)
    env = BraxToGymnaxWrapper(env=brax_env, episode_length=1000)
    env_params = env.default_params

    rng = jax.random.PRNGKey(SEED)
    rngs = jax.random.split(rng, NUM_SEEDS)

    train_fn = make_train(
        env,
        env_params,
        num_steps=int(2e6),
        train_freq=1000,
        num_envs=5,
        confidence=0.9,
        cvar_limit=500.0,
    )
    train_vjit = jax.jit(jax.vmap(train_fn))
    runner_states, all_metrics = jax.block_until_ready(train_vjit(rngs))

    log.save_local(
        algo_name="cppo",
        env_name=brax_env.name,
        metrics=all_metrics,
    )

    log.save_wandb(
        project="EcoAnt",
        algo_name="cppo",
        env_name=brax_env.name,
        metrics=all_metrics,
    )
