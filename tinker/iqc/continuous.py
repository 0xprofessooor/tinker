"""Implicit Quantile Control (IQC) continuous state space algorithm."""

import time
import flashbax as fbx
import jax
from jax import numpy as jnp
from flax import struct, nnx
from gymnax.environments.environment import Environment, EnvParams, EnvState
import optax
from safenax.wrappers import LogWrapper
from safenax.portfolio_optimization.po_garch import (
    PortfolioOptimizationGARCH,
    GARCHParams,
)
from tinker import log


class StateModel(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        embedding_dim: int = 64,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.embedding_dim = embedding_dim

        self.encoder = nnx.Sequential(
            nnx.Linear(obs_dim + action_dim, 256, rngs=rngs),
            nnx.LayerNorm(256, rngs=rngs),
            nnx.silu,
            nnx.Linear(256, obs_dim * embedding_dim, rngs=rngs),
        )

        self.cosine_net = nnx.Sequential(
            nnx.Linear(embedding_dim, embedding_dim, rngs=rngs),
            nnx.relu,
        )

        self.decoder = nnx.Sequential(
            nnx.Linear(embedding_dim, 256, rngs=rngs),
            nnx.silu,
            nnx.Linear(256, 1, rngs=rngs),
        )

    def __call__(self, obs: jax.Array, action: jax.Array, tau: jax.Array) -> jax.Array:
        batch_size = obs.shape[0]
        inputs = jnp.concatenate([obs, action], axis=-1)
        psi = self.encoder(inputs)  # (batch_size, obs_dim * embedding_dim)
        psi = psi.reshape(batch_size, self.obs_dim, self.embedding_dim)

        tau_expanded = tau[..., None]  # (batch_size, obs_dim, 1)
        i = jnp.arange(self.embedding_dim)[None, None, :]  # (1, 1, embedding_dim)
        cosine_input = jnp.cos(
            jnp.pi * i * tau_expanded
        )  # (batch_size, obs_dim, embedding_dim)
        phi = self.cosine_net(cosine_input)  # (batch_size, obs_dim, embedding_dim)

        h = psi * phi  # (batch_size, obs_dim, embedding_dim)
        out = self.decoder(h)  # (batch_size, obs_dim, 1)
        return out.squeeze(-1)  # (batch_size, obs_dim)


@struct.dataclass
class Transition:
    state: EnvState
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    done: jax.Array
    info: dict


@struct.dataclass
class DynamicConfig:
    rng: jax.Array
    env_params: EnvParams
    lr: jax.Array = 3e-4
    adam_eps: jax.Array = 1e-12
    max_grad_norm: jax.Array = 1.0


def make_train(
    env: Environment,
    num_steps: int,
    num_envs: int,
    train_freq: int,
    buffer_size: int,
    batch_size: int,
    num_epochs: int,
    embedding_dim: int = 64,
    anneal_lr: bool = False,
):
    num_updates = num_steps // train_freq
    env = LogWrapper(env)

    def train(config: DynamicConfig):
        obs_dim = env.observation_space(config.env_params).shape[0]
        action_dim = env.action_space(config.env_params).shape[0]
        rng, model_rng, dummy_key = jax.random.split(config.rng, 3)

        model = StateModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            embedding_dim=embedding_dim,
            rngs=nnx.Rngs(model_rng),
        )

        # INIT OPTIMIZER
        if anneal_lr:
            schedule = optax.linear_schedule(
                init_value=config.lr,
                end_value=0.0,
                transition_steps=num_updates * num_epochs,
            )
            tx = optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adam(learning_rate=schedule, eps=config.adam_eps),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adam(config.lr, eps=config.adam_eps),
            )
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
        model_graphdef, model_state = nnx.split(model)
        opt_graphdef, opt_state = nnx.split(optimizer)

        # INIT BUFFER
        buffer = fbx.make_flat_buffer(
            max_length=buffer_size,
            min_length=batch_size,
            sample_batch_size=batch_size,
            add_batch_size=num_envs,
        )
        buffer = buffer.replace(
            init=jax.jit(buffer.init),
            add=jax.jit(buffer.add, donate_argnums=0),
            sample=jax.jit(buffer.sample),
            can_sample=jax.jit(buffer.can_sample),
        )

        # Create dummy transition for buffer init
        _, dummy_state = env.reset(dummy_key, config.env_params)
        dummy_action = env.action_space(config.env_params).sample(dummy_key)
        dummy_obs, dummy_next_state, dummy_reward, dummy_done, dummy_info = env.step(
            dummy_key, dummy_state, dummy_action, config.env_params
        )

        dummy_transition = Transition(
            state=dummy_next_state,
            obs=dummy_obs,
            action=dummy_action,
            reward=dummy_reward,
            done=dummy_done,
            info=dummy_info,
        )
        buffer_state = buffer.init(dummy_transition)

        # INIT ENV
        init_rngs = jax.random.split(rng, num_envs + 1)
        rng = init_rngs[0]
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(
            init_rngs[1:], config.env_params
        )

        runner_state = (env_state, model_state, opt_state, buffer_state, obsv, rng)

        def _update_step(runner_state: tuple, update_idx: int):
            def _env_step(runner_state: tuple, _):
                env_state, model_state, opt_state, buffer_state, obsv, rng = (
                    runner_state
                )
                rngs = jax.random.split(rng, num_envs + 1)
                rng = rngs[0]
                action_rngs = rngs[1:]
                action = jax.vmap(
                    env.action_space(config.env_params).sample, in_axes=0
                )(action_rngs)

                rng_step = jax.random.split(rng, num_envs + 1)
                rng = rng_step[0]
                rng_step = rng_step[1:]
                next_obsv, next_env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, config.env_params)

                transition = Transition(
                    state=env_state,
                    obs=obsv,
                    action=action,
                    reward=reward,
                    done=done,
                    info=info,
                )

                buffer_state = buffer.add(buffer_state, transition)

                runner_state = (
                    next_env_state,
                    model_state,
                    opt_state,
                    buffer_state,
                    next_obsv,
                    rng,
                )

                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, length=train_freq
            )

            def _update_epoch(runner_state: tuple, _):
                env_state, model_state, opt_state, buffer_state, obsv, rng = (
                    runner_state
                )
                rng, rng_sample, rng_tau = jax.random.split(rng, 3)
                batch = buffer.sample(buffer_state, rng_sample).experience
                tau = jax.random.uniform(rng_tau, shape=(batch_size, obs_dim))

                def loss_fn(model: StateModel) -> jax.Array:
                    """Multivariate pinball loss"""
                    obs = batch.first.obs
                    target_obs = batch.second.obs
                    action = batch.first.action

                    pred_obs = model(obs, action, tau)
                    delta = target_obs - pred_obs
                    loss = jnp.mean(
                        jnp.sum(
                            delta * (tau - (delta < 0).astype(jnp.float32)), axis=-1
                        )
                    )
                    return loss

                model = nnx.merge(model_graphdef, model_state)
                optimizer = nnx.merge(opt_graphdef, opt_state)
                loss, grads = nnx.value_and_grad(loss_fn)(model)
                optimizer.update(model, grads)

                _, model_state = nnx.split(model)
                _, opt_state = nnx.split(optimizer)

                runner_state = (
                    env_state,
                    model_state,
                    opt_state,
                    buffer_state,
                    obsv,
                    rng,
                )
                return runner_state, loss

            runner_state, loss = jax.lax.scan(
                _update_epoch, runner_state, None, length=num_epochs
            )
            metrics = {
                "num_updates": update_idx,
                "loss": loss.mean(),
                "episode_return": traj_batch.info["returned_episode_returns"].mean(),
                "episode_length": traj_batch.info["returned_episode_lengths"].mean(),
            }
            return runner_state, metrics

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, jnp.arange(num_updates)
        )
        return runner_state, metrics

    return train


if __name__ == "__main__":
    NUM_SEEDS = 1
    SEED = 0

    rng = jax.random.PRNGKey(SEED)
    rng_train, rng_env = jax.random.split(rng)
    garch_params = {
        "APPL": GARCHParams(
            mu=5e-4,
            omega=1e-5,
            alpha=jnp.array([0.05]),
            beta=jnp.array([0.9]),
            initial_price=1.0,
        ),
        "BTC": GARCHParams(
            mu=1.5e-3,
            omega=1e-4,
            alpha=jnp.array([0.15]),
            beta=jnp.array([0.8]),
            initial_price=1.0,
        ),
    }
    episode_length = 1000
    env = PortfolioOptimizationGARCH(
        rng=rng_env,
        garch_params=garch_params,
        num_samples=episode_length,
        num_trajectories=10_000,
    )
    env_params = env.default_params.replace(max_steps=episode_length)
    train_fn = make_train(
        env=env,
        num_steps=100_000,
        num_envs=1,
        train_freq=1000,
        buffer_size=10_000,
        batch_size=64,
        num_epochs=10,
    )

    rngs = jax.random.split(rng_train, NUM_SEEDS)
    dynamic_config = DynamicConfig(
        rng=rngs,
        env_params=jax.tree.map(lambda x: jnp.stack([x] * NUM_SEEDS), env_params),
        lr=jnp.ones(NUM_SEEDS) * 3e-4,
        adam_eps=jnp.ones(NUM_SEEDS) * 1e-12,
        max_grad_norm=jnp.ones(NUM_SEEDS) * 1.0,
    )
    train_vjit = jax.jit(jax.vmap(train_fn))
    start_time = time.perf_counter()
    runner_states, all_metrics = jax.block_until_ready(train_vjit(dynamic_config))
    runtime = time.perf_counter() - start_time
    print(f"Runtime: {runtime:.2f}s")

    log.save_local(
        algo_name="iqc",
        env_name=env.name,
        metrics=all_metrics,
    )

    log.save_wandb(
        project="test",
        algo_name="iqc",
        env_name=env.name,
        metrics=all_metrics,
    )
