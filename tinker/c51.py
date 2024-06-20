"""C51 distributional RL algorithm."""

import os
from typing import Callable, Tuple

import chex
import flashbax as fbx
import flax
import jax
import numpy as np
import optax
import wandb
from flashbax.buffers.trajectory_buffer import BufferState
from flax import linen as nn
from flax.training.train_state import TrainState
from gymnax import EnvState
from gymnax.environments.environment import Environment
from gymnax.wrappers import LogWrapper
from jax import numpy as jnp


class ZNetwork(nn.Module):
    """C51 network to generate a return distribution over a set of atoms.

    :param action_dim: The number of actions in the environment.
    :param n_atoms: The number of atoms in the distribution (e.g. 51).
    """

    action_dim: int
    n_atoms: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(120)(x)
        x = nn.relu(x)
        x = nn.Dense(84)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim * self.n_atoms)(x)
        if x.ndim > 1:
            x = x.reshape((x.shape[0], self.action_dim, self.n_atoms))
        else:
            x = x.reshape((self.action_dim, self.n_atoms))
        x = nn.softmax(x, axis=-1)  # pmfs
        return x


@chex.dataclass(frozen=True)
class Transition:
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    next_obs: jnp.ndarray
    done: jnp.ndarray


class C51TrainState(TrainState):
    target_params: flax.core.FrozenDict
    env_steps: int
    atoms: jnp.ndarray


RunnerState = Tuple[jnp.ndarray, EnvState, BufferState, C51TrainState, chex.PRNGKey]


@jax.jit
def linear_schedule(start: float, end: float, duration: int, step: int) -> jnp.ndarray:
    slope = (end - start) / duration
    return jnp.clip(slope * step + start, end)


def make_train(
    env: Environment,
    num_steps: int,
    num_atoms: int,
    value_min: float,
    value_max: float,
    buffer_size: int,
    batch_size: int,
    learning_starts: int,
    train_freq: int,
    target_update_freq: int,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_duration: int,
    lr: float,
    gamma: float,
    tau: float,
    log_wandb: bool = False,
    log_freq: int = 100,
) -> Callable[[chex.PRNGKey], tuple]:
    """Generate a jitted JAX C51 train function.

    :param env: The environment to train on.
    :param num_steps: The number of training steps to run.
    :param num_atoms: The number of atoms in the distribution (e.g. 51).
    :param value_min: The minimum value of the distribution.
    :param value_max: The maximum value of the distribution.
    :param buffer_size: The maximum size of the replay buffer.
    :param batch_size: The size of the batch sampled from the replay buffer.
    :param learning_starts: The number of steps before training starts.
    :param train_freq: The number of steps between training updates.
    :param target_update_freq: The number of steps between target network updates.
    :param epsilon_start: The starting epsilon value for epsilon greedy exploration.
    :param epsilon_end: The ending epsilon value for epsilon greedy exploration.
    :param epsilon_duration: The number of steps to linearly decay epsilon.
    :param lr: The learning rate for the optimizer.
    :param gamma: The discount factor.
    :param tau: The polyak averaging factor for target network updates.
    :param log_wandb: Whether to log to wandb.
    :param log_freq: The frequency to log metrics to wandb.
    """
    env = LogWrapper(env)

    def train(key: chex.PRNGKey) -> dict:
        # INIT DUMMY VARIABLES
        dummy_key = jax.random.PRNGKey(0)
        _action = env.action_space().sample(dummy_key)
        _, _state = env.reset(dummy_key)
        _obs, _, _reward, _done, _ = env.step(dummy_key, _state, _action)
        _transition = Transition(
            obs=_obs,
            action=_action,
            reward=_reward,
            next_obs=_obs,
            done=_done,
        )

        # INIT BUFFER
        buffer = fbx.make_flat_buffer(
            max_length=buffer_size,
            min_length=batch_size,
            sample_batch_size=batch_size,
        )
        buffer: fbx.trajectory_buffer.TrajectoryBuffer = buffer.replace(
            init=jax.jit(buffer.init),
            add=jax.jit(buffer.add, donate_argnums=0),
            sample=jax.jit(buffer.sample),
            can_sample=jax.jit(buffer.can_sample),
        )
        buffer_state = buffer.init(_transition)

        # INIT NETWORK
        network = ZNetwork(action_dim=env.action_space().n, n_atoms=num_atoms)
        network_params = network.init(dummy_key, _obs)

        # INIT OPTIMIZER
        tx = optax.adam(learning_rate=lr)

        # INIT TRAIN STATE
        train_state = C51TrainState(
            apply_fn=network.apply,
            params=network_params,
            atoms=jnp.asarray(np.linspace(value_min, value_max, num=num_atoms)),
            target_params=jax.tree_map(lambda x: jnp.copy(x), network_params),
            opt_state=tx.init(network_params),
            env_steps=0,
            step=0,
            tx=tx,
        )

        # INIT ENV
        key, subkey = jax.random.split(key)
        obs, env_state = env.reset(subkey)

        def eps_greedy_exploration(key: chex.PRNGKey, q_vals: jnp.ndarray, step: int):
            key, subkey = jax.random.split(
                key, 2
            )  # a key for sampling random actions and one for picking
            eps = linear_schedule(epsilon_start, epsilon_end, epsilon_duration, step)
            greedy_actions = jnp.argmax(q_vals, axis=-1)  # get the greedy actions
            actions = jnp.where(
                jax.random.uniform(subkey, greedy_actions.shape)
                < eps,  # pick the actions that should be random
                jax.random.randint(
                    key, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]
                ),  # sample random actions,
                greedy_actions,
            )
            return actions

        def update(
            train_state: C51TrainState, buffer_state: BufferState, key: chex.PRNGKey
        ) -> Tuple[C51TrainState, jnp.ndarray]:
            transitions = buffer.sample(buffer_state, key).experience

            next_pmfs = network.apply(
                train_state.target_params, transitions.second.obs
            )  # (batch_size, num_actions, num_atoms)
            next_vals = (next_pmfs * train_state.atoms).sum(
                axis=-1
            )  # (batch_size, num_actions)
            next_actions = jnp.argmax(next_vals, axis=-1)  # (batch_size,)
            next_pmfs = next_pmfs[
                jnp.arange(next_pmfs.shape[0]), next_actions
            ]  # (batch_size, num_atoms)
            atoms = jnp.expand_dims(train_state.atoms, axis=0)
            atoms = jnp.broadcast_to(atoms, (batch_size, num_atoms))
            rewards = transitions.first.reward.reshape(batch_size, 1)
            dones = (1 - transitions.first.done).reshape(batch_size, 1)
            next_atoms = rewards + gamma * atoms * dones  # (batch_size, num_atoms)

            # Projection Step
            delta_z = train_state.atoms[1] - train_state.atoms[0]
            tz = jnp.clip(next_atoms, value_min, value_max)
            b = (tz - value_min) / delta_z
            l = jnp.clip(jnp.floor(b), a_min=0, a_max=num_atoms - 1)
            u = jnp.clip(jnp.ceil(b), a_min=0, a_max=num_atoms - 1)
            d_m_l = (u + (l == u).astype(jnp.float32) - b) * next_pmfs
            d_m_u = (b - l) * next_pmfs
            target_pmfs = jnp.zeros_like(next_pmfs)

            def project_to_bins(i, val):
                val = val.at[i, l[i].astype(jnp.int32)].add(d_m_l[i])
                val = val.at[i, u[i].astype(jnp.int32)].add(d_m_u[i])
                return val

            target_pmfs = jax.lax.fori_loop(
                0, target_pmfs.shape[0], project_to_bins, target_pmfs
            )

            def loss_fn(params: jnp.ndarray) -> jnp.ndarray:
                pmfs = network.apply(params, transitions.first.obs)
                old_pmfs = pmfs[
                    np.arange(pmfs.shape[0]), transitions.first.action.squeeze()
                ]

                old_pmfs_l = jnp.clip(old_pmfs, a_min=1e-5, a_max=1 - 1e-5)
                loss = (-(target_pmfs * jnp.log(old_pmfs_l)).sum(-1)).mean()
                return loss

            loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, loss

        def train_loop(carry: RunnerState, xs: jnp.ndarray) -> tuple:
            obs, env_state, buffer_state, train_state, key = carry

            # STEP ENV
            key, key_a, key_s = jax.random.split(key, 3)
            pmfs = network.apply(train_state.params, obs)
            q_vals = (pmfs * train_state.atoms).sum(axis=-1)
            action = eps_greedy_exploration(key_a, q_vals, train_state.env_steps)
            next_obs, env_state, reward, done, info = env.step(key_s, env_state, action)

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
            train_state = train_state.replace(env_steps=train_state.env_steps + 1)
            is_learning_step = (
                (buffer.can_sample(buffer_state))
                & (train_state.env_steps >= learning_starts)
                & (train_state.env_steps % train_freq == 0)
            )
            key, subkey = jax.random.split(key)
            train_state, loss = jax.lax.cond(
                is_learning_step,
                lambda train_state, buffer_state, key: update(
                    train_state, buffer_state, key
                ),
                lambda train_state, buffer_state, key: (
                    train_state,
                    jnp.array(0.0),
                ),  # do nothing
                train_state,
                buffer_state,
                subkey,
            )

            # UPDATE TARGET NETWORK
            is_target_update_step = train_state.env_steps % target_update_freq == 0
            train_state = jax.lax.cond(
                is_target_update_step,
                lambda train_state: train_state.replace(
                    target_params=optax.incremental_update(
                        train_state.params,
                        train_state.target_params,
                        tau,
                    )
                ),
                lambda train_state: train_state,
                operand=train_state,
            )

            metrics = {
                "timesteps": train_state.env_steps,
                "updates": train_state.step,
                "loss": loss,
                "returns": info["returned_episode_returns"],
                "episode_lengths": info["returned_episode_lengths"],
                "dones": info["returned_episode"],
                "PMF": pmfs[action],
            }

            # LOGGING WITH WANDB
            if log_wandb:

                def callback(metrics):
                    if metrics["timesteps"] % log_freq == 0:
                        wandb.log(metrics)

                jax.debug.callback(callback, metrics)

            runner_state = (next_obs, env_state, buffer_state, train_state, key)

            return runner_state, metrics

        # RUN TRAINING LOOP
        runner_state = (obs, env_state, buffer_state, train_state, key)
        runner_state, metrics = jax.lax.scan(
            train_loop, runner_state, None, length=num_steps
        )
        return metrics, runner_state

    return train


def run(
    env: Environment,
    num_steps: int,
    network: ZNetwork,
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
    :return pmfs: List of episode rollout Z distribution PMFs.
    :return actions: List of episode rollout actions.
    :return rewards: List of episode rollout rewards.
    :return next_state: List of episode rollout next states.
    :return next_obs: List of episode rollout next observations.
    :return dones: List of episode rollout done flags.
    """
    # Reset the environment
    atoms = jnp.asarray(np.linspace(0, 1, num=network.n_atoms))
    rng_reset, rng_episode = jax.random.split(rng_input)
    obs, state = env.reset(rng_reset)

    def policy_step(state_input, tmp):
        """lax.scan compatible step transition in jax env."""
        obs, state, rng = state_input
        rng, rng_step, rng_net = jax.random.split(rng, 3)
        pmfs = network.apply(network_params, obs)
        q_vals = (pmfs * atoms).sum(axis=-1)
        action = jnp.argmax(q_vals, axis=-1)
        next_obs, next_state, reward, done, _ = env.step(rng_step, state, action)
        carry = [next_obs, next_state, rng]
        return carry, [
            state,
            obs,
            pmfs,
            action,
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
    from gymnax.environments import CartPole

    env = CartPole()
    train = make_train(
        env=env,
        num_steps=int(1e6),
        num_atoms=51,
        value_min=0,
        value_max=500,
        buffer_size=10000,
        batch_size=32,
        learning_starts=1000,
        train_freq=10,
        target_update_freq=500,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_duration=25e4,
        lr=2.5e-4,
        gamma=0.99,
        tau=0.95,
        log_wandb=True,
    )
    train_jit = jax.jit(train)

    wandb.login(os.environ.get("WANDB_KEY"))
    wandb.init(
        project="Tinker",
        tags=["C51", f"{env.name.upper()}", f"jax_{jax.__version__}"],
        name=f"c51_{env.name}",
        mode="online",
    )
