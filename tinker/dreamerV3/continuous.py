"""DreamerV3 with Continuous Action Space.

A single-file, end-to-end JIT-compilable JAX implementation of DreamerV3
for continuous-action gymnax environments. Follows the tinker project
conventions (make_train / DynamicConfig / LogWrapper / vmap over seeds).

Key DreamerV3 components:
- RSSM world model (discrete stochastic + deterministic recurrent state)
- Symlog/symexp transforms and TwoHot categorical regression
- Imagination-based actor-critic with lambda returns
- Replay-based value learning (repval)
- Slow EMA value target and return-percentile advantage normalization
- Free-bits KL loss
- LR warmup (1000 steps)
"""

import time
from typing import NamedTuple, Tuple

import distrax
from flax import nnx, struct
import jax
import jax.numpy as jnp
import numpy as np
import optax
from gymnax.environments.environment import Environment, EnvParams
from safenax.wrappers import LogWrapper, BraxToGymnaxWrapper
from safenax import EcoAntV2

from tinker import log


# ---------------------------------------------------------------------------
# Helper math
# ---------------------------------------------------------------------------


def symlog(x):
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x):
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))


def twohot_encode(target, bins):
    """Encode scalar targets into two-hot vectors over *bins*.

    The bins are already in symexp-transformed space, so targets are
    compared directly against bin values without additional transformation.
    """
    target = jnp.squeeze(target, axis=-1)
    below = jnp.sum((bins <= target[..., None]).astype(jnp.int32), axis=-1) - 1
    above = len(bins) - jnp.sum((bins > target[..., None]).astype(jnp.int32), axis=-1)
    below = jnp.clip(below, 0, len(bins) - 1)
    above = jnp.clip(above, 0, len(bins) - 1)
    equal = below == above
    dist_to_below = jnp.where(equal, 1.0, jnp.abs(bins[below] - target))
    dist_to_above = jnp.where(equal, 1.0, jnp.abs(bins[above] - target))
    total = dist_to_below + dist_to_above
    weight_below = dist_to_above / total
    weight_above = dist_to_below / total
    oh_below = jax.nn.one_hot(below, len(bins))
    oh_above = jax.nn.one_hot(above, len(bins))
    return oh_below * weight_below[..., None] + oh_above * weight_above[..., None]


def twohot_decode(logits, bins):
    """Decode logits over bins back to scalar using weighted average.

    The bins are already in symexp-transformed space, so the weighted
    average directly gives the predicted value (no additional symexp).
    """
    probs = jax.nn.softmax(logits, axis=-1)
    return (probs * bins).sum(axis=-1, keepdims=True)


def twohot_log_prob(logits, target, bins):
    """Log-prob of target under a TwoHot distribution."""
    mixed_target = twohot_encode(target, bins)
    log_pred = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
    return (mixed_target * log_pred).sum(axis=-1)


def make_symexp_bins(num_bins):
    """Create symmetric symexp-spaced bins."""
    if num_bins % 2 == 1:
        half = jnp.linspace(-20, 0, (num_bins - 1) // 2 + 1)
        half = symexp(half)
        return jnp.concatenate([half, -jnp.flip(half[:-1])], axis=0)
    half = jnp.linspace(-20, 0, num_bins // 2)
    half = symexp(half)
    return jnp.concatenate([half, -jnp.flip(half)], axis=0)


def categorical_kl(logits_p, logits_q):
    """KL(p || q) for categorical distributions."""
    log_p = jax.nn.log_softmax(logits_p, axis=-1)
    log_q = jax.nn.log_softmax(logits_q, axis=-1)
    p = jax.nn.softmax(logits_p, axis=-1)
    return (p * (log_p - log_q)).sum(axis=-1)


def gumbel_softmax(rng, logits, hard=True):
    """Straight-through Gumbel-Softmax sampling."""
    u = jax.random.uniform(rng, logits.shape, minval=1e-6, maxval=1.0 - 1e-6)
    gumbels = -jnp.log(-jnp.log(u))
    y_soft = jax.nn.softmax(logits + gumbels, axis=-1)
    if hard:
        idx = jnp.argmax(y_soft, axis=-1)
        y_hard = jax.nn.one_hot(idx, logits.shape[-1])
        return y_hard - jax.lax.stop_gradient(y_soft) + y_soft
    return y_soft


def lambda_return(last, term, reward, value, bootstrap, discount, gae_lambda):
    """Compute lambda returns. All inputs: (B, T, 1). Returns: (B, T-1, 1)."""
    live = (1 - term[:, 1:]) * discount
    cont = (1 - last[:, 1:]) * gae_lambda
    interm = reward[:, 1:] + (1 - cont) * live * bootstrap[:, 1:]

    def _scan_fn(carry, i):
        out = interm[:, i] + live[:, i] * cont[:, i] * carry
        return out, out

    T = live.shape[1]
    _, outs = jax.lax.scan(_scan_fn, bootstrap[:, -1], jnp.arange(T - 1, -1, -1))
    return jnp.flip(outs, axis=0).transpose(1, 0, 2)


# ---------------------------------------------------------------------------
# Weight initialization matching reference: std = 1.1368 / sqrt(fan_in)
# ---------------------------------------------------------------------------


def dreamer_kernel_init(key, shape, dtype=jnp.float32):
    """Truncated normal init: std = 1.1368 / sqrt(fan_in), clipped to [-2std, 2std]."""
    fan_in = shape[0] if len(shape) >= 2 else shape[-1]
    std = 1.1368 * jnp.sqrt(1.0 / fan_in)
    return jax.random.truncated_normal(key, -2.0, 2.0, shape, dtype) * std


def dreamer_bias_init(key, shape, dtype=jnp.float32):
    return jnp.zeros(shape, dtype)


# ---------------------------------------------------------------------------
# NNX building blocks
# ---------------------------------------------------------------------------


class RMSNormLayer(nnx.Module):
    def __init__(self, dim, rngs, eps=1e-4):
        self.scale = nnx.Param(jnp.ones(dim))
        self.eps = eps

    def __call__(self, x):
        rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return x / rms * self.scale


class MLPBlock(nnx.Module):
    """Linear -> RMSNorm -> SiLU."""

    def __init__(self, in_dim, out_dim, rngs):
        self.linear = nnx.Linear(
            in_dim,
            out_dim,
            rngs=rngs,
            kernel_init=dreamer_kernel_init,
            bias_init=dreamer_bias_init,
        )
        self.norm = RMSNormLayer(out_dim, rngs)

    def __call__(self, x):
        return jax.nn.silu(self.norm(self.linear(x)))


class MLPStack(nnx.Module):
    """Stack of MLPBlocks."""

    def __init__(self, in_dim, hidden, num_layers, rngs, symlog_inputs=False):
        self.symlog_inputs = symlog_inputs
        layers = []
        d = in_dim
        for _ in range(num_layers):
            layers.append(MLPBlock(d, hidden, rngs))
            d = hidden
        self.layers = nnx.List(layers)

    def __call__(self, x):
        if self.symlog_inputs:
            x = symlog(x)
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# RSSM components
# ---------------------------------------------------------------------------


class DeterNet(nnx.Module):
    """GRU-style deterministic transition."""

    def __init__(self, deter, hidden, stoch_flat, action_dim, rngs):
        self.deter_proj = MLPBlock(deter, hidden, rngs)
        self.stoch_proj = MLPBlock(stoch_flat, hidden, rngs)
        self.action_proj = MLPBlock(action_dim, hidden, rngs)
        self.hidden_layer = MLPBlock(3 * hidden + deter, deter, rngs)
        self.gru = nnx.Linear(
            deter,
            3 * deter,
            rngs=rngs,
            kernel_init=dreamer_kernel_init,
            bias_init=dreamer_bias_init,
        )
        self.deter = deter

    def __call__(self, stoch_flat, deter, action):
        action_normed = action / jnp.clip(jnp.abs(action), a_min=1.0)
        deter_embed = self.deter_proj(deter)
        stoch_embed = self.stoch_proj(stoch_flat)
        action_embed = self.action_proj(action_normed)
        x = jnp.concatenate([deter_embed, stoch_embed, action_embed, deter], axis=-1)
        x = self.hidden_layer(x)
        gates = self.gru(x)
        reset, cand, update = jnp.split(gates, 3, axis=-1)
        reset = jax.nn.sigmoid(reset)
        cand = jnp.tanh(reset * cand)
        update = jax.nn.sigmoid(update - 1)
        return update * cand + (1 - update) * deter


class ObsNet(nnx.Module):
    """Posterior: [deter, embed] -> stoch logits."""

    def __init__(self, in_dim, hidden, out_dim, rngs):
        self.block = MLPBlock(in_dim, hidden, rngs)
        self.out = nnx.Linear(
            hidden,
            out_dim,
            rngs=rngs,
            kernel_init=dreamer_kernel_init,
            bias_init=dreamer_bias_init,
        )

    def __call__(self, deter, embed):
        x = jnp.concatenate([deter, embed], axis=-1)
        return self.out(self.block(x))


class ImgNet(nnx.Module):
    """Prior: deter -> stoch logits."""

    def __init__(self, deter_dim, hidden, out_dim, rngs):
        self.block1 = MLPBlock(deter_dim, hidden, rngs)
        self.block2 = MLPBlock(hidden, hidden, rngs)
        self.out = nnx.Linear(
            hidden,
            out_dim,
            rngs=rngs,
            kernel_init=dreamer_kernel_init,
            bias_init=dreamer_bias_init,
        )

    def __call__(self, deter):
        return self.out(self.block2(self.block1(deter)))


# ---------------------------------------------------------------------------
# Full DreamerV3 model
# ---------------------------------------------------------------------------


class DreamerV3(nnx.Module):
    """Complete DreamerV3 model for continuous action spaces."""

    def __init__(
        self,
        obs_dim,
        action_dim,
        rngs,
        stoch=32,
        discrete=16,
        deter=512,
        hidden=256,
        num_bins=255,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.stoch = stoch
        self.discrete = discrete
        self.deter = deter
        self.hidden = hidden
        self.num_bins = num_bins
        self.flat_stoch = stoch * discrete
        self.feat_size = self.flat_stoch + deter
        self.unimix = 0.01

        # Encoder
        self.encoder = MLPStack(obs_dim, hidden, 3, rngs, symlog_inputs=True)

        # RSSM
        self.deter_net = DeterNet(deter, hidden, self.flat_stoch, action_dim, rngs)
        self.obs_net = ObsNet(deter + hidden, hidden, stoch * discrete, rngs)
        self.img_net = ImgNet(deter, hidden, stoch * discrete, rngs)

        # Decoder
        self.decoder_mlp = MLPStack(self.feat_size, hidden, 3, rngs)
        self.decoder_out = nnx.Linear(
            hidden,
            obs_dim,
            rngs=rngs,
            kernel_init=dreamer_kernel_init,
            bias_init=dreamer_bias_init,
        )

        # Reward head (outscale=0.0 -> zero init)
        self.reward_mlp = MLPStack(self.feat_size, hidden, 1, rngs)
        self.reward_out = nnx.Linear(
            hidden,
            num_bins,
            rngs=rngs,
            kernel_init=nnx.initializers.zeros_init(),
            bias_init=dreamer_bias_init,
        )

        # Continuation head (outscale=1.0 -> normal init)
        self.cont_mlp = MLPStack(self.feat_size, hidden, 1, rngs)
        self.cont_out = nnx.Linear(
            hidden,
            1,
            rngs=rngs,
            kernel_init=dreamer_kernel_init,
            bias_init=dreamer_bias_init,
        )

        # Actor (outscale=0.01: init weights then scale by 0.01)
        self.actor_mlp = MLPStack(self.feat_size, hidden, 3, rngs)
        self.actor_out = nnx.Linear(
            hidden,
            action_dim * 2,
            rngs=rngs,
            kernel_init=dreamer_kernel_init,
            bias_init=dreamer_bias_init,
        )
        self.actor_out.kernel[...] = self.actor_out.kernel[...] * 0.01

        # Critic (outscale=0.0 -> zero init)
        self.critic_mlp = MLPStack(self.feat_size, hidden, 3, rngs)
        self.critic_out = nnx.Linear(
            hidden,
            num_bins,
            rngs=rngs,
            kernel_init=nnx.initializers.zeros_init(),
            bias_init=dreamer_bias_init,
        )

    def _unimix(self, logits):
        probs = jax.nn.softmax(logits, axis=-1)
        uniform = self.unimix / logits.shape[-1]
        probs = probs * (1.0 - self.unimix) + uniform
        return jnp.log(probs)

    def encode(self, obs):
        return self.encoder(obs)

    def obs_step(self, stoch, deter, action, embed, reset, rng):
        """Single posterior RSSM step."""
        stoch = jnp.where(reset[..., None, None], 0.0, stoch)
        deter = jnp.where(reset[..., None], 0.0, deter)
        action = jnp.where(reset[..., None], 0.0, action)
        stoch_flat = stoch.reshape(*stoch.shape[:-2], self.flat_stoch)
        deter = self.deter_net(stoch_flat, deter, action)
        logits = self.obs_net(deter, embed)
        logits = logits.reshape(*logits.shape[:-1], self.stoch, self.discrete)
        logits = self._unimix(logits)
        new_stoch = gumbel_softmax(rng, logits, hard=True)
        return new_stoch, deter, logits

    def img_step(self, stoch, deter, action, rng):
        """Single prior RSSM step (no observation)."""
        stoch_flat = stoch.reshape(*stoch.shape[:-2], self.flat_stoch)
        deter = self.deter_net(stoch_flat, deter, action)
        logits = self.img_net(deter)
        logits = logits.reshape(*logits.shape[:-1], self.stoch, self.discrete)
        logits = self._unimix(logits)
        new_stoch = gumbel_softmax(rng, logits, hard=True)
        return new_stoch, deter

    def prior(self, deter):
        logits = self.img_net(deter)
        logits = logits.reshape(*logits.shape[:-1], self.stoch, self.discrete)
        return self._unimix(logits)

    def get_feat(self, stoch, deter):
        stoch_flat = stoch.reshape(*stoch.shape[:-2], self.flat_stoch)
        return jnp.concatenate([stoch_flat, deter], axis=-1)

    def decode(self, feat):
        return self.decoder_out(self.decoder_mlp(feat))

    def reward(self, feat):
        return self.reward_out(self.reward_mlp(feat))

    def cont(self, feat):
        return self.cont_out(self.cont_mlp(feat))

    def actor(self, feat):
        """Returns (mean, std) for continuous Normal distribution."""
        out = self.actor_out(self.actor_mlp(feat))
        mean, std_raw = jnp.split(out, 2, axis=-1)
        mean = jnp.tanh(mean)
        std = 0.9 * jax.nn.sigmoid(std_raw + 2.0) + 0.1  # [0.1, 1.0]
        return mean, std

    def critic(self, feat):
        return self.critic_out(self.critic_mlp(feat))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@struct.dataclass
class Transition:
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    done: jax.Array
    info: dict


@struct.dataclass
class DynamicConfig:
    """Holds dynamic configuration parameters for DreamerV3 training.

    :param rng: Random number generator key.
    :param env_params: Environment parameters.
    :param lr: Learning rate.
    :param kl_free: Free nats for KL loss.
    :param kl_dyn_scale: Scale for dynamics KL loss (default 1.0).
    :param kl_rep_scale: Scale for representation KL loss (default 0.1).
    :param horizon: Effective planning horizon (discount = 1 - 1/horizon).
    :param gae_lambda: Lambda for GAE-style lambda returns.
    :param entropy_coeff: Policy entropy regularization weight.
    :param slow_target_frac: EMA rate for slow value target.
    :param repval_scale: Loss scale for replay-based value learning.
    :param warmup_steps: Linear LR warmup steps.
    """

    rng: jax.Array
    env_params: EnvParams
    lr: jax.Array
    kl_free: jax.Array
    kl_dyn_scale: jax.Array
    kl_rep_scale: jax.Array
    horizon: jax.Array
    gae_lambda: jax.Array
    entropy_coeff: jax.Array
    slow_target_frac: jax.Array
    repval_scale: jax.Array
    warmup_steps: jax.Array


class ReturnEMAState(NamedTuple):
    low: jax.Array
    high: jax.Array


def init_return_ema():
    return ReturnEMAState(low=jnp.array(0.0), high=jnp.array(0.0))


def update_return_ema(state, returns, alpha=0.01):
    flat = returns.reshape(-1)
    q_low = jnp.percentile(flat, 5)
    q_high = jnp.percentile(flat, 95)
    new_low = alpha * q_low + (1 - alpha) * state.low
    new_high = alpha * q_high + (1 - alpha) * state.high
    return ReturnEMAState(low=new_low, high=new_high)


def return_scale(state):
    return jnp.clip(state.high - state.low, a_min=1.0)


# ---------------------------------------------------------------------------
# Parameter group keys for gradient isolation
# ---------------------------------------------------------------------------

ACTOR_KEYS = {"actor_mlp", "actor_out"}
CRITIC_KEYS = {"critic_mlp", "critic_out"}


# ---------------------------------------------------------------------------
# make_train
# ---------------------------------------------------------------------------


def make_train(
    env: Environment,
    num_steps: int,
    num_envs: int,
    train_freq: int,
    num_epochs: int = 1,
    batch_size: int = 16,
    batch_length: int = 64,
    imag_horizon: int = 15,
    stoch: int = 32,
    discrete: int = 16,
    deter: int = 512,
    hidden: int = 256,
    num_bins: int = 255,
    use_model_grads: bool = True,
):
    """Generate a jitted JAX DreamerV3 train function for continuous actions.

    :param env: Gymnax environment.
    :param num_steps: Total number of environment steps.
    :param num_envs: Number of parallel environments.
    :param train_freq: Steps between training updates.
    :param num_epochs: Gradient updates per collection phase. Each epoch
        samples a fresh batch from the replay buffer and updates all
        parameters (world model, actor, critic) jointly.
    :param batch_size: Number of trajectory slices per training batch.
    :param batch_length: Length of each trajectory slice.
    :param imag_horizon: Imagination rollout length.
    :param stoch: Number of stochastic state groups.
    :param discrete: Categories per stochastic group.
    :param deter: Deterministic state dimension.
    :param hidden: Hidden layer width.
    :param num_bins: Number of bins for TwoHot distributions.
    :param use_model_grads: If True, use stochastic backpropagation through
        the world model for the policy gradient (paper eq. 11). If False,
        use REINFORCE (score function estimator) as in r2dreamer.
    """
    num_updates = num_steps // train_freq
    env = LogWrapper(env)
    bins = make_symexp_bins(num_bins)

    def train(config: DynamicConfig) -> Tuple[dict, dict]:
        rng = config.rng
        discount = 1.0 - 1.0 / config.horizon

        obs_shape = env.observation_space(config.env_params).shape
        obs_dim = obs_shape[0] if len(obs_shape) == 1 else int(np.prod(obs_shape))
        action_dim = env.action_space(config.env_params).shape[0]

        # Init model
        rng, model_rng = jax.random.split(rng)
        model = DreamerV3(
            obs_dim,
            action_dim,
            nnx.Rngs(model_rng),
            stoch=stoch,
            discrete=discrete,
            deter=deter,
            hidden=hidden,
            num_bins=num_bins,
        )
        graphdef, params = nnx.split(model)
        slow_params = jax.tree_util.tree_map(jnp.copy, params)

        # Optimizer with LR warmup
        warmup_fn = lambda step: jnp.minimum(step / config.warmup_steps, 1.0)
        tx = optax.chain(
            optax.clip_by_global_norm(100.0),
            optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
            optax.scale_by_schedule(warmup_fn),
            optax.scale(-config.lr),
        )
        opt_state = tx.init(params)

        # Init envs
        rng, env_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(env_rng, num_envs)
        obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(
            reset_rngs, config.env_params
        )

        rssm_stoch = jnp.zeros((num_envs, stoch, discrete))
        rssm_deter = jnp.zeros((num_envs, deter))
        prev_action = jnp.zeros((num_envs, action_dim))

        # Replay buffer
        buf_size = min(num_steps, 100_000)
        buf = {
            "obs": jnp.zeros((buf_size, obs_dim)),
            "action": jnp.zeros((buf_size, action_dim)),
            "reward": jnp.zeros((buf_size,)),
            "done": jnp.zeros((buf_size,), dtype=jnp.bool_),
            "stoch": jnp.zeros((buf_size, stoch, discrete)),
            "deter": jnp.zeros((buf_size, deter)),
        }
        buf_ptr = jnp.array(0, dtype=jnp.int32)
        buf_count = jnp.array(0, dtype=jnp.int32)
        ret_ema = init_return_ema()

        # ===================================================================
        def _update_step(carry, _):
            (
                params,
                slow_params,
                opt_state,
                env_state,
                obs,
                rssm_stoch,
                rssm_deter,
                prev_action,
                buf,
                buf_ptr,
                buf_count,
                ret_ema,
                rng,
            ) = carry

            model = nnx.merge(graphdef, params)

            # ----- COLLECT -----
            def _env_step(carry, _):
                (env_state, obs, stoch_state, deter_state, prev_action, rng) = carry
                rng, act_rng, step_rng, rssm_rng = jax.random.split(rng, 4)

                embed = model.encode(obs)
                reset = jnp.zeros(num_envs, dtype=jnp.bool_)
                rssm_rngs = jax.random.split(rssm_rng, num_envs)
                new_stoch, new_deter, _ = jax.vmap(model.obs_step)(
                    stoch_state, deter_state, prev_action, embed, reset, rssm_rngs
                )

                feat = model.get_feat(new_stoch, new_deter)
                mean, std = model.actor(feat)
                dist = distrax.Independent(
                    distrax.Normal(mean, std), reinterpreted_batch_ndims=1
                )
                action = jnp.clip(dist.sample(seed=act_rng), -1.0, 1.0)

                step_rngs = jax.random.split(step_rng, num_envs)
                next_obs, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(step_rngs, env_state, action, config.env_params)

                transition = Transition(
                    obs=obs, action=action, reward=reward, done=done, info=info
                )
                carry = (env_state, next_obs, new_stoch, new_deter, action, rng)
                return carry, (transition, new_stoch, new_deter)

            init_carry = (env_state, obs, rssm_stoch, rssm_deter, prev_action, rng)
            final_carry, (traj, traj_stoch, traj_deter) = jax.lax.scan(
                _env_step, init_carry, None, train_freq
            )
            (env_state, obs, rssm_stoch, rssm_deter, prev_action, rng) = final_carry

            # ----- FILL BUFFER -----
            flat_obs = traj.obs.reshape(-1, obs_dim)
            flat_actions = traj.action.reshape(-1, action_dim)
            flat_rewards = traj.reward.reshape(-1)
            flat_dones = traj.done.reshape(-1)
            flat_stoch = traj_stoch.reshape(-1, stoch, discrete)
            flat_deter = traj_deter.reshape(-1, deter)
            n_new = flat_obs.shape[0]

            def _store(carry, i):
                buffer, ptr, count = carry
                idx = ptr % buf_size
                buffer = {
                    "obs": buffer["obs"].at[idx].set(flat_obs[i]),
                    "action": buffer["action"].at[idx].set(flat_actions[i]),
                    "reward": buffer["reward"].at[idx].set(flat_rewards[i]),
                    "done": buffer["done"].at[idx].set(flat_dones[i]),
                    "stoch": buffer["stoch"].at[idx].set(flat_stoch[i]),
                    "deter": buffer["deter"].at[idx].set(flat_deter[i]),
                }
                return (buffer, ptr + 1, jnp.minimum(count + 1, buf_size)), None

            (buf, buf_ptr, buf_count), _ = jax.lax.scan(
                _store, (buf, buf_ptr, buf_count), jnp.arange(n_new)
            )

            # ----- SAMPLE & TRAIN (num_epochs gradient steps) -----
            def _train_epoch(train_carry, _):
                params, slow_params, opt_state, ret_ema, rng = train_carry
                rng, sample_rng, train_rng = jax.random.split(rng, 3)
                valid = jnp.maximum(buf_count - batch_length - 1, 1)
                starts = jax.random.randint(sample_rng, (batch_size,), 0, valid)

                def _get_slice(start):
                    idx = (start + jnp.arange(batch_length)) % buf_size
                    return (
                        buf["obs"][idx],
                        buf["action"][idx],
                        buf["reward"][idx],
                        buf["done"][idx],
                        buf["stoch"][idx[0]],
                        buf["deter"][idx[0]],
                    )

                (
                    sampled_obs,
                    sampled_actions,
                    sampled_rewards,
                    sampled_dones,
                    init_stoch,
                    init_deter,
                ) = jax.vmap(_get_slice)(starts)

                B, T = sampled_obs.shape[:2]
                BT = B * T

                # === Helper: run posterior rollout (shared by WM and imagination) ===
                def _posterior_rollout(model, rng):
                    embed = model.encode(sampled_obs.reshape(BT, -1)).reshape(B, T, -1)

                    def _scan(carry, t):
                        prev_stoch, prev_deter, rng = carry
                        rng, scan_rng = jax.random.split(rng)
                        scan_rngs = jax.random.split(scan_rng, B)
                        is_first = jnp.where(
                            t == 0,
                            jnp.ones(B, dtype=jnp.bool_),
                            sampled_dones[:, jnp.maximum(t - 1, 0)],
                        )
                        new_stoch, new_deter, logits = jax.vmap(model.obs_step)(
                            prev_stoch,
                            prev_deter,
                            sampled_actions[:, jnp.maximum(t - 1, 0)],
                            embed[:, t],
                            is_first,
                            scan_rngs,
                        )
                        return (new_stoch, new_deter, rng), (
                            new_stoch,
                            new_deter,
                            logits,
                        )

                    _, (post_stoch, post_deter, post_logits) = jax.lax.scan(
                        _scan, (init_stoch, init_deter, rng), jnp.arange(T)
                    )
                    post_stoch = jnp.moveaxis(post_stoch, 0, 1)
                    post_deter = jnp.moveaxis(post_deter, 0, 1)
                    post_logits = jnp.moveaxis(post_logits, 0, 1)
                    return embed, post_stoch, post_deter, post_logits

                # === Helper: imagination rollout ===
                def _imagine(model, post_stoch, post_deter, rng):
                    imag_start_stoch = jax.lax.stop_gradient(
                        post_stoch.reshape(BT, stoch, discrete)
                    )
                    imag_start_deter = jax.lax.stop_gradient(
                        post_deter.reshape(BT, deter)
                    )

                    def _imag(carry, _):
                        s, d, rng = carry
                        rng, action_rng, step_rng = jax.random.split(rng, 3)
                        feat = model.get_feat(s, d)
                        mean, std = model.actor(feat)
                        noise = jax.random.normal(action_rng, mean.shape)
                        action = jnp.clip(mean + std * noise, -1.0, 1.0)
                        step_rngs = jax.random.split(step_rng, BT)
                        ns, nd = jax.vmap(model.img_step)(s, d, action, step_rngs)
                        return (ns, nd, rng), (feat, action)

                    _, (imag_feat, imag_actions) = jax.lax.scan(
                        _imag,
                        (imag_start_stoch, imag_start_deter, rng),
                        None,
                        imag_horizon + 1,
                    )
                    return jnp.moveaxis(imag_feat, 0, 1), jnp.moveaxis(
                        imag_actions, 0, 1
                    )

                # ===============================================================
                # LOSS 1: World Model (encoder, RSSM, decoder, reward, cont heads)
                # Only updates WM params. Actor/critic not in the compute graph.
                # ===============================================================
                def _wm_loss_fn(params, slow_params, rng):
                    model = nnx.merge(graphdef, params)
                    slow_model = nnx.merge(graphdef, slow_params)
                    rng, rssm_rng = jax.random.split(rng)
                    embed, post_stoch, post_deter, post_logits = _posterior_rollout(
                        model, rssm_rng
                    )
                    prior_logits = jax.vmap(jax.vmap(model.prior))(post_deter)

                    dyn_kl = jnp.clip(
                        categorical_kl(
                            jax.lax.stop_gradient(post_logits), prior_logits
                        ).sum(-1),
                        a_min=config.kl_free,
                    )
                    rep_kl = jnp.clip(
                        categorical_kl(
                            post_logits, jax.lax.stop_gradient(prior_logits)
                        ).sum(-1),
                        a_min=config.kl_free,
                    )

                    feat_flat = model.get_feat(post_stoch, post_deter).reshape(BT, -1)
                    decoded = model.decode(feat_flat).reshape(B, T, -1)
                    recon_loss = jnp.mean((decoded - symlog(sampled_obs)) ** 2)
                    reward_logits = model.reward(feat_flat).reshape(B, T, -1)
                    rew_loss = -jnp.mean(
                        twohot_log_prob(reward_logits, sampled_rewards[..., None], bins)
                    )
                    cont_logits = model.cont(feat_flat).reshape(B, T)
                    cont_target = 1.0 - sampled_dones.astype(jnp.float32)
                    cont_loss = -jnp.mean(
                        cont_target * jax.nn.log_sigmoid(cont_logits)
                        + (1 - cont_target) * jax.nn.log_sigmoid(-cont_logits)
                    )

                    wm_loss = (
                        config.kl_dyn_scale * jnp.mean(dyn_kl)
                        + config.kl_rep_scale * jnp.mean(rep_kl)
                        + recon_loss
                        + rew_loss
                        + cont_loss
                    )

                    wm_metrics = {
                        "dyn_loss": jnp.mean(dyn_kl),
                        "rep_loss": jnp.mean(rep_kl),
                        "recon_loss": recon_loss,
                        "rew_loss": rew_loss,
                        "cont_loss": cont_loss,
                    }
                    return wm_loss, (post_stoch, post_deter, wm_metrics)

                # ===============================================================
                # LOSS 2: Actor (pathwise gradients through frozen WM + critic)
                # Only updates actor params. WM/critic params are stop_gradient'd.
                # ===============================================================
                def _actor_loss_fn(params, slow_params, ret_ema, rng):
                    # Freeze WM and critic params so actor loss can't update them.
                    # stop_gradient must be applied INSIDE the grad function.
                    frozen = jax.lax.stop_gradient(params)
                    merged_dict = {}
                    for k in params.keys():
                        if k in ACTOR_KEYS:
                            merged_dict[k] = params[k]  # actor: live gradients
                        else:
                            merged_dict[k] = frozen[k]  # WM/critic: frozen
                    model = nnx.merge(graphdef, nnx.State(merged_dict))

                    rng, rssm_rng, imag_rng = jax.random.split(rng, 3)
                    _, post_stoch, post_deter, _ = _posterior_rollout(model, rssm_rng)
                    imag_feat, imag_actions = _imagine(
                        model, post_stoch, post_deter, imag_rng
                    )
                    imag_steps = imag_horizon + 1

                    # Reward/cont with gradients flowing to actor through dynamics
                    imag_feat_flat = imag_feat.reshape(-1, imag_feat.shape[-1])
                    if use_model_grads:
                        imag_reward = twohot_decode(
                            model.reward(imag_feat_flat).reshape(
                                BT, imag_steps, num_bins
                            ),
                            bins,
                        )
                        imag_cont = jax.nn.sigmoid(
                            model.cont(imag_feat_flat).reshape(BT, imag_steps, 1)
                        )
                        # Value WITH gradients: actor can see beyond the horizon
                        imag_value = twohot_decode(
                            model.critic(imag_feat_flat).reshape(
                                BT, imag_steps, num_bins
                            ),
                            bins,
                        )
                    else:
                        imag_feat_flat_sg = jax.lax.stop_gradient(imag_feat_flat)
                        imag_reward = twohot_decode(
                            model.reward(imag_feat_flat_sg).reshape(
                                BT, imag_steps, num_bins
                            ),
                            bins,
                        )
                        imag_cont = jax.nn.sigmoid(
                            model.cont(imag_feat_flat_sg).reshape(BT, imag_steps, 1)
                        )
                        imag_value = twohot_decode(
                            model.critic(imag_feat_flat_sg).reshape(
                                BT, imag_steps, num_bins
                            ),
                            bins,
                        )

                    weight = jnp.cumprod(
                        jax.lax.stop_gradient(imag_cont) * discount, axis=1
                    )

                    if use_model_grads:
                        # Lambda returns with active value gradients for actor
                        imag_returns = lambda_return(
                            jnp.zeros_like(imag_cont),
                            1.0 - imag_cont,
                            imag_reward,
                            imag_value,
                            imag_value,
                            discount,
                            config.gae_lambda,
                        )
                    else:
                        imag_returns = lambda_return(
                            jnp.zeros_like(imag_cont),
                            1.0 - imag_cont,
                            imag_reward,
                            jax.lax.stop_gradient(imag_value),
                            jax.lax.stop_gradient(imag_value),
                            discount,
                            config.gae_lambda,
                        )

                    new_ret_ema = update_return_ema(
                        ret_ema, jax.lax.stop_gradient(imag_returns)
                    )
                    scale = return_scale(new_ret_ema)

                    # Entropy from stopped features (closed-form gradient)
                    imag_feat_sg = jax.lax.stop_gradient(imag_feat)
                    imag_mean, imag_std = model.actor(imag_feat_sg)
                    pi = distrax.Independent(
                        distrax.Normal(imag_mean, imag_std), reinterpreted_batch_ndims=1
                    )
                    entropy = pi.entropy()[:, :-1, None]

                    if use_model_grads:
                        policy_loss = -jnp.mean(
                            jax.lax.stop_gradient(weight[:, :-1]) * imag_returns / scale
                        ) - config.entropy_coeff * jnp.mean(entropy)
                    else:
                        advantages = (
                            jax.lax.stop_gradient(imag_returns)
                            - jax.lax.stop_gradient(imag_value[:, :-1])
                        ) / scale
                        imag_actions_sg = jax.lax.stop_gradient(imag_actions)
                        logpi = pi.log_prob(imag_actions_sg)[:, :-1, None]
                        policy_loss = jnp.mean(
                            jax.lax.stop_gradient(weight[:, :-1])
                            * -(logpi * advantages + config.entropy_coeff * entropy)
                        )

                    actor_metrics = {
                        "policy_loss": policy_loss,
                        "imag_reward": jnp.mean(jax.lax.stop_gradient(imag_reward)),
                        "imag_value": jnp.mean(jax.lax.stop_gradient(imag_value)),
                        "entropy": jnp.mean(entropy),
                    }
                    return policy_loss, (
                        actor_metrics,
                        new_ret_ema,
                        jax.lax.stop_gradient(imag_feat),
                        jax.lax.stop_gradient(imag_returns),
                        jax.lax.stop_gradient(imag_cont),
                    )

                # ===============================================================
                # LOSS 3: Critic (TwoHot regression on stopped features/returns)
                # Only updates critic params.
                # ===============================================================
                def _critic_loss_fn(
                    params,
                    slow_params,
                    imag_feat_sg,
                    imag_returns_sg,
                    imag_cont_sg,
                    post_stoch_sg,
                    post_deter_sg,
                    sampled_rewards_sg,
                    sampled_dones_sg,
                ):
                    # Freeze everything except critic params
                    frozen = jax.lax.stop_gradient(params)
                    merged_dict = {}
                    for k in params.keys():
                        if k in CRITIC_KEYS:
                            merged_dict[k] = params[k]
                        else:
                            merged_dict[k] = frozen[k]
                    model = nnx.merge(graphdef, nnx.State(merged_dict))
                    slow_model = nnx.merge(graphdef, slow_params)
                    imag_steps = imag_horizon + 1

                    # --- Prepare features for a single batched forward pass ---
                    imag_feat_flat_sg = imag_feat_sg.reshape(-1, imag_feat_sg.shape[-1])
                    repval_feat_flat = model.get_feat(
                        post_stoch_sg, post_deter_sg
                    ).reshape(BT, -1)
                    combined_feats = jnp.concatenate(
                        [imag_feat_flat_sg, repval_feat_flat], axis=0
                    )

                    # Single forward pass through critic and slow critic
                    combined_logits = model.critic(combined_feats)
                    combined_slow_logits = slow_model.critic(combined_feats)

                    # Split back into imagination and repval components
                    n_imag = imag_feat_flat_sg.shape[0]
                    imag_logits, repval_logits_for_targets = jnp.split(
                        combined_logits, [n_imag], axis=0
                    )
                    imag_slow_logits, repval_slow_logits = jnp.split(
                        combined_slow_logits, [n_imag], axis=0
                    )

                    # --- Imagination value loss ---
                    weight = jnp.cumprod(imag_cont_sg * discount, axis=1)
                    value_logits = imag_logits.reshape(BT, imag_steps, num_bins)
                    imag_slow_value = twohot_decode(
                        imag_slow_logits.reshape(BT, imag_steps, num_bins), bins
                    )
                    returns_padded = jnp.concatenate(
                        [imag_returns_sg, jnp.zeros_like(imag_returns_sg[:, -1:])], 1
                    )
                    value_loss_returns = -twohot_log_prob(
                        value_logits, returns_padded, bins
                    )[:, :-1]
                    value_loss_slow = -twohot_log_prob(
                        value_logits, jax.lax.stop_gradient(imag_slow_value), bins
                    )[:, :-1]
                    imag_value_loss = jnp.mean(
                        jax.lax.stop_gradient(weight[:, :-1, 0])
                        * (value_loss_returns + value_loss_slow)
                    )

                    # --- Repval: critic loss on real replay data ---
                    repval_value = twohot_decode(
                        jax.lax.stop_gradient(repval_logits_for_targets).reshape(
                            B, T, num_bins
                        ),
                        bins,
                    )
                    repval_slow_value = twohot_decode(
                        repval_slow_logits.reshape(B, T, num_bins), bins
                    )
                    repval_is_last = sampled_dones_sg.astype(jnp.float32)[..., None]
                    repval_bootstrap = jnp.zeros((B, T, 1))
                    repval_returns = lambda_return(
                        repval_is_last,
                        repval_is_last,
                        sampled_rewards_sg[..., None],
                        repval_value,
                        repval_bootstrap,
                        discount,
                        config.gae_lambda,
                    )
                    repval_returns_padded = jnp.concatenate(
                        [repval_returns, jnp.zeros_like(repval_returns[:, -1:])], 1
                    )
                    repval_weight = (1.0 - repval_is_last)[:, :-1]
                    repval_value_logits = repval_logits_for_targets.reshape(
                        B, T, num_bins
                    )
                    repval_loss = jnp.mean(
                        repval_weight[..., 0]
                        * (
                            -twohot_log_prob(
                                repval_value_logits,
                                jax.lax.stop_gradient(repval_returns_padded),
                                bins,
                            )[:, :-1]
                            - twohot_log_prob(
                                repval_value_logits,
                                jax.lax.stop_gradient(repval_slow_value[..., 0:1]),
                                bins,
                            )[:, :-1]
                        )
                    )

                    total_critic_loss = (
                        imag_value_loss + config.repval_scale * repval_loss
                    )
                    return total_critic_loss, {
                        "value_loss": imag_value_loss,
                        "repval_loss": repval_loss,
                    }

                # ===============================================================
                # Compute gradients for each component separately
                # ===============================================================
                rng, wm_rng, actor_rng = jax.random.split(train_rng, 3)

                # 1. World model
                (_, (post_stoch, post_deter, wm_metrics)), wm_grads = (
                    jax.value_and_grad(_wm_loss_fn, has_aux=True)(
                        params, slow_params, wm_rng
                    )
                )

                # 2. Actor
                (
                    (
                        _,
                        (
                            actor_metrics,
                            ret_ema,
                            imag_feat_sg,
                            imag_returns_sg,
                            imag_cont_sg,
                        ),
                    ),
                    actor_grads,
                ) = jax.value_and_grad(_actor_loss_fn, has_aux=True)(
                    params, slow_params, ret_ema, actor_rng
                )

                # 3. Critic (uses stopped outputs from actor + WM steps)
                post_stoch_sg = jax.lax.stop_gradient(post_stoch)
                post_deter_sg = jax.lax.stop_gradient(post_deter)
                (critic_loss, critic_metrics), critic_grads = jax.value_and_grad(
                    _critic_loss_fn, has_aux=True
                )(
                    params,
                    slow_params,
                    imag_feat_sg,
                    imag_returns_sg,
                    imag_cont_sg,
                    post_stoch_sg,
                    post_deter_sg,
                    jax.lax.stop_gradient(sampled_rewards),
                    jax.lax.stop_gradient(sampled_dones),
                )

                # Combine: each gradient set only has non-zero values for its
                # component (WM grads zero for actor/critic, actor grads zero for
                # WM/critic due to stop_gradient, critic grads zero for WM/actor
                # due to stopped features)
                grads = jax.tree_util.tree_map(
                    lambda w, a, c: w + a + c, wm_grads, actor_grads, critic_grads
                )

                updates, opt_state = tx.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)

                # Slow target EMA
                slow_params = jax.tree_util.tree_map(
                    lambda s, v: config.slow_target_frac * v
                    + (1 - config.slow_target_frac) * s,
                    slow_params,
                    params,
                )

                epoch_metrics = {
                    **wm_metrics,
                    **actor_metrics,
                    **critic_metrics,
                    "total_loss": (
                        wm_metrics["dyn_loss"]
                        + wm_metrics["rep_loss"]
                        + wm_metrics["recon_loss"]
                        + wm_metrics["rew_loss"]
                        + wm_metrics["cont_loss"]
                        + actor_metrics["policy_loss"]
                        + critic_loss
                    ),
                }
                train_carry = (params, slow_params, opt_state, ret_ema, rng)
                return train_carry, epoch_metrics

            # Run num_epochs gradient steps
            train_carry = (params, slow_params, opt_state, ret_ema, rng)
            train_carry, epoch_metrics = jax.lax.scan(
                _train_epoch, train_carry, None, num_epochs
            )
            params, slow_params, opt_state, ret_ema, rng = train_carry

            # Use metrics from last epoch for logging
            loss_metrics = jax.tree_util.tree_map(lambda x: x[-1], epoch_metrics)

            metrics = {
                "episode_return": traj.info["returned_episode_returns"].mean(),
                "episode_length": traj.info["returned_episode_lengths"].mean(),
                **loss_metrics,
            }
            carry = (
                params,
                slow_params,
                opt_state,
                env_state,
                obs,
                rssm_stoch,
                rssm_deter,
                prev_action,
                buf,
                buf_ptr,
                buf_count,
                ret_ema,
                rng,
            )
            return carry, metrics

        rng, loop_rng = jax.random.split(rng)
        init_carry = (
            params,
            slow_params,
            opt_state,
            env_state,
            obs,
            rssm_stoch,
            rssm_deter,
            prev_action,
            buf,
            buf_ptr,
            buf_count,
            ret_ema,
            loop_rng,
        )
        final_carry, metrics = jax.lax.scan(_update_step, init_carry, None, num_updates)
        return final_carry, metrics

    return train


if __name__ == "__main__":
    SEED = 0
    NUM_RUNS = 5

    brax_env = EcoAntV2(battery_limit=500.0)
    env = BraxToGymnaxWrapper(env=brax_env, episode_length=1000)
    env_params = [env.default_params] * NUM_RUNS

    rng = jax.random.PRNGKey(SEED)
    rngs = jax.random.split(rng, NUM_RUNS)

    dynamic_config = DynamicConfig(
        rng=rngs,
        env_params=jax.tree.map(lambda *xs: jnp.stack(xs), *env_params),
        lr=jnp.ones(NUM_RUNS) * 4e-5,
        kl_free=jnp.ones(NUM_RUNS) * 1.0,
        kl_dyn_scale=jnp.ones(NUM_RUNS) * 1.0,
        kl_rep_scale=jnp.ones(NUM_RUNS) * 0.1,
        horizon=jnp.ones(NUM_RUNS) * 333,
        gae_lambda=jnp.ones(NUM_RUNS) * 0.95,
        entropy_coeff=jnp.ones(NUM_RUNS) * 3e-4,
        slow_target_frac=jnp.ones(NUM_RUNS) * 0.02,
        repval_scale=jnp.ones(NUM_RUNS) * 0.3,
        warmup_steps=jnp.ones(NUM_RUNS) * 1000,
    )

    # num_envs=25, train_freq=100 → 2500 env steps per collection.
    # num_epochs=50 → 50 grad steps per collection.
    # Train ratio = 50 / 2500 = 0.02 (vs old 0.0004).
    # Total grad steps = (1M / 100) * 50 = 500k.
    train_fn = make_train(
        env,
        num_steps=int(500_000),
        num_envs=25,
        train_freq=100,
        num_epochs=50,
        use_model_grads=True,
    )
    train_vjit = jax.jit(jax.vmap(train_fn))
    start_time = time.perf_counter()
    runner_states, all_metrics = jax.block_until_ready(train_vjit(dynamic_config))
    runtime = time.perf_counter() - start_time
    print(f"Runtime: {runtime:.2f}s")

    log.save_local(
        algo_name="dreamerV3",
        env_name=brax_env.name,
        metrics=all_metrics,
    )

    log.save_wandb(
        project="EcoAnt",
        algo_name="dreamerV3",
        env_name=brax_env.name,
        metrics=all_metrics,
    )
