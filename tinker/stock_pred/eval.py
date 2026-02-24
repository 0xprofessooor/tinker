"""Evaluate StateModel's ability to capture GARCH non-stationary volatility.

Run with:  uv run -m tinker.thinker.evaluate

Metrics produced
----------------
1. Median-prediction RMSE  (τ = 0.50)
2. Volatility tracking     Pearson r and Spearman p between the predicted
                           90 %-interval width and the rolling realised vol
3. Quantile calibration    Empirical coverage at each τ level (ideal: coverage ≈ τ)
4. Mean calibration error  Mean absolute deviation from ideal coverage

Plots saved to results/thinker_garch_eval.png
"""

import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from flax import nnx
from safenax.wrappers import LogWrapper
from safenax.portfolio_optimization.po_garch import (
    PortfolioOptimizationGARCH,
    GARCHParams,
)
from tinker.stock_pred.run import StateModel, DynamicConfig, make_train

# ── Hyperparameters ───────────────────────────────────────────────────────────
TRAIN_SEED = 0
EVAL_SEED = 42  # Different seed → different GARCH trajectories for eval
NUM_TRAIN_STEPS = 1_000_000
EPISODE_LENGTH = 1_000
NUM_EVAL_EPISODES = 10
EMBEDDING_DIM = 64
RNN_HIDDEN_DIM = 128
ROLLING_WINDOW = 50  # Steps for rolling realised-volatility estimate
# 19 evenly-spaced quantile levels from 0.05 to 0.95
EVAL_TAUS = jnp.linspace(0.05, 0.95, 19)


# ── Correlation helpers (no scipy required) ───────────────────────────────────
def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    xc, yc = x - x.mean(), y - y.mean()
    denom = np.sqrt((xc**2).sum() * (yc**2).sum())
    return float((xc * yc).sum() / denom) if denom > 1e-8 else 0.0


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    def _rank(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a)
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(a)) + 1.0
        return r

    return _pearson(_rank(x), _rank(y))


# ── Environment factory ───────────────────────────────────────────────────────
def build_env(
    rng, episode_length: int = EPISODE_LENGTH, num_trajectories: int = 10_000
):
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
    env = PortfolioOptimizationGARCH(
        rng=rng,
        garch_params=garch_params,
        num_samples=episode_length,
        num_trajectories=num_trajectories,
    )
    env_params = env.default_params.replace(max_steps=episode_length)
    return env, env_params


# ── Training ──────────────────────────────────────────────────────────────────
def train_model(rng, env, env_params):
    """Train the StateModel and return (runner_states, metrics)."""
    train_fn = make_train(
        env=env,
        num_steps=NUM_TRAIN_STEPS,
        num_envs=1,
        train_freq=1000,
        buffer_size=5_000,
        batch_size=64,
        num_epochs=10,
        embedding_dim=EMBEDDING_DIM,
        rnn_hidden_dim=RNN_HIDDEN_DIM,
    )
    dynamic_config = DynamicConfig(
        rng=rng[None],
        env_params=jax.tree.map(lambda x: jnp.stack([x]), env_params),
        lr=jnp.ones(1) * 3e-4,
        adam_eps=jnp.ones(1) * 1e-12,
        max_grad_norm=jnp.ones(1) * 1.0,
        vol_loss_coeff=jnp.ones(1) * 0.5,
    )
    print("Training model …")
    t0 = time.perf_counter()
    runner_states, metrics = jax.block_until_ready(
        jax.jit(jax.vmap(train_fn))(dynamic_config)
    )
    print(f"  Done in {time.perf_counter() - t0:.1f}s")
    return runner_states, metrics


# ── Evaluation ────────────────────────────────────────────────────────────────
def _make_predict_fn(model_graphdef, model_state, obs_dim: int):
    """Return a JIT-compiled step function.

    Given the current GRU hidden state, observation, and action, advances the
    GRU and returns (h_next, pred_quantiles) where pred_quantiles has shape
    (obs_dim, num_taus) — one quantile per obs dimension per tau level.

    The trick: we replicate (h, obs, action) across the tau dimension so that
    a single model forward-pass produces all quantiles simultaneously.
    """
    num_taus = len(EVAL_TAUS)

    @jax.jit
    def predict(h, obs, action):
        m = nnx.merge(model_graphdef, model_state)

        # Replicate inputs across the tau batch dimension
        h_b = jnp.repeat(h[None], num_taus, axis=0)  # (num_taus, H)
        obs_b = jnp.repeat(obs[None], num_taus, axis=0)  # (num_taus, obs_dim)
        act_b = jnp.repeat(action[None], num_taus, axis=0)  # (num_taus, act_dim)
        # Each row of tau_b holds the same scalar τ repeated across obs dimensions
        tau_b = jnp.repeat(EVAL_TAUS[:, None], obs_dim, axis=1)  # (num_taus, obs_dim)

        h_nexts, quants, _ = m(h_b, obs_b, act_b, tau_b)  # quants: (num_taus, obs_dim)
        # All h_nexts are identical (τ doesn't affect the GRU); take the first
        return h_nexts[0], quants.T  # (H,), (obs_dim, num_taus)

    return predict


def run_evaluation(model_graphdef, model_state, eval_env, env_params, rng):
    """Roll out the model on held-out GARCH trajectories.

    Returns
    -------
    pred_q   : np.ndarray  (T_total, obs_dim, num_taus)
    next_obs : np.ndarray  (T_total, obs_dim)
    ep_idx   : np.ndarray  (T_total,)  which episode each step belongs to
    """
    env = LogWrapper(eval_env)
    obs_dim = env.observation_space(env_params).shape[0]
    predict = _make_predict_fn(model_graphdef, model_state, obs_dim)

    all_pred_q = []
    all_next_obs = []
    all_ep_idx = []

    for ep in range(NUM_EVAL_EPISODES):
        rng, rng_reset = jax.random.split(rng)
        obs, env_state = env.reset(rng_reset, env_params)
        h = jnp.zeros((RNN_HIDDEN_DIM,))

        ep_pred_q = []
        ep_next_obs = []

        for _ in range(EPISODE_LENGTH - 1):
            rng, rng_act, rng_step = jax.random.split(rng, 3)
            action = env.action_space(env_params).sample(rng_act)

            h_next, pred_q = predict(h, obs, action)  # (H,), (obs_dim, num_taus)

            next_obs, next_env_state, _, done, _ = env.step(
                rng_step, env_state, action, env_params
            )

            ep_pred_q.append(np.array(pred_q))
            ep_next_obs.append(np.array(next_obs))

            # Reset hidden state on episode termination
            h = jnp.where(done, jnp.zeros_like(h_next), h_next)
            obs, env_state = next_obs, next_env_state

            if bool(done):
                break

        T_ep = len(ep_pred_q)
        all_pred_q.append(np.stack(ep_pred_q))
        all_next_obs.append(np.stack(ep_next_obs))
        all_ep_idx.extend([ep] * T_ep)
        print(f"  Episode {ep + 1}/{NUM_EVAL_EPISODES}: {T_ep} steps")

    return (
        np.concatenate(all_pred_q, axis=0),  # (T_total, obs_dim, num_taus)
        np.concatenate(all_next_obs, axis=0),  # (T_total, obs_dim)
        np.array(all_ep_idx),  # (T_total,)
    )


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(pred_q: np.ndarray, next_obs: np.ndarray) -> dict:
    """Compute all evaluation metrics.

    Parameters
    ----------
    pred_q   : (T, obs_dim, num_taus)
    next_obs : (T, obs_dim)
    """
    T, obs_dim, num_taus = pred_q.shape
    taus_np = np.array(EVAL_TAUS)

    i05 = int(np.argmin(np.abs(taus_np - 0.05)))
    i25 = int(np.argmin(np.abs(taus_np - 0.25)))
    i50 = int(np.argmin(np.abs(taus_np - 0.50)))
    i75 = int(np.argmin(np.abs(taus_np - 0.75)))
    i95 = int(np.argmin(np.abs(taus_np - 0.95)))

    # 1. Quantile coverage: fraction of actuals below the τ-th predicted quantile
    coverage = np.array(
        [
            (next_obs < pred_q[:, :, i]).mean(axis=0)  # (obs_dim,)
            for i in range(num_taus)
        ]
    ).T  # (obs_dim, num_taus)

    # 2. Predicted 90 %-interval width (proxy for predicted conditional vol)
    pred_spread = pred_q[:, :, i95] - pred_q[:, :, i05]  # (T, obs_dim)

    # 3. Rolling realised volatility (std of actual obs in a sliding window)
    realized_vol = np.zeros((T, obs_dim))
    for t in range(T):
        window = next_obs[max(0, t - ROLLING_WINDOW + 1) : t + 1]
        if len(window) > 1:
            realized_vol[t] = window.std(axis=0)

    # 4. Correlation between predicted spread and realised vol
    pearson_corr = np.array(
        [_pearson(pred_spread[:, d], realized_vol[:, d]) for d in range(obs_dim)]
    )
    spearman_corr = np.array(
        [_spearman(pred_spread[:, d], realized_vol[:, d]) for d in range(obs_dim)]
    )

    # 5. RMSE of median prediction (τ = 0.5)
    rmse = np.sqrt(((pred_q[:, :, i50] - next_obs) ** 2).mean(axis=0))  # (obs_dim,)

    # 6. Mean 90 %-interval width (sharpness — smaller is better given calibration)
    mean_width = pred_spread.mean(axis=0)  # (obs_dim,)

    # 7. Mean absolute calibration error
    calib_error = np.abs(coverage - taus_np[None, :]).mean(axis=1)  # (obs_dim,)

    return dict(
        coverage=coverage,
        pred_spread=pred_spread,
        realized_vol=realized_vol,
        pearson_corr=pearson_corr,
        spearman_corr=spearman_corr,
        rmse=rmse,
        mean_width=mean_width,
        calib_error=calib_error,
        i05=i05,
        i25=i25,
        i50=i50,
        i75=i75,
        i95=i95,
    )


# ── Report ────────────────────────────────────────────────────────────────────
def print_report(metrics: dict, obs_dim: int) -> None:
    taus_np = np.array(EVAL_TAUS)
    coverage = metrics["coverage"]

    sep = "=" * 68
    print(f"\n{sep}")
    print("EVALUATION REPORT: StateModel on GARCH Environment")
    print(sep)

    print("\n── 1. Median-Prediction RMSE (τ = 0.5) ──────────────────────────")
    for d in range(obs_dim):
        print(f"   dim {d}: {metrics['rmse'][d]:.6f}")

    print("\n── 2. Volatility Tracking ────────────────────────────────────────")
    print("   (Predicted 90 %-interval width vs rolling realised vol)")
    print(f"   {'Dim':>4}  {'Pearson r':>10}  {'Spearman ρ':>11}  {'Mean width':>11}")
    for d in range(obs_dim):
        print(
            f"   {d:>4}  {metrics['pearson_corr'][d]:>10.4f}"
            f"  {metrics['spearman_corr'][d]:>11.4f}"
            f"  {metrics['mean_width'][d]:>11.6f}"
        )

    print("\n── 3. Quantile Calibration (empirical coverage vs ideal τ) ──────")
    stride = max(1, len(taus_np) // 9)
    header = f"   {'τ':>5}  " + "  ".join(f"dim_{d}" for d in range(obs_dim))
    print(header)
    for i in range(0, len(taus_np), stride):
        tau = taus_np[i]
        vals = "  ".join(
            f"{coverage[d, i]:.3f}({coverage[d, i] - tau:+.3f})" for d in range(obs_dim)
        )
        print(f"   {tau:>5.2f}  {vals}")

    print("\n── 4. Mean Absolute Calibration Error (lower is better) ─────────")
    for d in range(obs_dim):
        print(f"   dim {d}: {metrics['calib_error'][d]:.4f}")

    print(f"\n{sep}\n")


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_results(
    pred_q: np.ndarray,
    next_obs: np.ndarray,
    ep_idx: np.ndarray,
    metrics: dict,
    obs_dim: int,
    save_path: str,
) -> None:
    taus_np = np.array(EVAL_TAUS)
    i05, i25, i50, i75, i95 = (
        metrics["i05"],
        metrics["i25"],
        metrics["i50"],
        metrics["i75"],
        metrics["i95"],
    )

    # Time-series plots use only episode 0 to keep things readable
    mask = ep_idx == 0
    T_ep = int(mask.sum())
    t = np.arange(T_ep)

    n_panels = obs_dim + 2  # per-dim prediction + vol tracking + calibration
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels))

    # ── Per-dim: predicted intervals vs actual ──
    for d in range(obs_dim):
        ax = axes[d]
        ax.fill_between(
            t,
            pred_q[mask, d, i05],
            pred_q[mask, d, i95],
            alpha=0.15,
            color="steelblue",
            label="90 % interval",
        )
        ax.fill_between(
            t,
            pred_q[mask, d, i25],
            pred_q[mask, d, i75],
            alpha=0.30,
            color="steelblue",
            label="50 % interval",
        )
        ax.plot(
            t,
            pred_q[mask, d, i50],
            color="steelblue",
            lw=1.2,
            label="Median pred (τ=0.5)",
        )
        ax.plot(t, next_obs[mask, d], "r.", ms=2, alpha=0.4, label="Actual next obs")
        ax.set_title(
            f"Obs dim {d}  —  Predicted quantile intervals vs Actual\n"
            f"RMSE(τ=0.5) = {metrics['rmse'][d]:.5f}   "
            f"Pearson r(vol) = {metrics['pearson_corr'][d]:.3f}"
        )
        ax.set_ylabel("Value")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(alpha=0.3)

    # ── Volatility-tracking panel ──
    ax = axes[obs_dim]
    for d in range(obs_dim):
        ps = metrics["pred_spread"][mask, d]
        rv = metrics["realized_vol"][mask, d]

        def _norm(x):
            s = x.std()
            return (x - x.mean()) / s if s > 1e-8 else x

        ax.plot(
            t,
            _norm(ps),
            lw=1.2,
            label=f"Pred spread dim {d}  (r={metrics['pearson_corr'][d]:.3f}, ρ={metrics['spearman_corr'][d]:.3f})",
        )
        ax.plot(t, _norm(rv), "--", lw=1.0, alpha=0.7, label=f"Realised vol dim {d}")

    ax.set_title(
        "Volatility Tracking: Predicted 90 %-interval Width vs Rolling Realised Vol\n"
        "(both normalised to zero-mean unit-variance for visual comparison)"
    )
    ax.set_xlabel("Time step")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── Calibration panel ──
    ax = axes[obs_dim + 1]
    ax.plot([0, 1], [0, 1], "k--", lw=2, label="Ideal calibration")
    for d in range(obs_dim):
        ax.plot(
            taus_np,
            metrics["coverage"][d],
            "o-",
            ms=4,
            label=f"dim {d}  (MAE={metrics['calib_error'][d]:.3f})",
        )
    ax.set_xlabel("τ (quantile level)")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Quantile Calibration Plot")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {save_path}")
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = jax.random.PRNGKey(TRAIN_SEED)
    rng_train, rng_env_train, rng_env_eval, rng_eval = jax.random.split(rng, 4)

    # ── Train ──
    train_env, env_params = build_env(rng_env_train)
    runner_states, train_metrics = train_model(rng_train, train_env, env_params)
    print(f"  Final loss:           {np.array(train_metrics['loss'][0, -1]):.6f}")
    print(
        f"  Final episode return: {np.array(train_metrics['episode_return'][0, -1]):.4f}"
    )

    # ── Reconstruct trained model ──
    # runner_states[1] is the batched model_state (first dim = num_seeds)
    trained_model_state = jax.tree.map(lambda x: x[0], runner_states[1])

    # Create a fresh model with the same architecture to obtain the graphdef.
    # nnx.merge(graphdef, state) uses path-based matching, so the graphdef from
    # any model with identical hyperparameters is compatible with the trained state.
    logged_env = LogWrapper(train_env)
    obs_dim = logged_env.observation_space(env_params).shape[0]
    action_dim = logged_env.action_space(env_params).shape[0]

    dummy_model = StateModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        embedding_dim=EMBEDDING_DIM,
        rnn_hidden_dim=RNN_HIDDEN_DIM,
        rngs=nnx.Rngs(rng_train),
    )
    model_graphdef, _ = nnx.split(dummy_model)

    # ── Evaluate on held-out trajectories ──
    eval_env, _ = build_env(rng_env_eval)  # Different RNG → different GARCH paths
    print(f"\nEvaluating on {NUM_EVAL_EPISODES} held-out episodes …")
    pred_q, next_obs, ep_idx = run_evaluation(
        model_graphdef, trained_model_state, eval_env, env_params, rng_eval
    )

    # ── Report ──
    metrics = compute_metrics(pred_q, next_obs)
    print_report(metrics, obs_dim)

    # ── Plot ──
    plot_results(
        pred_q,
        next_obs,
        ep_idx,
        metrics,
        obs_dim,
        save_path="results/thinker_garch_eval.png",
    )
