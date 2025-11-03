from typing import Dict, Tuple
from enum import Enum
from gymnax.environments.environment import Environment, EnvParams, EnvState
from gymnax.environments import spaces
from flax import struct
import chex
import jax
from jax import numpy as jnp
from matplotlib import pyplot as plt


class BinanceFeeTier(Enum):
    REGULAR = 0.001
    VIP_1 = 0.001
    VIP_2 = 0.001
    VIP_3 = 0.0006
    VIP_4 = 0.00052
    VIP_5 = 0.00031
    VIP_6 = 0.00029
    VIP_7 = 0.00028
    VIP_8 = 0.00025
    VIP_9 = 0.00023


@struct.dataclass
class GARCHParams:
    omega: float  # Constant term in variance equation
    alpha: chex.Array  # ARCH coefficients (length q)
    beta: chex.Array  # GARCH coefficients (length p)
    mu: float  # Mean return
    initial_price: float  # Starting price for the asset


@struct.dataclass
class VecGARCHParams:
    omega: chex.Array  # Constant term in variance equation
    alpha: chex.Array  # ARCH coefficients (length q)
    beta: chex.Array  # GARCH coefficients (length p)
    mu: chex.Array  # Mean return
    initial_price: chex.Array  # Starting price for the asset


@struct.dataclass
class EnvState:
    step: int
    time: int
    prices: chex.Array  # Current prices for all assets
    returns: chex.Array  # Current returns for all assets
    volatilities: chex.Array  # Current volatilities for all assets
    holdings: chex.Array  # Current holdings
    values: chex.Array  # Current values
    total_value: float


@struct.dataclass
class EnvParams:
    max_steps: int
    initial_cash: float
    taker_fee: float
    gas_fee: float
    trade_threshold: float
    garch_params: Dict[str, GARCHParams]  # GARCH params for each asset


@jax.jit
def _sample_garch(carry, x):
    """
    JIT-compiled step function for jax.lax.scan.

    This function is vectorized to process all assets in parallel for a single time step.

    Args:
        carry: A tuple containing (params, garch_state)
            - params: A GARCHParams pytree where each field is a (num_assets, ...) array.
            - garch_state: A tuple (last_vols, last_returns, last_price)
                - last_vols: (num_assets, p) array of past volatilities
                - last_returns: (num_assets, q) array of past returns
                - last_price: (num_assets,) array of prices from t-1
        x: (num_assets,) array of standard normal random shocks for time t.

    Returns:
        new_carry: The updated carry for time t+1
        y_t: A tuple (new_return, new_vol, new_price) of outputs to stack
            - new_return: (num_assets,) array of returns for time t
            - new_vol: (num_assets,) array of volatilities for time t
            - new_price: (num_assets,) array of prices for time t
    """
    # Unpack carry
    params: VecGARCHParams = carry[0]
    garch_state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] = carry[1]
    last_vols, last_returns, last_price = garch_state
    noise = x  # (num_assets,)

    # GARCH(p,q) equations (all vectorized)
    # sigma_t^2 = omega + sum(alpha * residuals^2) + sum(beta * vols^2)
    residuals = last_returns - params.mu[:, None]  # (num_assets, q)
    arch_term = (params.alpha * residuals**2).sum(axis=-1)
    garch_term = (params.beta * last_vols**2).sum(axis=-1)
    variance = params.omega + arch_term + garch_term
    new_vol = jnp.sqrt(jnp.maximum(variance, 1e-8))  # Ensure positive variance

    # r_t = mu + sigma_t * epsilon_t
    new_return = params.mu + new_vol * noise
    new_price = last_price * jnp.exp(new_return)

    # Update GARCH state
    y = (new_return, new_vol, new_price)
    new_last_vols = jnp.roll(last_vols, shift=1, axis=-1)
    new_last_vols = new_last_vols.at[:, 0].set(new_vol)
    new_last_returns = jnp.roll(last_returns, shift=1, axis=-1)
    new_last_returns = new_last_returns.at[:, 0].set(new_return)
    new_carry = (params, (new_last_vols, new_last_returns, new_price))
    return new_carry, y


class PortfolioOptimizationGARCH(Environment):
    """Portfolio optimization environment with GARCH-simulated asset returns."""

    def __init__(
        self,
        rng: chex.PRNGKey,
        garch_params: Dict[str, GARCHParams],
        step_size: int = 1,
        total_samples: int = 10_000_000,
    ):
        """
        Initialize GARCH portfolio environment.

        Args:
            rng: Random key for generating GARCH paths
            garch_params: Dict mapping asset names to GARCHParams
            step_size: Step size for sampling (if subsampling the data)
            total_samples: Total number of time steps to generate
        """
        super().__init__()
        self.asset_names = sorted(garch_params.keys())
        self.num_assets = len(self.asset_names)
        self.step_size = step_size
        self.total_samples = total_samples

        # Store individual GARCH params for default_params property
        self._garch_params = {name: garch_params[name] for name in self.asset_names}

        # Stack GARCHParams into vectorized arrays for parallel processing
        # Each field becomes (num_assets, ...) shaped
        max_p = max(len(gp.beta) for gp in garch_params.values())
        max_q = max(len(gp.alpha) for gp in garch_params.values())

        # Pad alpha and beta to same length for vectorization
        omega_vec = jnp.array(
            [garch_params[name].omega for name in self.asset_names], dtype=jnp.float32
        )
        mu_vec = jnp.array(
            [garch_params[name].mu for name in self.asset_names], dtype=jnp.float32
        )
        initial_price_vec = jnp.array(
            [garch_params[name].initial_price for name in self.asset_names],
            dtype=jnp.float32,
        )

        alpha_list = []
        beta_list = []
        for name in self.asset_names:
            gp = garch_params[name]
            # Pad with zeros to max length
            alpha_padded = jnp.pad(gp.alpha, (0, max_q - len(gp.alpha)))
            beta_padded = jnp.pad(gp.beta, (0, max_p - len(gp.beta)))
            alpha_list.append(alpha_padded)
            beta_list.append(beta_padded)

        alpha_vec = jnp.stack(alpha_list, axis=0)  # (num_assets, max_q)
        beta_vec = jnp.stack(beta_list, axis=0)  # (num_assets, max_p)

        stacked_params = VecGARCHParams(
            omega=omega_vec,
            alpha=alpha_vec,
            beta=beta_vec,
            mu=mu_vec,
            initial_price=initial_price_vec,
        )

        # Initialize GARCH state with unconditional variance
        alpha_sum = alpha_vec.sum(axis=-1)  # (num_assets,)
        beta_sum = beta_vec.sum(axis=-1)  # (num_assets,)
        denominator = 1 - alpha_sum - beta_sum
        uncond_var = jnp.where(denominator < 1e-6, omega_vec, omega_vec / denominator)
        uncond_vol = jnp.sqrt(jnp.maximum(uncond_var, 1e-8))

        # Initialize state with burn-in using unconditional moments
        initial_vols = jnp.tile(uncond_vol[:, None], (1, max_p))  # (num_assets, p)
        initial_returns = jnp.tile(mu_vec[:, None], (1, max_q))  # (num_assets, q)
        initial_prices = initial_price_vec  # (num_assets,)

        initial_state = (initial_vols, initial_returns, initial_prices)

        # Generate noise for all time steps: (total_samples, num_assets)
        rng, noise_rng = jax.random.split(rng)
        noise = jax.random.normal(noise_rng, (total_samples, self.num_assets))

        # Run GARCH simulation
        _, outputs = jax.lax.scan(_sample_garch, (stacked_params, initial_state), noise)

        # Unpack outputs: each is (total_samples, num_assets)
        self.returns, self.volatilities, self.prices = outputs

    @property
    def name(self) -> str:
        return "PortfolioOptimizationGARCH"

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(
            max_steps=10000,
            initial_cash=1000.0,
            taker_fee=BinanceFeeTier.REGULAR.value,
            gas_fee=0.0,
            trade_threshold=1.0,
            garch_params=self._garch_params,
        )

    def action_space(self, params: EnvParams) -> spaces.Box:
        """Action space: portfolio weights (including cash)."""
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self.num_assets + 1,),  # +1 for cash
            dtype=jnp.float32,
        )

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation: recent returns and volatilities for all assets."""
        # Features: returns and volatilities for each asset over lookback window
        obs_shape = (self.step_size * self.prices.shape[1] * 2,)
        return spaces.Box(
            low=-jnp.inf, high=jnp.inf, shape=obs_shape, dtype=jnp.float32
        )

    def get_obs(self, state: EnvState, params: EnvParams) -> chex.Array:
        """Get observation from current state."""
        # Extract recent returns and volatilities from pre-generated path
        start_time_idx = jnp.maximum(0, state.time - self.step_size + 1)
        start_indices = (start_time_idx, 0)
        slice_sizes = (self.step_size, self.num_assets)
        returns_window = jax.lax.dynamic_slice(
            self.returns,
            start_indices,
            slice_sizes,
        )
        vols_window = jax.lax.dynamic_slice(
            self.volatilities,
            start_indices,
            slice_sizes,
        )

        obs = jnp.concatenate([returns_window.flatten(), vols_window.flatten()])
        return obs

    def reward(
        self, state: EnvState, next_state: EnvState, params: EnvParams
    ) -> chex.Array:
        """Log return of portfolio value."""
        log_return = jnp.log(next_state.total_value) - jnp.log(state.total_value)
        return log_return

    def is_terminal(self, state: EnvState, params: EnvParams) -> chex.Array:
        """Check if episode is done."""
        max_steps_reached = state.step >= params.max_steps
        portfolio_bankrupt = state.total_value <= 0
        return jnp.logical_or(max_steps_reached, portfolio_bankrupt)

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: chex.Array, params: EnvParams
    ) -> tuple[chex.Array, EnvState, chex.Array, chex.Array, dict]:
        """Execute one environment step with pre-generated GARCH prices."""
        time = state.time + self.step_size
        prices = jnp.concatenate([jnp.array([1.0]), self.prices[time, :]])
        returns = jnp.concatenate([jnp.array([0.0]), self.returns[time, :]])
        volatilities = jnp.concatenate([jnp.array([0.0]), self.volatilities[time, :]])

        # Normalize action to portfolio weights
        weights = jax.nn.softmax(action)

        ############### UPDATE PORTFOLIO WITH FEES ###############
        values = state.holdings * prices
        total_value = jnp.sum(values)
        asset_values = values[1:]
        asset_weights = weights[1:]
        new_asset_values_no_fee = total_value * asset_weights
        deltas_no_fee = new_asset_values_no_fee - asset_values
        num_trades = jnp.sum(jnp.abs(deltas_no_fee) > params.trade_threshold)
        gas_cost = params.gas_fee * num_trades

        # Split buy and sell orders
        buy_indices = deltas_no_fee > params.trade_threshold
        sell_indices = deltas_no_fee < -params.trade_threshold
        no_trade_indices = jnp.concatenate(
            [jnp.array([False]), jnp.abs(deltas_no_fee) <= params.trade_threshold]
        )
        buy_weights = jnp.where(buy_indices, asset_weights, 0.0)
        sell_weights = jnp.where(sell_indices, asset_weights, 0.0)
        current_buy_values = jnp.where(buy_indices, asset_values, 0.0)
        current_sell_values = jnp.where(sell_indices, asset_values, 0.0)

        # Calculate new portfolio value after fees
        fee_param = params.taker_fee / (1 - params.taker_fee)
        numerator = (
            total_value
            - gas_cost
            + fee_param * (jnp.sum(current_buy_values) - jnp.sum(current_sell_values))
        )
        denominator = 1 + fee_param * (jnp.sum(buy_weights) - jnp.sum(sell_weights))
        new_total_value = numerator / denominator
        new_values = new_total_value * weights
        adj_new_values = jnp.where(no_trade_indices, values, new_values)
        delta_values = new_values - adj_new_values
        delta_cash = jnp.sum(delta_values)
        adj_new_values = adj_new_values.at[0].add(delta_cash)
        new_holdings = adj_new_values / prices

        next_state = EnvState(
            step=state.step + 1,
            time=time,
            prices=prices,
            returns=returns,
            volatilities=volatilities,
            holdings=new_holdings,
            values=adj_new_values,
            total_value=new_total_value,
        )

        obs = self.get_obs(next_state, params)
        reward = self.reward(state, next_state, params)
        done = self.is_terminal(next_state, params)
        info = {}
        return obs, next_state, reward, done, info

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment and pre-generate GARCH paths."""
        episode_length = params.max_steps * self.step_size
        max_start = self.prices.shape[0] - episode_length
        min_start = self.step_size
        time = jax.random.randint(key, (), min_start, max_start)
        prices = jnp.concatenate([jnp.array([1.0]), self.prices[time, :]])
        returns = jnp.concatenate([jnp.array([0.0]), self.returns[time, :]])
        volatilities = jnp.concatenate([jnp.array([0.0]), self.volatilities[time, :]])
        holdings = jnp.zeros(self.num_assets + 1)
        holdings = holdings.at[0].set(params.initial_cash)
        values = holdings * prices
        state = EnvState(
            step=0,
            time=time,
            prices=prices,
            returns=returns,
            volatilities=volatilities,
            holdings=holdings,
            values=values,
            total_value=jnp.sum(values),
        )
        obs = self.get_obs(state, params)
        return obs, state

    def plot_garch(self):
        """Plot the generated GARCH price paths, returns, and volatilities for all assets."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # Plot prices
        for i, name in enumerate(self.asset_names):
            axes[0].plot(self.prices[:, i], label=f"{name}")
        axes[0].set_title("GARCH-Simulated Asset Prices")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Price")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot returns
        for i, name in enumerate(self.asset_names):
            axes[1].plot(self.returns[:, i], label=f"{name}", alpha=0.7)
        axes[1].set_title("GARCH Returns")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Return")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color="k", linestyle="--", linewidth=0.5)

        # Plot volatilities
        for i, name in enumerate(self.asset_names):
            axes[2].plot(self.volatilities[:, i], label=f"{name}")
        axes[2].set_title("GARCH Conditional Volatilities")
        axes[2].set_xlabel("Time")
        axes[2].set_ylabel("Volatility")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    rng = jax.random.PRNGKey(1)
    garch_params = {
        "BTC": GARCHParams(
            mu=0,
            omega=0.0000001110,
            alpha=jnp.array([0.165]),
            beta=jnp.array([0.8]),
            initial_price=66084.0,
        ),
        "ETH": GARCHParams(
            mu=0,
            omega=0.0000004817,
            alpha=jnp.array([0.15]),
            beta=jnp.array([0.8]),
            initial_price=2629.79,
        ),
    }
    env = PortfolioOptimizationGARCH(rng, garch_params)

    env.plot_garch()

    obs, state = env.reset(rng, env.default_params)
    action = jnp.array([0.999995, 0.000003, 0.0000002])  # Example action
    next_obs, next_state, reward, done, info = env.step_env(
        rng, state, action, env.default_params
    )
