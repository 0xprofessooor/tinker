from typing import Dict, Tuple
from enum import Enum
from gymnax.environments.environment import Environment, EnvParams, EnvState
from gymnax.environments import spaces
from flax import struct
import chex
import polars as pl
import jax
from jax import numpy as jnp


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


class KLineFeatures(Enum):
    CLOSE = 0
    OPEN = 1
    HIGH = 2
    LOW = 3
    VOLUME = 4
    TAKER_BUY_VOLUME = 5
    NUM_TRADES = 6


@struct.dataclass
class EnvState:
    step: int
    time: int
    prices: chex.Array
    portfolio: chex.Array
    portfolio_value: float


@struct.dataclass
class EnvParams:
    step_size: int
    max_steps: int
    initial_cash: float
    taker_fee: float
    gas_fee: float


class PortfolioOptimizationV0(Environment):
    def __init__(self, data_paths: Dict[str, str]):
        super().__init__()
        data_dict = {key: load_binance_klines(path) for key, path in data_paths.items()}
        self.assets = sorted(data_dict.keys())
        self.data = jnp.stack(
            [data_dict[asset] for asset in self.assets], axis=1
        )  # shape (num_rows, num_assets, num_features)

    @property
    def name(self) -> str:
        return "PortfolioOptimizationV0"

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(
            step_size=3600,
            max_steps=2160,
            initial_cash=1000.0,
            taker_fee=BinanceFeeTier.REGULAR.value,
            gas_fee=0.0,
        )

    def action_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(
            low=-jnp.inf, high=jnp.inf, shape=(len(self.data) + 1,), dtype=jnp.float32
        )

    def observation_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(
            low=-jnp.inf, high=jnp.inf, shape=(len(self.data),), dtype=jnp.float32
        )

    def get_obs(self, state: EnvState, params: EnvParams) -> chex.Array:
        step_data = self.data[
            state.time - (params.step_size - 1) : state.time + 1, :, :
        ]
        return step_data

    def reward(
        self, state: EnvState, next_state: EnvState, params: EnvParams
    ) -> chex.Array:
        pass

    def is_terminal(self, state: EnvState, params: EnvParams) -> chex.Array:
        return jax.lax.cond(
            state.step >= params.max_steps,
            lambda _: jnp.array(True),
            lambda _: jnp.array(False),
            None,
        )

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: chex.Array, params: EnvParams
    ) -> tuple[chex.Array, EnvState, chex.Array, chex.Array, dict]:
        time = state.time + params.step_size
        prices = jnp.concatenate(
            [jnp.array([1.0]), self.data[time, :, KLineFeatures.CLOSE.value]]
        )

        # nomralize action
        action = jax.nn.softmax(action)

        # update portfolio
        current_values = state.portfolio * prices
        target_values = action * state.portfolio_value
        delta_values = target_values - current_values
        asset_deltas = delta_values[1:]
        asset_sell_values = jnp.clip(-delta_values, a_min=0.0)
        asset_buy_values = jnp.clip(delta_values, a_min=0.0)
        total_sell_value = jnp.sum(asset_sell_values)
        total_buy_value = jnp.sum(asset_buy_values)

        sell_fees = total_sell_value * params.taker_fee
        buy_fees = total_buy_value * params.taker_fee

        net_cash_from_sells = total_sell_value - sell_fees
        net_cash_for_buys = total_buy_value + buy_fees

        new_asset_portfolio = jnp.where(
            asset_deltas < 0,
            state.portfolio.at[1:]
            + (asset_deltas / prices[1:]) * (1 - params.taker_fee),
            jnp.where(
                asset_deltas > 0,
                state.portfolio.at[1:]
                + (asset_deltas / prices[1:]) * (1 - params.taker_fee),
                state.portfolio.at[1:],
            ),
        )

        num_trades = jnp.sum(jnp.abs(asset_deltas) > 1e-6)
        total_gas_fee = num_trades * params.gas_fee

        new_cash = (
            state.portfolio[0] + net_cash_from_sells - net_cash_for_buys - total_gas_fee
        )
        new_portfolio = jnp.concatenate([jnp.array([new_cash]), new_asset_portfolio])

        portfolio_value = jnp.sum(new_portfolio * prices)
        next_state = EnvState(
            step=state.step + 1,
            time=time,
            prices=prices,
            portfolio=new_portfolio,
            portfolio_value=portfolio_value,
        )
        obs = self.get_obs(next_state, params)
        reward = self.reward(state, next_state, params)
        done = self.is_terminal(next_state, params)
        info = {}
        return obs, next_state, reward, done, info

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        episode_length = params.max_steps * params.step_size
        max_start = self.data.shape[0] - episode_length
        min_start = params.step_size
        time = jax.random.randint(key, (), min_start, max_start)
        prices = jnp.concatenate(
            [jnp.array([1.0]), self.data[time, :, KLineFeatures.CLOSE.value]]
        )
        portfolio = jnp.zeros(len(self.assets) + 1)
        portfolio = portfolio.at[0].set(params.initial_cash)
        state = EnvState(
            step=0,
            time=time,
            prices=prices,
            portfolio=portfolio,
            portfolio_value=params.initial_cash,
        )
        obs = self.get_obs(state, params)
        return obs, state


def load_binance_klines(filepath: str) -> chex.Array:
    df = pl.read_csv(filepath)
    data = df.select(
        (pl.col("close")),
        (pl.col("open")),
        (pl.col("high")),
        (pl.col("low")),
        (pl.col("quote_asset_volume")),
        (pl.col("taker_buy_quote_volume")),
        (pl.col("number_of_trades")),
    )
    data = data.to_jax()
    return data


if __name__ == "__main__":
    env = PortfolioOptimizationV0(
        data_paths={"BTC": "data/BTCUSDT_2024-10-15_2025-10-15_1s.csv"}
    )
    key = jax.random.PRNGKey(0)
    print(env.data.shape)
    env.reset(key, env.default_params)
