"""
Q-Micro :: data.synthetic_generator
--------------------------------------
Synthetic order-flow generator with heterogeneous agents and regime switching.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from core.order import Order, Side, OrderType
from core.exchange_simulator import ExchangeSimulator


class Regime(str, Enum):
    CALM = "CALM"
    VOLATILE = "VOLATILE"
    TRENDING = "TRENDING"


@dataclass
class RegimeParams:
    vol: float          # per-step return std
    arrival_rate: float  # avg orders per step
    drift: float = 0.0


REGIME_PARAMS = {
    Regime.CALM: RegimeParams(vol=0.0008, arrival_rate=3.0, drift=0.0),
    Regime.VOLATILE: RegimeParams(vol=0.0035, arrival_rate=6.0, drift=0.0),
    Regime.TRENDING: RegimeParams(vol=0.0015, arrival_rate=4.0, drift=0.0006),
}


class Agent:
    """Base class for a synthetic market participant."""

    def __init__(self, trader_id: str):
        self.trader_id = trader_id

    def act(self, ex: ExchangeSimulator, symbol: str, mid: float, rng: random.Random) -> List[Order]:
        raise NotImplementedError


class NoiseTrader(Agent):
    """Submits random small limit/market orders around the touch — no information."""

    def act(self, ex, symbol, mid, rng) -> List[Order]:
        side = rng.choice([Side.BUY, Side.SELL])
        qty = rng.randint(50, 300)
        if rng.random() < 0.3:
            return [Order(side=side, quantity=qty, order_type=OrderType.MARKET, trader_id=self.trader_id)]
        offset = rng.uniform(-0.05, 0.05)
        price = round(mid + offset if side == Side.BUY else mid - offset, 2)
        return [Order(side=side, price=price, quantity=qty, trader_id=self.trader_id)]


class InformedTrader(Agent):
    """Holds a noisy private signal about future fundamental value and trades on it."""

    def __init__(self, trader_id: str, signal_strength: float = 0.6):
        super().__init__(trader_id)
        self.signal_strength = signal_strength
        self._fair_value: Optional[float] = None

    def set_fair_value(self, fair_value: float) -> None:
        self._fair_value = fair_value

    def act(self, ex, symbol, mid, rng) -> List[Order]:
        if self._fair_value is None or rng.random() > self.signal_strength:
            return []
        side = Side.BUY if self._fair_value > mid else Side.SELL
        qty = rng.randint(200, 800)
        return [Order(side=side, quantity=qty, order_type=OrderType.MARKET, trader_id=self.trader_id)]


class MarketMakerAgent(Agent):
    """Quotes both sides symmetrically around mid, skewing on inventory."""

    def __init__(self, trader_id: str, half_spread: float = 0.05, quote_size: int = 400):
        super().__init__(trader_id)
        self.half_spread = half_spread
        self.quote_size = quote_size
        self.inventory = 0

    def act(self, ex, symbol, mid, rng) -> List[Order]:
        skew = -0.01 * self.inventory / max(self.quote_size, 1)
        bid_px = round(mid - self.half_spread + skew, 2)
        ask_px = round(mid + self.half_spread + skew, 2)
        return [
            Order(side=Side.BUY, price=bid_px, quantity=self.quote_size, trader_id=self.trader_id),
            Order(side=Side.SELL, price=ask_px, quantity=self.quote_size, trader_id=self.trader_id),
        ]

    def on_fill(self, side: Side, qty: float) -> None:
        self.inventory += qty if side == Side.BUY else -qty


class InstitutionalTrader(Agent):
    """Periodically executes a large parent order sliced over time (feeds execution algos)."""

    def __init__(self, trader_id: str, side: Side, total_qty: float, slices: int):
        super().__init__(trader_id)
        self.side = side
        self.remaining = total_qty
        self.slice_qty = total_qty / slices

    def act(self, ex, symbol, mid, rng) -> List[Order]:
        if self.remaining <= 0:
            return []
        qty = min(self.slice_qty, self.remaining)
        self.remaining -= qty
        return [Order(side=self.side, quantity=qty, order_type=OrderType.MARKET, trader_id=self.trader_id)]


class SyntheticMarketGenerator:
    """Drives a population of agents against an ExchangeSimulator, step by step."""

    def __init__(self, symbol: str = "SYNTH", start_price: float = 100.0, seed: int = 42):
        self.symbol = symbol
        self.ex = ExchangeSimulator(symbols=[symbol])
        self.rng = random.Random(seed)
        self.fair_value = start_price
        self.regime = Regime.CALM
        self.agents: List[Agent] = []
        self._seed_book(start_price)

    def _seed_book(self, start_price: float) -> None:
        for i in range(1, 6):
            self.ex.submit_order(self.symbol, Order(
                side=Side.BUY, price=round(start_price - 0.05 * i, 2), quantity=500, trader_id="seed"))
            self.ex.submit_order(self.symbol, Order(
                side=Side.SELL, price=round(start_price + 0.05 * i, 2), quantity=500, trader_id="seed"))

    def add_agent(self, agent: Agent) -> None:
        self.agents.append(agent)

    def set_regime(self, regime: Regime) -> None:
        self.regime = regime

    def step(self) -> dict:
        params = REGIME_PARAMS[self.regime]
        self.fair_value *= (1 + self.rng.gauss(params.drift, params.vol))
        for agent in self.agents:
            if isinstance(agent, InformedTrader):
                agent.set_fair_value(self.fair_value)

        mid = self.ex.market_data(self.symbol)["mid"] or self.fair_value
        n_active = self.rng.poisson(params.arrival_rate) if hasattr(self.rng, "poisson") else int(params.arrival_rate)
        for _ in range(max(1, n_active)):
            agent = self.rng.choice(self.agents)
            for order in agent.act(self.ex, self.symbol, mid, self.rng):
                trades = self.ex.submit_order(self.symbol, order)
                if isinstance(agent, MarketMakerAgent):
                    for t in trades:
                        filled_side = Side.BUY if t.buyer_id == agent.trader_id else Side.SELL
                        agent.on_fill(filled_side, t.quantity)

        return self.ex.market_data(self.symbol)

    def run(self, n_steps: int) -> List[dict]:
        return [self.step() for _ in range(n_steps)]