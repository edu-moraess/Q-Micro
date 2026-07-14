"""
Q-Micro :: simulation.execution_env
---------------------------------------
Gym-like environment: agent must liquidate/acquire a parent order over
a fixed horizon, minimizing PnL-adjusted execution cost.

State  = [inventory_frac, spread, volatility, depth_imbalance, ofi, time_frac]
Action = {0: WAIT, 1: PASSIVE_SLICE, 2: AGGRESSIVE_SLICE, 3: CANCEL_AND_WIDEN}
Reward = -(execution_cost_this_step) - risk_aversion * inventory_risk
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from core.order import Order, Side, OrderType
from data.synthetic_generator import SyntheticMarketGenerator, Regime


@dataclass
class ExecutionEnvConfig:
    target_qty: float = 5000.0
    horizon: int = 100
    side: Side = Side.SELL
    slice_frac: float = 0.05      # fraction of target_qty per PASSIVE_SLICE
    aggressive_mult: float = 2.0  # AGGRESSIVE_SLICE = slice_frac * aggressive_mult
    risk_aversion: float = 0.01
    start_price: float = 100.0
    seed: int = 0


class ExecutionEnv:
    ACTIONS = ["WAIT", "PASSIVE_SLICE", "AGGRESSIVE_SLICE", "CANCEL_AND_WIDEN"]

    def __init__(self, config: ExecutionEnvConfig):
        self.cfg = config
        self.reset()

    def reset(self) -> List[float]:
        self.gen = SyntheticMarketGenerator(start_price=self.cfg.start_price, seed=self.cfg.seed)
        self.gen.set_regime(Regime.VOLATILE)
        self.t = 0
        self.remaining = self.cfg.target_qty
        self.decision_price = self.cfg.start_price
        self.cash_flow = 0.0
        return self._state()

    def _state(self) -> List[float]:
        md = self.gen.ex.market_data(self.gen.symbol)
        mid = md["mid"] or self.cfg.start_price
        spread = md["spread"] or 0.0
        ofi = md["ofi"] or 0.0
        depth = md["depth"]
        bid_depth = sum(q for _, q in depth["bids"]) or 1.0
        ask_depth = sum(q for _, q in depth["asks"]) or 1.0
        depth_imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)

        inventory_frac = self.remaining / self.cfg.target_qty
        time_frac = self.t / self.cfg.horizon
        # crude realized vol proxy: relative spread
        volatility = spread / mid if mid else 0.0

        return [inventory_frac, spread, volatility, depth_imbalance, ofi, time_frac]

    def step(self, action: int) -> Tuple[List[float], float, bool, dict]:
        self.gen.step()  # background market noise advances one tick
        md_before = self.gen.ex.market_data(self.gen.symbol)
        mid_before = md_before["mid"] or self.cfg.start_price

        qty = 0.0
        order_type = OrderType.LIMIT
        price = mid_before

        if action == 1:  # PASSIVE_SLICE
            qty = min(self.remaining, self.cfg.slice_frac * self.cfg.target_qty)
            offset = md_before["spread"] / 2 if md_before["spread"] else 0.02
            price = round(mid_before - offset if self.cfg.side == Side.SELL else mid_before + offset, 2)
        elif action == 2:  # AGGRESSIVE_SLICE
            qty = min(self.remaining, self.cfg.slice_frac * self.cfg.aggressive_mult * self.cfg.target_qty)
            order_type = OrderType.MARKET
        elif action == 3:  # CANCEL_AND_WIDEN — treated as a no-op passive re-quote, smaller size
            qty = min(self.remaining, 0.25 * self.cfg.slice_frac * self.cfg.target_qty)
            offset = (md_before["spread"] or 0.02) * 1.5
            price = round(mid_before - offset if self.cfg.side == Side.SELL else mid_before + offset, 2)
        # action == 0 -> WAIT, qty stays 0

        execution_cost = 0.0
        if qty > 0:
            order = Order(side=self.cfg.side, quantity=qty, price=price if order_type == OrderType.LIMIT else None,
                          order_type=order_type, trader_id="rl_agent")
            trades = self.gen.ex.submit_order(self.gen.symbol, order)
            filled = sum(t.quantity for t in trades)
            self.remaining -= filled
            side_sign = -1 if self.cfg.side == Side.SELL else 1
            for t in trades:
                execution_cost += side_sign * (self.decision_price - t.price) * t.quantity
                self.cash_flow += t.price * t.quantity * (1 if self.cfg.side == Side.SELL else -1)

        self.t += 1
        done = self.t >= self.cfg.horizon or self.remaining <= 1e-6

        inventory_risk = (self.remaining / self.cfg.target_qty) ** 2
        reward = execution_cost - self.cfg.risk_aversion * inventory_risk

        if done and self.remaining > 1e-6:
            # forced liquidation penalty at unfavorable price (market order)
            md_now = self.gen.ex.market_data(self.gen.symbol)
            mid_now = md_now["mid"] or mid_before
            penalty = -abs(mid_now - self.decision_price) * self.remaining
            reward += penalty
            self.remaining = 0.0

        info = {"remaining": self.remaining, "t": self.t}
        return self._state(), reward, done, info