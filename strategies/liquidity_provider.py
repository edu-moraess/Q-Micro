
"""Q-Micro :: strategies.liquidity_provider — passive multi-level liquidity provision."""

from __future__ import annotations
from typing import List
from core.order import Order, Side
from core.exchange_simulator import ExchangeSimulator


class LiquidityProviderStrategy:
    """Lays down resting orders at multiple price levels away from mid (ladder)."""

    def __init__(self, symbol: str, trader_id: str = "lp_strategy",
                 n_levels: int = 3, level_gap: float = 0.05, size_per_level: float = 200):
        self.symbol = symbol
        self.trader_id = trader_id
        self.n_levels = n_levels
        self.level_gap = level_gap
        self.size_per_level = size_per_level

    def place_ladder(self, ex: ExchangeSimulator) -> List[Order]:
        md = ex.market_data(self.symbol)
        mid = md["mid"]
        if mid is None:
            return []

        placed = []
        for lvl in range(1, self.n_levels + 1):
            bid = Order(side=Side.BUY, price=round(mid - lvl * self.level_gap, 2),
                        quantity=self.size_per_level, trader_id=self.trader_id)
            ask = Order(side=Side.SELL, price=round(mid + lvl * self.level_gap, 2),
                        quantity=self.size_per_level, trader_id=self.trader_id)
            ex.submit_order(self.symbol, bid)
            ex.submit_order(self.symbol, ask)
            placed.extend([bid, ask])
        return placed