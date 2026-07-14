"""Q-Micro :: execution.twap — Time-Weighted Average Price scheduler."""

from __future__ import annotations
from typing import List
from core.order import Order, Side, OrderType


def twap_schedule(total_qty: float, n_slices: int, side: Side, trader_id: str = "twap_algo") -> List[Order]:
    """Splits a parent order into n_slices equal MARKET child orders, one per interval."""
    if n_slices <= 0:
        raise ValueError("n_slices must be positive.")
    slice_qty = total_qty / n_slices
    return [
        Order(side=side, quantity=slice_qty, order_type=OrderType.MARKET, trader_id=trader_id)
        for _ in range(n_slices)
    ]


class TWAPExecutor:
    """Stateful TWAP executor: call .next_slice() once per scheduling interval."""

    def __init__(self, total_qty: float, n_slices: int, side: Side, trader_id: str = "twap_algo"):
        self.side = side
        self.trader_id = trader_id
        self.slice_qty = total_qty / n_slices
        self.slices_left = n_slices
        self.executed = 0.0

    def next_slice(self) -> Order:
        if self.slices_left <= 0:
            raise StopIteration("TWAP schedule complete.")
        self.slices_left -= 1
        self.executed += self.slice_qty
        return Order(side=self.side, quantity=self.slice_qty,
                      order_type=OrderType.MARKET, trader_id=self.trader_id)