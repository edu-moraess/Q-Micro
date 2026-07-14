"""
Q-Micro :: core.order
----------------------
Defines the Order primitive used throughout the exchange simulator.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum
from time import time_ns
from typing import Optional


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP = "STOP"
    CANCEL = "CANCEL"


class OrderStatus(str, Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


_id_counter = itertools.count(1)


def _next_order_id() -> int:
    return next(_id_counter)


@dataclass
class Order:
    """
    A single order sent to the exchange.

    LIMIT orders require `price`.
    STOP orders require `stop_price` (trigger level) and are converted
    to MARKET by ExchangeSimulator once triggered.
    """

    side: Side
    quantity: float
    price: Optional[float] = None          # required for LIMIT
    order_type: OrderType = OrderType.LIMIT
    trader_id: str = "anonymous"
    stop_price: Optional[float] = None     # required for STOP
    order_id: int = field(default_factory=_next_order_id)
    timestamp: int = field(default_factory=time_ns)
    filled_quantity: float = 0.0
    status: OrderStatus = OrderStatus.NEW

    def __post_init__(self) -> None:
        if self.order_type == OrderType.LIMIT and self.price is None:
            raise ValueError("LIMIT orders require a price.")
        if self.order_type == OrderType.STOP and self.stop_price is None:
            raise ValueError("STOP orders require a stop_price.")
        if self.quantity <= 0:
            raise ValueError("Order quantity must be positive.")

    @property
    def remaining_quantity(self) -> float:
        return self.quantity - self.filled_quantity

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED)

    def fill(self, quantity: float) -> None:
        if quantity <= 0 or quantity > self.remaining_quantity + 1e-9:
            raise ValueError("Invalid fill quantity.")
        self.filled_quantity += quantity
        self.status = (
            OrderStatus.FILLED
            if self.remaining_quantity <= 1e-9
            else OrderStatus.PARTIALLY_FILLED
        )

    def cancel(self) -> None:
        if self.status in (OrderStatus.FILLED, OrderStatus.CANCELLED):
            return
        self.status = OrderStatus.CANCELLED

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Order(id={self.order_id}, {self.side.value} "
            f"{self.remaining_quantity}/{self.quantity} @ {self.price}, "
            f"type={self.order_type.value}, status={self.status.value})"
        )