from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import uuid
import time

class OrderSide(Enum):
    BUY = auto()
    SELL = auto()

class OrderType(Enum):
    MARKET = auto()
    LIMIT = auto()
    STOP = auto()
    CANCEL = auto()

@dataclass
class Order:
    """
    Represents a single order in the market.

    Attributes:
        order_id: Unique identifier for the order.
        timestamp: Time when the order was placed (in seconds since epoch).
        trader_id: Identifier for the trader who placed the order.
        side: BUY or SELL.
        price: Price of the order (for MARKET orders, this is the best available price).
        quantity: Number of shares to buy/sell.
        order_type: Type of the order (MARKET, LIMIT, STOP, CANCEL).
        filled: Amount of the order that has been filled.
        status: Current status of the order (OPEN, FILLED, CANCELLED, PARTIAL).
    """
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    trader_id: str = "TRADER_0"
    side: OrderSide = OrderSide.BUY
    price: float = 0.0
    quantity: int = 0
    order_type: OrderType = OrderType.LIMIT
    filled: int = 0
    status: str = "OPEN"

    def __post_init__(self):
        if self.quantity < 0:
            raise ValueError("Quantity must be non-negative.")
        if self.price < 0:
            raise ValueError("Price must be non-negative.")

    def is_buy(self) -> bool:
        return self.side == OrderSide.BUY

    def is_sell(self) -> bool:
        return self.side == OrderSide.SELL

    def is_market_order(self) -> bool:
        return self.order_type == OrderType.MARKET

    def is_limit_order(self) -> bool:
        return self.order_type == OrderType.LIMIT

    def is_cancel_order(self) -> bool:
        return self.order_type == OrderType.CANCEL

    def remaining_quantity(self) -> int:
        return self.quantity - self.filled

    def fill(self, fill_quantity: int) -> None:
        if fill_quantity > self.remaining_quantity():
            raise ValueError("Fill quantity exceeds remaining quantity.")
        self.filled += fill_quantity
        if self.filled == self.quantity:
            self.status = "FILLED"
        else:
            self.status = "PARTIAL"

    def cancel(self) -> None:
        self.status = "CANCELLED"