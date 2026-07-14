from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import bisect
from .order import Order, OrderSide, OrderType

@dataclass
class OrderBook:
    """
    Represents a Limit Order Book (LOB) with buy and sell sides.

    Attributes:
        buy_orders: Dict mapping price levels to lists of buy orders (sorted descending).
        sell_orders: Dict mapping price levels to lists of sell orders (sorted ascending).
        trades: List of executed trades (for history).
    """
    buy_orders: Dict[float, List[Order]] = field(default_factory=dict)
    sell_orders: Dict[float, List[Order]] = field(default_factory=dict)
    trades: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        self.buy_orders = {}
        self.sell_orders = {}
        self.trades = []

    def add_order(self, order: Order) -> None:
        """Add a new order to the book."""
        if order.is_buy():
            target = self.buy_orders
        else:
            target = self.sell_orders

        if order.price not in target:
            target[order.price] = []

        bisect.insort(target[order.price], order, key=lambda x: x.timestamp)

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by its ID. Returns True if successful."""
        for price_level in list(self.buy_orders.keys()):
            for i, order in enumerate(self.buy_orders[price_level]):
                if order.order_id == order_id and order.status == "OPEN":
                    order.cancel()
                    self.buy_orders[price_level].pop(i)
                    if not self.buy_orders[price_level]:
                        del self.buy_orders[price_level]
                    return True

        for price_level in list(self.sell_orders.keys()):
            for i, order in enumerate(self.sell_orders[price_level]):
                if order.order_id == order_id and order.status == "OPEN":
                    order.cancel()
                    self.sell_orders[price_level].pop(i)
                    if not self.sell_orders[price_level]:
                        del self.sell_orders[price_level]
                    return True

        return False

    def get_best_bid(self) -> Optional[float]:
        """Return the best bid price (highest buy price)."""
        if not self.buy_orders:
            return None
        return max(self.buy_orders.keys())

    def get_best_ask(self) -> Optional[float]:
        """Return the best ask price (lowest sell price)."""
        if not self.sell_orders:
            return None
        return min(self.sell_orders.keys())

    def get_mid_price(self) -> Optional[float]:
        """Return the mid price between best bid and ask."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid is None or best_ask is None:
            return None
        return (best_bid + best_ask) / 2

    def get_spread(self) -> Optional[float]:
        """Return the spread (best ask - best bid)."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid is None or best_ask is None:
            return None
        return best_ask - best_bid

    def get_depth(self, side: OrderSide, levels: int = 5) -> List[Tuple[float, int]]:
        """
        Return the depth of the order book for a given side.

        Args:
            side: BUY or SELL.
            levels: Number of price levels to return.

        Returns:
            List of (price, total_quantity) tuples, sorted by price.
        """
        if side == OrderSide.BUY:
            price_levels = sorted(self.buy_orders.keys(), reverse=True)
        else:
            price_levels = sorted(self.sell_orders.keys())

        depth = []
        for price in price_levels[:levels]:
            orders = self.buy_orders[price] if side == OrderSide.BUY else self.sell_orders[price]
            total_quantity = sum(order.remaining_quantity() for order in orders)
            depth.append((price, total_quantity))

        return depth

    def get_order_book_state(self) -> Dict:
        """Return the current state of the order book."""
        return {
            "best_bid": self.get_best_bid(),
            "best_ask": self.get_best_ask(),
            "mid_price": self.get_mid_price(),
            "spread": self.get_spread(),
            "buy_depth": self.get_depth(OrderSide.BUY),
            "sell_depth": self.get_depth(OrderSide.SELL),
        }

    def __str__(self) -> str:
        """String representation of the order book."""
        buy_side = "\n".join(
            [f"{price:8.2f} | {sum(o.remaining_quantity() for o in orders):6d}"
             for price, orders in sorted(self.buy_orders.items(), reverse=True)[:5]]
        )
        sell_side = "\n".join(
            [f"{price:8.2f} | {sum(o.remaining_quantity() for o in orders):6d}"
             for price, orders in sorted(self.sell_orders.items())[:5]]
        )

        return (
            f"=== ORDER BOOK ===\n"
            f"BID SIDE:\n{buy_side}\n\n"
            f"ASK SIDE:\n{sell_side}\n"
            f"Best Bid: {self.get_best_bid():.2f} | Best Ask: {self.get_best_ask():.2f} | "
            f"Mid: {self.get_mid_price():.2f} | Spread: {self.get_spread():.2f}"
        )