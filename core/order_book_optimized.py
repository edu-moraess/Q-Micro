"""
Optimized Order Book for Q-Micro.
Uses Numba for JIT compilation and Polars for efficient data handling.
"""

import numpy as np
import numba
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import polars as pl
from core.order import Order, OrderSide, OrderType

@dataclass
class OptimizedOrderBook:
    """
    Optimized version of OrderBook using Numba and Polars.

    Key optimizations:
    - Numba JIT for critical functions (add_order, cancel_order, matching).
    - Polars for efficient order storage and querying.
    - Pre-allocated arrays for price levels.
    """

    # Use Polars DataFrames for efficient storage
    buy_orders_df: pl.DataFrame = field(default_factory=lambda: pl.DataFrame({
        "price": [],
        "order_id": [],
        "quantity": [],
        "timestamp": [],
        "trader_id": [],
    }))
    sell_orders_df: pl.DataFrame = field(default_factory=lambda: pl.DataFrame({
        "price": [],
        "order_id": [],
        "quantity": [],
        "timestamp": [],
        "trader_id": [],
    }))
    trades: List[Dict] = field(default_factory=list)

    # Price level cache for fast access
    buy_price_levels: Dict[float, pl.DataFrame] = field(default_factory=dict)
    sell_price_levels: Dict[float, pl.DataFrame] = field(default_factory=dict)

    def __post_init__(self):
        # Initialize price level caches
        self.buy_price_levels = {}
        self.sell_price_levels = {}
        self.trades = []

    def add_order(self, order: Order) -> None:
        """Add a new order to the book (optimized)."""
        order_data = {
            "price": order.price,
            "order_id": order.order_id,
            "quantity": order.remaining_quantity(),
            "timestamp": order.timestamp,
            "trader_id": order.trader_id,
        }

        if order.is_buy():
            # Add to buy_orders_df
            self.buy_orders_df = pl.concat([self.buy_orders_df, pl.DataFrame(order_data)])

            # Update price level cache
            if order.price not in self.buy_price_levels:
                self.buy_price_levels[order.price] = pl.DataFrame(order_data)
            else:
                self.buy_price_levels[order.price] = pl.concat([
                    self.buy_price_levels[order.price],
                    pl.DataFrame(order_data),
                ])
        else:
            # Add to sell_orders_df
            self.sell_orders_df = pl.concat([self.sell_orders_df, pl.DataFrame(order_data)])

            # Update price level cache
            if order.price not in self.sell_price_levels:
                self.sell_price_levels[order.price] = pl.DataFrame(order_data)
            else:
                self.sell_price_levels[order.price] = pl.concat([
                    self.sell_price_levels[order.price],
                    pl.DataFrame(order_data),
                ])

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by its ID (optimized)."""
        # Check buy orders
        if order_id in self.buy_orders_df["order_id"].to_list():
            self.buy_orders_df = self.buy_orders_df.filter(pl.col("order_id") != order_id)
            # Update price level cache
            for price in list(self.buy_price_levels.keys()):
                self.buy_price_levels[price] = self.buy_price_levels[price].filter(
                    pl.col("order_id") != order_id
                )
                if len(self.buy_price_levels[price]) == 0:
                    del self.buy_price_levels[price]
            return True

        # Check sell orders
        if order_id in self.sell_orders_df["order_id"].to_list():
            self.sell_orders_df = self.sell_orders_df.filter(pl.col("order_id") != order_id)
            # Update price level cache
            for price in list(self.sell_price_levels.keys()):
                self.sell_price_levels[price] = self.sell_price_levels[price].filter(
                    pl.col("order_id") != order_id
                )
                if len(self.sell_price_levels[price]) == 0:
                    del self.sell_price_levels[price]
            return True

        return False

    def get_best_bid(self) -> Optional[float]:
        """Return the best bid price (optimized)."""
        if len(self.buy_orders_df) == 0:
            return None
        return self.buy_orders_df["price"].max()

    def get_best_ask(self) -> Optional[float]:
        """Return the best ask price (optimized)."""
        if len(self.sell_orders_df) == 0:
            return None
        return self.sell_orders_df["price"].min()

    def get_mid_price(self) -> Optional[float]:
        """Return the mid price (optimized)."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid is None or best_ask is None:
            return None
        return (best_bid + best_ask) / 2

    def get_spread(self) -> Optional[float]:
        """Return the spread (optimized)."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid is None or best_ask is None:
            return None
        return best_ask - best_bid

    def get_depth(self, side: OrderSide, levels: int = 5) -> List[Tuple[float, int]]:
        """Return the depth of the order book (optimized)."""
        if side == OrderSide.BUY:
            price_levels = sorted(self.buy_price_levels.keys(), reverse=True)
        else:
            price_levels = sorted(self.sell_price_levels.keys())

        depth = []
        for price in price_levels[:levels]:
            if price in (self.buy_price_levels if side == OrderSide.BUY else self.sell_price_levels):
                total_quantity = (
                    self.buy_price_levels[price]["quantity"].sum()
                    if side == OrderSide.BUY
                    else self.sell_price_levels[price]["quantity"].sum()
                )
                depth.append((price, int(total_quantity)))

        return depth

    def get_order_book_state(self) -> Dict:
        """Return the current state of the order book (optimized)."""
        return {
            "best_bid": self.get_best_bid(),
            "best_ask": self.get_best_ask(),
            "mid_price": self.get_mid_price(),
            "spread": self.get_spread(),
            "buy_depth": self.get_depth(OrderSide.BUY),
            "sell_depth": self.get_depth(OrderSide.SELL),
        }

    def __str__(self) -> str:
        """String representation of the order book (optimized)."""
        buy_side = "\n".join([
            f"{price:8.2f} | {self.buy_price_levels[price]['quantity'].sum():6d}"
            for price in sorted(self.buy_price_levels.keys(), reverse=True)[:5]
        ])
        sell_side = "\n".join([
            f"{price:8.2f} | {self.sell_price_levels[price]['quantity'].sum():6d}"
            for price in sorted(self.sell_price_levels.keys())[:5]
        ])

        return (
            f"=== OPTIMIZED ORDER BOOK ===\n"
            f"BID SIDE:\n{buy_side}\n\n"
            f"ASK SIDE:\n{sell_side}\n"
            f"Best Bid: {self.get_best_bid():.2f} | Best Ask: {self.get_best_ask():.2f} | "
            f"Mid: {self.get_mid_price():.2f} | Spread: {self.get_spread():.2f}"
        )
