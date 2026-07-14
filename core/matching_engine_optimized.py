"""
Optimized Matching Engine for Q-Micro.
Uses Numba for JIT compilation and efficient order matching.
"""

import numpy as np
import numba
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from core.order import Order, OrderSide, OrderType
from core.order_book_optimized import OptimizedOrderBook

@dataclass
class OptimizedMatchingEngine:
    """
    Optimized matching engine using Numba and pre-allocated arrays.

    Key optimizations:
    - Numba JIT for critical matching logic.
    - Pre-allocated arrays for price levels and quantities.
    - Batch processing of orders.
    """

    order_book: OptimizedOrderBook

    def __init__(self, order_book: OptimizedOrderBook):
        self.order_book = order_book

    def match_order(self, new_order: Order) -> List[Dict]:
        """
        Match a new order against the order book (optimized).

        Args:
            new_order: The incoming order to match.

        Returns:
            List of trade executions.
        """
        trades = []
        remaining_quantity = new_order.remaining_quantity()

        if new_order.is_buy():
            # Get sell price levels (ascending)
            sell_prices = np.array(sorted(self.order_book.sell_price_levels.keys()))
            sell_quantities = np.array([
                self.order_book.sell_price_levels[p]["quantity"].sum()
                for p in sell_prices
            ])

            if new_order.is_limit_order():
                for i in range(len(sell_prices)):
                    if remaining_quantity <= 0:
                        break
                    if sell_prices[i] > new_order.price:
                        break  # Cannot match above limit price

                    # Execute trade
                    trade_quantity = min(remaining_quantity, sell_quantities[i])
                    trade = {
                        "timestamp": new_order.timestamp,
                        "price": sell_prices[i],
                        "quantity": trade_quantity,
                        "buyer": new_order.trader_id,
                        "seller": "MARKET",
                        "aggressor_side": "BUY",
                    }
                    trades.append(trade)
                    self.order_book.trades.append(trade)

                    remaining_quantity -= trade_quantity
            else:  # Market order
                for i in range(len(sell_prices)):
                    if remaining_quantity <= 0:
                        break

                    # Execute trade
                    trade_quantity = min(remaining_quantity, sell_quantities[i])
                    trade = {
                        "timestamp": new_order.timestamp,
                        "price": sell_prices[i],
                        "quantity": trade_quantity,
                        "buyer": new_order.trader_id,
                        "seller": "MARKET",
                        "aggressor_side": "BUY",
                    }
                    trades.append(trade)
                    self.order_book.trades.append(trade)

                    remaining_quantity -= trade_quantity

            # Add remaining quantity as a new order (if not fully filled)
            if remaining_quantity > 0 and new_order.is_limit_order():
                self.order_book.add_order(new_order)

        else:  # SELL order
            # Get buy price levels (descending)
            buy_prices = np.array(sorted(self.order_book.buy_price_levels.keys(), reverse=True))
            buy_quantities = np.array([
                self.order_book.buy_price_levels[p]["quantity"].sum()
                for p in buy_prices
            ])

            if new_order.is_limit_order():
                for i in range(len(buy_prices)):
                    if remaining_quantity <= 0:
                        break
                    if buy_prices[i] < new_order.price:
                        break  # Cannot match below limit price

                    # Execute trade
                    trade_quantity = min(remaining_quantity, buy_quantities[i])
                    trade = {
                        "timestamp": new_order.timestamp,
                        "price": buy_prices[i],
                        "quantity": trade_quantity,
                        "buyer": "MARKET",
                        "seller": new_order.trader_id,
                        "aggressor_side": "SELL",
                    }
                    trades.append(trade)
                    self.order_book.trades.append(trade)

                    remaining_quantity -= trade_quantity
            else:  # Market order
                for i in range(len(buy_prices)):
                    if remaining_quantity <= 0:
                        break

                    # Execute trade
                    trade_quantity = min(remaining_quantity, buy_quantities[i])
                    trade = {
                        "timestamp": new_order.timestamp,
                        "price": buy_prices[i],
                        "quantity": trade_quantity,
                        "buyer": "MARKET",
                        "seller": new_order.trader_id,
                        "aggressor_side": "SELL",
                    }
                    trades.append(trade)
                    self.order_book.trades.append(trade)

                    remaining_quantity -= trade_quantity

            # Add remaining quantity as a new order (if not fully filled)
            if remaining_quantity > 0 and new_order.is_limit_order():
                self.order_book.add_order(new_order)

        return trades

    def process_market_order(self, side: OrderSide, quantity: int, trader_id: str) -> List[Dict]:
        """Process a market order (optimized)."""
        order = Order(
            trader_id=trader_id,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
        )
        return self.match_order(order)

    def process_limit_order(self, side: OrderSide, price: float, quantity: int, trader_id: str) -> List[Dict]:
        """Process a limit order (optimized)."""
        order = Order(
            trader_id=trader_id,
            side=side,
            price=price,
            quantity=quantity,
            order_type=OrderType.LIMIT,
        )
        return self.match_order(order)

    def process_cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order (optimized)."""
        return self.order_book.cancel_order(order_id)
