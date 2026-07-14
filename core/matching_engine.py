from typing import List, Dict
from .order import Order, OrderSide, OrderType
from .order_book import OrderBook

class MatchingEngine:
    """
    Implements a matching engine for a limit order book.

    Rules:
    1. Price Priority: Orders with better prices are matched first.
    2. Time Priority: Orders at the same price are matched in order of arrival.
    """

    def __init__(self, order_book: OrderBook):
        self.order_book = order_book

    def match_order(self, new_order: Order) -> List[Dict]:
        """
        Match a new order against the order book.

        Args:
            new_order: The incoming order to match.

        Returns:
            List of trade executions (dicts with trade details).
        """
        trades = []
        remaining_quantity = new_order.remaining_quantity()

        if new_order.is_buy():
            price_levels = sorted(self.order_book.sell_orders.keys())
            for price in price_levels:
                if remaining_quantity <= 0:
                    break
                if new_order.is_limit_order() and price > new_order.price:
                    break  # Cannot match above limit price for buy

                sell_orders = self.order_book.sell_orders[price]
                for sell_order in sell_orders[:]:
                    if remaining_quantity <= 0:
                        break
                    if sell_order.status != "OPEN":
                        continue

                    trade_quantity = min(remaining_quantity, sell_order.remaining_quantity())
                    trade_price = price

                    new_order.fill(trade_quantity)
                    sell_order.fill(trade_quantity)

                    trade = {
                        "timestamp": max(new_order.timestamp, sell_order.timestamp),
                        "price": trade_price,
                        "quantity": trade_quantity,
                        "buyer": new_order.trader_id,
                        "seller": sell_order.trader_id,
                        "aggressor_side": "BUY",
                    }
                    trades.append(trade)
                    self.order_book.trades.append(trade)

                    remaining_quantity -= trade_quantity

                    if sell_order.status == "FILLED":
                        sell_orders.remove(sell_order)
                        if not sell_orders:
                            del self.order_book.sell_orders[price]

            if new_order.remaining_quantity() > 0 and new_order.is_limit_order():
                self.order_book.add_order(new_order)

        else:  # SELL order
            price_levels = sorted(self.order_book.buy_orders.keys(), reverse=True)
            for price in price_levels:
                if remaining_quantity <= 0:
                    break
                if new_order.is_limit_order() and price < new_order.price:
                    break  # Cannot match below limit price for sell

                buy_orders = self.order_book.buy_orders[price]
                for buy_order in buy_orders[:]:
                    if remaining_quantity <= 0:
                        break
                    if buy_order.status != "OPEN":
                        continue

                    trade_quantity = min(remaining_quantity, buy_order.remaining_quantity())
                    trade_price = price

                    new_order.fill(trade_quantity)
                    buy_order.fill(trade_quantity)

                    trade = {
                        "timestamp": max(new_order.timestamp, buy_order.timestamp),
                        "price": trade_price,
                        "quantity": trade_quantity,
                        "buyer": buy_order.trader_id,
                        "seller": new_order.trader_id,
                        "aggressor_side": "SELL",
                    }
                    trades.append(trade)
                    self.order_book.trades.append(trade)

                    remaining_quantity -= trade_quantity

                    if buy_order.status == "FILLED":
                        buy_orders.remove(buy_order)
                        if not buy_orders:
                            del self.order_book.buy_orders[price]

            if new_order.remaining_quantity() > 0 and new_order.is_limit_order():
                self.order_book.add_order(new_order)

        return trades

    def process_market_order(self, side: OrderSide, quantity: int, trader_id: str) -> List[Dict]:
        """
        Process a market order (executes immediately at best available prices).

        Args:
            side: BUY or SELL.
            quantity: Order quantity.
            trader_id: Trader placing the order.

        Returns:
            List of trade executions.
        """
        order = Order(
            trader_id=trader_id,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
        )
        return self.match_order(order)

    def process_limit_order(self, side: OrderSide, price: float, quantity: int, trader_id: str) -> List[Dict]:
        """
        Process a limit order.

        Args:
            side: BUY or SELL.
            price: Limit price.
            quantity: Order quantity.
            trader_id: Trader placing the order.

        Returns:
            List of trade executions (empty if no immediate match).
        """
        order = Order(
            trader_id=trader_id,
            side=side,
            price=price,
            quantity=quantity,
            order_type=OrderType.LIMIT,
        )
        return self.match_order(order)

    def process_cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        return self.order_book.cancel_order(order_id)