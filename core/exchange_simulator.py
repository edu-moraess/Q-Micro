from typing import List, Dict
from .order import Order, OrderSide, OrderType
from .order_book import OrderBook
from .matching_engine import MatchingEngine

class ExchangeSimulator:
    """
    Simulates a stock exchange with a limit order book and matching engine.
    """

    def __init__(self):
        self.order_book = OrderBook()
        self.matching_engine = MatchingEngine(self.order_book)
        self.traders = {}
        self.order_history = []
        self.trade_history = []

    def add_trader(self, trader_id: str, trader_type: str = "NOISE") -> None:
        """Register a new trader."""
        self.traders[trader_id] = {"type": trader_type, "orders": []}

    def submit_order(self, trader_id: str, side: OrderSide, price: float, quantity: int,
                     order_type: OrderType = OrderType.LIMIT) -> Order:
        """
        Submit a new order to the exchange.

        Args:
            trader_id: ID of the trader.
            side: BUY or SELL.
            price: Order price (for MARKET orders, use 0 or best available).
            quantity: Order quantity.
            order_type: Type of order (LIMIT, MARKET, etc.).

        Returns:
            The submitted Order object.
        """
        if trader_id not in self.traders:
            self.add_trader(trader_id)

        order = Order(
            trader_id=trader_id,
            side=side,
            price=price,
            quantity=quantity,
            order_type=order_type,
        )

        self.traders[trader_id]["orders"].append(order)
        self.order_history.append(order)

        if order_type == OrderType.MARKET:
            self.matching_engine.process_market_order(side, quantity, trader_id)
        else:
            self.matching_engine.match_order(order)

        return order

    def get_order_book_state(self) -> Dict:
        """Return the current state of the order book."""
        return self.order_book.get_order_book_state()

    def get_trade_history(self, limit: int = 100) -> List[Dict]:
        """Return the last N trades."""
        return self.order_book.trades[-limit:]

    def get_order_history(self, limit: int = 100) -> List[Order]:
        """Return the last N orders."""
        return self.order_history[-limit:]

    def reset(self) -> None:
        """Reset the exchange state."""
        self.order_book = OrderBook()
        self.matching_engine = MatchingEngine(self.order_book)
        self.order_history = []
        self.trade_history = []