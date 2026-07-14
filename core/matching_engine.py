"""
Q-Micro :: core.matching_engine
---------------------------------
Price-time priority matching engine (strict price priority, FIFO within level).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time_ns
from typing import List, Optional

from core.order import Order, OrderType, Side, OrderStatus
from core.order_book import OrderBook


@dataclass
class Trade:
    price: float
    quantity: float
    aggressor_side: Side
    buy_order_id: int
    sell_order_id: int
    buyer_id: str
    seller_id: str
    timestamp: int = field(default_factory=time_ns)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Trade(px={self.price}, qty={self.quantity}, "
            f"buy={self.buy_order_id}<-{self.buyer_id}, "
            f"sell={self.sell_order_id}<-{self.seller_id})"
        )


class MatchingEngine:
    def __init__(self, order_book: OrderBook):
        self.book = order_book
        self.trade_history: List[Trade] = []

    # ---------------------------------------------------------------- #
    def submit(self, order: Order) -> List[Trade]:
        if order.order_type == OrderType.CANCEL:
            self.book.cancel_order(order.order_id)
            return []
        if order.order_type == OrderType.MARKET:
            return self._match_market(order)
        if order.order_type == OrderType.LIMIT:
            return self._match_limit(order)
        if order.order_type == OrderType.STOP:
            raise NotImplementedError(
                "STOP orders must be routed through ExchangeSimulator, "
                "which converts them to MARKET once triggered."
            )
        raise ValueError(f"Unsupported order_type: {order.order_type}")

    # ---------------------------------------------------------------- #
    def _opposing_side_touch(self, order: Order) -> bool:
        if order.side == Side.BUY:
            best_ask = self.book.best_ask()
            if best_ask is None:
                return False
            return order.order_type == OrderType.MARKET or order.price >= best_ask
        best_bid = self.book.best_bid()
        if best_bid is None:
            return False
        return order.order_type == OrderType.MARKET or order.price <= best_bid

    def _match_market(self, order: Order) -> List[Trade]:
        trades = self._cross(order)
        if order.is_active and order.remaining_quantity > 0:
            order.status = OrderStatus.CANCELLED  # unfilled residual is not resting
        return trades

    def _match_limit(self, order: Order) -> List[Trade]:
        trades = self._cross(order)
        if order.is_active and order.remaining_quantity > 1e-9:
            self.book.add_limit_order(order)
        return trades

    def _cross(self, order: Order) -> List[Trade]:
        trades: List[Trade] = []
        book = self.book._asks if order.side == Side.BUY else self.book._bids

        while order.remaining_quantity > 1e-9 and self._opposing_side_touch(order):
            touch_price = self.book.best_ask() if order.side == Side.BUY else self.book.best_bid()
            level = book[touch_price]

            while level and order.remaining_quantity > 1e-9:
                resting = level[0]
                if not resting.is_active:
                    level.popleft()
                    continue

                fill_qty = min(order.remaining_quantity, resting.remaining_quantity)
                order.fill(fill_qty)
                resting.fill(fill_qty)

                buy_order = order if order.side == Side.BUY else resting
                sell_order = resting if order.side == Side.BUY else order

                trades.append(Trade(
                    price=touch_price,
                    quantity=fill_qty,
                    aggressor_side=order.side,
                    buy_order_id=buy_order.order_id,
                    sell_order_id=sell_order.order_id,
                    buyer_id=buy_order.trader_id,
                    seller_id=sell_order.trader_id,
                ))

                if not resting.is_active:
                    level.popleft()

            if not level:
                del book[touch_price]

        self.trade_history.extend(trades)
        return trades

    # ---------------------------------------------------------------- #
    def last_trade_price(self) -> Optional[float]:
        return self.trade_history[-1].price if self.trade_history else None

    def trades_to_records(self) -> List[dict]:
        return [
            {
                "timestamp": t.timestamp,
                "price": t.price,
                "volume": t.quantity,
                "buyer": t.buyer_id,
                "seller": t.seller_id,
                "aggressor": t.aggressor_side.value,
            }
            for t in self.trade_history
        ]