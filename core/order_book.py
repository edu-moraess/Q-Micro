"""
Q-Micro :: core.order_book
----------------------------
Limit Order Book (LOB) with price-time priority.

Design:
  - self._bids / self._asks: price -> deque[Order]  (FIFO within a level)
  - self._bid_heap: max-heap of active bid prices (stored negated)
  - self._ask_heap: min-heap of active ask prices
  - lazy deletion: stale heap entries are skipped/popped on read
"""

from __future__ import annotations

import heapq
from collections import deque, defaultdict
from typing import Deque, Dict, List, Optional, Tuple

from core.order import Order, Side


class OrderBook:
    def __init__(self, symbol: str = "SYNTH"):
        self.symbol = symbol
        self._bids: Dict[float, Deque[Order]] = defaultdict(deque)
        self._asks: Dict[float, Deque[Order]] = defaultdict(deque)
        self._bid_heap: List[float] = []   # negated prices (max-heap)
        self._ask_heap: List[float] = []   # prices (min-heap)
        self._orders_index: Dict[int, Order] = {}

    # ---------------------------------------------------------------- #
    # Mutation
    # ---------------------------------------------------------------- #
    def add_limit_order(self, order: Order) -> None:
        if order.price is None:
            raise ValueError("add_limit_order requires a priced order.")
        book, heap, sign = (
            (self._bids, self._bid_heap, -1.0) if order.side == Side.BUY
            else (self._asks, self._ask_heap, 1.0)
        )
        if order.price not in book:
            heapq.heappush(heap, sign * order.price)
        book[order.price].append(order)
        self._orders_index[order.order_id] = order

    def cancel_order(self, order_id: int) -> bool:
        order = self._orders_index.get(order_id)
        if order is None or not order.is_active:
            return False
        order.cancel()
        book = self._bids if order.side == Side.BUY else self._asks
        level = book.get(order.price)
        if level and order in level:
            level.remove(order)
            if not level:
                del book[order.price]
        return True

    # ---------------------------------------------------------------- #
    # Top-of-book (lazy deletion of stale/empty levels)
    # ---------------------------------------------------------------- #
    def best_bid(self) -> Optional[float]:
        while self._bid_heap:
            price = -self._bid_heap[0]
            if price in self._bids and self._bids[price]:
                return price
            heapq.heappop(self._bid_heap)
        return None

    def best_ask(self) -> Optional[float]:
        while self._ask_heap:
            price = self._ask_heap[0]
            if price in self._asks and self._asks[price]:
                return price
            heapq.heappop(self._ask_heap)
        return None

    def mid_price(self) -> Optional[float]:
        bid, ask = self.best_bid(), self.best_ask()
        return None if (bid is None or ask is None) else (bid + ask) / 2.0

    def spread(self) -> Optional[float]:
        bid, ask = self.best_bid(), self.best_ask()
        return None if (bid is None or ask is None) else ask - bid

    def spread_bps(self) -> Optional[float]:
        s, m = self.spread(), self.mid_price()
        return None if (s is None or not m) else 10_000.0 * s / m

    # ---------------------------------------------------------------- #
    # Depth / imbalance
    # ---------------------------------------------------------------- #
    def depth(self, levels: int = 5) -> Dict[str, List[Tuple[float, float]]]:
        bid_prices = sorted({p for p in self._bids if self._bids[p]}, reverse=True)[:levels]
        ask_prices = sorted({p for p in self._asks if self._asks[p]})[:levels]
        bids = [(p, sum(o.remaining_quantity for o in self._bids[p])) for p in bid_prices]
        asks = [(p, sum(o.remaining_quantity for o in self._asks[p])) for p in ask_prices]
        return {"bids": bids, "asks": asks}

    def order_flow_imbalance(self, levels: int = 5) -> Optional[float]:
        """OFI = (BuyVolume - SellVolume) / (BuyVolume + SellVolume), top-N levels."""
        d = self.depth(levels)
        buy_vol = sum(q for _, q in d["bids"])
        sell_vol = sum(q for _, q in d["asks"])
        total = buy_vol + sell_vol
        return None if total == 0 else (buy_vol - sell_vol) / total

    def queue_position(self, order_id: int) -> Optional[int]:
        order = self._orders_index.get(order_id)
        if order is None:
            return None
        book = self._bids if order.side == Side.BUY else self._asks
        level = book.get(order.price)
        if not level:
            return None
        for i, o in enumerate(level):
            if o.order_id == order_id:
                return i
        return None

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"OrderBook({self.symbol}, best_bid={self.best_bid()}, "
            f"best_ask={self.best_ask()}, spread={self.spread()})"
        )