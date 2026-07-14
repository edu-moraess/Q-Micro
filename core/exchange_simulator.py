"""
Q-Micro :: core.exchange_simulator
-------------------------------------
Top-level façade wiring Order / OrderBook / MatchingEngine together,
with STOP-order trigger routing and simple market-data/trade-tape access.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from core.order import Order, OrderType, Side
from core.order_book import OrderBook
from core.matching_engine import MatchingEngine, Trade


class ExchangeSimulator:
    def __init__(self, symbols: Optional[List[str]] = None):
        symbols = symbols or ["SYNTH"]
        self.books: Dict[str, OrderBook] = {s: OrderBook(s) for s in symbols}
        self.engines: Dict[str, MatchingEngine] = {
            s: MatchingEngine(self.books[s]) for s in symbols
        }
        self._pending_stops: Dict[str, List[Order]] = {s: [] for s in symbols}

    # ---------------------------------------------------------------- #
    def submit_order(self, symbol: str, order: Order) -> List[Trade]:
        if symbol not in self.engines:
            raise KeyError(f"Unknown symbol: {symbol}")
        if order.order_type == OrderType.STOP:
            self._pending_stops[symbol].append(order)
            return []
        trades = self.engines[symbol].submit(order)
        self._check_stop_triggers(symbol)
        return trades

    def cancel_order(self, symbol: str, order_id: int) -> bool:
        return self.books[symbol].cancel_order(order_id)

    # ---------------------------------------------------------------- #
    def _check_stop_triggers(self, symbol: str) -> None:
        last = self.engines[symbol].last_trade_price()
        if last is None:
            return
        still_pending = []
        for stop in self._pending_stops[symbol]:
            triggered = (
                (stop.side == Side.BUY and last >= stop.stop_price) or
                (stop.side == Side.SELL and last <= stop.stop_price)
            )
            if triggered:
                live = Order(
                    side=stop.side,
                    quantity=stop.remaining_quantity,
                    order_type=OrderType.MARKET,
                    trader_id=stop.trader_id,
                )
                self.engines[symbol].submit(live)
            else:
                still_pending.append(stop)
        self._pending_stops[symbol] = still_pending

    # ---------------------------------------------------------------- #
    def market_data(self, symbol: str, depth_levels: int = 5) -> dict:
        book = self.books[symbol]
        return {
            "symbol": symbol,
            "best_bid": book.best_bid(),
            "best_ask": book.best_ask(),
            "mid": book.mid_price(),
            "spread": book.spread(),
            "spread_bps": book.spread_bps(),
            "ofi": book.order_flow_imbalance(depth_levels),
            "depth": book.depth(depth_levels),
        }

    def trade_tape(self, symbol: str) -> List[dict]:
        return self.engines[symbol].trades_to_records()