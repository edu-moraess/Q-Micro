
"""Q-Micro :: strategies.market_maker — inventory-aware quoting strategy."""

from __future__ import annotations
from core.order import Order, Side
from core.exchange_simulator import ExchangeSimulator


class MarketMakerStrategy:
    def __init__(self, symbol: str, trader_id: str = "mm_strategy",
                 target_spread: float = 0.10, quote_size: float = 300,
                 max_inventory: float = 2000, inventory_skew_coef: float = 0.02):
        self.symbol = symbol
        self.trader_id = trader_id
        self.target_spread = target_spread
        self.quote_size = quote_size
        self.max_inventory = max_inventory
        self.inventory_skew_coef = inventory_skew_coef
        self.inventory = 0.0

    def quote(self, ex: ExchangeSimulator) -> None:
        md = ex.market_data(self.symbol)
        mid = md["mid"]
        if mid is None:
            return

        # Cancel-and-requote is out of scope for this skeleton; assumes fresh book each call.
        skew = -self.inventory_skew_coef * (self.inventory / self.max_inventory)
        half = self.target_spread / 2.0
        bid_px = round(mid - half + skew, 2)
        ask_px = round(mid + half + skew, 2)

        size = self.quote_size
        if abs(self.inventory) >= self.max_inventory:
            size *= 0.25  # shrink quotes when at risk limit

        trades_bid = ex.submit_order(self.symbol, Order(
            side=Side.BUY, price=bid_px, quantity=size, trader_id=self.trader_id))
        trades_ask = ex.submit_order(self.symbol, Order(
            side=Side.SELL, price=ask_px, quantity=size, trader_id=self.trader_id))

        for t in trades_bid + trades_ask:
            if t.buyer_id == self.trader_id:
                self.inventory += t.quantity
            if t.seller_id == self.trader_id:
                self.inventory -= t.quantity