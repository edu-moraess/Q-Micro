from .orderbook_reconstructor import OrderBookReconstructor
from .event_dispatcher import EventDispatcher

class MarketStream:
    def __init__(self, symbol: str, dispatcher: EventDispatcher):
        self.symbol = symbol.lower()
        self.dispatcher = dispatcher
        self.orderbook = OrderBookReconstructor(symbol)

    async def handle_message(self, data: dict):
        stream = data.get("e", "")
        if stream == "depthUpdate":
            if data.get("U") <= self.orderbook.last_update_id <= data.get("u"):
                self.orderbook.update(data)
                self.dispatcher.dispatch("orderbook", self.orderbook.top_of_book())
        elif stream == "trade":
            # Normaliza trade
            trade = {
                "price": float(data["p"]),
                "quantity": float(data["q"]),
                "side": "BUY" if data["m"] else "SELL",  # m = market maker side?
                "timestamp": data["T"]
            }
            self.dispatcher.dispatch("trade", trade)
        elif stream == "bookTicker":
            # best bid/ask
            ticker = {
                "bid": float(data["b"]),
                "ask": float(data["a"]),
                "bid_qty": float(data["B"]),
                "ask_qty": float(data["A"])
            }
            self.dispatcher.dispatch("ticker", ticker)
        # ... outros streams