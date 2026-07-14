import sortedcontainers  # pip install sortedcontainers
from collections import deque

class OrderBookReconstructor:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bids = sortedcontainers.SortedDict(negate=True)  # maior preço primeiro
        self.asks = sortedcontainers.SortedDict()              # menor preço primeiro
        self.last_update_id = None

    def snapshot(self, data: dict):
        """Aplica o snapshot inicial (evento depth@100ms)."""
        self.bids.clear()
        self.asks.clear()
        for p, q in data["bids"]:
            self.bids[float(p)] = float(q)
        for p, q in data["asks"]:
            self.asks[float(p)] = float(q)
        self.last_update_id = data["lastUpdateId"]

    def update(self, data: dict):
        """Processa um evento depthUpdate (diff)."""
        for p, q in data["b"]:
            price = float(p)
            qty = float(q)
            if qty == 0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = qty
        for p, q in data["a"]:
            price = float(p)
            qty = float(q)
            if qty == 0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = qty
        self.last_update_id = data["u"]

    def top_of_book(self) -> dict:
        best_bid = self.bids.peekitem(0) if self.bids else (None, None)
        best_ask = self.asks.peekitem(0) if self.asks else (None, None)
        return {"bid": best_bid[0], "bid_qty": best_bid[1],
                "ask": best_ask[0], "ask_qty": best_ask[1]}

    def depth(self, levels=5) -> dict:
        """Retorna os N melhores níveis de cada lado."""
        bids = list(self.bids.items())[:levels]
        asks = list(self.asks.items())[:levels]
        return {"bids": bids, "asks": asks}