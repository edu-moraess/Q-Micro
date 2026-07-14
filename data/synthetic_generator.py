"""
Synthetic Market Data Generator for Q-Micro.
Generates realistic order flow, traders, and market regimes.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from enum import Enum, auto
from dataclasses import dataclass
import random
import time

class TraderType(Enum):
    NOISE = auto()
    INFORMED = auto()
    MARKET_MAKER = auto()
    INSTITUTIONAL = auto()

@dataclass
class Trader:
    trader_id: str
    trader_type: TraderType
    cash: float = 1_000_000.0
    inventory: int = 0
    private_info: Optional[float] = None  # For informed traders

class SyntheticMarketGenerator:
    """
    Generates synthetic market data with different trader types.
    """

    def __init__(self, n_traders: int = 10, initial_price: float = 100.0):
        self.n_traders = n_traders
        self.initial_price = initial_price
        self.traders = self._initialize_traders()
        self.current_price = initial_price
        self.volatility = 0.01  # 1% volatility
        self.time_step = 0

    def _initialize_traders(self) -> List[Trader]:
        """Initialize traders with different types."""
        traders = []
        for i in range(self.n_traders):
            trader_type = random.choice(list(TraderType))
            trader = Trader(
                trader_id=f"TRADER_{i}",
                trader_type=trader_type,
                private_info=random.uniform(-0.05, 0.05) if trader_type == TraderType.INFORMED else None
            )
            traders.append(trader)
        return traders

    def generate_order(self, trader: Trader) -> Dict:
        """Generate a single order for a trader."""
        if trader.trader_type == TraderType.NOISE:
            return self._generate_noise_order(trader)
        elif trader.trader_type == TraderType.INFORMED:
            return self._generate_informed_order(trader)
        elif trader.trader_type == TraderType.MARKET_MAKER:
            return self._generate_market_maker_order(trader)
        elif trader.trader_type == TraderType.INSTITUTIONAL:
            return self._generate_institutional_order(trader)
        else:
            raise ValueError(f"Unknown trader type: {trader.trader_type}")

    def _generate_noise_order(self, trader: Trader) -> Dict:
        """Generate a random order (Noise Trader)."""
        side = random.choice(["BUY", "SELL"])
        price = self.current_price * (1 + random.uniform(-0.005, 0.005))  # Small deviation
        quantity = random.randint(1, 100)
        return {
            "trader_id": trader.trader_id,
            "side": side,
            "price": price,
            "quantity": quantity,
            "order_type": "LIMIT",
            "timestamp": time.time()
        }

    def _generate_informed_order(self, trader: Trader) -> Dict:
        """Generate an order based on private information (Informed Trader)."""
        # Informed traders know the future price direction
        future_price = self.current_price * (1 + trader.private_info)
        if future_price > self.current_price:
            side = "BUY"
        else:
            side = "SELL"
        price = self.current_price * (1 + random.uniform(-0.002, 0.002))
        quantity = random.randint(10, 500)
        return {
            "trader_id": trader.trader_id,
            "side": side,
            "price": price,
            "quantity": quantity,
            "order_type": "LIMIT",
            "timestamp": time.time()
        }

    def _generate_market_maker_order(self, trader: Trader) -> Dict:
        """Generate orders to provide liquidity (Market Maker)."""
        spread = 0.001  # 0.1% spread
        bid_price = self.current_price * (1 - spread/2)
        ask_price = self.current_price * (1 + spread/2)
        side = random.choice(["BUY", "SELL"])
        price = bid_price if side == "BUY" else ask_price
        quantity = random.randint(50, 200)
        return {
            "trader_id": trader.trader_id,
            "side": side,
            "price": price,
            "quantity": quantity,
            "order_type": "LIMIT",
            "timestamp": time.time()
        }

    def _generate_institutional_order(self, trader: Trader) -> Dict:
        """Generate large orders (Institutional Trader)."""
        side = random.choice(["BUY", "SELL"])
        price = self.current_price * (1 + random.uniform(-0.001, 0.001))
        quantity = random.randint(500, 2000)
        return {
            "trader_id": trader.trader_id,
            "side": side,
            "price": price,
            "quantity": quantity,
            "order_type": "LIMIT",
            "timestamp": time.time()
        }

    def generate_order_flow(self, n_orders: int = 100) -> List[Dict]:
        """Generate a flow of orders from all traders."""
        orders = []
        for _ in range(n_orders):
            trader = random.choice(self.traders)
            order = self.generate_order(trader)
            orders.append(order)
            # Update current price based on order flow
            if order["side"] == "BUY":
                self.current_price *= (1 + random.uniform(0, 0.0001))
            else:
                self.current_price *= (1 - random.uniform(0, 0.0001))
        return orders