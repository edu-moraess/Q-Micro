"""
Q-Micro :: data.market_data_loader
-------------------------------------
Loads real or synthetic reference market data (e.g. to calibrate the
SyntheticMarketGenerator's volatility/regime parameters).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None


@dataclass
class MarketDataConfig:
    source: str = "csv"       # "csv" | "synthetic"
    path: Optional[str] = None
    price_col: str = "close"


class MarketDataLoader:
    def __init__(self, config: MarketDataConfig):
        self.config = config

    def load(self):
        if self.config.source == "csv":
            if pd is None:
                raise ImportError("pandas is required for CSV loading.")
            if not self.config.path:
                raise ValueError("config.path is required for source='csv'.")
            df = pd.read_csv(self.config.path)
            return df

        if self.config.source == "synthetic":
            return self._synthetic_series()

        raise ValueError(f"Unknown source: {self.config.source}")

    def _synthetic_series(self, n: int = 1000, start: float = 100.0, vol: float = 0.01):
        import random
        prices: List[float] = [start]
        rng = random.Random(7)
        for _ in range(n - 1):
            prices.append(prices[-1] * (1 + rng.gauss(0, vol)))
        if pd is not None:
            return pd.DataFrame({self.config.price_col: prices})
        return prices

    @staticmethod
    def realized_volatility(prices, window: int = 20):
        if pd is None:
            raise ImportError("pandas is required for realized_volatility.")
        returns = pd.Series(prices).pct_change().dropna()
        return returns.rolling(window).std()