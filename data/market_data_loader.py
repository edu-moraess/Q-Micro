"""
Market Data Loader for Q-Micro.
Loads real market data from CSV, Parquet, or APIs.
"""

import pandas as pd
from typing import Optional, Union
from pathlib import Path

class MarketDataLoader:
    """
    Loads historical market data from files or APIs.
    Supports CSV, Parquet, and JSON formats.
    """

    def __init__(self, data_dir: str = "data/raw/"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_csv(self, filename: str) -> pd.DataFrame:
        """Load data from a CSV file."""
        filepath = self.data_dir / filename
        return pd.read_csv(filepath)

    def load_parquet(self, filename: str) -> pd.DataFrame:
        """Load data from a Parquet file."""
        filepath = self.data_dir / filename
        return pd.read_parquet(filepath)

    def load_json(self, filename: str) -> pd.DataFrame:
        """Load data from a JSON file."""
        filepath = self.data_dir / filename
        return pd.read_json(filepath)

    def save_to_parquet(self, df: pd.DataFrame, filename: str) -> None:
        """Save DataFrame to Parquet."""
        filepath = self.data_dir / filename
        df.to_parquet(filepath)