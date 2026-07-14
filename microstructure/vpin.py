
"""
VPIN (Volume-Synchronized Probability of Informed Trading) for Q-Micro.
Implements the VPIN metric as described in Easley, López de Prado, and O'Hara (2012).
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from collections import deque


@dataclass
class VPIN:
    """
    Computes VPIN (Volume-Synchronized Probability of Informed Trading).
    
    VPIN is calculated as:
    VPIN = |Sum_{i=1 to N} (V_i^b - V_i^a)| / Sum_{i=1 to N} (V_i^b + V_i^a)
    
    Where:
    - V_i^b: Buy volume in bucket i
    - V_i^a: Sell volume in bucket i
    - N: Number of volume buckets
    """
    bucket_size: int = 100  # Number of trades per bucket
    
    def compute_vpin(
        self,
        buy_volumes: Union[List[float], np.ndarray, pd.Series],
        sell_volumes: Union[List[float], np.ndarray, pd.Series],
    ) -> float:
        """
        Compute VPIN from buy and sell volumes.
        
        Args:
            buy_volumes: List or array of buy volumes (per trade).
            sell_volumes: List or array of sell volumes (per trade).
        
        Returns:
            VPIN value.
        """
        if isinstance(buy_volumes, pd.Series):
            buy_volumes = buy_volumes.values
        if isinstance(sell_volumes, pd.Series):
            sell_volumes = sell_volumes.values
        
        buy_volumes = np.asarray(buy_volumes)
        sell_volumes = np.asarray(sell_volumes)
        
        if len(buy_volumes) != len(sell_volumes):
            raise ValueError("buy_volumes and sell_volumes must have the same length")
        
        # Split into buckets
        n_buckets = len(buy_volumes) // self.bucket_size
        if n_buckets == 0:
            return 0.0
        
        buy_buckets = np.sum(buy_volumes[:n_buckets * self.bucket_size].reshape(n_buckets, self.bucket_size), axis=1)
        sell_buckets = np.sum(sell_volumes[:n_buckets * self.bucket_size].reshape(n_buckets, self.bucket_size), axis=1)
        
        # Compute VPIN
        numerator = np.abs(np.sum(buy_buckets - sell_buckets))
        denominator = np.sum(buy_buckets + sell_buckets)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def compute_vpin_from_trades(self, trades: List[Dict]) -> float:
        """
        Compute VPIN directly from a list of trades.
        
        Args:
            trades: List of trade dicts with keys: "side", "quantity".
        
        Returns:
            VPIN value.
        """
        buy_volumes = []
        sell_volumes = []
        
        for trade in trades:
            if trade["side"] == "BUY":
                buy_volumes.append(trade["quantity"])
                sell_volumes.append(0)
            else:
                buy_volumes.append(0)
                sell_volumes.append(trade["quantity"])
        
        return self.compute_vpin(buy_volumes, sell_volumes)
    
    def rolling_vpin(
        self,
        buy_volumes: Union[List[float], np.ndarray, pd.Series],
        sell_volumes: Union[List[float], np.ndarray, pd.Series],
        window_size: int = 1000,
    ) -> np.ndarray:
        """
        Compute rolling VPIN over a sliding window.
        
        Args:
            buy_volumes: List or array of buy volumes.
            sell_volumes: List or array of sell volumes.
            window_size: Size of the rolling window (number of trades).
        
        Returns:
            Array of rolling VPIN values.
        """
        if isinstance(buy_volumes, pd.Series):
            buy_volumes = buy_volumes.values
        if isinstance(sell_volumes, pd.Series):
            sell_volumes = sell_volumes.values
        
        buy_volumes = np.asarray(buy_volumes)
        sell_volumes = np.asarray(sell_volumes)
        
        if len(buy_volumes) != len(sell_volumes):
            raise ValueError("buy_volumes and sell_volumes must have the same length")
        
        n = len(buy_volumes)
        rolling_vpins = np.zeros(n - window_size + 1)
        
        for i in range(n - window_size + 1):
            window_buy = buy_volumes[i:i + window_size]
            window_sell = sell_volumes[i:i + window_size]
            rolling_vpins[i] = self.compute_vpin(window_buy, window_sell)
        
        return rolling_vpins