"""
Liquidity Metrics for Q-Micro.
Implements Amihud Illiquidity, Order Flow Imbalance, and other liquidity measures.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
from dataclasses import dataclass


@dataclass
class LiquidityMetrics:
    """
    Computes various liquidity metrics from trade and order book data.
    """
    
    @staticmethod
    def compute_amihud_illiquidity(
        returns: Union[np.ndarray, pd.Series],
        volumes: Union[np.ndarray, pd.Series],
    ) -> Union[np.ndarray, pd.Series]:
        """
        Compute Amihud (2002) illiquidity measure:
        ILLIQ = |Return| / Volume
        
        Args:
            returns: Array or Series of asset returns.
            volumes: Array or Series of trading volumes.
        
        Returns:
            Amihud illiquidity measure.
        """
        if isinstance(returns, pd.Series):
            returns = returns.values
        if isinstance(volumes, pd.Series):
            volumes = volumes.values
        
        illiquidity = np.abs(returns) / np.maximum(volumes, 1e-6)
        return illiquidity
    
    @staticmethod
    def compute_order_flow_imbalance(
        buy_volumes: Union[np.ndarray, pd.Series],
        sell_volumes: Union[np.ndarray, pd.Series],
    ) -> Union[np.ndarray, pd.Series]:
        """
        Compute Order Flow Imbalance (OFI):
        OFI = (Buy Volume - Sell Volume) / (Buy Volume + Sell Volume)
        
        Args:
            buy_volumes: Array or Series of buy volumes.
            sell_volumes: Array or Series of sell volumes.
        
        Returns:
            Order Flow Imbalance.
        """
        if isinstance(buy_volumes, pd.Series):
            buy_volumes = buy_volumes.values
        if isinstance(sell_volumes, pd.Series):
            sell_volumes = sell_volumes.values
        
        total_volume = buy_volumes + sell_volumes
        ofi = np.divide(
            buy_volumes - sell_volumes,
            np.maximum(total_volume, 1e-6),
        )
        return ofi
    
    @staticmethod
    def compute_bid_ask_spread(
        bid_prices: Union[np.ndarray, pd.Series],
        ask_prices: Union[np.ndarray, pd.Series],
    ) -> Union[np.ndarray, pd.Series]:
        """
        Compute bid-ask spread as a fraction of mid-price.
        
        Args:
            bid_prices: Array or Series of bid prices.
            ask_prices: Array or Series of ask prices.
        
        Returns:
            Bid-ask spread (as a fraction of mid-price).
        """
        if isinstance(bid_prices, pd.Series):
            bid_prices = bid_prices.values
        if isinstance(ask_prices, pd.Series):
            ask_prices = ask_prices.values
        
        mid_price = (bid_prices + ask_prices) / 2
        spread = (ask_prices - bid_prices) / np.maximum(mid_price, 1e-6)
        return spread
    
    @staticmethod
    def compute_depth(
        bid_volumes: Union[np.ndarray, pd.Series],
        ask_volumes: Union[np.ndarray, pd.Series],
        levels: int = 5,
    ) -> Dict[str, Union[np.ndarray, pd.Series]]:
        """
        Compute order book depth (total volume at each price level).
        
        Args:
            bid_volumes: 2D array or DataFrame of bid volumes (price levels x time).
            ask_volumes: 2D array or DataFrame of ask volumes (price levels x time).
            levels: Number of price levels to consider.
        
        Returns:
            Dict with bid_depth and ask_depth.
        """
        if isinstance(bid_volumes, pd.DataFrame):
            bid_volumes = bid_volumes.values
        if isinstance(ask_volumes, pd.DataFrame):
            ask_volumes = ask_volumes.values
        
        bid_depth = np.sum(bid_volumes[:levels, :], axis=0)
        ask_depth = np.sum(ask_volumes[:levels, :], axis=0)
        
        return {"bid_depth": bid_depth, "ask_depth": ask_depth}


# Aliases for convenience
AmihudIlliquidity = LiquidityMetrics.compute_amihud_illiquidity
OrderFlowImbalance = LiquidityMetrics.compute_order_flow_imbalance
