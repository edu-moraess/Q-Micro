"""
Spread Model for Q-Micro.
Estimates bid-ask spread dynamics based on volatility, liquidity, and order flow.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SpreadModel:
    """
    Models the bid-ask spread as a function of:
    - Volatility (sigma)
    - Liquidity (order book depth)
    - Order flow imbalance
    
    The spread is estimated as:
    spread = base_spread + alpha * volatility + beta * (1 / liquidity) + gamma * |order_flow_imbalance|
    """
    base_spread: float = 0.001  # 0.1% base spread
    alpha: float = 0.5         # Volatility weight
    beta: float = 0.3          # Illiquidity weight
    gamma: float = 0.2         # Order flow imbalance weight
    
    def estimate_spread(
        self,
        volatility: float,
        liquidity: float,
        order_flow_imbalance: float,
    ) -> float:
        """
        Estimate the spread based on volatility, liquidity, and order flow.
        
        Args:
            volatility: Standard deviation of returns (daily).
            liquidity: Total volume at best bid/ask (normalized).
            order_flow_imbalance: (Buy Volume - Sell Volume) / Total Volume.
        
        Returns:
            Estimated spread (as a fraction of mid-price).
        """
        if liquidity <= 0:
            liquidity = 1e-6  # Avoid division by zero
        
        spread = (
            self.base_spread +
            self.alpha * volatility +
            self.beta * (1 / liquidity) +
            self.gamma * abs(order_flow_imbalance)
        )
        return spread
    
    def fit(
        self,
        spreads: np.ndarray,
        volatilities: np.ndarray,
        liquidities: np.ndarray,
        ofis: np.ndarray,
    ) -> None:
        """
        Fit the spread model parameters using linear regression.
        
        Args:
            spreads: Observed spreads (target).
            volatilities: Observed volatilities.
            liquidities: Observed liquidities.
            ofis: Observed order flow imbalances.
        """
        from sklearn.linear_model import LinearRegression
        
        X = np.column_stack([
            np.ones(len(volatilities)),
            volatilities,
            1 / np.maximum(liquidities, 1e-6),
            np.abs(ofis),
        ])
        y = spreads
        
        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)
        
        self.base_spread = model.coef_[0]
        self.alpha = model.coef_[1]
        self.beta = model.coef_[2]
        self.gamma = model.coef_[3]
    
    def get_parameters(self) -> Dict[str, float]:
        """Return the current model parameters."""
        return {
            "base_spread": self.base_spread,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
        }
