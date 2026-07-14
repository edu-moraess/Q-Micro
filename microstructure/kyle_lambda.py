"""
Kyle's Lambda for Q-Micro.
Estimates the price impact of order flow using Kyle's Lambda model.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression


@dataclass
class KyleLambda:
    """
    Implements Kyle's Lambda model:
    PriceImpact = lambda * OrderFlow + epsilon
    
    Where:
    - OrderFlow = (Buy Volume - Sell Volume) / Total Volume
    - lambda: Market depth parameter (slope of the impact curve)
    """
    lambda_: float = 0.01  # Default market depth
    
    def estimate_impact(self, order_flow: float) -> float:
        """
        Estimate price impact from order flow.
        
        Args:
            order_flow: Net order flow (normalized).
        
        Returns:
            Estimated price impact.
        """
        return self.lambda_ * order_flow
    
    def fit(
        self,
        order_flows: Union[np.ndarray, pd.Series],
        price_impacts: Union[np.ndarray, pd.Series],
    ) -> None:
        """
        Fit Kyle's Lambda using linear regression.
        
        Args:
            order_flows: Array or Series of order flows.
            price_impacts: Array or Series of observed price impacts.
        """
        if isinstance(order_flows, pd.Series):
            order_flows = order_flows.values
        if isinstance(price_impacts, pd.Series):
            price_impacts = price_impacts.values
        
        X = order_flows.reshape(-1, 1)
        y = price_impacts
        
        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)
        self.lambda_ = model.coef_[0]
    
    def get_lambda(self) -> float:
        """Return the current lambda parameter."""
        return self.lambda_
