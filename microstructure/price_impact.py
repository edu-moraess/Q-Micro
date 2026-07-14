
"""
Price Impact Model for Q-Micro.
Implements linear and nonlinear price impact models.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression


@dataclass
class PriceImpactModel:
    """
    Models price impact as a function of order size and market conditions.
    
    Supports:
    - Linear impact: PriceImpact = lambda * OrderSize
    - Square-root impact: PriceImpact = lambda * sqrt(OrderSize)
    - Logarithmic impact: PriceImpact = lambda * log(1 + OrderSize)
    """
    impact_type: str = "linear"  # "linear", "sqrt", or "log"
    lambda_: float = 0.001      # Market depth parameter
    
    def compute_impact(self, order_size: float, volume: float) -> float:
        """
        Compute price impact for a given order size.
        
        Args:
            order_size: Size of the order (in shares).
            volume: Average daily volume (for normalization).
        
        Returns:
            Price impact (as a fraction of price).
        """
        normalized_size = order_size / max(volume, 1e-6)
        
        if self.impact_type == "linear":
            impact = self.lambda_ * normalized_size
        elif self.impact_type == "sqrt":
            impact = self.lambda_ * np.sqrt(normalized_size)
        elif self.impact_type == "log":
            impact = self.lambda_ * np.log(1 + normalized_size)
        else:
            raise ValueError(f"Unknown impact type: {self.impact_type}")
        
        return impact
    
    def fit(
        self,
        order_sizes: np.ndarray,
        impacts: np.ndarray,
        volumes: Optional[np.ndarray] = None,
        impact_type: str = "linear",
    ) -> None:
        """
        Fit the price impact model to observed data.
        
        Args:
            order_sizes: Array of order sizes.
            impacts: Array of observed price impacts.
            volumes: Array of average daily volumes (optional).
            impact_type: Type of impact model to fit.
        """
        self.impact_type = impact_type
        
        if volumes is None:
            volumes = np.ones_like(order_sizes)
        
        normalized_sizes = order_sizes / np.maximum(volumes, 1e-6)
        
        if impact_type == "linear":
            X = normalized_sizes.reshape(-1, 1)
        elif impact_type == "sqrt":
            X = np.sqrt(normalized_sizes).reshape(-1, 1)
        elif impact_type == "log":
            X = np.log(1 + normalized_sizes).reshape(-1, 1)
        else:
            raise ValueError(f"Unknown impact type: {impact_type}")
        
        model = LinearRegression(fit_intercept=False)
        model.fit(X, impacts)
        self.lambda_ = model.coef_[0]
    
    def get_parameters(self) -> Dict[str, Union[str, float]]:
        """Return the current model parameters."""
        return {
            "impact_type": self.impact_type,
            "lambda": self.lambda_,
        }