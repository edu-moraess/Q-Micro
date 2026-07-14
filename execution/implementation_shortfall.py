
"""
Implementation Shortfall Execution Algorithm for Q-Micro.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy.optimize import minimize


@dataclass
class ImplementationShortfall:
    """
    Implements the Implementation Shortfall execution algorithm.
    
    Implementation Shortfall minimizes the total execution cost, which includes:
    - Execution cost (difference between execution price and decision price)
    - Market impact (price movement due to our trading)
    - Timing risk (opportunity cost of not executing immediately)
    
    The algorithm uses optimization to find the optimal execution schedule.
    
    Attributes:
        total_quantity: Total quantity to execute.
        start_time: Start time of the execution.
        end_time: End time of the execution.
        decision_price: Price at which the decision to trade was made.
        lambda_: Market impact parameter (from Kyle's Lambda or PriceImpactModel).
        sigma: Volatility of the asset (for timing risk).
        risk_aversion: Risk aversion parameter (higher = more aggressive execution).
    """
    total_quantity: int
    start_time: datetime
    end_time: datetime
    decision_price: float
    lambda_: float = 0.01  # Market impact parameter
    sigma: float = 0.02     # Daily volatility
    risk_aversion: float = 0.5
    n_slices: int = 10
    
    def __post_init__(self):
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be after start_time")
        if self.n_slices <= 0:
            raise ValueError("n_slices must be positive")
    
    def _compute_execution_cost(
        self,
        quantities: np.ndarray,
        prices: np.ndarray,
    ) -> float:
        """
        Compute the execution cost for a given set of quantities and prices.
        
        Args:
            quantities: Array of quantities executed in each slice.
            prices: Array of execution prices in each slice.
        
        Returns:
            Total execution cost.
        """
        return np.sum((prices - self.decision_price) * quantities)
    
    def _compute_market_impact(
        self,
        quantities: np.ndarray,
    ) -> float:
        """
        Compute the market impact cost for a given set of quantities.
        
        Args:
            quantities: Array of quantities executed in each slice.
        
        Returns:
            Total market impact cost.
        """
        # Market impact: lambda * (cumulative quantity)^2
        cumulative_quantity = np.cumsum(quantities)
        return self.lambda_ * np.sum(cumulative_quantity ** 2)
    
    def _compute_timing_risk(
        self,
        quantities: np.ndarray,
        time_fracs: np.ndarray,
    ) -> float:
        """
        Compute the timing risk (opportunity cost) for a given execution schedule.
        
        Args:
            quantities: Array of quantities executed in each slice.
            time_fracs: Array of time fractions (0 to 1) for each slice.
        
        Returns:
            Total timing risk.
        """
        # Timing risk: 0.5 * sigma^2 * T * (1 - time_frac)^2 * (remaining_quantity)^2
        T = (self.end_time - self.start_time).total_seconds() / 86400  # Convert to days
        remaining_quantity = self.total_quantity - np.cumsum(quantities)
        timing_risk = 0.5 * (self.sigma ** 2) * T * np.sum(
            (1 - time_fracs) ** 2 * (remaining_quantity ** 2)
        )
        return timing_risk
    
    def _total_cost(self, quantities: np.ndarray) -> float:
        """
        Compute the total cost (execution + market impact + timing risk).
        
        Args:
            quantities: Array of quantities executed in each slice.
        
        Returns:
            Total cost.
        """
        # For simplicity, assume prices are constant (decision_price + impact)
        # In practice, prices would be estimated from the market
        time_fracs = np.linspace(0, 1, self.n_slices + 1)[1:]  # Time fractions for each slice
        
        # Execution cost: assume prices move linearly with cumulative quantity
        cumulative_quantity = np.cumsum(quantities)
        prices = self.decision_price + self.lambda_ * cumulative_quantity
        execution_cost = self._compute_execution_cost(quantities, prices)
        
        market_impact = self._compute_market_impact(quantities)
        timing_risk = self._compute_timing_risk(quantities, time_fracs)
        
        total_cost = execution_cost + market_impact + self.risk_aversion * timing_risk
        return total_cost
    
    def optimize_execution(self) -> np.ndarray:
        """
        Find the optimal execution schedule using optimization.
        
        Returns:
            Array of optimal quantities for each slice.
        """
        # Initial guess: equal slices
        x0 = np.ones(self.n_slices) * (self.total_quantity / self.n_slices)
        
        # Constraints: sum(quantities) = total_quantity, quantities >= 0
        constraints = (
            {"type": "eq", "fun": lambda x: np.sum(x) - self.total_quantity},
        )
        bounds = [(0, self.total_quantity) for _ in range(self.n_slices)]
        
        result = minimize(
            self._total_cost,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        
        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")
        
        return result.x
    
    def get_execution_plan(self) -> List[Dict]:
        """
        Generate the full execution plan.
        
        Returns:
            List of dicts with keys: "time", "quantity", "side".
        """
        optimal_quantities = self.optimize_execution()
        slice_times = self.get_slice_times()
        
        execution_plan = []
        for time, quantity in zip(slice_times, optimal_quantities):
            execution_plan.append({
                "time": time,
                "quantity": int(round(quantity)),
                "side": "BUY",  # Default; can be overridden
            })
        
        return execution_plan
    
    def get_slice_times(self) -> List[datetime]:
        """Return the timestamps for each execution slice."""
        delta = (self.end_time - self.start_time) / self.n_slices
        return [self.start_time + i * delta for i in range(1, self.n_slices + 1)]
    
    def execute(
        self,
        exchange_simulator,
        side: str = "BUY",
        price: Optional[float] = None,
    ) -> List[Dict]:
        """
        Execute the Implementation Shortfall strategy on a given exchange simulator.
        
        Args:
            exchange_simulator: Instance of ExchangeSimulator.
            side: "BUY" or "SELL".
            price: Limit price (for LIMIT orders; None for MARKET orders).
        
        Returns:
            List of trade executions.
        """
        from core.order import OrderSide, OrderType
        
        side_enum = OrderSide.BUY if side == "BUY" else OrderSide.SELL
        execution_plan = self.get_execution_plan()
        trades = []
        
        for slice in execution_plan:
            if price is None:
                trade = exchange_simulator.matching_engine.process_market_order(
                    side=side_enum,
                    quantity=slice["quantity"],
                    trader_id="IS_AGENT",
                )
            else:
                trade = exchange_simulator.matching_engine.process_limit_order(
                    side=side_enum,
                    price=price,
                    quantity=slice["quantity"],
                    trader_id="IS_AGENT",
                )
            trades.extend(trade)
        
        return trades