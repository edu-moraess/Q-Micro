
"""
Optimal Execution Algorithm for Q-Micro.
Implements Almgren-Chriss and other advanced optimal execution models.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy.optimize import minimize


@dataclass
class OptimalExecution:
    """
    Implements the Almgren-Chriss optimal execution model.
    
    The model minimizes the total execution cost, which includes:
    - Permanent market impact (linear or nonlinear)
    - Temporary market impact (price walk)
    - Volatility risk (variance of execution price)
    
    The optimal execution schedule is derived analytically for the linear case,
    or numerically for nonlinear cases.
    
    Attributes:
        total_quantity: Total quantity to execute.
        start_time: Start time of the execution.
        end_time: End time of the execution.
        decision_price: Price at which the decision to trade was made.
        sigma: Volatility of the asset.
        lambda_: Permanent market impact parameter.
        eta: Temporary market impact parameter.
        risk_aversion: Risk aversion parameter.
    """
    total_quantity: int
    start_time: datetime
    end_time: datetime
    decision_price: float
    sigma: float = 0.02      # Daily volatility
    lambda_: float = 0.01   # Permanent market impact
    eta: float = 0.005      # Temporary market impact
    risk_aversion: float = 0.5
    n_slices: int = 10
    
    def __post_init__(self):
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be after start_time")
        if self.n_slices <= 0:
            raise ValueError("n_slices must be positive")
    
    def _almgren_chriss_analytical(self) -> np.ndarray:
        """
        Compute the optimal execution schedule using the Almgren-Chriss analytical solution.
        
        For the linear case, the optimal execution rate is:
        x*(t) = (Q / T) + (sigma^2 / (2 * lambda_)) * (T - 2t) * (Q / T^2)
        
        Returns:
            Array of optimal quantities for each slice.
        """
        T = (self.end_time - self.start_time).total_seconds() / 86400  # Convert to days
        Q = self.total_quantity
        
        # Time points (as fractions of T)
        t = np.linspace(0, T, self.n_slices + 1)[1:]  # Exclude t=0
        
        # Optimal execution rate at each time t
        x_dot = (Q / T) + (self.sigma ** 2 / (2 * self.lambda_)) * (T - 2 * t) * (Q / T ** 2)
        
        # Quantities for each slice (integral of x_dot over the interval)
        dt = T / self.n_slices
        quantities = x_dot * dt
        
        # Adjust for rounding errors
        quantities = np.round(quantities).astype(int)
        remaining = Q - np.sum(quantities)
        quantities[-1] += remaining  # Add remaining to last slice
        
        return quantities
    
    def _total_cost(self, quantities: np.ndarray) -> float:
        """
        Compute the total cost for the Almgren-Chriss model.
        
        Args:
            quantities: Array of quantities executed in each slice.
        
        Returns:
            Total cost.
        """
        T = (self.end_time - self.start_time).total_seconds() / 86400
        dt = T / self.n_slices
        
        # Permanent impact: lambda * sum(cumulative_quantity)
        cumulative_quantity = np.cumsum(quantities)
        permanent_impact = self.lambda_ * np.sum(cumulative_quantity * dt)
        
        # Temporary impact: eta * sum(quantities^2)
        temporary_impact = self.eta * np.sum(quantities ** 2)
        
        # Volatility risk: sigma^2 * risk_aversion * sum((remaining_quantity * dt)^2)
        remaining_quantity = self.total_quantity - cumulative_quantity
        volatility_risk = (
            self.sigma ** 2 * self.risk_aversion * np.sum(remaining_quantity ** 2 * dt)
        )
        
        total_cost = permanent_impact + temporary_impact + volatility_risk
        return total_cost
    
    def optimize_execution(self) -> np.ndarray:
        """
        Find the optimal execution schedule using numerical optimization.
        
        Returns:
            Array of optimal quantities for each slice.
        """
        # Initial guess: Almgren-Chriss analytical solution
        x0 = self._almgren_chriss_analytical()
        
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
        Execute the Optimal Execution strategy on a given exchange simulator.
        
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
                    trader_id="OE_AGENT",
                )
            else:
                trade = exchange_simulator.matching_engine.process_limit_order(
                    side=side_enum,
                    price=price,
                    quantity=slice["quantity"],
                    trader_id="OE_AGENT",
                )
            trades.extend(trade)
        
        return trades