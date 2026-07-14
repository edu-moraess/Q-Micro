"""
VWAP (Volume-Weighted Average Price) Execution Algorithm for Q-Micro.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class VWAP:
    """
    Implements the VWAP execution algorithm.
    
    VWAP divides a large order into slices proportional to the expected volume
    in each time interval. The goal is to match the volume profile of the market.
    
    Attributes:
        total_quantity: Total quantity to execute.
        start_time: Start time of the execution.
        end_time: End time of the execution.
        volume_profile: Expected volume profile (list of volumes per interval).
    """
    total_quantity: int
    start_time: datetime
    end_time: datetime
    volume_profile: List[float]  # Expected volume in each interval
    
    def __post_init__(self):
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be after start_time")
        if len(self.volume_profile) == 0:
            raise ValueError("volume_profile must not be empty")
        if any(v < 0 for v in self.volume_profile):
            raise ValueError("volume_profile values must be non-negative")
    
    def get_slice_times(self) -> List[datetime]:
        """Return the timestamps for each execution slice."""
        n_slices = len(self.volume_profile)
        delta = (self.end_time - self.start_time) / n_slices
        return [self.start_time + i * delta for i in range(1, n_slices + 1)]
    
    def get_slice_quantities(self) -> List[int]:
        """
        Compute the quantity for each slice, proportional to the volume profile.
        
        Returns:
            List of quantities for each slice.
        """
        total_volume = sum(self.volume_profile)
        if total_volume == 0:
            return [self.total_quantity // len(self.volume_profile)] * len(self.volume_profile)
        
        weights = np.array(self.volume_profile) / total_volume
        quantities = (weights * self.total_quantity).astype(int)
        
        # Adjust for rounding errors
        remaining = self.total_quantity - sum(quantities)
        if remaining > 0:
            quantities[-1] += remaining
        elif remaining < 0:
            quantities[-1] += remaining  # Will be negative; adjust last slice
        
        return quantities.tolist()
    
    def get_execution_plan(self) -> List[Dict]:
        """
        Generate the full execution plan.
        
        Returns:
            List of dicts with keys: "time", "quantity", "side".
        """
        slice_times = self.get_slice_times()
        slice_quantities = self.get_slice_quantities()
        
        execution_plan = []
        for time, quantity in zip(slice_times, slice_quantities):
            execution_plan.append({
                "time": time,
                "quantity": quantity,
                "side": "BUY",  # Default; can be overridden
            })
        
        return execution_plan
    
    def execute(
        self,
        exchange_simulator,
        side: str = "BUY",
        price: Optional[float] = None,
    ) -> List[Dict]:
        """
        Execute the VWAP strategy on a given exchange simulator.
        
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
                    trader_id="VWAP_AGENT",
                )
            else:
                trade = exchange_simulator.matching_engine.process_limit_order(
                    side=side_enum,
                    price=price,
                    quantity=slice["quantity"],
                    trader_id="VWAP_AGENT",
                )
            trades.extend(trade)
        
        return trades