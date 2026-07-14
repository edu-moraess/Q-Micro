
"""
TWAP (Time-Weighted Average Price) Execution Algorithm for Q-Micro.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class TWAP:
    """
    Implements the TWAP execution algorithm.
    
    TWAP divides a large order into equal-sized slices over a specified time horizon.
    The goal is to minimize market impact by spreading the order evenly in time.
    
    Attributes:
        total_quantity: Total quantity to execute.
        start_time: Start time of the execution.
        end_time: End time of the execution.
        n_slices: Number of slices to divide the order into.
    """
    total_quantity: int
    start_time: datetime
    end_time: datetime
    n_slices: int = 10
    
    def __post_init__(self):
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be after start_time")
        if self.n_slices <= 0:
            raise ValueError("n_slices must be positive")
    
    def get_slice_times(self) -> List[datetime]:
        """Return the timestamps for each execution slice."""
        delta = (self.end_time - self.start_time) / self.n_slices
        return [self.start_time + i * delta for i in range(1, self.n_slices + 1)]
    
    def get_slice_quantity(self) -> int:
        """Return the quantity for each slice."""
        return self.total_quantity // self.n_slices
    
    def get_execution_plan(self) -> List[Dict]:
        """
        Generate the full execution plan.
        
        Returns:
            List of dicts with keys: "time", "quantity", "side".
        """
        slice_quantity = self.get_slice_quantity()
        slice_times = self.get_slice_times()
        
        execution_plan = []
        remaining_quantity = self.total_quantity
        
        for i, time in enumerate(slice_times):
            if i == self.n_slices - 1:
                # Last slice: execute remaining quantity
                quantity = remaining_quantity
            else:
                quantity = slice_quantity
            
            execution_plan.append({
                "time": time,
                "quantity": quantity,
                "side": "BUY",  # Default; can be overridden
            })
            remaining_quantity -= quantity
        
        return execution_plan
    
    def execute(
        self,
        exchange_simulator,
        side: str = "BUY",
        price: Optional[float] = None,
    ) -> List[Dict]:
        """
        Execute the TWAP strategy on a given exchange simulator.
        
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
                # Market order
                trade = exchange_simulator.matching_engine.process_market_order(
                    side=side_enum,
                    quantity=slice["quantity"],
                    trader_id="TWAP_AGENT",
                )
            else:
                # Limit order
                trade = exchange_simulator.matching_engine.process_limit_order(
                    side=side_enum,
                    price=price,
                    quantity=slice["quantity"],
                    trader_id="TWAP_AGENT",
                )
            trades.extend(trade)
        
        return trades