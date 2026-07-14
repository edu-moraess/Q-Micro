"""Q-Micro :: execution.implementation_shortfall — Perold (1988) IS decomposition."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class ISResult:
    execution_cost: float     # (avg_fill_price - decision_price) * signed_qty
    market_impact: float      # portion attributable to price moving while executing
    timing_risk: float        # opportunity cost from delay/non-execution
    total_cost: float


def implementation_shortfall(decision_price: float, avg_fill_price: float,
                              side_sign: int, filled_qty: float, target_qty: float,
                              final_price: float) -> ISResult:
    """
    side_sign: +1 for BUY, -1 for SELL.
    Cost = ExecutionCost + MarketImpact + TimingRisk (Perold decomposition, simplified).
    """
    execution_cost = side_sign * (avg_fill_price - decision_price) * filled_qty
    unfilled = target_qty - filled_qty
    timing_risk = side_sign * (final_price - decision_price) * unfilled
    # Market impact approximated as the execution_cost net of pure drift
    drift_component = side_sign * (final_price - decision_price) * filled_qty
    market_impact = execution_cost - drift_component

    total = execution_cost + timing_risk
    return ISResult(execution_cost=execution_cost, market_impact=market_impact,
                     timing_risk=timing_risk, total_cost=total)


def shortfall_bps(is_result: ISResult, decision_price: float, target_qty: float) -> float:
    notional = decision_price * target_qty
    return 10_000.0 * is_result.total_cost / notional if notional else 0.0