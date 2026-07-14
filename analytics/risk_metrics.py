
"""Q-Micro :: analytics.risk_metrics — VaR, drawdown, volatility."""

from __future__ import annotations
from typing import List


def historical_var(returns: List[float], confidence: float = 0.95) -> float:
    sorted_r = sorted(returns)
    idx = int((1 - confidence) * len(sorted_r))
    return sorted_r[max(idx, 0)]


def max_drawdown(equity_curve: List[float]) -> float:
    peak = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        peak = max(peak, v)
        dd = (v - peak) / peak if peak != 0 else 0.0
        max_dd = min(max_dd, dd)
    return max_dd


def realized_volatility(returns: List[float], periods_per_year: int = 252) -> float:
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    return (var ** 0.5) * (periods_per_year ** 0.5)