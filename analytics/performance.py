"""Q-Micro :: analytics.performance — PnL, Sharpe, slippage."""

from __future__ import annotations
from typing import List


def pnl_series(prices: List[float], positions: List[float]) -> List[float]:
    """positions[t] is the held position over (t, t+1]."""
    return [positions[i] * (prices[i + 1] - prices[i]) for i in range(len(prices) - 1)]


def sharpe_ratio(returns: List[float], periods_per_year: int = 252) -> float:
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    std = var ** 0.5
    if std == 0:
        return 0.0
    return (mean / std) * (periods_per_year ** 0.5)


def slippage_bps(decision_price: float, avg_fill_price: float, side_sign: int) -> float:
    return 10_000.0 * side_sign * (avg_fill_price - decision_price) / decision_price