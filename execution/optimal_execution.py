"""Q-Micro :: execution.optimal_execution — Almgren-Chriss optimal execution trajectory."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


@dataclass
class AlmgrenChrissParams:
    volatility: float          # sigma, per-period
    temp_impact: float         # eta (temporary impact coefficient)
    perm_impact: float         # gamma (permanent impact coefficient)
    risk_aversion: float       # lambda (investor risk aversion)


def optimal_trajectory(total_qty: float, n_periods: int, params: AlmgrenChrissParams) -> List[float]:
    """
    Returns the remaining-inventory trajectory x_0..x_n (length n_periods+1)
    that minimizes E[Cost] + risk_aversion * Var[Cost] under Almgren-Chriss (2000).
    """
    if np is None:
        raise ImportError("numpy is required for optimal_trajectory.")
    kappa_sq = (params.risk_aversion * params.volatility ** 2) / max(params.temp_impact, 1e-12)
    kappa = kappa_sq ** 0.5

    tau = 1.0  # normalized period length
    if kappa * tau < 1e-6:
        # risk-neutral limit -> linear (TWAP-like) trajectory
        return [total_qty * (1 - k / n_periods) for k in range(n_periods + 1)]

    trajectory = []
    denom = np.sinh(kappa * n_periods * tau)
    for k in range(n_periods + 1):
        remaining = total_qty * np.sinh(kappa * (n_periods - k) * tau) / denom
        trajectory.append(float(remaining))
    return trajectory


def trade_list(trajectory: List[float]) -> List[float]:
    """Converts an inventory trajectory into per-period trade sizes."""
    return [trajectory[i] - trajectory[i + 1] for i in range(len(trajectory) - 1)]


def expected_cost(trajectory: List[float], params: AlmgrenChrissParams) -> float:
    trades = trade_list(trajectory)
    temp_cost = sum(params.temp_impact * (q ** 2) for q in trades)
    perm_cost = sum(params.perm_impact * q * x for q, x in zip(trades, trajectory[:-1]))
    return temp_cost + perm_cost