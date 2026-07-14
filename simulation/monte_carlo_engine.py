
"""Q-Micro :: simulation.monte_carlo_engine — multi-path Monte Carlo over MarketSimulator."""

from __future__ import annotations
from typing import Callable, List
from simulation.market_simulator import MarketSimulator


def run_monte_carlo(n_paths: int, n_steps: int, start_price: float = 100.0,
                     metric_fn: Callable[[MarketSimulator], float] = None) -> List[float]:
    """
    Runs n_paths independent MarketSimulator instances (different seeds) and
    collects a scalar metric per path (e.g. final mid, realized spread, PnL).
    """
    if metric_fn is None:
        metric_fn = lambda sim: sim.history[-1]["mid"] if sim.history else start_price

    results = []
    for path in range(n_paths):
        sim = MarketSimulator(start_price=start_price, seed=1000 + path)
        sim.run(n_steps)
        results.append(metric_fn(sim))
    return results


def summary_stats(values: List[float]) -> dict:
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / max(n - 1, 1)
    sorted_v = sorted(values)
    return {
        "mean": mean,
        "std": var ** 0.5,
        "min": sorted_v[0],
        "max": sorted_v[-1],
        "p5": sorted_v[int(0.05 * n)],
        "p95": sorted_v[int(0.95 * n)],
    }