"""Q-Micro :: microstructure.price_impact — linear and square-root impact models."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ImpactParams:
    linear_coef: float = 0.0001
    sqrt_coef: float = 0.01
    daily_volume: float = 1_000_000.0


def linear_impact(order_size: float, params: ImpactParams) -> float:
    """Temporary impact ~ linear in participation rate."""
    return params.linear_coef * (order_size / params.daily_volume)


def sqrt_impact(order_size: float, volatility: float, params: ImpactParams) -> float:
    """
    Square-root law (Almgren et al.): impact ~ sigma * sqrt(order_size / ADV).
    Common empirical fit for large institutional orders.
    """
    participation = order_size / params.daily_volume
    return params.sqrt_coef * volatility * (participation ** 0.5)


def total_impact_cost(order_size: float, avg_price: float, volatility: float,
                       params: ImpactParams) -> float:
    """Expected $ cost from temporary impact over the life of the order."""
    impact_frac = sqrt_impact(order_size, volatility, params)
    return impact_frac * avg_price * order_size