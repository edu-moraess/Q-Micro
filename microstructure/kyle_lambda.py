"""Q-Micro :: microstructure.kyle_lambda — price impact coefficient (Kyle 1985)."""

from __future__ import annotations
from typing import List, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def estimate_kyle_lambda(price_changes: List[float], signed_volumes: List[float]) -> float:
    """
    OLS estimate of lambda in: delta_price = lambda * signed_volume + epsilon.
    signed_volume = buy_volume - sell_volume over the same interval as delta_price.
    """
    if np is None:
        raise ImportError("numpy is required for estimate_kyle_lambda.")
    x = np.asarray(signed_volumes, dtype=float)
    y = np.asarray(price_changes, dtype=float)
    if x.size < 2 or np.allclose(x.var(), 0.0):
        return 0.0
    beta = np.cov(x, y, bias=True)[0, 1] / x.var()
    return float(beta)


def price_impact(lam: float, order_flow: float) -> float:
    """PriceImpact = lambda * OrderFlow."""
    return lam * order_flow


def rolling_kyle_lambda(price_changes: List[float], signed_volumes: List[float],
                         window: int = 50) -> List[float]:
    if np is None:
        raise ImportError("numpy is required for rolling_kyle_lambda.")
    out = []
    for i in range(len(price_changes)):
        lo = max(0, i - window + 1)
        out.append(estimate_kyle_lambda(price_changes[lo:i + 1], signed_volumes[lo:i + 1]))
    return out