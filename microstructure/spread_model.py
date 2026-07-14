"""Q-Micro :: microstructure.spread_model — Spread = f(volatility, liquidity, order flow)."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class SpreadModelParams:
    alpha: float = 0.01      # base spread
    beta_vol: float = 2.5    # sensitivity to volatility
    beta_liq: float = -0.8   # sensitivity to depth (more depth -> tighter spread)
    beta_flow: float = 1.2   # sensitivity to |order flow imbalance|


def estimate_spread(volatility: float, depth: float, order_flow_imbalance: float,
                     params: SpreadModelParams = SpreadModelParams()) -> float:
    """
    Reduced-form spread estimate. `depth` should be total resting size near
    the touch; `order_flow_imbalance` in [-1, 1] from OrderBook.order_flow_imbalance().
    """
    depth_term = params.beta_liq * (1.0 / max(depth, 1.0))
    spread = (
        params.alpha
        + params.beta_vol * volatility
        + depth_term
        + params.beta_flow * abs(order_flow_imbalance)
    )
    return max(spread, 0.0001)