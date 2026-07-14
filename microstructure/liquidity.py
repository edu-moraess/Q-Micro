"""Q-Micro :: microstructure.liquidity — Amihud illiquidity and depth-based measures."""

from __future__ import annotations
from typing import List, Optional


def amihud_illiquidity(returns: List[float], dollar_volumes: List[float]) -> List[Optional[float]]:
    """ILLIQ_t = |return_t| / dollar_volume_t. Higher -> less liquid."""
    out = []
    for r, v in zip(returns, dollar_volumes):
        out.append(abs(r) / v if v > 0 else None)
    return out


def average_amihud(returns: List[float], dollar_volumes: List[float]) -> Optional[float]:
    vals = [v for v in amihud_illiquidity(returns, dollar_volumes) if v is not None]
    return sum(vals) / len(vals) if vals else None


def depth_at_touch(depth_snapshot: dict, side: str) -> float:
    """depth_snapshot as returned by OrderBook.depth(); side='bids'|'asks'."""
    levels = depth_snapshot.get(side, [])
    return levels[0][1] if levels else 0.0


def effective_spread(trade_price: float, mid_at_trade: float, side_sign: int) -> float:
    """
    Effective spread = 2 * side_sign * (trade_price - mid) / mid.
    side_sign = +1 for buyer-initiated, -1 for seller-initiated.
    """
    return 2.0 * side_sign * (trade_price - mid_at_trade) / mid_at_trade