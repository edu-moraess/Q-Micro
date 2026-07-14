"""Q-Micro :: execution.vwap — Volume-Weighted Average Price scheduler."""

from __future__ import annotations
from typing import List
from core.order import Order, Side, OrderType


def vwap_schedule(total_qty: float, volume_profile: List[float], side: Side,
                   trader_id: str = "vwap_algo") -> List[Order]:
    """
    volume_profile: expected fraction of ADV per interval (should sum to ~1.0),
    e.g. a U-shaped intraday profile. Child order sizes track expected volume.
    """
    total_weight = sum(volume_profile)
    if total_weight <= 0:
        raise ValueError("volume_profile must sum to a positive value.")
    return [
        Order(side=side, quantity=total_qty * (w / total_weight),
              order_type=OrderType.MARKET, trader_id=trader_id)
        for w in volume_profile
    ]


def realized_vwap(trades: List[dict]) -> float:
    """VWAP = sum(price * volume) / sum(volume) over a trade tape."""
    num = sum(t["price"] * t["volume"] for t in trades)
    den = sum(t["volume"] for t in trades)
    return num / den if den else 0.0