"""Q-Micro :: microstructure.vpin — Volume-Synchronized Probability of Informed Trading."""

from __future__ import annotations
from typing import List


def bucketize_by_volume(trades: List[dict], bucket_size: float) -> List[dict]:
    """
    Groups a trade tape (list of {'price','volume', ...}) into equal-volume
    buckets and classifies volume as buy/sell using the tick rule.
    """
    buckets: List[dict] = []
    cur_buy, cur_sell, cur_vol = 0.0, 0.0, 0.0
    last_price = None

    for t in trades:
        price, vol = t["price"], t["volume"]
        is_buy = last_price is None or price >= last_price
        last_price = price
        remaining = vol
        while remaining > 0:
            space = bucket_size - cur_vol
            take = min(space, remaining)
            if is_buy:
                cur_buy += take
            else:
                cur_sell += take
            cur_vol += take
            remaining -= take
            if cur_vol >= bucket_size - 1e-9:
                buckets.append({"buy_volume": cur_buy, "sell_volume": cur_sell})
                cur_buy, cur_sell, cur_vol = 0.0, 0.0, 0.0

    return buckets


def compute_vpin(trades: List[dict], bucket_size: float, window: int = 50) -> List[float]:
    """VPIN_t = average(|BuyVol - SellVol|) / bucket_size over the trailing `window` buckets."""
    buckets = bucketize_by_volume(trades, bucket_size)
    vpin_series = []
    for i in range(len(buckets)):
        lo = max(0, i - window + 1)
        window_buckets = buckets[lo:i + 1]
        imbalances = [abs(b["buy_volume"] - b["sell_volume"]) for b in window_buckets]
        vpin_series.append(sum(imbalances) / (len(window_buckets) * bucket_size))
    return vpin_series