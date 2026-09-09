"""Deterministic synthetic M1 data.

Used by the tests and by ``--synthetic`` runs so the whole pipeline can be
exercised with no network.  The generator deliberately builds SMC-shaped price
action -- trends, ranges, and stop runs beyond prior extremes followed by
reversals -- otherwise a random walk would produce almost no valid setups.
"""
from __future__ import annotations

import math
import random
from typing import List

from ..core.types import MS_MINUTE, Candle


def generate(n_minutes: int = 60 * 24 * 30, start: int = 1_704_067_200_000,
             price: float = 42_000.0, seed: int = 42,
             volatility: float = 0.00055) -> List[Candle]:
    rng = random.Random(seed)
    out: List[Candle] = []
    ts = start
    trend = 0.0
    regime_left = 0
    regime = "RANGE"
    range_hi = price * 1.004
    range_lo = price * 0.996
    base_vol = 120.0
    start_price = price

    for i in range(n_minutes):
        if regime_left <= 0:
            regime = rng.choices(["TREND_UP", "TREND_DOWN", "RANGE"],
                                 weights=[0.3, 0.3, 0.4])[0]
            regime_left = rng.randint(120, 900)
            trend = {"TREND_UP": 1.0, "TREND_DOWN": -1.0, "RANGE": 0.0}[regime]
            span = price * rng.uniform(0.002, 0.006)
            range_hi, range_lo = price + span, price - span
        regime_left -= 1

        drift = trend * volatility * 0.09 * price
        noise = rng.gauss(0, volatility) * price
        # Session-of-day volatility profile (quiet Asia, active London/NY).
        hour = ((ts // 3_600_000) % 24)
        season = 0.6 + 0.8 * (0.5 + 0.5 * math.sin((hour - 6) / 24 * 2 * math.pi))
        step = (drift + noise) * season

        # Weak anchor to the starting price keeps a long series inside a
        # realistic band instead of compounding into an exponential ramp.
        step -= 0.00004 * (price - start_price)

        if regime == "RANGE":
            if price > range_hi:
                step -= (price - range_hi) * 0.35
            elif price < range_lo:
                step += (range_lo - price) * 0.35

        open_ = price
        close = price + step
        wick = abs(rng.gauss(0, volatility * 0.9)) * price * season

        # Periodic stop-run: spike beyond a recent extreme, then reject.
        sweep = (i > 60 and rng.random() < 0.012)
        if sweep and len(out) > 60:
            recent_hi = max(c.high for c in out[-60:])
            recent_lo = min(c.low for c in out[-60:])
            if rng.random() < 0.5:
                high = recent_hi + wick * 1.8 + price * 0.0004
                low = min(open_, close) - wick * 0.3
                close = open_ - abs(step) * 1.2
            else:
                low = recent_lo - wick * 1.8 - price * 0.0004
                high = max(open_, close) + wick * 0.3
                close = open_ + abs(step) * 1.2
        else:
            high = max(open_, close) + wick
            low = min(open_, close) - wick

        high = max(high, open_, close)
        low = min(low, open_, close)
        vol = base_vol * season * (1.0 + abs(step) / (volatility * price + 1e-9)) \
            * rng.uniform(0.6, 1.5)
        out.append(Candle(ts, ts + MS_MINUTE, round(open_, 1), round(high, 1),
                          round(low, 1), round(close, 1), round(vol, 2)))
        price = close
        ts += MS_MINUTE
    return out
