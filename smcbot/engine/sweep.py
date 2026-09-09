"""Sections 10-11 -- manipulation / liquidity sweep detection and scoring.

A sweep is *not* just "price traded through a level".  It is: wick beyond the
level, then a close back on the original side, reasonably fast.  Both the
single-candle version (wick + close back on the same bar) and the multi-bar
version (poke out, reclaim within ``max_return_bars``) are detected.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..config import SweepConfig
from ..core.series import CandleSeries
from ..core.types import LiquidityLevel, Sweep
from .liquidity import LiquidityEngine


class SweepEngine:
    def __init__(self, tf: str, cfg: SweepConfig):
        self.tf = tf
        self.cfg = cfg
        self.sweeps: List[Sweep] = []
        # level uid -> (bar_index, extreme_price) while price is outside a level
        self._pending: Dict[int, Tuple[int, float]] = {}
        self._bar = 0

    # ------------------------------------------------------------------
    def update(self, series: CandleSeries, liq: LiquidityEngine) -> List[Sweep]:
        c = series.last
        if c is None:
            return []
        self._bar += 1
        atr = series.atr or 1e-9
        vol_ma = series.vol_ma() or 1e-9
        out: List[Sweep] = []

        lookback = self.cfg.lookback_bars.get(self.tf, 10)
        span = max(self.cfg.max_penetration_atr * atr, 4 * atr)
        # Only levels the bar could plausibly interact with -- a bisect window
        # rather than a full scan of the (few hundred) live levels.
        candidates = liq.around(c.close, span + c.range)

        for level in candidates:
            key = level.uid
            if level.confirmed_at > c.close_time:
                continue                      # not yet knowable -- would be look-ahead
            outside = (c.high > level.price) if level.is_high else (c.low < level.price)
            extreme = c.high if level.is_high else c.low
            if outside:
                prev = self._pending.get(key)
                if prev is None:
                    self._pending[key] = (self._bar, extreme)
                else:
                    best = max(prev[1], extreme) if level.is_high else min(prev[1], extreme)
                    self._pending[key] = (prev[0], best)

            back_inside = (c.close < level.price) if level.is_high else (c.close > level.price)
            pend = self._pending.get(key)
            if pend and back_inside:
                start_bar, extreme_price = pend
                bars = self._bar - start_bar
                del self._pending[key]
                if bars > self.cfg.max_return_bars:
                    continue
                sweep = self._build(c, level, extreme_price, bars, atr, vol_ma, liq)
                if sweep and sweep.score >= self.cfg.min_score:
                    self.sweeps.append(sweep)
                    out.append(sweep)

        # Forget stale pokes that never reclaimed.
        for key in [k for k, v in self._pending.items()
                    if self._bar - v[0] > lookback]:
            del self._pending[key]
        self.sweeps = self.sweeps[-60:]
        return out

    # ------------------------------------------------------------------
    def _build(self, c, level: LiquidityLevel, extreme: float, bars: int,
               atr: float, vol_ma: float, liq: LiquidityEngine) -> Optional[Sweep]:
        penetration = abs(extreme - level.price)
        pen_atr = penetration / atr
        if pen_atr < self.cfg.min_penetration_atr or pen_atr > self.cfg.max_penetration_atr:
            return None

        rng = c.range or 1e-9
        wick = c.upper_wick if level.is_high else c.lower_wick
        wick_ratio = wick / rng
        # How decisively the close rejected the level, normalised by the poke.
        close_back = abs(c.close - level.price) / max(penetration, 1e-9)
        vol_ratio = c.volume / vol_ma
        _, cluster_strength, is_cluster = liq.cluster_at(level.price, atr, level.is_high)

        score = self._score(level, pen_atr, wick_ratio, close_back, bars,
                            vol_ratio, is_cluster, cluster_strength)
        return Sweep(
            time=c.close_time, timeframe=self.tf, is_high_sweep=level.is_high,
            level=level, penetration=penetration, penetration_atr=round(pen_atr, 4),
            wick_ratio=round(wick_ratio, 4), close_back_ratio=round(close_back, 4),
            return_speed=float(bars), volume_ratio=round(vol_ratio, 3),
            score=round(score, 2), extreme_price=extreme,
        )

    def _score(self, level, pen_atr, wick_ratio, close_back, bars, vol_ratio,
               is_cluster, cluster_strength) -> float:
        """0-100.  Strong = major level + clean wick + fast, decisive reclaim."""
        # Level importance (0-30)
        s = min(level.strength / 12.0, 1.0) * 30.0
        # Penetration: enough to trip stops, not a genuine breakout (0-15)
        ideal = 0.45
        s += max(0.0, 1.0 - abs(pen_atr - ideal) / 1.2) * 15.0
        # Wick quality (0-20)
        s += min(wick_ratio / self.cfg.min_wick_ratio, 1.5) / 1.5 * 20.0
        # Rejection depth (0-15)
        s += min(close_back, 2.0) / 2.0 * 15.0
        # Speed (0-10)
        s += max(0.0, 1.0 - bars / max(self.cfg.max_return_bars, 1)) * 10.0
        # Participation (0-10)
        s += min(max(vol_ratio - 0.8, 0.0) / 1.2, 1.0) * 10.0
        if is_cluster:
            s += min(cluster_strength / 30.0, 1.0) * 10.0
        return max(0.0, min(100.0, s))

    # ------------------------------------------------------------------
    def recent(self, since: int, is_high: Optional[bool] = None) -> List[Sweep]:
        out = [s for s in self.sweeps if s.time >= since]
        if is_high is not None:
            out = [s for s in out if s.is_high_sweep is is_high]
        return out

    def best_recent(self, since: int, is_high: bool) -> Optional[Sweep]:
        r = self.recent(since, is_high)
        return max(r, key=lambda s: s.score) if r else None
