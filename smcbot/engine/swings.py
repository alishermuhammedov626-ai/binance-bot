"""Section 5 -- adaptive swing (fractal) detection.

A swing high at bar ``i`` needs ``right`` bars to its right that are lower.
Those bars do not exist yet at bar ``i``, so the pivot is *published* only at
bar ``i + right``.  ``Swing.confirmed_at`` records that publication time and is
the only timestamp downstream code is allowed to gate on -- using ``Swing.time``
as an availability check would leak the future by exactly ``right`` bars.
"""
from __future__ import annotations

from typing import List, Optional

from ..config import SwingConfig
from ..core.series import CandleSeries
from ..core.types import Swing


class SwingEngine:
    def __init__(self, timeframe: str, cfg: SwingConfig):
        self.tf = timeframe
        self.cfg = cfg
        self.highs: List[Swing] = []
        self.lows: List[Swing] = []
        self._scanned_to = -1          # absolute index of last scanned pivot

    # ------------------------------------------------------------------
    def _wings(self, series: CandleSeries) -> tuple:
        left = self.cfg.left.get(self.tf, 2)
        right = self.cfg.right.get(self.tf, 2)
        if self.cfg.adaptive and series.ready:
            vol = series.atr_pct
            if vol > self.cfg.vol_high:
                left += 1
                right += 1
            elif vol < self.cfg.vol_low and left > 1:
                left -= 1
                right -= 1
        return left, right

    def update(self, series: CandleSeries) -> List[Swing]:
        """Scan for pivots that became confirmed with the newest candle."""
        if len(series) < 3:
            return []
        left, right = self._wings(series)
        new: List[Swing] = []
        n = len(series)
        # Candidate pivot index: the newest bar that now has `right` bars after it.
        pivot = n - 1 - right
        if pivot < left:
            return []
        abs_pivot = series.abs_index(pivot)
        if abs_pivot <= self._scanned_to:
            return []
        self._scanned_to = abs_pivot

        c = series[pivot]
        left_bars = series.candles[pivot - left:pivot]
        right_bars = series.candles[pivot + 1:pivot + 1 + right]
        confirmed_at = series[-1].close_time
        atr = series.atr_at(pivot) or 1e-9

        if all(b.high < c.high for b in left_bars) and all(b.high < c.high for b in right_bars):
            depth = (c.high - min(b.low for b in left_bars + right_bars)) / atr
            s = Swing(c.open_time, c.high, True, confirmed_at, self.tf, abs_pivot,
                      strength=round(min(depth, 5.0), 3))
            self.highs.append(s)
            self.highs = self.highs[-self.cfg.max_swings:]
            new.append(s)
        if all(b.low > c.low for b in left_bars) and all(b.low > c.low for b in right_bars):
            depth = (max(b.high for b in left_bars + right_bars) - c.low) / atr
            s = Swing(c.open_time, c.low, False, confirmed_at, self.tf, abs_pivot,
                      strength=round(min(depth, 5.0), 3))
            self.lows.append(s)
            self.lows = self.lows[-self.cfg.max_swings:]
            new.append(s)
        return new

    # ------------------------------------------------------------------
    def last_high(self, before: Optional[int] = None) -> Optional[Swing]:
        return self._last(self.highs, before)

    def last_low(self, before: Optional[int] = None) -> Optional[Swing]:
        return self._last(self.lows, before)

    @staticmethod
    def _last(items: List[Swing], before: Optional[int]) -> Optional[Swing]:
        if before is None:
            return items[-1] if items else None
        for s in reversed(items):
            if s.confirmed_at <= before:
                return s
        return None

    def recent_highs(self, n: int = 5) -> List[Swing]:
        return self.highs[-n:]

    def recent_lows(self, n: int = 5) -> List[Swing]:
        return self.lows[-n:]

    def highest_swing(self, n: int = 10) -> Optional[Swing]:
        w = self.highs[-n:]
        return max(w, key=lambda s: s.price) if w else None

    def lowest_swing(self, n: int = 10) -> Optional[Swing]:
        w = self.lows[-n:]
        return min(w, key=lambda s: s.price) if w else None
