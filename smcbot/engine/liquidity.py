"""Sections 3-4 -- the liquidity map and its strength model.

Levels are registered the moment they become *known* (a completed day/week, a
finished session, a confirmed swing) and are marked swept incrementally as
price trades through them.  Nothing here inspects a candle that has not closed.
"""
from __future__ import annotations

import bisect
from typing import Dict, List, Optional

from ..config import LiquidityConfig
from ..core.series import MarketBook
from ..core.types import LiquidityKind, LiquidityLevel, Session, Swing

_TF_SOURCE = {"M15": "M15", "M5": "M5", "M1": "M1"}


class LiquidityEngine:
    def __init__(self, cfg: LiquidityConfig):
        self.cfg = cfg
        self.levels: Dict[str, LiquidityLevel] = {}
        self._seen_periods: set = set()
        self._seen_swings: set = set()
        # Price-sorted index.  A level's price is immutable once created, so the
        # index only needs rebuilding when levels are added or evicted -- this
        # turns the per-bar "which levels are near price" scan into a bisect.
        self._sorted: List[LiquidityLevel] = []
        self._prices: List[float] = []
        self._dirty = True

    # ------------------------------------------------------------------
    def _index(self) -> None:
        if not self._dirty:
            return
        self._sorted = sorted(self.levels.values(), key=lambda l: l.price)
        self._prices = [l.price for l in self._sorted]
        self._dirty = False

    def around(self, price: float, span: float) -> List[LiquidityLevel]:
        """Every level within +/- ``span`` of ``price`` (O(log n + k))."""
        self._index()
        lo = bisect.bisect_left(self._prices, price - span)
        hi = bisect.bisect_right(self._prices, price + span)
        return self._sorted[lo:hi]

    def _add(self, key: str, level: LiquidityLevel) -> None:
        if key in self.levels:
            return
        self.levels[key] = level
        self._dirty = True
        if len(self.levels) > self.cfg.max_levels:
            # Drop the oldest *swept* levels first, then the oldest overall.
            dead = [k for k, v in self.levels.items() if not v.alive]
            for k in sorted(dead, key=lambda k: self.levels[k].created_at)[:len(self.levels) // 4]:
                del self.levels[k]
            while len(self.levels) > self.cfg.max_levels:
                oldest = min(self.levels, key=lambda k: self.levels[k].created_at)
                del self.levels[oldest]
            self._dirty = True

    def refresh(self, book: MarketBook, swings: Dict[str, "object"]) -> None:
        """Register everything that became visible with the newest M1 close."""
        now = book.now
        atr = book.m15.atr or book.m1.atr or 1e-9

        # ---- weekly (section 3) --------------------------------------
        pw = book.prev_week
        if pw and ("W", pw.start) not in self._seen_periods:
            self._seen_periods.add(("W", pw.start))
            self._add(f"PWH:{pw.start}", LiquidityLevel(
                pw.high, LiquidityKind.MAJOR_EXTERNAL, True, "WEEKLY",
                self.cfg.strength["WEEKLY"], pw.start, now, "PrevWeekHigh"))
            self._add(f"PWL:{pw.start}", LiquidityLevel(
                pw.low, LiquidityKind.MAJOR_EXTERNAL, False, "WEEKLY",
                self.cfg.strength["WEEKLY"], pw.start, now, "PrevWeekLow"))

        # ---- daily ---------------------------------------------------
        pd = book.prev_day
        if pd and ("D", pd.start) not in self._seen_periods:
            self._seen_periods.add(("D", pd.start))
            self._add(f"PDH:{pd.start}", LiquidityLevel(
                pd.high, LiquidityKind.PREVIOUS_HIGH, True, "DAILY",
                self.cfg.strength["DAILY"], pd.start, now, "PrevDayHigh"))
            self._add(f"PDL:{pd.start}", LiquidityLevel(
                pd.low, LiquidityKind.PREVIOUS_LOW, False, "DAILY",
                self.cfg.strength["DAILY"], pd.start, now, "PrevDayLow"))
        if book.day and ("DO", book.day.start) not in self._seen_periods:
            self._seen_periods.add(("DO", book.day.start))
            self._add(f"DO:{book.day.start}", LiquidityLevel(
                book.day.open, LiquidityKind.INTERNAL, True, "DAILY",
                self.cfg.strength["DAILY"] * 0.5, book.day.start, now, "DailyOpen"))

        # ---- sessions ------------------------------------------------
        for s in (Session.ASIA, Session.LONDON, Session.NEW_YORK):
            done = book.sessions.last_completed(s.value)
            if done and ("S", s.value, done.start) not in self._seen_periods:
                self._seen_periods.add(("S", s.value, done.start))
                self._add(f"{s.value}H:{done.start}", LiquidityLevel(
                    done.high, LiquidityKind.SESSION_HIGH, True, "SESSION",
                    self.cfg.strength["SESSION"], done.start, now, f"{s.value}High"))
                self._add(f"{s.value}L:{done.start}", LiquidityLevel(
                    done.low, LiquidityKind.SESSION_LOW, False, "SESSION",
                    self.cfg.strength["SESSION"], done.start, now, f"{s.value}Low"))

        # ---- timeframe swings ---------------------------------------
        for tf, engine in swings.items():
            for swing in engine.highs[-12:] + engine.lows[-12:]:
                key = (tf, swing.index, swing.is_high)
                if key in self._seen_swings or swing.confirmed_at > now:
                    continue
                self._seen_swings.add(key)
                kind = (LiquidityKind.MAJOR_EXTERNAL if tf == "M15"
                        else LiquidityKind.INTERNAL)
                self._add(f"{tf}:{'H' if swing.is_high else 'L'}:{swing.index}",
                          LiquidityLevel(
                              swing.price, kind, swing.is_high, tf,
                              self.cfg.strength[_TF_SOURCE[tf]], swing.time,
                              swing.confirmed_at, f"{tf}Swing"))
            self._tag_equals(engine.highs[-8:], True, tf, atr, now)
            self._tag_equals(engine.lows[-8:], False, tf, atr, now)

    def _tag_equals(self, swings: List[Swing], is_high: bool, tf: str,
                    atr: float, now: int) -> None:
        """Equal highs/lows -- a magnet of resting stops (section 3)."""
        tol = self.cfg.equal_tolerance_atr * atr
        for i in range(len(swings) - 1):
            a, b = swings[i], swings[i + 1]
            if b.confirmed_at > now or abs(a.price - b.price) > tol:
                continue
            key = f"EQ:{tf}:{'H' if is_high else 'L'}:{a.index}:{b.index}"
            price = max(a.price, b.price) if is_high else min(a.price, b.price)
            strength = self.cfg.strength[_TF_SOURCE[tf]] + 2.0
            self._add(key, LiquidityLevel(
                price,
                LiquidityKind.EQUAL_HIGH if is_high else LiquidityKind.EQUAL_LOW,
                is_high, tf, strength, a.time, b.confirmed_at,
                f"{tf}Equal{'High' if is_high else 'Low'}"))

    # ------------------------------------------------------------------
    def mark_sweeps(self, high: float, low: float, ts: int) -> List[LiquidityLevel]:
        """Flag levels traded through by the just-closed candle."""
        hit = []
        for lvl in self.levels.values():
            if not lvl.alive:
                continue
            if (lvl.is_high and high > lvl.price) or (not lvl.is_high and low < lvl.price):
                lvl.swept_at = ts
                hit.append(lvl)
        return hit

    # ------------------------------------------------------------------
    def alive(self, is_high: Optional[bool] = None) -> List[LiquidityLevel]:
        out = [l for l in self.levels.values() if l.alive]
        if is_high is not None:
            out = [l for l in out if l.is_high is is_high]
        return out

    def nearby(self, price: float, atr: float, is_high: Optional[bool] = None,
               max_atr: Optional[float] = None) -> List[LiquidityLevel]:
        max_atr = self.cfg.proximity_atr if max_atr is None else max_atr
        span = max_atr * atr
        out = [l for l in self.around(price, span)
               if l.alive and (is_high is None or l.is_high is is_high)]
        return sorted(out, key=lambda l: abs(l.price - price))

    def targets_above(self, price: float) -> List[LiquidityLevel]:
        return sorted([l for l in self.alive(True) if l.price > price],
                      key=lambda l: l.price)

    def targets_below(self, price: float) -> List[LiquidityLevel]:
        return sorted([l for l in self.alive(False) if l.price < price],
                      key=lambda l: l.price, reverse=True)

    def recently_swept(self, since: int, is_high: Optional[bool] = None
                       ) -> List[LiquidityLevel]:
        out = [l for l in self.levels.values()
               if l.swept_at is not None and l.swept_at >= since]
        if is_high is not None:
            out = [l for l in out if l.is_high is is_high]
        return sorted(out, key=lambda l: -l.strength)

    # ------------------------------------------------------------------
    def cluster_at(self, price: float, atr: float, is_high: Optional[bool] = None
                   ) -> tuple:
        """Section 4 -- how much liquidity is stacked around ``price``.

        Returns ``(cluster_size, total_strength, is_cluster)``.
        """
        tol = self.cfg.cluster_tolerance_atr * atr
        members = [l for l in self.around(price, tol)
                   if is_high is None or l.is_high is is_high]
        total = sum(l.strength for l in members)
        return len(members), total, len(members) >= 2

    def strongest_near(self, price: float, atr: float, is_high: bool
                       ) -> Optional[LiquidityLevel]:
        near = self.nearby(price, atr, is_high, self.cfg.cluster_tolerance_atr * 2)
        return max(near, key=lambda l: l.strength) if near else None
