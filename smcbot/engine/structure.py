"""Sections 6-8, 13-14, 16 -- market structure.

Two structures run in parallel on every timeframe, exactly as the spec asks:

* **external** -- built from *major* pivots (wider fractal).  It answers "what
  is the market doing" and drives BULLISH / BEARISH / RANGE.
* **internal** -- built from ordinary pivots.  It answers "what is price doing
  inside the current leg" and is what CHOCH/BOS entries are read from, so a
  setup is available even while the external range holds (section 7).

Breaks are confirmed on candle *close* only -- an intrabar wick through a swing
is not a break.
"""
from __future__ import annotations

from dataclasses import replace
from typing import List, Optional

from ..config import StructureConfig, SwingConfig
from ..core.series import CandleSeries
from ..core.types import MarketState, StructureEvent, Swing
from .swings import SwingEngine


class _Leg:
    """One structural view (internal or external) over a swing stream."""

    def __init__(self, tf: str, engine: SwingEngine, internal: bool):
        self.tf = tf
        self.swings = engine
        self.internal = internal
        self.trend: MarketState = MarketState.UNKNOWN
        self.broken: set = set()
        self.events: List[StructureEvent] = []
        self.last_break_time: int = 0
        self.bars_since_break: int = 0

    # -- references -----------------------------------------------------
    def ref_high(self) -> Optional[Swing]:
        for s in reversed(self.swings.highs):
            if (s.index, True) not in self.broken:
                return s
        return None

    def ref_low(self) -> Optional[Swing]:
        for s in reversed(self.swings.lows):
            if (s.index, False) not in self.broken:
                return s
        return None

    def update(self, series: CandleSeries, atr: float) -> List[StructureEvent]:
        c = series.last
        out: List[StructureEvent] = []
        if c is None:
            return out
        self.bars_since_break += 1

        rh = self.ref_high()
        if rh is not None and rh.confirmed_at <= c.close_time and c.close > rh.price:
            self.broken.add((rh.index, True))
            kind = "BOS" if self.trend is MarketState.BULLISH else "CHOCH"
            ev = StructureEvent(c.close_time, self.tf, kind, True, rh.price,
                                self.internal, displacement=c.body / atr if atr else 0.0)
            self.trend = MarketState.BULLISH
            self._push(ev)
            out.append(ev)
            # Every older unbroken high below this close is broken too.
            for s in self.swings.highs:
                if s.price <= c.close:
                    self.broken.add((s.index, True))

        rl = self.ref_low()
        if rl is not None and rl.confirmed_at <= c.close_time and c.close < rl.price:
            self.broken.add((rl.index, False))
            kind = "BOS" if self.trend is MarketState.BEARISH else "CHOCH"
            ev = StructureEvent(c.close_time, self.tf, kind, False, rl.price,
                                self.internal, displacement=c.body / atr if atr else 0.0)
            self.trend = MarketState.BEARISH
            self._push(ev)
            out.append(ev)
            for s in self.swings.lows:
                if s.price >= c.close:
                    self.broken.add((s.index, False))
        return out

    def _push(self, ev: StructureEvent) -> None:
        self.events.append(ev)
        self.events = self.events[-80:]
        self.last_break_time = ev.time
        self.bars_since_break = 0

    # -- queries --------------------------------------------------------
    def last_event(self, kind: Optional[str] = None, bullish: Optional[bool] = None,
                   since: Optional[int] = None) -> Optional[StructureEvent]:
        for ev in reversed(self.events):
            if kind and ev.kind != kind:
                continue
            if bullish is not None and ev.bullish is not bullish:
                continue
            if since is not None and ev.time < since:
                break
            return ev
        return None

    def has_event(self, kind: str, bullish: bool, since: int) -> bool:
        return any(ev.kind == kind and ev.bullish is bullish and ev.time >= since
                   for ev in self.events)


class StructureEngine:
    """Owns the swing engines and both structural views for one timeframe."""

    def __init__(self, tf: str, cfg: StructureConfig, swing_cfg: SwingConfig):
        self.tf = tf
        self.cfg = cfg
        wide = replace(
            swing_cfg,
            left={k: v * 2 + 1 for k, v in swing_cfg.left.items()},
            right={k: v * 2 + 1 for k, v in swing_cfg.right.items()},
        )
        self.internal_swings = SwingEngine(tf, swing_cfg)
        self.external_swings = SwingEngine(tf, wide)
        self.internal = _Leg(tf, self.internal_swings, True)
        self.external = _Leg(tf, self.external_swings, False)
        self.state: MarketState = MarketState.UNKNOWN
        self.range_high: Optional[float] = None
        self.range_low: Optional[float] = None

    # ------------------------------------------------------------------
    def update(self, series: CandleSeries) -> List[StructureEvent]:
        self.internal_swings.update(series)
        self.external_swings.update(series)
        atr = series.atr
        events = self.external.update(series, atr) + self.internal.update(series, atr)
        self._update_state(series)
        return events

    def _update_state(self, series: CandleSeries) -> None:
        """Section 6/8 -- external break sets trend, silence sets RANGE."""
        ext = self.external
        if ext.trend is MarketState.UNKNOWN:
            self.state = MarketState.UNKNOWN
        elif ext.bars_since_break >= self.cfg.range_lookback_bars:
            self.state = MarketState.RANGE
        else:
            self.state = ext.trend

        hi = self.external_swings.highest_swing(6)
        lo = self.external_swings.lowest_swing(6)
        look = self.cfg.range_lookback_bars
        self.range_high = hi.price if hi else series.highest(look)
        self.range_low = lo.price if lo else series.lowest(look)
        if self.range_high is not None and self.range_low is not None:
            if self.range_high <= self.range_low:
                self.range_high, self.range_low = series.highest(look), series.lowest(look)

    # ------------------------------------------------------------------
    @property
    def equilibrium(self) -> Optional[float]:
        if self.range_high is None or self.range_low is None:
            return None
        return (self.range_high + self.range_low) / 2.0

    def range_position(self, price: float) -> Optional[float]:
        """0.0 at range low, 1.0 at range high (section 9)."""
        if self.range_high is None or self.range_low is None:
            return None
        span = self.range_high - self.range_low
        if span <= 0:
            return None
        return max(0.0, min(1.0, (price - self.range_low) / span))

    def zone_label(self, price: float) -> str:
        pos = self.range_position(price)
        if pos is None:
            return "UNKNOWN"
        if pos > 0.5:
            return "PREMIUM"
        if pos < 0.5:
            return "DISCOUNT"
        return "EQUILIBRIUM"

    def in_midpoint_block(self, price: float) -> bool:
        """Section 8/43 -- no entries around the middle of a range."""
        pos = self.range_position(price)
        if pos is None:
            return False
        return abs(pos - 0.5) <= self.cfg.range_midpoint_block

    def recent_choch(self, bullish: bool, now: int, bars: Optional[int] = None,
                     tf_ms: int = 900_000) -> Optional[StructureEvent]:
        max_age = (bars or self.cfg.choch_max_age_bars.get(self.tf, 20)) * tf_ms
        ev = self.internal.last_event("CHOCH", bullish)
        if ev and now - ev.time <= max_age:
            return ev
        return None

    def recent_bos(self, bullish: bool, now: int, bars: Optional[int] = None,
                   tf_ms: int = 900_000) -> Optional[StructureEvent]:
        max_age = (bars or self.cfg.choch_max_age_bars.get(self.tf, 20)) * tf_ms
        ev = self.internal.last_event("BOS", bullish)
        if ev and now - ev.time <= max_age:
            return ev
        return None

    def recent_shift(self, bullish: bool, now: int, tf_ms: int = 900_000
                     ) -> Optional[StructureEvent]:
        """Either a CHOCH or a BOS in the requested direction (section 13/14)."""
        choch = self.recent_choch(bullish, now, tf_ms=tf_ms)
        bos = self.recent_bos(bullish, now, tf_ms=tf_ms)
        if choch and bos:
            return choch if choch.time >= bos.time else bos
        return choch or bos
