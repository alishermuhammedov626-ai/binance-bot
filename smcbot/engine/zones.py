"""Sections 17-20 -- FVG, Order Block, Breaker and retest handling.

All three are reduced to a common :class:`Zone` (a price band with a quality
score) so the entry logic can treat them uniformly, while the individual
objects keep their own mitigation state.
"""
from __future__ import annotations

from typing import List, Optional

from ..config import ZoneConfig
from ..core.series import CandleSeries
from ..core.types import TF_MS, FVG, OrderBlock, StructureEvent, Zone

_TF_RANK = {"M15": 3.0, "M5": 2.0, "M1": 1.0}


class ZoneEngine:
    def __init__(self, tf: str, cfg: ZoneConfig):
        self.tf = tf
        self.cfg = cfg
        self.fvgs: List[FVG] = []
        self.obs: List[OrderBlock] = []

    # ------------------------------------------------------------------
    def update(self, series: CandleSeries, events: List[StructureEvent],
               sweep_recent: bool = False) -> None:
        self._detect_fvg(series)
        self._detect_ob(series, events, sweep_recent)
        self._mitigate(series)
        self._expire(series)

    # ---- FVG (section 17) --------------------------------------------
    def _detect_fvg(self, series: CandleSeries) -> None:
        if len(series) < 3:
            return
        c1, c2, c3 = series[-3], series[-2], series[-1]
        atr = series.atr or 1e-9
        min_size = self.cfg.min_fvg_atr * atr
        disp = c2.body / atr

        if c3.low > c1.high and (c3.low - c1.high) >= min_size:
            self.fvgs.append(FVG(c2.open_time, self.tf, True, c3.low, c1.high,
                                 c3.close_time, displacement=round(disp, 3)))
        elif c1.low > c3.high and (c1.low - c3.high) >= min_size:
            self.fvgs.append(FVG(c2.open_time, self.tf, False, c1.low, c3.high,
                                 c3.close_time, displacement=round(disp, 3)))
        self.fvgs = self.fvgs[-self.cfg.max_zones:]

    # ---- Order block (section 18) ------------------------------------
    def _detect_ob(self, series: CandleSeries, events: List[StructureEvent],
                   sweep_recent: bool) -> None:
        if not events or len(series) < 4:
            return
        atr = series.atr or 1e-9
        for ev in events:
            look = min(self.cfg.ob_lookback, len(series) - 1)
            window = series.candles[-look:]
            # Last opposite-colour candle before the impulse that broke structure.
            origin = None
            for c in reversed(window[:-1]):
                if ev.bullish and c.bearish:
                    origin = c
                    break
                if not ev.bullish and c.bullish:
                    origin = c
                    break
            if origin is None:
                continue
            if any(o.time == origin.open_time and o.bullish is ev.bullish
                   for o in self.obs):
                continue
            ob = OrderBlock(
                time=origin.open_time, timeframe=self.tf, bullish=ev.bullish,
                top=origin.high, bottom=origin.low,
                created_at=series[-1].close_time,
                displacement=ev.displacement, has_bos=True,
                from_sweep=sweep_recent,
            )
            ob.strength = self._ob_strength(ob, atr)
            self.obs.append(ob)
        self.obs = self.obs[-self.cfg.max_zones:]

    def _ob_strength(self, ob: OrderBlock, atr: float) -> float:
        s = 40.0
        s += min(ob.displacement / 2.0, 1.0) * 25.0
        if ob.has_bos:
            s += 15.0
        if ob.from_sweep:
            s += 15.0
        if ob.size <= 0 or ob.size / atr > 3.0:
            s -= 10.0                      # a bloated block is a weak block
        if ob.is_breaker:
            s += 5.0
        return max(0.0, min(100.0, s))

    # ---- Breaker (section 19) ----------------------------------------
    def flip_breakers(self, series: CandleSeries, events: List[StructureEvent]) -> int:
        """An OB violated *and* followed by an opposing shift flips role."""
        c = series.last
        if c is None or not events:
            return 0
        flipped = 0
        for ev in events:
            for ob in self.obs:
                if ob.is_breaker or ob.bullish is ev.bullish:
                    continue
                violated = (c.close > ob.top) if ev.bullish else (c.close < ob.bottom)
                if not violated:
                    continue
                ob.is_breaker = True
                ob.bullish = ev.bullish
                ob.mitigated_at = None
                ob.created_at = c.close_time
                ob.strength = self._ob_strength(ob, series.atr or 1e-9)
                flipped += 1
        return flipped

    # ---- mitigation & expiry (sections 17-20) ------------------------
    def _mitigate(self, series: CandleSeries) -> None:
        c = series.last
        if c is None:
            return
        for g in self.fvgs:
            if g.mitigated_at or g.created_at >= c.close_time:
                continue
            if g.bullish:
                filled = max(0.0, g.top - max(c.low, g.bottom))
            else:
                filled = max(0.0, min(c.high, g.top) - g.bottom)
            g.filled_ratio = max(g.filled_ratio,
                                 filled / g.size if g.size > 0 else 1.0)
            if g.filled_ratio >= self.cfg.mitigation_ratio:
                g.mitigated_at = c.close_time
        for ob in self.obs:
            if ob.mitigated_at or ob.created_at >= c.close_time:
                continue
            # A block is spent once price closes through it against its bias.
            if (ob.bullish and c.close < ob.bottom) or (not ob.bullish and c.close > ob.top):
                ob.mitigated_at = c.close_time
            elif (ob.bullish and c.low <= ob.mid) or (not ob.bullish and c.high >= ob.mid):
                ob.mitigated_at = c.close_time

    def _expire(self, series: CandleSeries) -> None:
        c = series.last
        if c is None:
            return
        max_age = self.cfg.max_zone_age_bars.get(self.tf, 60) * TF_MS[self.tf]
        cutoff = c.close_time - max_age
        self.fvgs = [g for g in self.fvgs if g.created_at >= cutoff][-self.cfg.max_zones:]
        self.obs = [o for o in self.obs if o.created_at >= cutoff][-self.cfg.max_zones:]

    # ------------------------------------------------------------------
    def zones(self, bullish: bool, fresh_only: bool = True) -> List[Zone]:
        """Unified, quality-ranked POI list for one direction."""
        out: List[Zone] = []
        rank = _TF_RANK.get(self.tf, 1.0)
        for g in self.fvgs:
            if g.bullish is not bullish or (fresh_only and g.mitigated_at):
                continue
            q = 45.0 + min(g.displacement / 2.0, 1.0) * 25.0 + rank * 5.0
            q -= g.filled_ratio * 15.0
            out.append(Zone(g.top, g.bottom, self.tf, "FVG", bullish,
                            g.created_at, round(max(0.0, min(100.0, q)), 2)))
        for ob in self.obs:
            if ob.bullish is not bullish or (fresh_only and ob.mitigated_at):
                continue
            q = ob.strength * 0.8 + rank * 5.0
            out.append(Zone(ob.top, ob.bottom, self.tf,
                            "BREAKER" if ob.is_breaker else "OB", bullish,
                            ob.created_at, round(max(0.0, min(100.0, q)), 2)))
        out.sort(key=lambda z: (-z.quality, -z.created_at))
        return out

    def best_zone(self, bullish: bool, price: float,
                  max_distance: Optional[float] = None) -> Optional[Zone]:
        """Best unmitigated zone that price has not yet passed through."""
        cands = []
        for z in self.zones(bullish):
            if bullish and z.top > price + (max_distance or float("inf")):
                continue
            if not bullish and z.bottom < price - (max_distance or float("inf")):
                continue
            # For a long we want the zone at or below price, and vice versa.
            if bullish and z.bottom > price:
                continue
            if not bullish and z.top < price:
                continue
            cands.append(z)
        if not cands:
            return None
        return max(cands, key=lambda z: (z.quality, -abs(z.mid - price)))

    def in_zone(self, price: float, zone: Zone, atr: float) -> bool:
        tol = self.cfg.retest_tolerance_atr * atr
        return (zone.bottom - tol) <= price <= (zone.top + tol)

    def has_breaker(self, bullish: bool) -> bool:
        return any(o.is_breaker and o.bullish is bullish and not o.mitigated_at
                   for o in self.obs)
