"""Wires every engine together into one time-synchronised analysis object.

Update order per closed candle is deliberate and is the second half of the
anti-look-ahead guarantee:

1. structure (which advances the swing engines) -- swings confirm with lag;
2. sweep detection against the liquidity map **as it stood before this bar**;
3. mark levels that this bar traded through;
4. zones (FVG / OB / breaker) from the structure events this bar produced;
5. only then register the *new* levels this bar created.

Step 5 last means a level can never be created and swept by the same candle,
which would otherwise fabricate sweeps that no live bot could have traded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from ..config import Config
from ..core.series import MarketBook
from ..core.types import (Candle, MarketState, Session, StructureEvent, Sweep)
from .displacement import Displacement, DisplacementEngine
from .liquidity import LiquidityEngine
from .structure import StructureEngine
from .sweep import SweepEngine
from .zones import ZoneEngine

TIMEFRAMES = ("M15", "M5", "M1")


@dataclass
class TimeframeView:
    tf: str
    structure: StructureEngine
    zones: ZoneEngine
    sweeps: SweepEngine
    last_events: List[StructureEvent] = field(default_factory=list)
    last_sweeps: List[Sweep] = field(default_factory=list)


class SMCContext:
    """The bot's entire view of the market at the current instant."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.book = MarketBook(cfg.sessions)
        self.liquidity = LiquidityEngine(cfg.liquidity)
        self.displacement = DisplacementEngine(cfg.displacement)
        self.views: Dict[str, TimeframeView] = {
            tf: TimeframeView(
                tf,
                StructureEngine(tf, cfg.structure, cfg.swing),
                ZoneEngine(tf, cfg.zone),
                SweepEngine(tf, cfg.sweep),
            )
            for tf in TIMEFRAMES
        }
        self.funding_rate: float = cfg.execution.funding_rate
        self.spread_bps: float = 1.0
        self.bars_processed = 0

    # ------------------------------------------------------------------
    def on_m1(self, candle: Candle) -> List[str]:
        closed = self.book.push_m1(candle)
        self.bars_processed += 1
        for tf in closed:
            self._update_tf(tf)
        return closed

    def _update_tf(self, tf: str) -> None:
        view = self.views[tf]
        series = self.book.series[tf]
        c = series.last
        if c is None:
            return

        events = view.structure.update(series)                       # 1
        view.last_events = events
        view.last_sweeps = view.sweeps.update(series, self.liquidity)  # 2
        self.liquidity.mark_sweeps(c.high, c.low, c.close_time)        # 3
        swept_recently = bool(view.last_sweeps)
        view.zones.update(series, events, swept_recently)              # 4
        view.zones.flip_breakers(series, events)
        self.liquidity.refresh(self.book, {                            # 5
            t: self.views[t].structure.internal_swings for t in TIMEFRAMES
        })

    # ------------------------------------------------------------------
    @property
    def now(self) -> int:
        return self.book.now

    @property
    def price(self) -> float:
        return self.book.price

    def atr(self, tf: str) -> float:
        return self.book.series[tf].atr

    def ready(self) -> bool:
        return self.book.ready(self.cfg.filters.min_bars_ready)

    def session(self) -> Session:
        return self.book.current_session()

    def market_state(self) -> MarketState:
        return self.views["M15"].structure.state

    def displacement_of(self, tf: str, bullish: bool) -> Displacement:
        return self.displacement.measure(self.book.series[tf], bullish)

    def volatility_pct(self, tf: str = "M15") -> float:
        return self.book.series[tf].atr_pct

    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        """Human-readable state, used by the dashboard (section 98)."""
        m15, m5, m1 = (self.views[t].structure for t in TIMEFRAMES)
        alive = self.liquidity.alive()
        return {
            "time": self.now,
            "price": self.price,
            "market_state": self.market_state().value,
            "session": self.session().value,
            "m15": {"external": m15.external.trend.value,
                    "internal": m15.internal.trend.value,
                    "range_high": m15.range_high, "range_low": m15.range_low,
                    "zone": m15.zone_label(self.price)},
            "m5": {"internal": m5.internal.trend.value},
            "m1": {"internal": m1.internal.trend.value},
            "atr": {tf: round(self.atr(tf), 4) for tf in TIMEFRAMES},
            "liquidity_alive": len(alive),
            "liquidity_above": len(self.liquidity.targets_above(self.price)),
            "liquidity_below": len(self.liquidity.targets_below(self.price)),
            "last_sweep": max(
                (s.time for v in self.views.values() for s in v.sweeps.sweeps),
                default=None),
        }
