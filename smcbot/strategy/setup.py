"""Sections 20-24, 40-43, 67-70, 80-86 -- setup construction and lifecycle.

The engine never "forces" a trade.  It grows candidates through the state
machine of section 67 and drops them the moment their premise dies:

    WAIT -> LIQUIDITY_FOUND -> SWEEP -> M15_STRUCTURE_SHIFT -> M5_CONFIRMATION
         -> ZONE_FOUND -> M5_RETEST -> M1_CONFIRMATION -> ENTRY

Only the last transition produces a tradable :class:`Setup`; everything before
it is bookkeeping that can expire (section 69) or be invalidated (section 68).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..config import Config
from ..core.types import (TF_MS, LiquidityKind, LiquidityLevel, MarketState, SetupState, Setup,
                          Side, StructureEvent, Sweep, Zone)
from ..engine.context import SMCContext
from .risk import build_stop
from .scoring import Evidence, classify, priority_rank, score
from .targets import select_targets

MINUTE = 60_000


@dataclass
class Candidate:
    """A setup in flight -- not yet tradable."""

    setup_id: str
    side: Side
    state: SetupState
    created_at: int
    expires_at: int
    sweep: Sweep
    kind: str                                   # REVERSAL | CONTINUATION | RANGE_REVERSAL
    m15_shift: Optional[StructureEvent] = None
    m5_shift: Optional[StructureEvent] = None
    m5_bos: Optional[StructureEvent] = None
    m1_shift: Optional[StructureEvent] = None
    m1_sweep: Optional[Sweep] = None
    zone: Optional[Zone] = None
    zone_found_at: int = 0
    retested_at: int = 0
    history: List[str] = field(default_factory=list)
    invalid_reason: str = ""

    def to(self, state: SetupState, ts: int) -> None:
        if self.state is not state:
            self.history.append(f"{ts}:{state.value}")
            self.state = state

    @property
    def bullish(self) -> bool:
        return self.side is Side.BUY


class SignalEngine:
    """Turns the analysis context into setups, one M1 bar at a time."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.candidates: List[Candidate] = []
        self.traded_ids: set = set()
        self.expired: int = 0
        self.invalidated: int = 0
        self.rejections: Dict[str, int] = {}

    # ------------------------------------------------------------------
    def reject(self, reason: str) -> None:
        self.rejections[reason] = self.rejections.get(reason, 0) + 1

    def on_bar(self, ctx: SMCContext) -> List[Setup]:
        """Advance every candidate.  Returns setups that reached ENTRY."""
        if not ctx.ready():
            return []
        now = ctx.now
        self._spawn(ctx)
        self._prune(ctx, now)

        ready: List[Setup] = []
        for cand in list(self.candidates):
            setup = self._advance(ctx, cand)
            if setup is not None:
                ready.append(setup)
        ready.sort(key=lambda s: -s.smc_score)
        return ready

    # ---- candidate creation (sections 40-43, 81-85) ------------------
    def _spawn(self, ctx: SMCContext) -> None:
        now = ctx.now
        if len(self.candidates) >= self.cfg.setup.max_active_setups:
            return
        state = ctx.market_state()
        m15 = ctx.views["M15"]

        # Sweeps that closed on this bar, on M15 or M5 (section 84 allows a
        # continuation setup to originate from an M5 sweep inside an M15 trend).
        fresh: List[Sweep] = []
        for tf in ("M15", "M5"):
            fresh.extend(ctx.views[tf].last_sweeps)
        for sweep in fresh:
            side = Side.BUY if not sweep.is_high_sweep else Side.SELL
            kind = self._classify_setup(ctx, sweep, side, state)
            if kind is None:
                continue
            sid = self._setup_id(ctx, sweep, side)
            if sid in self.traded_ids or any(c.setup_id == sid for c in self.candidates):
                continue
            ttl = self.cfg.setup.m15_ttl_minutes * MINUTE
            cand = Candidate(sid, side, SetupState.SWEEP, now, now + ttl, sweep, kind)
            cand.history.append(f"{now}:SWEEP@{sweep.timeframe}")
            self.candidates.append(cand)
            if len(self.candidates) >= self.cfg.setup.max_active_setups:
                return

    def _classify_setup(self, ctx: SMCContext, sweep: Sweep, side: Side,
                        state: MarketState) -> Optional[str]:
        """Reversal, continuation or range reversal -- or nothing at all."""
        lvl = sweep.level
        major = lvl.timeframe in ("WEEKLY", "DAILY", "SESSION")
        m15 = ctx.views["M15"].structure

        if state is MarketState.RANGE:
            pos = m15.range_position(ctx.price)
            if pos is None:
                return "REVERSAL" if major else None
            # Section 43: only the extremes of a range are tradable.
            if m15.in_midpoint_block(ctx.price):
                self.reject("range_midpoint")
                return None
            if side is Side.BUY and pos > 0.5:
                self.reject("range_wrong_half")
                return None
            if side is Side.SELL and pos < 0.5:
                self.reject("range_wrong_half")
                return None
            return "RANGE_REVERSAL"

        aligned = ((state is MarketState.BULLISH and side is Side.BUY) or
                   (state is MarketState.BEARISH and side is Side.SELL))
        if aligned:
            return "CONTINUATION"
        # Section 3 taxonomy: weekly/daily/session *and* M15 external swings and
        # equal highs/lows are all "major" resting liquidity.
        major = major or lvl.kind in (LiquidityKind.MAJOR_EXTERNAL,
                                      LiquidityKind.EQUAL_HIGH,
                                      LiquidityKind.EQUAL_LOW)
        if major:
            return "REVERSAL"       # section 42/85 -- countertrend needs a major grab
        if self.cfg.filters.allow_counter_trend_minor:
            return "REVERSAL"
        self.reject("counter_trend_minor_liquidity")
        return None

    def _setup_id(self, ctx: SMCContext, sweep: Sweep, side: Side) -> str:
        """Section 70 -- one trade per liquidity+direction, not per candle."""
        atr = ctx.atr("M15") or 1.0
        band = max(self.cfg.setup.dedupe_atr * atr, 1e-9)
        bucket = int(sweep.level.price / band)
        return f"{side.value}:{sweep.level.kind.value}:{bucket}"

    # ---- lifecycle (sections 68-69) ----------------------------------
    def _prune(self, ctx: SMCContext, now: int) -> None:
        keep: List[Candidate] = []
        for c in self.candidates:
            reason = self._invalidation(ctx, c, now)
            if reason:
                c.invalid_reason = reason
                if reason == "expired":
                    self.expired += 1
                else:
                    self.invalidated += 1
                self.reject(f"invalidated:{reason}")
                continue
            keep.append(c)
        self.candidates = keep

    def _invalidation(self, ctx: SMCContext, c: Candidate, now: int) -> str:
        if now > c.expires_at:
            return "expired"
        price = ctx.price
        # The sweep's own extreme is the premise: lose it and the idea is dead.
        if c.bullish and price < c.sweep.extreme_price:
            return "sweep_extreme_lost"
        if not c.bullish and price > c.sweep.extreme_price:
            return "sweep_extreme_lost"
        # Opposite structure break on M5 or M15 (section 68).
        for tf in ("M15", "M5"):
            for ev in ctx.views[tf].last_events:
                if ev.bullish is not c.bullish and ev.time > c.created_at:
                    return f"opposite_{tf}_{ev.kind}"
        if c.zone is not None:
            if c.bullish and price < c.zone.bottom - ctx.atr("M5"):
                return "zone_invalid"
            if not c.bullish and price > c.zone.top + ctx.atr("M5"):
                return "zone_invalid"
        return ""

    # ---- state machine ------------------------------------------------
    def _advance(self, ctx: SMCContext, c: Candidate) -> Optional[Setup]:
        now = ctx.now
        m15, m5, m1 = (ctx.views[t] for t in ("M15", "M5", "M1"))

        # 1. M15 structure shift (sections 13-14)
        if c.m15_shift is None:
            shift = m15.structure.recent_shift(c.bullish, now, TF_MS["M15"])
            if shift and shift.time >= c.sweep.time:
                c.m15_shift = shift
                c.to(SetupState.M15_STRUCTURE_SHIFT, now)
            elif c.sweep.timeframe == "M5" and c.kind == "CONTINUATION":
                pass    # a continuation setup may skip the M15 shift
            else:
                return None

        # 2. M5 confirmation (sections 15-16)
        if c.m5_shift is None:
            shift = m5.structure.recent_shift(c.bullish, now, TF_MS["M5"])
            if shift and shift.time >= c.sweep.time:
                c.m5_shift = shift
                c.m5_bos = m5.structure.recent_bos(c.bullish, now, tf_ms=TF_MS["M5"])
                c.to(SetupState.M5_CONFIRMATION, now)
            else:
                return None

        # 3. M5 zone (sections 17-19)
        atr5 = ctx.atr("M5") or 1e-9
        if c.zone is None:
            zone = m5.zones.best_zone(c.bullish, ctx.price)
            if zone is None:
                zone = m15.zones.best_zone(c.bullish, ctx.price)
            if zone is None:
                return None
            c.zone = zone
            c.zone_found_at = now
            c.to(SetupState.ZONE_FOUND, now)

        # 4. Retest (section 20) -- no chasing.
        if not c.retested_at:
            if m5.zones.in_zone(ctx.price, c.zone, atr5):
                c.retested_at = now
                c.to(SetupState.M5_RETEST, now)
            else:
                distance = (ctx.price - c.zone.top if c.bullish
                            else c.zone.bottom - ctx.price)
                if distance > self.cfg.filters.max_chase_atr * atr5:
                    self.reject("chase_protection")
                return None

        # 5. M1 confirmation (section 21)
        atr1 = ctx.atr("M1") or 1e-9
        m1_sweep = m1.sweeps.best_recent(c.retested_at - 10 * MINUTE,
                                         is_high=not c.bullish)
        m1_shift = m1.structure.recent_shift(c.bullish, now, TF_MS["M1"])
        stale = m1_shift is None or m1_shift.time < c.retested_at - 10 * MINUTE
        if stale:
            if self.cfg.filters.require_m1_confirmation:
                return None
            self.reject("m1_confirmation_missing_but_allowed")
        c.m1_sweep = m1_sweep
        c.m1_shift = m1_shift
        c.to(SetupState.M1_CONFIRMATION, now)

        return self._build_setup(ctx, c)

    # ---- entry construction (sections 22, 25-29) ---------------------
    def _build_setup(self, ctx: SMCContext, c: Candidate) -> Optional[Setup]:
        now = ctx.now
        price = ctx.price
        atr1, atr5 = ctx.atr("M1") or 1e-9, ctx.atr("M5") or 1e-9
        m1, m5 = ctx.views["M1"], ctx.views["M5"]

        # Entry: limit at a fresh M1 zone if price has not reached it yet,
        # otherwise a confirmation market entry (section 22).
        m1_zone = m1.zones.best_zone(c.bullish, price)
        entry, entry_type = price, "MARKET"
        if m1_zone is not None:
            target_price = m1_zone.top if c.bullish else m1_zone.bottom
            gap = (price - target_price) if c.bullish else (target_price - price)
            if 0 < gap <= self.cfg.filters.max_chase_atr * atr1:
                entry, entry_type = target_price, "LIMIT"

        # Stop: behind the M1 confirmation swing, M5 invalidation as fallback.
        m1_anchor = self._confirmation_swing(ctx, c, "M1")
        m5_anchor = self._confirmation_swing(ctx, c, "M5")
        stop_plan = build_stop(c.side, entry, m1_anchor, m5_anchor, atr1, atr5,
                               self.cfg.risk, self.cfg.execution,
                               sweep_extreme=c.sweep.extreme_price,
                               atr_ref=ctx.atr(self.cfg.risk.sl_atr_period_timeframe))
        if not stop_plan.valid:
            self.reject(f"stop:{stop_plan.reason or 'invalid'}")
            return None

        # Cost gate (off by default).  A round trip costs roughly
        # 2 * taker * notional, and notional is risk / stop_pct, so the fee
        # measured in R is ~ 2*taker / stop_pct.  Refusing a setup whose cost
        # eats too much of its own risk is a no-trade filter, not a stop that
        # has been dragged into structure.
        max_fee_r = self.cfg.filters.max_fee_r
        if max_fee_r > 0:
            stop_pct = stop_plan.distance / entry if entry else 0.0
            fills = 1 + max(1, sum(1 for v in self.cfg.risk.partial_tp.values() if v > 0))
            est_fee_r = (fills * self.cfg.execution.taker_fee / stop_pct
                         if stop_pct > 0 else float("inf"))
            if est_fee_r > max_fee_r:
                self.reject("fee_r_too_high")
                return None

        targets, rr, treason = select_targets(c.side, entry, stop_plan.price, atr5,
                                              ctx.liquidity, self.cfg.risk)
        if treason != "ok":
            self.reject(f"target:{treason}")
            return None

        ev = self._evidence(ctx, c, rr, entry)
        final, breakdown = score(ev, self.cfg.score)
        if final < self.cfg.score.valid:
            self.reject(f"score:{classify(final, self.cfg.score)}")
            return None

        setup = Setup(
            setup_id=c.setup_id, created_at=now, side=c.side, state=SetupState.ENTRY,
            entry=entry, stop=stop_plan.price, targets=targets, rr=rr,
            smc_score=final, score_breakdown=breakdown, zone=c.zone,
            entry_type=entry_type,
            expires_at=now + self.cfg.setup.m1_ttl_bars * MINUTE,
            reasons=list(c.history),
        )
        atr15 = ctx.atr("M15") or 1e-9
        setup.meta = {
            # Recording only -- how stale the sweep was by the time the entry
            # formed, and how far price had travelled away from its extreme.
            # Nothing reads these back; they exist so the sweep can be audited.
            "sweep_age_min": round((now - c.sweep.time) / MINUTE, 1),
            "sweep_distance_atr": round(
                abs(entry - c.sweep.extreme_price) / atr15, 3),
            "sweep_penetration_atr": c.sweep.penetration_atr,
            "sweep_wick_ratio": c.sweep.wick_ratio,
            "sweep_return_bars": c.sweep.return_speed,
            "sweep_volume_ratio": c.sweep.volume_ratio,
            "sweep_level_kind": c.sweep.level.kind.value,
            "kind": c.kind,
            "classification": classify(final, self.cfg.score),
            "stop_source": stop_plan.source,
            "sweep_score": c.sweep.score,
            "sweep_level": c.sweep.level.label,
            "sweep_tf": c.sweep.timeframe,
            "zone_kind": c.zone.kind if c.zone else "",
            "market_state": ctx.market_state().value,
            "session": ctx.session().value,
            "priority": priority_rank(ev),
        }
        setup.features = {}          # filled by the feature builder
        setup.meta["evidence"] = ev
        return setup

    def _confirmation_swing(self, ctx: SMCContext, c: Candidate, tf: str
                            ) -> Optional[float]:
        """Extreme of the confirmation leg -- what the stop must sit behind."""
        series = ctx.book.series[tf]
        since = c.m1_shift.time if (tf == "M1" and c.m1_shift) else c.retested_at
        bars = [x for x in series.window(40) if x.close_time >= since - 5 * TF_MS[tf]]
        if not bars:
            bars = series.window(6)
        if not bars:
            return None
        return min(b.low for b in bars) if c.bullish else max(b.high for b in bars)

    # ---- evidence assembly (section 44) ------------------------------
    def _evidence(self, ctx: SMCContext, c: Candidate, rr: float,
                  entry: float) -> Evidence:
        now = ctx.now
        m15, m5, m1 = (ctx.views[t] for t in ("M15", "M5", "M1"))
        lvl: LiquidityLevel = c.sweep.level
        atr15 = ctx.atr("M15") or 1e-9
        size, cluster_strength, is_cluster = ctx.liquidity.cluster_at(
            lvl.price, atr15, lvl.is_high)

        disp5 = ctx.displacement_of("M5", c.bullish)
        disp1 = ctx.displacement_of("M1", c.bullish)
        disp_bonus = max(disp5.bonus, disp1.bonus)

        state = ctx.market_state()
        aligned = ((state is MarketState.BULLISH and c.bullish) or
                   (state is MarketState.BEARISH and not c.bullish))

        zone_pos = m15.structure.range_position(entry)
        pd_ok = False
        if zone_pos is not None:
            pd_ok = (c.bullish and zone_pos < 0.5) or (not c.bullish and zone_pos > 0.5)

        return Evidence(
            side=c.side,
            weekly_liquidity=lvl.timeframe == "WEEKLY",
            daily_liquidity=lvl.timeframe == "DAILY",
            session_liquidity=lvl.timeframe == "SESSION",
            liquidity_cluster=is_cluster,
            cluster_strength=cluster_strength,
            liquidity_strength=lvl.strength,
            m15_external_aligned=aligned,
            m15_sweep_score=c.sweep.score if c.sweep.timeframe == "M15" else c.sweep.score * 0.75,
            m15_internal_choch=bool(c.m15_shift and c.m15_shift.kind == "CHOCH"),
            m15_bos=bool(c.m15_shift and c.m15_shift.kind == "BOS"),
            m5_confirmation=c.m5_shift is not None,
            m5_internal_choch=bool(c.m5_shift and c.m5_shift.kind == "CHOCH"),
            m5_bos=bool(c.m5_bos),
            m5_sweep_score=c.sweep.score if c.sweep.timeframe == "M5" else 0.0,
            zone_kind=c.zone.kind if c.zone else "",
            zone_quality=c.zone.quality if c.zone else 0.0,
            m1_confirmation=c.m1_shift is not None,
            m1_sweep_score=c.m1_sweep.score if c.m1_sweep else 0.0,
            m1_choch=bool(c.m1_shift and c.m1_shift.kind == "CHOCH"),
            m1_bos=bool(c.m1_shift and c.m1_shift.kind == "BOS"),
            displacement_bonus=disp_bonus,
            displacement_label=disp5.label if disp5.bonus >= disp1.bonus else disp1.label,
            premium_discount_ok=pd_ok,
            range_position=zone_pos,
            rr=rr,
            setup_kind=c.kind,
            extras={"cluster_size": float(size),
                    "m5_disp": disp5.body_atr, "m1_disp": disp1.body_atr},
        )

    # ------------------------------------------------------------------
    def mark_traded(self, setup_id: str) -> None:
        self.traded_ids.add(setup_id)
        self.candidates = [c for c in self.candidates if c.setup_id != setup_id]

    def drop(self, setup_id: str) -> None:
        self.candidates = [c for c in self.candidates if c.setup_id != setup_id]
