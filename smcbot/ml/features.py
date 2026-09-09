"""Section 48 -- the ML feature vector, and section 54 -- no leakage.

Every value here is computed from candles that have already closed and from
objects whose ``confirmed_at`` is <= now.  Nothing reads a future swing, a
future candle, or the trade's own outcome.  Keys beginning with ``_`` are
metadata (time, side, score) carried alongside the vector for bookkeeping and
are excluded from the model input by :func:`vectorise`.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..core.types import MarketState, Session, Setup, Side
from ..engine.context import SMCContext

_STATE_CODE = {MarketState.BEARISH: -1.0, MarketState.RANGE: 0.0,
               MarketState.BULLISH: 1.0, MarketState.UNKNOWN: 0.0}
_SESSION_CODE = {Session.ASIA: 1.0, Session.LONDON: 2.0,
                 Session.NEW_YORK: 3.0, Session.OFF: 0.0}
_KIND_CODE = {"CONTINUATION": 1.0, "REVERSAL": 2.0, "RANGE_REVERSAL": 3.0}
_ZONE_CODE = {"": 0.0, "FVG": 1.0, "OB": 2.0, "BREAKER": 3.0}

DAY_MS = 86_400_000


def _safe(a: float, b: float) -> float:
    return a / b if b else 0.0


def build_features(ctx: SMCContext, setup: Setup,
                   recent_results: Optional[List[int]] = None) -> Dict[str, float]:
    ev = setup.meta.get("evidence")
    now = ctx.now
    price = ctx.price
    side_sign = setup.side.sign
    atr15 = ctx.atr("M15") or 1e-9
    atr5 = ctx.atr("M5") or 1e-9
    atr1 = ctx.atr("M1") or 1e-9
    m15, m5, m1 = (ctx.views[t] for t in ("M15", "M5", "M1"))
    book = ctx.book

    f: Dict[str, float] = {}

    # -- metadata (never fed to the model) --------------------------------
    f["_time"] = float(now)
    f["_side"] = float(side_sign)
    f["_smc_score"] = float(setup.smc_score)
    f["_setup_id"] = 0.0

    # -- structure (section 48) -------------------------------------------
    f["market_state"] = _STATE_CODE.get(ctx.market_state(), 0.0) * side_sign
    f["m15_external_trend"] = _STATE_CODE.get(m15.structure.external.trend, 0.0) * side_sign
    f["m15_internal_trend"] = _STATE_CODE.get(m15.structure.internal.trend, 0.0) * side_sign
    f["m5_internal_trend"] = _STATE_CODE.get(m5.structure.internal.trend, 0.0) * side_sign
    f["m1_internal_trend"] = _STATE_CODE.get(m1.structure.internal.trend, 0.0) * side_sign
    f["is_range"] = 1.0 if ctx.market_state() is MarketState.RANGE else 0.0

    bullish = setup.side is Side.BUY
    for tf, view, tf_ms in (("m15", m15, 900_000), ("m5", m5, 300_000),
                            ("m1", m1, 60_000)):
        choch = view.structure.internal.last_event("CHOCH", bullish)
        bos = view.structure.internal.last_event("BOS", bullish)
        f[f"{tf}_choch"] = 1.0 if choch else 0.0
        f[f"{tf}_bos"] = 1.0 if bos else 0.0
        f[f"{tf}_choch_age"] = _safe(now - choch.time, tf_ms) if choch else 999.0
        f[f"{tf}_bos_age"] = _safe(now - bos.time, tf_ms) if bos else 999.0
        f[f"{tf}_bars_since_ext_break"] = float(view.structure.external.bars_since_break)

    # -- liquidity & sweep -------------------------------------------------
    if ev is not None:
        f["liquidity_strength"] = float(ev.liquidity_strength)
        f["liquidity_weekly"] = 1.0 if ev.weekly_liquidity else 0.0
        f["liquidity_daily"] = 1.0 if ev.daily_liquidity else 0.0
        f["liquidity_session"] = 1.0 if ev.session_liquidity else 0.0
        f["liquidity_cluster"] = 1.0 if ev.liquidity_cluster else 0.0
        f["cluster_strength"] = float(ev.cluster_strength)
        f["cluster_size"] = float(ev.extras.get("cluster_size", 0.0))
        f["sweep_score"] = float(ev.m15_sweep_score)
        f["m5_sweep_score"] = float(ev.m5_sweep_score)
        f["m1_sweep_score"] = float(ev.m1_sweep_score)
        f["zone_kind"] = _ZONE_CODE.get(ev.zone_kind, 0.0)
        f["zone_quality"] = float(ev.zone_quality)
        f["displacement_bonus"] = float(ev.displacement_bonus)
        f["m5_displacement"] = float(ev.extras.get("m5_disp", 0.0))
        f["m1_displacement"] = float(ev.extras.get("m1_disp", 0.0))
        f["setup_kind"] = _KIND_CODE.get(ev.setup_kind, 0.0)
        f["premium_discount_ok"] = 1.0 if ev.premium_discount_ok else 0.0
        f["range_position"] = (ev.range_position if ev.range_position is not None
                               else 0.5)
    sweep_meta = setup.meta
    f["sweep_tf"] = {"M15": 3.0, "M5": 2.0, "M1": 1.0}.get(
        sweep_meta.get("sweep_tf", ""), 0.0)

    # -- geometry ----------------------------------------------------------
    risk = abs(setup.entry - setup.stop)
    f["sl_distance_atr"] = _safe(risk, atr1)
    f["sl_distance_pct"] = _safe(risk, setup.entry)
    f["rr"] = float(setup.rr)
    if setup.targets:
        f["tp1_distance_atr"] = _safe(abs(setup.targets[0].price - setup.entry), atr5)
        f["tp1_rr"] = float(setup.targets[0].rr)
        f["tp1_probability"] = float(setup.targets[0].probability)
        f["n_targets"] = float(len(setup.targets))
        f["tp_far_rr"] = float(max(t.rr for t in setup.targets))
    f["entry_is_limit"] = 1.0 if setup.entry_type == "LIMIT" else 0.0

    # -- volatility / participation ---------------------------------------
    f["atr_pct_m15"] = ctx.volatility_pct("M15")
    f["atr_pct_m5"] = ctx.volatility_pct("M5")
    f["atr_pct_m1"] = ctx.volatility_pct("M1")
    f["atr_ratio_m1_m15"] = _safe(atr1, atr15)
    last1 = book.m1.last
    f["volume_ratio_m1"] = _safe(last1.volume, book.m1.vol_ma()) if last1 else 0.0
    last5 = book.m5.last
    f["volume_ratio_m5"] = _safe(last5.volume, book.m5.vol_ma()) if last5 else 0.0
    f["spread_bps"] = float(ctx.spread_bps)
    f["funding_rate"] = float(ctx.funding_rate)

    # -- location relative to reference levels -----------------------------
    def dist(level: Optional[float]) -> float:
        return _safe((price - level) * side_sign, atr15) if level else 0.0

    f["dist_daily_open"] = dist(book.day.open if book.day else None)
    f["dist_prev_day_high"] = dist(book.prev_day.high if book.prev_day else None)
    f["dist_prev_day_low"] = dist(book.prev_day.low if book.prev_day else None)
    f["dist_prev_week_high"] = dist(book.prev_week.high if book.prev_week else None)
    f["dist_prev_week_low"] = dist(book.prev_week.low if book.prev_week else None)
    f["dist_range_high"] = dist(m15.structure.range_high)
    f["dist_range_low"] = dist(m15.structure.range_low)

    # -- calendar ------------------------------------------------------------
    f["session"] = _SESSION_CODE.get(ctx.session(), 0.0)
    f["hour"] = float((now % DAY_MS) // 3_600_000)
    f["day_of_week"] = float(((now // DAY_MS) + 4) % 7)   # 0 = Monday

    # -- recent performance (past only) ---------------------------------------
    seq = recent_results or []
    f["recent_wins"] = float(sum(1 for r in seq[-5:] if r > 0))
    f["recent_losses"] = float(sum(1 for r in seq[-5:] if r <= 0))
    f["last_result"] = float(seq[-1]) if seq else 0.0

    return f


FEATURE_NAMES_CACHE: List[str] = []


def feature_names(rows: List[Dict[str, float]]) -> List[str]:
    """Stable, sorted list of model-visible feature names across all rows."""
    names = set()
    for r in rows:
        names.update(k for k in r if not k.startswith("_"))
    return sorted(names)


def vectorise(row: Dict[str, float], names: List[str]) -> List[float]:
    return [float(row.get(n, 0.0)) for n in names]


def vectorise_all(rows: List[Dict[str, float]], names: List[str]) -> List[List[float]]:
    return [vectorise(r, names) for r in rows]
