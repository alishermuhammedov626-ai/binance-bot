"""Sections 25-26, 32-34, 78, 92 -- stops, sizing and liquidation safety.

The stop is placed behind the structure that invalidates the idea, never at a
fixed percentage; the *position size* is what absorbs a wide stop, never the
stop being dragged into the structure (section 78).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..config import ExecutionConfig, RiskConfig
from ..core.types import Side


@dataclass
class StopPlan:
    price: float
    source: str            # M1_SWING | M5_SWING | ATR_FALLBACK
    distance: float
    distance_atr: float
    valid: bool
    reason: str = ""


@dataclass
class SizePlan:
    qty: float
    notional: float
    margin: float
    risk_usdt: float
    risk_pct: float
    liquidation_price: float
    valid: bool
    reason: str = ""


def round_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return math.floor(value / step + 1e-9) * step


def round_tick(price: float, tick: float, direction: int = 0) -> float:
    if tick <= 0:
        return price
    units = price / tick
    if direction > 0:
        units = math.ceil(units - 1e-9)
    elif direction < 0:
        units = math.floor(units + 1e-9)
    else:
        units = round(units)
    return round(units * tick, 10)


def build_stop(side: Side, entry: float, m1_swing: Optional[float],
               m5_swing: Optional[float], atr_m1: float, atr_m5: float,
               cfg: RiskConfig, execution: ExecutionConfig,
               sweep_extreme: Optional[float] = None) -> StopPlan:
    """Section 25 -- behind the confirmation swing, with an ATR/tick buffer.

    ``cfg.stop_mode`` selects which structure the stop hides behind.  All three
    modes place the stop *beyond* an invalidation level and never inside it
    (section 78); they differ only in which level counts as invalidation.
    """
    buffer_ = max(cfg.sl_atr_buffer * max(atr_m1, 1e-9),
                  cfg.sl_tick_buffer * execution.tick_size)

    def make(anchor: float, source: str) -> StopPlan:
        price = anchor - buffer_ if side is Side.BUY else anchor + buffer_
        price = round_tick(price, execution.tick_size, -side.sign)
        dist = abs(entry - price)
        return StopPlan(price, source, dist,
                        dist / atr_m1 if atr_m1 else 0.0, True)

    mode = getattr(cfg, "stop_mode", "M1_SWING")
    if mode == "M5_SWING" and m5_swing is not None:
        anchor = m5_swing
        plan = make(anchor, "M5_SWING")
        return _finalise(plan, side, entry, cfg)
    if mode == "SWEEP_EXTREME" and sweep_extreme is not None:
        plan = make(sweep_extreme, "SWEEP_EXTREME")
        return _finalise(plan, side, entry, cfg)

    plan: Optional[StopPlan] = None
    if m1_swing is not None:
        candidate = make(m1_swing, "M1_SWING")
        # A stop the size of a rounding error is not an invalidation level.
        if candidate.distance_atr >= cfg.min_sl_atr:
            plan = candidate
        elif m5_swing is not None:
            plan = make(m5_swing, "M5_SWING")
        else:
            plan = candidate
    elif m5_swing is not None:
        plan = make(m5_swing, "M5_SWING")

    if plan is None:
        anchor = entry - 1.5 * atr_m5 * side.sign
        plan = make(anchor, "ATR_FALLBACK")

    return _finalise(plan, side, entry, cfg)


def _finalise(plan: StopPlan, side: Side, entry: float,
              cfg: RiskConfig) -> StopPlan:
    """Shared validity checks for every stop mode."""
    # The stop must be on the losing side of entry.
    if (side is Side.BUY and plan.price >= entry) or \
       (side is Side.SELL and plan.price <= entry):
        return StopPlan(plan.price, plan.source, 0.0, 0.0, False, "stop_wrong_side")
    if plan.distance_atr > cfg.max_sl_atr:
        plan.valid = False
        plan.reason = "stop_too_wide"
    if plan.distance <= 0:
        plan.valid = False
        plan.reason = "stop_zero_distance"
    return plan


def liquidation_price(side: Side, entry: float, leverage: float,
                      maintenance_margin_rate: float) -> float:
    """Isolated-margin liquidation approximation (section 34).

    long:  entry * (1 - 1/L + mmr)      short: entry * (1 + 1/L - mmr)
    Conservative: it ignores the extra wallet balance that would push the real
    liquidation further away, so the safety check errs towards refusing trades.
    """
    if leverage <= 0:
        return 0.0
    edge = 1.0 / leverage - maintenance_margin_rate
    return entry * (1.0 - edge) if side is Side.BUY else entry * (1.0 + edge)


def size_position(side: Side, entry: float, stop: float, equity: float,
                  risk_pct: float, cfg: RiskConfig,
                  execution: ExecutionConfig) -> SizePlan:
    """Section 32 -- size from risk and stop distance, never from leverage."""
    sl_distance = abs(entry - stop)
    liq = liquidation_price(side, entry, cfg.leverage, cfg.maintenance_margin_rate)
    if sl_distance <= 0:
        return SizePlan(0, 0, 0, 0, 0, liq, False, "zero_stop_distance")

    risk_usdt = equity * risk_pct
    raw_qty = risk_usdt / sl_distance
    qty = round_step(raw_qty, execution.qty_step)
    notional = qty * entry
    margin = notional / cfg.leverage if cfg.leverage else notional

    if qty < execution.min_qty:
        return SizePlan(0, 0, 0, risk_usdt, risk_pct, liq, False, "below_min_qty")
    if notional < execution.min_notional:
        return SizePlan(0, 0, 0, risk_usdt, risk_pct, liq, False, "below_min_notional")
    if margin > equity:
        # Cap by available margin rather than silently over-leveraging.
        qty = round_step(equity * cfg.leverage / entry, execution.qty_step)
        notional = qty * entry
        margin = notional / cfg.leverage
        if qty < execution.min_qty:
            return SizePlan(0, 0, 0, risk_usdt, risk_pct, liq, False, "insufficient_margin")

    # Section 34: the stop must trigger comfortably before liquidation.
    liq_distance = abs(entry - liq)
    if liq_distance < cfg.liquidation_safety * sl_distance:
        return SizePlan(qty, notional, margin, risk_usdt, risk_pct, liq, False,
                        "liquidation_too_close")
    return SizePlan(round(qty, 8), round(notional, 4), round(margin, 4),
                    round(risk_usdt, 4), risk_pct, round(liq, 4), True, "ok")


def rr_of(side: Side, entry: float, stop: float, target: float) -> float:
    risk = abs(entry - stop)
    if risk <= 0:
        return 0.0
    reward = (target - entry) if side is Side.BUY else (entry - target)
    return round(reward / risk, 4)
