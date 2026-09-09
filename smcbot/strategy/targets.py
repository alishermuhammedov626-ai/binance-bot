"""Sections 27-29, 79 -- liquidity-based take profits.

Targets are never a fixed percentage.  The engine collects the resting
liquidity in the trade's direction, ranks it by reachability, and keeps the
hierarchy M1 internal -> M5 -> M15 swing -> daily -> weekly.
"""
from __future__ import annotations

import math
from typing import List

from ..config import RiskConfig
from ..core.types import LiquidityKind, LiquidityLevel, Side, Target
from ..engine.liquidity import LiquidityEngine

_TIER = {"M1": 1, "M5": 2, "M15": 3, "SESSION": 3, "DAILY": 4, "WEEKLY": 5}


def _probability(distance_atr: float, strength: float) -> float:
    """Crude reachability prior: closer and stronger magnets get hit more often.

    This is only a ranking heuristic -- the ML filter is what actually estimates
    win probability, and it never sees this number as ground truth.
    """
    reach = math.exp(-max(distance_atr, 0.0) / 8.0)
    pull = 0.55 + 0.45 * min(strength / 10.0, 1.0)
    return round(max(0.02, min(0.97, reach * pull)), 4)


def build_targets(side: Side, entry: float, stop: float, atr: float,
                  liq: LiquidityEngine, cfg: RiskConfig,
                  max_targets: int = 3) -> List[Target]:
    risk = abs(entry - stop)
    if risk <= 0 or atr <= 0:
        return []
    levels = (liq.targets_above(entry) if side is Side.BUY
              else liq.targets_below(entry))

    seen_tiers = {}
    out: List[Target] = []
    for lvl in levels:
        distance = abs(lvl.price - entry)
        if distance <= 0:
            continue
        rr = distance / risk
        if rr < 0.4:
            continue                      # noise level, not a target
        tier = _TIER.get(lvl.timeframe, 1)
        t = Target(price=lvl.price, liquidity=lvl, rr=round(rr, 3),
                   distance=distance,
                   probability=_probability(distance / atr, lvl.strength))
        # Keep the strongest magnet per tier, nearest first.
        prev = seen_tiers.get(tier)
        if prev is None or (lvl.strength > prev.liquidity.strength
                            and t.distance < prev.distance * 1.5):
            seen_tiers[tier] = t
    out = [seen_tiers[k] for k in sorted(seen_tiers)]
    out.sort(key=lambda t: t.distance)
    return out[:max_targets]


def atr_targets(side: Side, entry: float, stop: float, atr: float,
                cfg: RiskConfig) -> List[Target]:
    """Fixed ATR-multiple targets, for comparison against liquidity targets."""
    risk = abs(entry - stop)
    if risk <= 0 or atr <= 0:
        return []
    out: List[Target] = []
    for mult in cfg.tp_atr_multiples:
        distance = mult * atr
        price = entry + distance * side.sign
        level = LiquidityLevel(price, LiquidityKind.INTERNAL, side is Side.BUY,
                               "ATR", 5.0, 0, 0, f"ATRx{mult}")
        out.append(Target(price=price, liquidity=level,
                          rr=round(distance / risk, 3), distance=distance,
                          probability=_probability(mult, 5.0)))
    return out


def percent_targets(side: Side, entry: float, stop: float,
                    cfg: RiskConfig) -> List[Target]:
    """Targets a fixed percentage of the *underlying price* away.

    Deliberately not scaled by leverage: leverage changes the margin a position
    needs, never the distance price must travel to reach a target.
    """
    risk = abs(entry - stop)
    if risk <= 0:
        return []
    out: List[Target] = []
    for pct in cfg.tp_percent_levels:
        distance = entry * pct / 100.0
        price = entry + distance * side.sign
        level = LiquidityLevel(price, LiquidityKind.INTERNAL, side is Side.BUY,
                               "PCT", 5.0, 0, 0, f"{pct}%")
        out.append(Target(price=price, liquidity=level,
                          rr=round(distance / risk, 3), distance=distance,
                          probability=_probability(distance / max(risk, 1e-9), 5.0)))
    return out


def select_targets(side: Side, entry: float, stop: float, atr: float,
                   liq: LiquidityEngine, cfg: RiskConfig) -> tuple:
    """Return ``(targets, final_rr, reason)`` honouring the minimum-RR rule.

    ``final_rr`` is the *partial-weighted* RR actually expected from the ladder,
    not the RR of the furthest target, so a 1.2R first target cannot be dressed
    up by a distant TP3 the trade will rarely reach.
    """
    if cfg.tp_mode == "ATR":
        targets = atr_targets(side, entry, stop, atr, cfg)
    elif cfg.tp_mode == "PERCENT":
        targets = percent_targets(side, entry, stop, cfg)
    else:
        targets = build_targets(side, entry, stop, atr, liq, cfg)
    if not targets:
        return [], 0.0, "no_liquidity_target"

    weights = [cfg.partial_tp["tp1"], cfg.partial_tp["tp2"], cfg.partial_tp["tp3"]]
    used = targets[:len(weights)]
    w = weights[:len(used)]
    total_w = sum(w) or 1.0
    weighted_rr = sum(t.rr * wi for t, wi in zip(used, w)) / total_w

    best_rr = max(t.rr for t in used)
    if best_rr < cfg.min_rr:
        return used, round(weighted_rr, 3), "rr_below_minimum"
    # Section 79: the *first* target must itself be worth taking.
    if used[0].rr < cfg.min_first_target_rr and len(used) == 1:
        return used, round(weighted_rr, 3), "target_too_close"
    return used, round(weighted_rr, 3), "ok"
