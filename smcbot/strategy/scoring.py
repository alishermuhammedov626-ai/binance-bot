"""Sections 44-45, 86 -- the SMC score engine.

Points are awarded per the specification's table, then normalised to 0-100.
The normaliser divides by the *maximum attainable* score rather than by a fixed
constant, so adding a new feature to the table does not silently deflate every
historical score (which would break the ML labels).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from ..config import ScoreConfig
from ..core.types import Side


@dataclass
class Evidence:
    """Everything the setup builder found, in one flat record."""

    side: Side
    weekly_liquidity: bool = False
    daily_liquidity: bool = False
    session_liquidity: bool = False
    liquidity_cluster: bool = False
    cluster_strength: float = 0.0
    liquidity_strength: float = 0.0
    m15_external_aligned: bool = False
    m15_sweep_score: float = 0.0            # 0-100
    m15_internal_choch: bool = False
    m15_bos: bool = False
    m5_confirmation: bool = False
    m5_internal_choch: bool = False
    m5_bos: bool = False
    m5_sweep_score: float = 0.0
    zone_kind: str = ""                     # FVG | OB | BREAKER
    zone_quality: float = 0.0
    m1_confirmation: bool = False
    m1_sweep_score: float = 0.0
    m1_choch: bool = False
    m1_bos: bool = False
    displacement_bonus: float = 0.0
    displacement_label: str = "WEAK"
    premium_discount_ok: bool = False
    range_position: Optional[float] = None
    rr: float = 0.0
    setup_kind: str = "CONTINUATION"        # CONTINUATION | REVERSAL | RANGE_REVERSAL
    extras: Dict[str, float] = field(default_factory=dict)


def score(ev: Evidence, cfg: ScoreConfig) -> tuple:
    """Return ``(final_score_0_100, breakdown)``."""
    b: Dict[str, float] = {}
    m: Dict[str, float] = {}     # maximum attainable, same keys

    def award(key: str, value: float, maximum: float) -> None:
        b[key] = round(value, 3)
        m[key] = maximum

    # A level belongs to exactly one tier, so the three tiers of section 44 are
    # scored as a single mutually-exclusive component.  Summing them into the
    # denominator instead would make a perfect setup unable to exceed ~87.
    if ev.weekly_liquidity:
        liq_pts = cfg.weekly_liquidity
    elif ev.daily_liquidity:
        liq_pts = cfg.daily_liquidity
    elif ev.session_liquidity:
        liq_pts = cfg.session_liquidity
    else:
        # Intraday swing liquidity still counts, scaled by its section-4 strength.
        liq_pts = cfg.weekly_liquidity * min(ev.liquidity_strength / 10.0, 1.0)
    award("liquidity_source", liq_pts,
          max(cfg.weekly_liquidity, cfg.daily_liquidity, cfg.session_liquidity))
    award("liquidity_cluster", cfg.liquidity_cluster if ev.liquidity_cluster else 0.0,
          cfg.liquidity_cluster)
    award("m15_external_structure",
          cfg.m15_external_structure if ev.m15_external_aligned else 0.0,
          cfg.m15_external_structure)
    # The sweep is the single heaviest ingredient and scales with its quality.
    award("m15_sweep", cfg.m15_sweep * min(ev.m15_sweep_score, 100.0) / 100.0,
          cfg.m15_sweep)
    award("m15_internal_choch",
          cfg.m15_internal_choch if ev.m15_internal_choch else 0.0,
          cfg.m15_internal_choch)
    award("m5_confirmation", cfg.m5_confirmation if ev.m5_confirmation else 0.0,
          cfg.m5_confirmation)
    award("m5_internal_choch",
          cfg.m5_internal_choch if ev.m5_internal_choch else 0.0,
          cfg.m5_internal_choch)
    award("m5_bos", cfg.m5_bos if ev.m5_bos else 0.0, cfg.m5_bos)
    zone_pts = 0.0
    if ev.zone_kind:
        zone_pts = cfg.zone * max(0.4, min(ev.zone_quality, 100.0) / 100.0)
        if ev.zone_kind == "BREAKER":
            zone_pts += cfg.breaker_bonus
    award("zone", zone_pts, cfg.zone + cfg.breaker_bonus)
    award("m1_confirmation", cfg.m1_confirmation if ev.m1_confirmation else 0.0,
          cfg.m1_confirmation)
    award("displacement", min(ev.displacement_bonus, cfg.displacement_max),
          cfg.displacement_max)
    award("premium_discount",
          cfg.premium_discount if ev.premium_discount_ok else 0.0,
          cfg.premium_discount)
    rr_pts = 0.0
    if ev.rr >= 2.5:
        rr_pts = cfg.rr_bonus
    elif ev.rr >= 2.0:
        rr_pts = cfg.rr_bonus * 0.6
    elif ev.rr >= 1.5:
        rr_pts = cfg.rr_bonus * 0.3
    award("rr", rr_pts, cfg.rr_bonus)

    raw = sum(b.values())
    max_raw = sum(m.values()) or 1.0
    final = round(100.0 * raw / max_raw, 2)
    b["_raw"] = round(raw, 3)
    b["_max"] = round(max_raw, 3)
    b["_final"] = final
    return final, b


def classify(final: float, cfg: ScoreConfig) -> str:
    """Section 45 bands."""
    if final >= cfg.a_plus:
        return "A_PLUS"
    if final >= cfg.high_quality:
        return "HIGH_QUALITY"
    if final >= cfg.valid:
        return "VALID"
    if final >= cfg.watchlist:
        return "WATCHLIST"
    return "NO_TRADE"


def priority_rank(ev: Evidence) -> tuple:
    """Section 86 -- tie-break ordering when several setups compete."""
    return (
        ev.cluster_strength,
        ev.m15_sweep_score,
        1.0 if (ev.m15_internal_choch or ev.m15_bos) else 0.0,
        1.0 if ev.premium_discount_ok else 0.0,
        1.0 if ev.m5_confirmation else 0.0,
        1.0 if ev.m1_confirmation else 0.0,
        ev.zone_quality,
        ev.displacement_bonus,
        ev.rr,
    )
