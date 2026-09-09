"""Every tunable parameter of the bot lives here.

Nothing in the engines hard-codes a threshold; the backtest, the optimiser and
the live runner all take a :class:`Config`.  Values are the documented defaults
from the specification -- section numbers are quoted next to each block.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, get_type_hints


@dataclass
class SwingConfig:
    """Section 5 -- adaptive fractal detection."""

    left: Dict[str, int] = field(default_factory=lambda: {"M15": 2, "M5": 2, "M1": 2})
    right: Dict[str, int] = field(default_factory=lambda: {"M15": 2, "M5": 2, "M1": 2})
    adaptive: bool = True
    # When realised volatility (ATR / price) exceeds ``vol_high`` the pivots need
    # one more bar of confirmation; below ``vol_low`` they need one less.
    vol_high: float = 0.0035
    vol_low: float = 0.0010
    max_swings: int = 240


@dataclass
class LiquidityConfig:
    """Sections 3-4."""

    strength: Dict[str, float] = field(
        default_factory=lambda: {
            "WEEKLY": 10.0,
            "DAILY": 9.0,
            "SESSION": 8.0,
            "M15": 7.0,
            "M5": 5.0,
            "M1": 3.0,
        }
    )
    equal_tolerance_atr: float = 0.12      # equal high/low tolerance in ATR
    cluster_tolerance_atr: float = 0.25    # levels within this distance cluster
    cluster_bonus: float = 7.0             # section 4
    max_levels: int = 400
    # A level is only tradable if price is within this many ATR of it.
    proximity_atr: float = 3.0


@dataclass
class SweepConfig:
    """Sections 10-11."""

    min_penetration_atr: float = 0.03
    max_penetration_atr: float = 2.5
    min_wick_ratio: float = 0.30           # wick beyond level / candle range
    max_return_bars: int = 3               # bars allowed to close back inside
    min_score: float = 35.0
    volume_lookback: int = 20
    lookback_bars: Dict[str, int] = field(
        default_factory=lambda: {"M15": 8, "M5": 10, "M1": 12}
    )


@dataclass
class DisplacementConfig:
    """Section 12 -- never a hard requirement."""

    strong_body_atr: float = 1.5
    medium_body_atr: float = 0.9
    strong_bonus: float = 10.0
    medium_bonus: float = 5.0
    lookback: int = 3
    volume_boost: float = 1.4              # volume vs average to count as strong


@dataclass
class ZoneConfig:
    """Sections 17-20."""

    min_fvg_atr: float = 0.10
    max_zone_age_bars: Dict[str, int] = field(
        default_factory=lambda: {"M15": 60, "M5": 90, "M1": 120}
    )
    max_zones: int = 40
    mitigation_ratio: float = 0.5          # >50% filled => mitigated
    retest_tolerance_atr: float = 0.15     # how close price must come to the zone
    ob_lookback: int = 12


@dataclass
class StructureConfig:
    """Sections 6-8."""

    range_lookback_bars: int = 60          # external high/low unbroken => RANGE
    range_midpoint_block: float = 0.15     # +/- 15% around EQ is a no-trade band
    internal_lookback: int = 30
    choch_max_age_bars: Dict[str, int] = field(
        default_factory=lambda: {"M15": 20, "M5": 24, "M1": 12}
    )


@dataclass
class ScoreConfig:
    """Sections 44-45 -- base points, then normalised to 100."""

    weekly_liquidity: float = 10.0
    daily_liquidity: float = 10.0
    session_liquidity: float = 8.0
    liquidity_cluster: float = 7.0
    m15_external_structure: float = 8.0
    m15_sweep: float = 15.0
    m15_internal_choch: float = 12.0
    m5_confirmation: float = 10.0
    m5_internal_choch: float = 10.0        # section 16
    m5_bos: float = 8.0
    zone: float = 5.0                      # FVG / OB / Breaker
    breaker_bonus: float = 3.0
    m1_confirmation: float = 10.0
    displacement_max: float = 10.0
    premium_discount: float = 5.0
    rr_bonus: float = 5.0
    max_points: float = 100.0              # denominator used by normalise
    # Section 45 classification bands
    watchlist: float = 60.0
    valid: float = 70.0
    high_quality: float = 80.0
    a_plus: float = 90.0


@dataclass
class RiskConfig:
    """Sections 25-36, 92."""

    leverage: float = 17.0
    risk_per_trade: float = 0.005          # 0.5% max (section 33)
    min_risk_per_trade: float = 0.0025
    max_risk_per_trade: float = 0.005
    max_open_positions: int = 1            # section 93
    min_rr: float = 1.5                    # section 29
    preferred_rr: float = 2.0
    strong_rr: float = 2.5
    # Where the stop comes from.  M1_SWING is the shipped behaviour; the other
    # two exist so the choice can be measured rather than assumed.
    stop_mode: str = "M1_SWING"            # M1_SWING | M5_SWING | SWEEP_EXTREME
    sl_atr_buffer: float = 0.25            # section 25
    sl_tick_buffer: float = 2.0            # in ticks
    min_sl_atr: float = 0.35               # too-tight M1 stop => fall back to M5
    max_sl_atr: float = 6.0
    liquidation_safety: float = 2.5        # SL must be >= 2.5x closer than liq price
    maintenance_margin_rate: float = 0.005
    daily_loss_limit: float = 0.02         # section 35
    consecutive_loss_cooldowns: Dict[str, int] = field(
        default_factory=lambda: {"3": 60, "5": 180}      # losses -> minutes
    )
    cooldown_minutes: int = 12             # section 38
    cooldown_after_loss_minutes: int = 25
    max_trades_per_day: int = 5            # section 37
    # At most this many trades in one session (0 = no session cap).  The cap is
    # an upper bound, never a quota: a session with no valid setup trades zero.
    max_trades_per_session: int = 0
    soft_trades_per_day: int = 4
    # LIQUIDITY is the shipped behaviour (section 27).  ATR places targets at
    # fixed ATR multiples instead, so the two can be compared head to head.
    # LIQUIDITY is the shipped behaviour.  ATR and PERCENT place targets at a
    # fixed distance instead, so the three can be compared head to head.
    # PERCENT is measured on the underlying price, never multiplied by
    # leverage -- leverage changes margin, not where price has to travel.
    tp_mode: str = "LIQUIDITY"             # LIQUIDITY | ATR | PERCENT
    tp_atr_multiples: list = field(default_factory=lambda: [1.0, 2.0, 3.0])
    tp_percent_levels: list = field(default_factory=lambda: [0.4, 0.7, 1.0])
    partial_tp: Dict[str, float] = field(
        default_factory=lambda: {"tp1": 0.30, "tp2": 0.30, "tp3": 0.40}   # section 30
    )
    move_to_be_after_tp1: bool = True
    be_offset_r: float = 0.05              # BE+ offset in R
    trailing_enabled: bool = True          # section 31
    trailing_after_tp: int = 1
    trailing_timeframe: str = "M5"
    # LEGACY is the shipped behaviour (trail after the first partial).  A-D are
    # R-triggered ladders; the stop only ever moves in the favourable
    # direction, and every trigger is evaluated on closed bars.
    #   A: +0.25R BE, +0.50R swing trail
    #   B: +0.50R BE, +0.75R swing trail
    #   C: +0.75R swing trail, no early breakeven
    #   D: swing trail from the start, every new higher low / lower high
    trailing_mode: str = "LEGACY"          # LEGACY | A | B | C | D
    trailing_buffer_atr: float = 0.10
    trailing_swing_timeframe: str = "M1"
    # Take the configured share at the final target and let the rest ride the
    # trailing stop, instead of closing the position there.
    trail_remainder: bool = False
    early_exit_on_invalidation: bool = True  # section 95


@dataclass
class FilterConfig:
    """Sections 46, 63-66, 77-79."""

    max_spread_bps: float = 6.0
    min_atr_pct: float = 0.0004            # dead market
    max_atr_pct: float = 0.020             # extreme volatility
    extreme_funding: float = 0.0008        # 0.08% per 8h
    max_chase_atr: float = 0.8             # section 77
    # Reject a setup whose estimated round-trip commission exceeds this share
    # of its own risk.  0 disables it, which is the shipped behaviour.
    max_fee_r: float = 0.0
    news_filter: bool = False              # section 66 -- OFF unless a feed exists
    min_bars_ready: Dict[str, int] = field(
        default_factory=lambda: {"M15": 120, "M5": 240, "M1": 300}
    )


@dataclass
class SessionConfig:
    """Section 39 -- UTC hour ranges, configurable."""

    asia: tuple = (0, 8)
    london: tuple = (7, 16)
    new_york: tuple = (12, 21)


@dataclass
class MLConfig:
    """Sections 47-54."""

    enabled: bool = True
    threshold: float = 0.55
    min_train_samples: int = 120
    n_estimators: int = 120
    learning_rate: float = 0.06
    max_depth: int = 3
    min_samples_leaf: int = 8
    subsample: float = 0.85
    seed: int = 7
    # Walk-forward (section 53)
    n_folds: int = 4
    # Two distinct embargoes.  ``embargo_samples`` drops setups from the tail of
    # each training block; ``embargo_minutes`` delays when a fold's model may be
    # used, so it never scores a trade that overlaps its own training data.
    embargo_samples: int = 5
    embargo_minutes: int = 720
    model_version: str = "ML_MODEL_v1"


@dataclass
class ExecutionConfig:
    """Sections 55-56, 74-76 -- backtest realism."""

    taker_fee: float = 0.0005
    maker_fee: float = 0.0002
    slippage_bps: float = 1.5
    execution_delay_bars: int = 1          # signal at close -> fill next bar
    funding_interval_hours: int = 8
    funding_rate: float = 0.0001
    tick_size: float = 0.1
    qty_step: float = 0.001
    min_qty: float = 0.001
    min_notional: float = 5.0
    limit_order_ttl_bars: int = 12
    use_server_side_stops: bool = True


@dataclass
class SetupConfig:
    """Sections 68-70 -- lifetimes and de-duplication."""

    m15_ttl_minutes: int = 90
    m5_ttl_minutes: int = 45
    m1_ttl_bars: int = 8
    max_active_setups: int = 6
    dedupe_atr: float = 0.6                # same level+direction inside this band


@dataclass
class Config:
    symbol: str = "BTCUSDT"
    initial_equity: float = 1000.0
    swing: SwingConfig = field(default_factory=SwingConfig)
    liquidity: LiquidityConfig = field(default_factory=LiquidityConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    displacement: DisplacementConfig = field(default_factory=DisplacementConfig)
    zone: ZoneConfig = field(default_factory=ZoneConfig)
    structure: StructureConfig = field(default_factory=StructureConfig)
    score: ScoreConfig = field(default_factory=ScoreConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    sessions: SessionConfig = field(default_factory=SessionConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    setup: SetupConfig = field(default_factory=SetupConfig)
    smc_engine_version: str = "SMC_ENGINE_v1"
    risk_engine_version: str = "RISK_ENGINE_v1"

    # ---------------- serialisation ----------------
    def to_dict(self) -> dict:
        return asdict(self)

    def dumps(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        return _build(cls, data)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.dumps())

    def with_overrides(self, **overrides: Any) -> "Config":
        """``cfg.with_overrides(**{"risk.min_rr": 2.0})`` -> new Config."""
        data = self.to_dict()
        for dotted, value in overrides.items():
            node = data
            parts = dotted.split(".")
            for part in parts[:-1]:
                node = node[part]
            if parts[-1] not in node:
                raise KeyError(f"unknown config key: {dotted}")
            node[parts[-1]] = value
        return Config.from_dict(data)


def _build(kls, data):
    """Rebuild a nested dataclass tree from plain dicts.

    ``from __future__ import annotations`` turns field types into strings, so we
    resolve them through ``get_type_hints`` rather than reading ``f.type``.
    """
    if not is_dataclass(kls) or not isinstance(data, dict):
        return data
    hints = get_type_hints(kls)
    kwargs = {}
    for f in fields(kls):
        if f.name not in data:
            continue
        value = data[f.name]
        ftype = hints.get(f.name)
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[f.name] = _build(ftype, value)
        elif ftype is tuple and isinstance(value, list):
            kwargs[f.name] = tuple(value)
        else:
            kwargs[f.name] = tuple(value) if isinstance(value, list) and ftype is tuple else value
    return kls(**kwargs)
