"""Core value objects shared by every engine.

Design rule that governs this whole package: an object created at time ``T``
may only depend on information that was public at ``T``.  Every dataclass here
therefore carries the timestamp at which it became *known* (``confirmed_at``),
which is not always the timestamp at which it *happened* (``time``).  A swing
high, for example, happens at its pivot candle but only becomes known a few
candles later once the right-hand side is complete.  Engines must filter on
``confirmed_at``; the backtest asserts this invariant.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


MS_MINUTE = 60_000
TF_MS = {"M1": MS_MINUTE, "M5": 5 * MS_MINUTE, "M15": 15 * MS_MINUTE}


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY


class MarketState(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    RANGE = "RANGE"
    UNKNOWN = "UNKNOWN"


class LiquidityKind(str, Enum):
    """Section 3 taxonomy."""

    MAJOR_EXTERNAL = "MAJOR_EXTERNAL"
    INTERNAL = "INTERNAL"
    EQUAL_HIGH = "EQUAL_HIGH"
    EQUAL_LOW = "EQUAL_LOW"
    PREVIOUS_HIGH = "PREVIOUS_HIGH"
    PREVIOUS_LOW = "PREVIOUS_LOW"
    SESSION_HIGH = "SESSION_HIGH"
    SESSION_LOW = "SESSION_LOW"


class Session(str, Enum):
    ASIA = "ASIA"
    LONDON = "LONDON"
    NEW_YORK = "NEW_YORK"
    LATE = "LATE"          # fourth block, so a day can hold four session slots
    OFF = "OFF"


class SetupState(str, Enum):
    """Section 67 trade state machine."""

    WAIT = "WAIT"
    LIQUIDITY_FOUND = "LIQUIDITY_FOUND"
    SWEEP = "SWEEP"
    M15_STRUCTURE_SHIFT = "M15_STRUCTURE_SHIFT"
    M5_CONFIRMATION = "M5_CONFIRMATION"
    ZONE_FOUND = "ZONE_FOUND"
    M5_RETEST = "M5_RETEST"
    M1_CONFIRMATION = "M1_CONFIRMATION"
    ENTRY = "ENTRY"
    POSITION_OPEN = "POSITION_OPEN"
    CLOSED = "CLOSED"
    EXPIRED = "EXPIRED"
    INVALIDATED = "INVALIDATED"


@dataclass(frozen=True)
class Candle:
    """A *closed* candle.  Open candles never enter the system."""

    open_time: int          # ms, inclusive
    close_time: int         # ms, exclusive upper bound of the bar
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def bullish(self) -> bool:
        return self.close > self.open

    @property
    def bearish(self) -> bool:
        return self.close < self.open

    @property
    def body_ratio(self) -> float:
        return self.body / self.range if self.range > 0 else 0.0


@dataclass(frozen=True)
class Swing:
    """Fractal pivot.  ``confirmed_at`` == close time of the last right-side bar."""

    time: int
    price: float
    is_high: bool
    confirmed_at: int
    timeframe: str
    index: int
    strength: float = 1.0


_LEVEL_SEQ = itertools.count(1)


@dataclass
class LiquidityLevel:
    price: float
    kind: LiquidityKind
    is_high: bool
    timeframe: str
    strength: float
    created_at: int
    confirmed_at: int
    label: str = ""
    swept_at: Optional[int] = None
    cluster_size: int = 1
    uid: int = field(default_factory=lambda: next(_LEVEL_SEQ))

    @property
    def alive(self) -> bool:
        return self.swept_at is None


@dataclass
class Sweep:
    """Liquidity grab: wick beyond a level, close back inside (sections 10-11)."""

    time: int
    timeframe: str
    is_high_sweep: bool            # True => sell-side setup (bought highs)
    level: LiquidityLevel
    penetration: float             # absolute distance beyond the level
    penetration_atr: float
    wick_ratio: float
    close_back_ratio: float
    return_speed: float            # bars taken to close back inside (lower = faster)
    volume_ratio: float
    score: float                   # 0-100 (section 11)
    extreme_price: float


@dataclass
class StructureEvent:
    """CHOCH / BOS event."""

    time: int
    timeframe: str
    kind: str                      # "CHOCH" | "BOS"
    bullish: bool
    level: float                   # broken swing price
    internal: bool
    displacement: float = 0.0


@dataclass
class FVG:
    """Fair value gap (section 17)."""

    time: int
    timeframe: str
    bullish: bool
    top: float
    bottom: float
    created_at: int
    displacement: float = 0.0
    mitigated_at: Optional[int] = None
    filled_ratio: float = 0.0

    @property
    def size(self) -> float:
        return self.top - self.bottom

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2.0


@dataclass
class OrderBlock:
    """Order block / breaker (sections 18-19)."""

    time: int
    timeframe: str
    bullish: bool
    top: float
    bottom: float
    created_at: int
    displacement: float = 0.0
    has_bos: bool = False
    from_sweep: bool = False
    is_breaker: bool = False
    mitigated_at: Optional[int] = None
    strength: float = 0.0

    @property
    def size(self) -> float:
        return self.top - self.bottom

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2.0


@dataclass
class Zone:
    """Unified POI (FVG, OB or breaker) used for retest / entry."""

    top: float
    bottom: float
    timeframe: str
    kind: str                      # FVG | OB | BREAKER
    bullish: bool
    created_at: int
    quality: float = 0.0

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2.0

    @property
    def size(self) -> float:
        return self.top - self.bottom

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top


@dataclass
class Target:
    price: float
    liquidity: LiquidityLevel
    rr: float
    distance: float
    probability: float


@dataclass
class Setup:
    """A fully formed trade candidate."""

    setup_id: str
    created_at: int
    side: Side
    state: SetupState
    entry: float
    stop: float
    targets: list = field(default_factory=list)     # list[Target]
    rr: float = 0.0
    smc_score: float = 0.0
    ml_probability: Optional[float] = None
    features: dict = field(default_factory=dict)
    score_breakdown: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)
    zone: Optional[Zone] = None
    entry_type: str = "MARKET"                      # MARKET | LIMIT
    expires_at: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def risk(self) -> float:
        return abs(self.entry - self.stop)


@dataclass
class Order:
    side: Side
    kind: str          # MARKET | LIMIT | STOP | TAKE_PROFIT
    price: float
    qty: float
    reduce_only: bool = False
    tag: str = ""


@dataclass
class Fill:
    time: int
    price: float
    qty: float
    fee: float
    tag: str


@dataclass
class Trade:
    """Section 71 trade journal record."""

    setup_id: str
    symbol: str
    side: Side
    entry_time: int
    entry_price: float
    qty: float
    stop: float
    targets: list                       # list[float]
    smc_score: float
    ml_probability: Optional[float]
    market_state: str
    session: str
    features: dict = field(default_factory=dict)
    exit_time: Optional[int] = None
    exit_price: Optional[float] = None
    result: str = "OPEN"                # WIN | LOSS | BREAKEVEN | OPEN
    pnl: float = 0.0
    fees: float = 0.0
    funding: float = 0.0
    mfe: float = 0.0                    # max favourable excursion, in R
    mae: float = 0.0                    # max adverse excursion, in R
    r_multiple: float = 0.0
    tp_hits: list = field(default_factory=list)
    exit_reason: str = ""
    holding_minutes: float = 0.0
    model_version: str = ""
    equity_after: float = 0.0
    fills: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)
