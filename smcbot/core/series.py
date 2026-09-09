"""Streaming, append-only market data structures.

The whole anti-look-ahead guarantee of the backtest rests on this file:

* Candles are pushed one M1 bar at a time, in chronological order.
* Higher timeframes are *aggregated* from that stream and a bar is published
  only once its final M1 constituent has closed.  A partially formed M15 bar is
  never visible to any engine.
* Daily / weekly / session extremes are likewise only rolled over when the
  period has actually finished.
* ``now`` is the close time of the newest M1 bar.  Any object with
  ``confirmed_at > now`` is by definition invisible.

Everything downstream reads from here and therefore inherits the guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .types import TF_MS, Candle, Session
from ..config import SessionConfig

DAY_MS = 86_400_000
WEEK_MS = 7 * DAY_MS
# 1970-01-01 was a Thursday, so the first Monday is epoch + 4 days.  Weekly
# levels therefore roll at Monday 00:00 UTC, matching exchange weekly candles.
_WEEK_ANCHOR = 4 * DAY_MS


def week_start(ts: int) -> int:
    return ((ts - _WEEK_ANCHOR) // WEEK_MS) * WEEK_MS + _WEEK_ANCHOR


def day_start(ts: int) -> int:
    return (ts // DAY_MS) * DAY_MS


class CandleSeries:
    """Append-only closed-candle series with incremental indicators."""

    def __init__(self, timeframe: str, atr_period: int = 14, max_len: int = 4000):
        self.timeframe = timeframe
        self.atr_period = atr_period
        self.max_len = max_len
        self.candles: List[Candle] = []
        self._tr: List[float] = []
        self._atr: List[float] = []
        self._vol_ma: List[float] = []
        self._vol_sum = 0.0
        self._vol_window = 20
        self.dropped = 0            # candles evicted by the ring buffer

    # ------------------------------------------------------------------
    def append(self, candle: Candle) -> None:
        if self.candles and candle.open_time <= self.candles[-1].open_time:
            raise ValueError(
                f"{self.timeframe}: out-of-order candle "
                f"{candle.open_time} <= {self.candles[-1].open_time}"
            )
        prev_close = self.candles[-1].close if self.candles else candle.open
        tr = max(
            candle.high - candle.low,
            abs(candle.high - prev_close),
            abs(candle.low - prev_close),
        )
        self.candles.append(candle)
        self._tr.append(tr)

        n = min(self.atr_period, len(self._tr))
        self._atr.append(sum(self._tr[-n:]) / n)

        self._vol_sum += candle.volume
        if len(self.candles) > self._vol_window:
            self._vol_sum -= self.candles[-self._vol_window - 1].volume
        self._vol_ma.append(self._vol_sum / min(len(self.candles), self._vol_window))

        if len(self.candles) > self.max_len:
            trim = len(self.candles) - self.max_len
            self.candles = self.candles[trim:]
            self._tr = self._tr[trim:]
            self._atr = self._atr[trim:]
            self._vol_ma = self._vol_ma[trim:]
            self.dropped += trim

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.candles)

    def __getitem__(self, i):
        return self.candles[i]

    def __iter__(self):
        return iter(self.candles)

    @property
    def ready(self) -> bool:
        return len(self.candles) > self.atr_period

    @property
    def last(self) -> Optional[Candle]:
        return self.candles[-1] if self.candles else None

    @property
    def now(self) -> int:
        return self.candles[-1].close_time if self.candles else 0

    @property
    def atr(self) -> float:
        return self._atr[-1] if self._atr else 0.0

    def atr_at(self, i: int) -> float:
        return self._atr[i] if self._atr else 0.0

    def vol_ma(self, i: int = -1) -> float:
        return self._vol_ma[i] if self._vol_ma else 0.0

    @property
    def atr_pct(self) -> float:
        c = self.last
        if not c or c.close <= 0:
            return 0.0
        return self.atr / c.close

    def window(self, n: int) -> List[Candle]:
        return self.candles[-n:] if n > 0 else []

    def highest(self, n: int) -> float:
        w = self.window(n)
        return max(c.high for c in w) if w else 0.0

    def lowest(self, n: int) -> float:
        w = self.window(n)
        return min(c.low for c in w) if w else 0.0

    def abs_index(self, i: int) -> int:
        """Index that survives ring-buffer eviction (stable across trims)."""
        return self.dropped + (i if i >= 0 else len(self.candles) + i)


@dataclass
class Period:
    """A completed or in-progress calendar period (day / week / session)."""

    start: int
    end: int
    high: float
    low: float
    open: float
    close: float
    label: str = ""

    def update(self, c: Candle) -> None:
        self.high = max(self.high, c.high)
        self.low = min(self.low, c.low)
        self.close = c.close
        self.end = c.close_time

    @classmethod
    def start_from(cls, c: Candle, start: int, label: str = "") -> "Period":
        return cls(start=start, end=c.close_time, high=c.high, low=c.low,
                   open=c.open, close=c.close, label=label)


class SessionTracker:
    """Section 39/40 -- per-session extremes, rolled over only on completion."""

    def __init__(self, cfg: SessionConfig):
        self.cfg = cfg
        self.current: Dict[str, Period] = {}
        self.completed: Dict[str, List[Period]] = {s.value: [] for s in Session}

    def session_of(self, ts: int) -> List[Session]:
        hour = (ts % DAY_MS) // 3_600_000
        out = []
        for name, (a, b) in (
            (Session.ASIA, self.cfg.asia),
            (Session.LONDON, self.cfg.london),
            (Session.NEW_YORK, self.cfg.new_york),
            (Session.LATE, self.cfg.late),
        ):
            if a == b:
                continue                    # empty block
            inside = a <= hour < b if a < b else (hour >= a or hour < b)
            if inside:
                out.append(name)
        return out or [Session.OFF]

    def primary_session(self, ts: int) -> Session:
        act = self.session_of(ts)
        # Overlaps resolve to the later-opening (more dominant) session.
        for pref in (Session.LATE, Session.NEW_YORK, Session.LONDON,
                     Session.ASIA):
            if pref in act:
                return pref
        return Session.OFF

    def update(self, c: Candle) -> None:
        active = {s.value for s in self.session_of(c.open_time)}
        for name in list(self.current):
            if name not in active:
                self.completed[name].append(self.current.pop(name))
                self.completed[name] = self.completed[name][-10:]
        for name in active:
            if name == Session.OFF.value:
                continue
            if name in self.current:
                self.current[name].update(c)
            else:
                self.current[name] = Period.start_from(c, c.open_time, name)

    def last_completed(self, name: str) -> Optional[Period]:
        lst = self.completed.get(name, [])
        return lst[-1] if lst else None

    def running(self, name: str) -> Optional[Period]:
        return self.current.get(name)


class MarketBook:
    """M1 stream in, synchronised M1/M5/M15 + calendar context out."""

    def __init__(self, session_cfg: Optional[SessionConfig] = None,
                 max_len: int = 4000):
        self.m1 = CandleSeries("M1", max_len=max_len)
        self.m5 = CandleSeries("M5", max_len=max_len)
        self.m15 = CandleSeries("M15", max_len=max_len)
        self.series: Dict[str, CandleSeries] = {
            "M1": self.m1, "M5": self.m5, "M15": self.m15
        }
        self._partial: Dict[str, Optional[List]] = {"M5": None, "M15": None}
        self.sessions = SessionTracker(session_cfg or SessionConfig())
        self.day: Optional[Period] = None
        self.week: Optional[Period] = None
        self.prev_days: List[Period] = []
        self.prev_weeks: List[Period] = []
        self.closed_timeframes: List[str] = []   # which TFs closed on last push

    # ------------------------------------------------------------------
    @property
    def now(self) -> int:
        return self.m1.now

    @property
    def price(self) -> float:
        c = self.m1.last
        return c.close if c else 0.0

    def ready(self, min_bars: Dict[str, int]) -> bool:
        return all(len(self.series[tf]) >= n for tf, n in min_bars.items())

    def push_m1(self, c: Candle) -> List[str]:
        """Feed one closed M1 candle.  Returns the timeframes that closed."""
        self.m1.append(c)
        closed = ["M1"]
        for tf in ("M5", "M15"):
            done = self._aggregate(tf, c)
            if done is not None:
                self.series[tf].append(done)
                closed.append(tf)
        self._roll_calendar(c)
        self.sessions.update(c)
        self.closed_timeframes = closed
        return closed

    def _aggregate(self, tf: str, c: Candle) -> Optional[Candle]:
        step = TF_MS[tf]
        bucket = (c.open_time // step) * step
        cur = self._partial[tf]
        if cur is None or cur[0] != bucket:
            cur = [bucket, c.open, c.high, c.low, c.close, c.volume]
            self._partial[tf] = cur
        else:
            cur[2] = max(cur[2], c.high)
            cur[3] = min(cur[3], c.low)
            cur[4] = c.close
            cur[5] += c.volume
        # The bar is published only when its last M1 constituent has closed.
        if c.close_time >= bucket + step:
            self._partial[tf] = None
            return Candle(bucket, bucket + step, cur[1], cur[2], cur[3], cur[4], cur[5])
        return None

    def _roll_calendar(self, c: Candle) -> None:
        ds, ws = day_start(c.open_time), week_start(c.open_time)
        if self.day is None or self.day.start != ds:
            if self.day is not None:
                self.prev_days.append(self.day)
                self.prev_days = self.prev_days[-15:]
            self.day = Period.start_from(c, ds, "DAY")
        else:
            self.day.update(c)
        if self.week is None or self.week.start != ws:
            if self.week is not None:
                self.prev_weeks.append(self.week)
                self.prev_weeks = self.prev_weeks[-8:]
            self.week = Period.start_from(c, ws, "WEEK")
        else:
            self.week.update(c)

    @property
    def prev_day(self) -> Optional[Period]:
        return self.prev_days[-1] if self.prev_days else None

    @property
    def prev_week(self) -> Optional[Period]:
        return self.prev_weeks[-1] if self.prev_weeks else None

    def current_session(self) -> Session:
        return self.sessions.primary_session(self.now - 1)


def build_book_from_m1(candles: Iterable[Candle], session_cfg=None) -> MarketBook:
    book = MarketBook(session_cfg)
    for c in candles:
        book.push_m1(c)
    return book
