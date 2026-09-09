"""M1 candle input: CSV files or Binance USD-M futures klines.

CSV format (header optional):
``open_time,open,high,low,close,volume`` with ``open_time`` in ms or ISO-8601.
Candles are validated: strictly increasing, no duplicates, gaps reported.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Iterable, Iterator, List, Optional

from ..core.types import MS_MINUTE, Candle

BINANCE_FAPI = "https://fapi.binance.com/fapi/v1/klines"


def _parse_time(value: str) -> int:
    value = value.strip()
    if value.isdigit():
        ts = int(value)
        return ts if ts > 10_000_000_000 else ts * 1000
    d = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return int(d.timestamp() * 1000)


def load_csv(path: str, limit: Optional[int] = None) -> List[Candle]:
    out: List[Candle] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row or row[0].strip().lower().startswith(("open_time", "time", "date")):
                continue
            try:
                ts = _parse_time(row[0])
                o, h, l, c = (float(row[i]) for i in (1, 2, 3, 4))
                v = float(row[5]) if len(row) > 5 and row[5] else 0.0
            except (ValueError, IndexError):
                continue
            out.append(Candle(ts, ts + MS_MINUTE, o, h, l, c, v))
            if limit and len(out) >= limit:
                break
    return validate(out)


def save_csv(path: str, candles: Iterable[Candle]) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    n = 0
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["open_time", "open", "high", "low", "close", "volume"])
        for c in candles:
            w.writerow([c.open_time, c.open, c.high, c.low, c.close, c.volume])
            n += 1
    return n


def validate(candles: List[Candle], drop_bad: bool = True) -> List[Candle]:
    """Sort, de-duplicate and sanity-check OHLC relationships."""
    candles = sorted(candles, key=lambda c: c.open_time)
    out: List[Candle] = []
    for c in candles:
        if out and c.open_time == out[-1].open_time:
            continue
        bad = not (c.low <= c.open <= c.high and c.low <= c.close <= c.high) or c.high < c.low
        if bad and drop_bad:
            continue
        out.append(c)
    return out


def gaps(candles: List[Candle]) -> List[tuple]:
    return [(a.close_time, b.open_time)
            for a, b in zip(candles, candles[1:]) if b.open_time != a.close_time]


class RateLimit(Exception):
    """429 / 418 from the exchange -- back off, do not hammer."""

    def __init__(self, retry_after: float, status: int):
        super().__init__(f"rate limited ({status}), retry after {retry_after:.0f}s")
        self.retry_after = retry_after
        self.status = status


class RequestRejected(Exception):
    """A 4xx the exchange will keep rejecting -- bad symbol, bad interval.

    Retrying cannot fix it, so it is raised straight through instead of
    burning six backoffs before reporting a typo.
    """


def _request(url: str, timeout: int) -> tuple:
    """One GET.  Returns ``(rows, used_weight)`` or raises :class:`RateLimit`."""
    req = urllib.request.Request(url, headers={"User-Agent": "smcbot/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            weight = resp.headers.get("X-MBX-USED-WEIGHT-1M")
            return json.loads(resp.read().decode()), int(weight or 0)
    except urllib.error.HTTPError as exc:
        if exc.code in (429, 418):
            # 418 means the IP is already banned; Retry-After is authoritative.
            retry_after = float(exc.headers.get("Retry-After") or
                                (60 if exc.code == 429 else 300))
            raise RateLimit(retry_after, exc.code) from exc
        if 400 <= exc.code < 500:
            detail = ""
            try:
                detail = exc.read().decode()[:200]
            except Exception:
                pass
            finally:
                exc.close()      # release the body; a long fetch must not leak fds
            raise RequestRejected(f"HTTP {exc.code} from the exchange: "
                                  f"{detail or exc.reason}") from exc
        raise


def fetch_binance(symbol: str = "BTCUSDT", interval: str = "1m",
                  start: Optional[str] = None, end: Optional[str] = None,
                  limit_total: int = 100_000, timeout: int = 30,
                  sleep_between: float = 0.25, max_retries: int = 6,
                  weight_ceiling: int = 1800,
                  progress: Optional[Callable[[int, int], None]] = None,
                  on_batch: Optional[Callable[[List[Candle]], None]] = None
                  ) -> List[Candle]:
    """Download klines from Binance USD-M futures, safely enough for a server.

    A year of M1 data is ~350 requests, so the naive loop gets rate limited and
    dies half way.  This one paces itself (``sleep_between``), watches the
    ``X-MBX-USED-WEIGHT-1M`` header and pauses before hitting the IP limit,
    retries 429/418/5xx and transport errors with exponential backoff, and can
    hand each batch to ``on_batch`` so a long download is written to disk as it
    goes instead of only at the end.
    """
    start_ms = _parse_time(start) if start else int(
        (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=7)).timestamp() * 1000)
    end_ms = _parse_time(end) if end else int(
        dt.datetime.now(dt.timezone.utc).timestamp() * 1000)

    out: List[Candle] = []
    cursor = start_ms
    while cursor < end_ms and len(out) < limit_total:
        params = urllib.parse.urlencode(
            {"symbol": symbol, "interval": interval, "startTime": cursor,
             "endTime": end_ms, "limit": 1500})
        url = f"{BINANCE_FAPI}?{params}"

        rows = None
        for attempt in range(max_retries):
            try:
                rows, weight = _request(url, timeout)
                if weight and weight > weight_ceiling:
                    time.sleep(20.0)          # cool down before the hard limit
                break
            except RateLimit as exc:
                if attempt == max_retries - 1:
                    raise
                time.sleep(exc.retry_after)
            except RequestRejected:
                raise                      # a typo does not get better on retry
            except (urllib.error.URLError, TimeoutError, OSError,
                    json.JSONDecodeError) as exc:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"giving up on {symbol} at {cursor}: {exc}") from exc
                time.sleep(min(2.0 ** attempt, 30.0))

        if not rows:
            break
        batch = [Candle(int(r[0]), int(r[6]) + 1, float(r[1]), float(r[2]),
                        float(r[3]), float(r[4]), float(r[5])) for r in rows]
        out.extend(batch)
        if on_batch:
            on_batch(batch)
        if progress:
            progress(len(out), batch[-1].open_time)

        cursor = int(rows[-1][0]) + MS_MINUTE
        if len(rows) < 1500:
            break
        if sleep_between:
            time.sleep(sleep_between)
    return validate(out)


def last_candle_time(path: str) -> Optional[int]:
    """Open time of the final candle in a CSV, for resuming a download."""
    if not os.path.exists(path):
        return None
    last = None
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if row and row[0].strip().isdigit():
                last = row[0]
    return _parse_time(last) if last else None


def append_csv(path: str, candles: Iterable[Candle]) -> int:
    """Append candles, writing the header only for a new file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fresh = not os.path.exists(path) or os.path.getsize(path) == 0
    n = 0
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if fresh:
            w.writerow(["open_time", "open", "high", "low", "close", "volume"])
        for c in candles:
            w.writerow([c.open_time, c.open, c.high, c.low, c.close, c.volume])
            n += 1
    return n


def stream(candles: Iterable[Candle]) -> Iterator[Candle]:
    """Chronological replay -- the only way candles enter the engines."""
    last = -1
    for c in candles:
        if c.open_time <= last:
            raise ValueError(f"non-monotonic candle at {c.open_time}")
        last = c.open_time
        yield c
