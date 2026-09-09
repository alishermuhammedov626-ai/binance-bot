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
import urllib.parse
import urllib.request
from typing import Iterable, Iterator, List, Optional

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


def fetch_binance(symbol: str = "BTCUSDT", interval: str = "1m",
                  start: Optional[str] = None, end: Optional[str] = None,
                  limit_total: int = 100_000, timeout: int = 30) -> List[Candle]:
    """Download M1 klines from Binance USD-M futures (public endpoint)."""
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
        req = urllib.request.Request(f"{BINANCE_FAPI}?{params}",
                                     headers={"User-Agent": "smcbot/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            rows = json.loads(resp.read().decode())
        if not rows:
            break
        for r in rows:
            out.append(Candle(int(r[0]), int(r[6]) + 1, float(r[1]), float(r[2]),
                              float(r[3]), float(r[4]), float(r[5])))
        cursor = int(rows[-1][0]) + MS_MINUTE
        if len(rows) < 1500:
            break
    return validate(out)


def stream(candles: Iterable[Candle]) -> Iterator[Candle]:
    """Chronological replay -- the only way candles enter the engines."""
    last = -1
    for c in candles:
        if c.open_time <= last:
            raise ValueError(f"non-monotonic candle at {c.open_time}")
        last = c.open_time
        yield c
