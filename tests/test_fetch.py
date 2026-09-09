"""Download hardening: rate limits, retries, resume.

A year of M1 candles is roughly 350 requests.  The naive loop gets rate limited
part way through and loses everything, which is exactly the failure mode that
matters when the job runs unattended on a server.  These tests mock the HTTP
layer (the exchange is not contacted) and pin the recovery behaviour.
"""
from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
import urllib.error
from unittest import mock

from smcbot.data import loader
from smcbot.data.loader import (RateLimit, RequestRejected, append_csv,
                                fetch_binance, last_candle_time)

MIN = 60_000
START = 1_704_067_200_000


def kline(open_time: int, price: float = 100.0):
    """One Binance kline row in the exchange's array format."""
    return [open_time, str(price), str(price + 1), str(price - 1),
            str(price + 0.5), "10.0", open_time + MIN - 1]


def batch(n: int, first: int = START):
    return [kline(first + i * MIN, 100 + i) for i in range(n)]


class FakeResponse:
    def __init__(self, rows, weight=0):
        self._body = json.dumps(rows).encode()
        self.headers = {"X-MBX-USED-WEIGHT-1M": str(weight)}

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def http_error(code, retry_after=None):
    """Build an HTTPError with a real body.

    Passing ``fp=None`` makes urllib allocate a TemporaryFile that nothing ever
    closes, which surfaces as a ResourceWarning on Python 3.13+.  An explicit
    BytesIO keeps the tests quiet and still lets the code under test read the
    error body.
    """
    headers = {"Retry-After": str(retry_after)} if retry_after else {}
    body = io.BytesIO(b'{"code":-1121,"msg":"Invalid symbol."}')
    return urllib.error.HTTPError("u", code, "err", headers, body)


class TestFetchRetries(unittest.TestCase):
    def setUp(self):
        self.slept = []
        patcher = mock.patch.object(loader.time, "sleep", self.slept.append)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_rate_limit_is_retried_after_the_requested_delay(self):
        responses = [http_error(429, retry_after=7), FakeResponse(batch(10))]

        def fake(req, timeout=None):
            r = responses.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        with mock.patch.object(loader.urllib.request, "urlopen", fake):
            out = fetch_binance("BTCUSDT", start=str(START), end=str(START + 100 * MIN),
                                sleep_between=0)
        self.assertEqual(len(out), 10)
        self.assertIn(7.0, self.slept, "must honour Retry-After")

    def test_ban_response_uses_a_long_default_backoff(self):
        responses = [http_error(418), FakeResponse(batch(5))]

        def fake(req, timeout=None):
            r = responses.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        with mock.patch.object(loader.urllib.request, "urlopen", fake):
            fetch_binance(start=str(START), end=str(START + 10 * MIN),
                          sleep_between=0)
        self.assertGreaterEqual(max(self.slept), 300.0)

    def test_transport_errors_back_off_exponentially(self):
        responses = [urllib.error.URLError("reset"), urllib.error.URLError("reset"),
                     FakeResponse(batch(3))]

        def fake(req, timeout=None):
            r = responses.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        with mock.patch.object(loader.urllib.request, "urlopen", fake):
            out = fetch_binance(start=str(START), end=str(START + 10 * MIN),
                                sleep_between=0)
        self.assertEqual(len(out), 3)
        self.assertEqual(self.slept[:2], [1.0, 2.0])

    def test_gives_up_after_max_retries_with_a_clear_error(self):
        def always_fail(req, timeout=None):
            raise urllib.error.URLError("down")

        with mock.patch.object(loader.urllib.request, "urlopen", always_fail):
            with self.assertRaises(RuntimeError) as ctx:
                fetch_binance(start=str(START), end=str(START + 10 * MIN),
                              max_retries=3, sleep_between=0)
        self.assertIn("giving up", str(ctx.exception))

    def test_persistent_rate_limit_surfaces_as_rate_limit(self):
        def always_429(req, timeout=None):
            raise http_error(429, retry_after=1)

        with mock.patch.object(loader.urllib.request, "urlopen", always_429):
            with self.assertRaises(RateLimit):
                fetch_binance(start=str(START), end=str(START + 10 * MIN),
                              max_retries=2, sleep_between=0)

    def test_high_used_weight_triggers_a_cooldown(self):
        with mock.patch.object(loader.urllib.request, "urlopen",
                               lambda req, timeout=None: FakeResponse(batch(4), weight=9999)):
            fetch_binance(start=str(START), end=str(START + 10 * MIN),
                          sleep_between=0, weight_ceiling=100)
        self.assertIn(20.0, self.slept, "must pause before the IP weight limit")

    def test_client_errors_fail_fast_without_retrying(self):
        """A bad symbol is a typo, not a transient fault."""
        calls = []

        def bad_request(req, timeout=None):
            calls.append(1)
            raise http_error(400)

        with mock.patch.object(loader.urllib.request, "urlopen", bad_request):
            with self.assertRaises(RequestRejected):
                fetch_binance(symbol="NOTACOIN", start=str(START),
                              end=str(START + 10 * MIN), max_retries=6,
                              sleep_between=0)
        self.assertEqual(len(calls), 1, "must not retry a 4xx")
        self.assertEqual(self.slept, [])


class TestIncrementalWrite(unittest.TestCase):
    def test_on_batch_receives_each_page(self):
        pages = [FakeResponse(batch(1500, START)),
                 FakeResponse(batch(200, START + 1500 * MIN))]
        seen = []
        with mock.patch.object(loader.time, "sleep", lambda *_: None):
            with mock.patch.object(loader.urllib.request, "urlopen",
                                   lambda req, timeout=None: pages.pop(0)):
                out = fetch_binance(start=str(START),
                                    end=str(START + 5000 * MIN),
                                    sleep_between=0, on_batch=seen.append)
        self.assertEqual([len(p) for p in seen], [1500, 200])
        self.assertEqual(len(out), 1700)

    def test_append_csv_writes_the_header_once(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.csv")
            from smcbot.core.types import Candle
            a = [Candle(START, START + MIN, 1, 2, 0.5, 1.5, 3)]
            b = [Candle(START + MIN, START + 2 * MIN, 1, 2, 0.5, 1.5, 3)]
            append_csv(path, a)
            append_csv(path, b)
            with open(path) as fh:
                lines = fh.read().strip().splitlines()
            self.assertEqual(len(lines), 3)
            self.assertTrue(lines[0].startswith("open_time"))
            self.assertEqual(len(loader.load_csv(path)), 2)

    def test_resume_point_is_the_last_candle(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.csv")
            self.assertIsNone(last_candle_time(path))
            from smcbot.core.types import Candle
            append_csv(path, [Candle(START + i * MIN, START + (i + 1) * MIN,
                                     1, 2, 0.5, 1.5, 3) for i in range(5)])
            self.assertEqual(last_candle_time(path), START + 4 * MIN)


if __name__ == "__main__":
    unittest.main(verbosity=2)
