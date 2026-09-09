"""Unit tests for the SMC primitives: swings, structure, liquidity, FVG, sweeps."""
from __future__ import annotations

import unittest

from smcbot.config import (Config, LiquidityConfig, StructureConfig, SweepConfig,
                           SwingConfig, ZoneConfig)
from smcbot.core.series import CandleSeries, MarketBook, day_start, week_start
from smcbot.core.types import Candle, LiquidityKind, MarketState
from smcbot.engine.liquidity import LiquidityEngine
from smcbot.engine.structure import StructureEngine
from smcbot.engine.swings import SwingEngine
from smcbot.engine.zones import ZoneEngine

MIN = 60_000


def candles_from(prices, start=0, wick=0.5, volume=10.0, step=MIN):
    """One candle per price, with a symmetric wick."""
    out = []
    t = start
    for i, p in enumerate(prices):
        o = prices[i - 1] if i else p
        out.append(Candle(t, t + step, o, max(o, p) + wick, min(o, p) - wick,
                          p, volume))
        t += step
    return out


def ohlc(rows, start=0, step=MIN):
    out = []
    t = start
    for o, h, l, c, v in rows:
        out.append(Candle(t, t + step, o, h, l, c, v))
        t += step
    return out


class TestSwings(unittest.TestCase):
    def test_pivot_is_confirmed_only_after_right_bars(self):
        cfg = SwingConfig(adaptive=False, left={"M1": 2}, right={"M1": 2})
        series, eng = CandleSeries("M1"), SwingEngine("M1", cfg)
        rows = [(10, 11, 9, 10, 1), (10, 12, 9, 11, 1), (11, 16, 10, 15, 1),
                (15, 15, 10, 11, 1), (11, 12, 9, 10, 1), (10, 11, 8, 9, 1),
                (9, 10, 7, 8, 1)]
        found = []
        for i, c in enumerate(ohlc(rows)):
            series.append(c)
            for sw in eng.update(series):
                found.append((i, sw))
        highs = [(i, sw) for i, sw in found if sw.is_high]
        self.assertEqual(len(highs), 1)
        bar, swing = highs[0]
        self.assertEqual(bar, 4, "pivot at bar 2 must only confirm at bar 4")
        self.assertAlmostEqual(swing.price, 16)

    def test_no_pivot_in_monotonic_series(self):
        cfg = SwingConfig(adaptive=False, left={"M1": 2}, right={"M1": 2})
        series, eng = CandleSeries("M1"), SwingEngine("M1", cfg)
        for c in candles_from(list(range(20))):
            series.append(c)
            eng.update(series)
        self.assertEqual(eng.highs, [])
        self.assertEqual(eng.lows, [])

    def test_adaptive_wings_widen_with_volatility(self):
        cfg = SwingConfig(adaptive=True, left={"M1": 2}, right={"M1": 2},
                          vol_high=0.0001)
        series = CandleSeries("M1")
        for c in candles_from([100, 130, 90, 140, 80, 150] * 4):
            series.append(c)
        eng = SwingEngine("M1", cfg)
        left, right = eng._wings(series)
        self.assertEqual((left, right), (3, 3))


class TestStructure(unittest.TestCase):
    def _engine(self):
        return StructureEngine("M1", StructureConfig(),
                               SwingConfig(adaptive=False, left={"M1": 1},
                                           right={"M1": 1}))

    def test_choch_then_bos_in_both_directions(self):
        """A hand-built zig-zag: down leg, then reversal up.

        Expected: the first break of a swing low flips an unknown trend
        (CHOCH bearish), the next one continues it (BOS bearish), then the
        first break of a swing high flips it back (CHOCH bullish) before the
        next continues (BOS bullish).
        """
        eng = self._engine()
        series = CandleSeries("M1")
        rows = [                       # o,     h,    l,    c
            (100, 101, 98, 99, 1), (99, 99, 95, 96, 1), (99, 103, 99, 101, 1),
            (97, 97, 92, 94, 1), (96, 101, 96, 98, 1), (95, 95, 90, 91, 1),
            (94, 99, 94, 96, 1), (99, 105, 99, 103, 1), (102, 103, 97, 100, 1),
            (104, 110, 104, 108, 1), (106, 108, 102, 105, 1),
            (109, 115, 109, 113, 1),
        ]
        for c in ohlc(rows):
            series.append(c)
            eng.update(series)
        kinds = [(e.kind, e.bullish) for e in eng.internal.events]
        for expected in (("CHOCH", False), ("BOS", False),
                         ("CHOCH", True), ("BOS", True)):
            self.assertIn(expected, kinds, f"missing {expected} in {kinds}")
        self.assertLess(kinds.index(("CHOCH", True)), kinds.index(("BOS", True)))
        self.assertLess(kinds.index(("CHOCH", False)), kinds.index(("BOS", False)))

    def test_wick_through_swing_is_not_a_break(self):
        eng = self._engine()
        series = CandleSeries("M1")
        rows = [(100, 101, 99, 100, 1), (100, 106, 99, 100, 1),
                (100, 101, 99, 100, 1), (100, 101, 99, 100, 1),
                # a big wick above the swing high but closing back below
                (100, 120, 99, 100, 1), (100, 101, 99, 100, 1)]
        for c in ohlc(rows):
            series.append(c)
            eng.update(series)
        self.assertEqual([e for e in eng.internal.events if e.bullish], [])

    def test_premium_discount_and_midpoint_block(self):
        eng = self._engine()
        eng.range_high, eng.range_low = 200.0, 100.0
        self.assertEqual(eng.zone_label(180), "PREMIUM")
        self.assertEqual(eng.zone_label(120), "DISCOUNT")
        self.assertAlmostEqual(eng.equilibrium, 150.0)
        self.assertAlmostEqual(eng.range_position(125), 0.25)
        self.assertTrue(eng.in_midpoint_block(152))
        self.assertFalse(eng.in_midpoint_block(190))

    def test_range_state_after_long_silence(self):
        """External trend + no external break for N bars => RANGE (section 8)."""
        cfg = StructureConfig(range_lookback_bars=5)
        eng = StructureEngine("M1", cfg, SwingConfig(adaptive=False,
                                                     left={"M1": 1}, right={"M1": 1}))
        series = CandleSeries("M1")
        for c in ohlc([(100, 101, 99, 100, 1)] * 10):
            series.append(c)
        eng.external.trend = MarketState.BULLISH

        eng.external.bars_since_break = 1
        eng._update_state(series)
        self.assertEqual(eng.state, MarketState.BULLISH)

        eng.external.bars_since_break = 20
        eng._update_state(series)
        self.assertEqual(eng.state, MarketState.RANGE)

    def test_state_is_unknown_before_any_external_break(self):
        eng = self._engine()
        series = CandleSeries("M1")
        for c in ohlc([(100, 101, 99, 100, 1)] * 10):
            series.append(c)
            eng.update(series)
        self.assertEqual(eng.state, MarketState.UNKNOWN)


class TestLiquidity(unittest.TestCase):
    def test_previous_day_and_week_levels_appear_after_rollover(self):
        book = MarketBook()
        eng = LiquidityEngine(LiquidityConfig())
        start = 1_704_067_200_000            # Mon 2024-01-01 00:00 UTC
        price = 100.0
        for i in range(60 * 24 * 2 + 10):    # two full days plus a bit
            t = start + i * MIN
            price += 0.01 if i % 2 else -0.005
            book.push_m1(Candle(t, t + MIN, price, price + 1, price - 1, price, 5))
            eng.refresh(book, {})
        labels = {l.label for l in eng.levels.values()}
        self.assertIn("PrevDayHigh", labels)
        self.assertIn("PrevDayLow", labels)
        self.assertIn("DailyOpen", labels)

    def test_sweep_marking_and_cluster(self):
        eng = LiquidityEngine(LiquidityConfig())
        from smcbot.core.types import LiquidityLevel
        eng._add("a", LiquidityLevel(100.0, LiquidityKind.PREVIOUS_HIGH, True,
                                     "DAILY", 9.0, 0, 0, "PDH"))
        eng._add("b", LiquidityLevel(100.4, LiquidityKind.EQUAL_HIGH, True,
                                     "M15", 9.0, 0, 0, "EQH"))
        size, strength, is_cluster = eng.cluster_at(100.2, atr=4.0, is_high=True)
        self.assertEqual(size, 2)
        self.assertTrue(is_cluster)
        self.assertAlmostEqual(strength, 18.0)
        self.assertEqual(len(eng.alive(True)), 2)
        eng.mark_sweeps(high=101.0, low=99.0, ts=123)
        self.assertEqual(len(eng.alive(True)), 0)

    def test_targets_are_ordered_outward(self):
        from smcbot.core.types import LiquidityLevel
        eng = LiquidityEngine(LiquidityConfig())
        for i, p in enumerate([110.0, 105.0, 120.0]):
            eng._add(f"h{i}", LiquidityLevel(p, LiquidityKind.INTERNAL, True,
                                             "M5", 5.0, 0, 0))
        self.assertEqual([l.price for l in eng.targets_above(100)],
                         [105.0, 110.0, 120.0])
        self.assertEqual(eng.targets_below(100), [])


class TestZones(unittest.TestCase):
    def test_bullish_fvg_detection_and_mitigation(self):
        eng = ZoneEngine("M1", ZoneConfig(min_fvg_atr=0.0))
        series = CandleSeries("M1")
        rows = [(100, 101, 99, 100, 1), (101, 110, 100, 109, 5),
                (109, 112, 105, 111, 3)]           # gap: c1.high 101 < c3.low 105
        for c in ohlc(rows):
            series.append(c)
            eng.update(series, [])
        self.assertEqual(len(eng.fvgs), 1)
        gap = eng.fvgs[0]
        self.assertTrue(gap.bullish)
        self.assertAlmostEqual(gap.bottom, 101)
        self.assertAlmostEqual(gap.top, 105)
        self.assertIsNone(gap.mitigated_at)

        # Price trades back through the gap -> mitigated.
        series.append(Candle(3 * MIN, 4 * MIN, 106, 107, 100, 101, 2))
        eng.update(series, [])
        self.assertIsNotNone(eng.fvgs[0].mitigated_at)

    def test_bearish_fvg(self):
        eng = ZoneEngine("M1", ZoneConfig(min_fvg_atr=0.0))
        series = CandleSeries("M1")
        for c in ohlc([(110, 111, 109, 110, 1), (109, 110, 100, 101, 5),
                       (101, 104, 99, 100, 3)]):
            series.append(c)
            eng.update(series, [])
        self.assertEqual(len(eng.fvgs), 1)
        self.assertFalse(eng.fvgs[0].bullish)
        self.assertAlmostEqual(eng.fvgs[0].bottom, 104)
        self.assertAlmostEqual(eng.fvgs[0].top, 109)

    def test_no_fvg_when_candles_overlap(self):
        eng = ZoneEngine("M1", ZoneConfig(min_fvg_atr=0.0))
        series = CandleSeries("M1")
        for c in ohlc([(100, 105, 99, 104, 1), (104, 108, 103, 107, 1),
                       (107, 110, 104, 109, 1)]):
            series.append(c)
            eng.update(series, [])
        self.assertEqual(eng.fvgs, [])

    def test_order_block_is_last_opposite_candle_before_the_break(self):
        from smcbot.core.types import StructureEvent
        eng = ZoneEngine("M1", ZoneConfig())
        series = CandleSeries("M1")
        for c in ohlc([(100, 101, 99, 100, 1), (100, 101, 95, 96, 2),
                       (96, 108, 96, 107, 6), (107, 112, 106, 111, 5)]):
            series.append(c)
        ev = StructureEvent(series[-1].close_time, "M1", "BOS", True, 101, True, 2.0)
        eng.update(series, [ev])
        self.assertEqual(len(eng.obs), 1)
        ob = eng.obs[0]
        self.assertTrue(ob.bullish)
        self.assertAlmostEqual(ob.bottom, 95)     # the down candle's range
        self.assertAlmostEqual(ob.top, 101)


class TestSweeps(unittest.TestCase):
    def test_sweep_requires_close_back_inside(self):
        from smcbot.core.types import LiquidityLevel
        from smcbot.engine.sweep import SweepEngine
        series = CandleSeries("M1")
        for c in ohlc([(100, 101, 99, 100, 10)] * 20):
            series.append(c)
        liq = LiquidityEngine(LiquidityConfig())
        liq._add("h", LiquidityLevel(101.0, LiquidityKind.PREVIOUS_HIGH, True,
                                     "DAILY", 9.0, 0, 0, "PDH"))
        eng = SweepEngine("M1", SweepConfig(min_score=0.0))

        # Closes above the level: a breakout, not a sweep.
        series.append(Candle(21 * MIN, 22 * MIN, 100, 102, 99.5, 101.8, 30))
        self.assertEqual(eng.update(series, liq), [])

        # Wick above, close back below: a sweep.
        series2 = CandleSeries("M1")
        for c in ohlc([(100, 101, 99, 100, 10)] * 20):
            series2.append(c)
        eng2 = SweepEngine("M1", SweepConfig(min_score=0.0))
        series2.append(Candle(21 * MIN, 22 * MIN, 100, 102.0, 99.5, 99.8, 30))
        liq2 = LiquidityEngine(LiquidityConfig())
        liq2._add("h", LiquidityLevel(101.0, LiquidityKind.PREVIOUS_HIGH, True,
                                      "DAILY", 9.0, 0, 0, "PDH"))
        sweeps = eng2.update(series2, liq2)
        self.assertEqual(len(sweeps), 1)
        self.assertTrue(sweeps[0].is_high_sweep)
        self.assertGreater(sweeps[0].score, 0)


class TestCalendar(unittest.TestCase):
    def test_week_starts_on_monday_utc(self):
        import datetime as dt
        for iso in ("2024-01-01", "2024-01-04", "2024-01-07"):
            t = int(dt.datetime.fromisoformat(iso).replace(
                tzinfo=dt.timezone.utc).timestamp() * 1000)
            start = dt.datetime.fromtimestamp(week_start(t) / 1000, dt.timezone.utc)
            self.assertEqual(start.weekday(), 0)
        self.assertEqual(day_start(1_704_067_200_000 + 3600_000),
                         1_704_067_200_000)

    def test_session_classification(self):
        from smcbot.config import SessionConfig
        from smcbot.core.series import SessionTracker
        from smcbot.core.types import Session
        tr = SessionTracker(SessionConfig())
        base = 1_704_067_200_000
        self.assertIn(Session.ASIA, tr.session_of(base + 2 * 3_600_000))
        self.assertIn(Session.LONDON, tr.session_of(base + 9 * 3_600_000))
        self.assertEqual(tr.primary_session(base + 14 * 3_600_000),
                         Session.NEW_YORK)


if __name__ == "__main__":
    unittest.main(verbosity=2)
