"""The tests that matter most: proof the bot cannot see the future.

Three independent angles:

1. **Prefix determinism** -- decisions taken during the first K bars must be
   byte-identical whether the engine is later fed 2K bars or stopped at K.
2. **Future mutation** -- replacing every candle after bar K with completely
   different data must not change a single decision taken at or before K.
   This is the strongest statement: if any engine peeked ahead, the mutated
   future would change the past.
3. **Structural invariants** -- swings only publish after their right-hand
   bars, higher timeframes never expose an unfinished bar, and no object is
   ever consulted before its ``confirmed_at``.
"""
from __future__ import annotations

import random
import unittest

from smcbot.backtest.engine import Backtester
from smcbot.config import Config
from smcbot.core.series import MarketBook
from smcbot.core.types import TF_MS, Candle
from smcbot.data.synthetic import generate
from smcbot.engine.context import SMCContext
from smcbot.strategy.setup import SignalEngine


def fingerprint(bt: Backtester):
    """Everything the bot decided, in order."""
    return [
        (t.setup_id, t.side.value, round(t.entry_price, 6), round(t.stop, 6),
         t.entry_time, t.exit_time, round(t.pnl, 6), t.exit_reason,
         round(t.smc_score, 4))
        for t in bt.trades
    ]


def entry_fingerprint(bt: Backtester, cut_time: int):
    """Trades *opened* at or before ``cut_time``, described by entry-time facts.

    Keying on entry rather than exit is what makes the mutation test sharp: a
    trade that opens before the cut and closes after it still gets compared, so
    a leak that changes *whether* a trade is taken cannot hide behind an exit
    that happens in the rewritten future.
    """
    return [
        (t.setup_id, t.side.value, round(t.entry_price, 6), round(t.stop, 6),
         t.entry_time, round(t.smc_score, 4), round(t.targets[0], 6) if t.targets else None)
        for t in bt.trades if t.entry_time <= cut_time
    ]



def mutate_future(candles, cut, seed=1234):
    """Replace every candle from ``cut`` onwards with a completely different path."""
    rng = random.Random(seed)
    out = list(candles[:cut])
    price = candles[cut - 1].close
    for c in candles[cut:]:
        price *= (1 + rng.gauss(-0.0005, 0.004))
        out.append(Candle(
            c.open_time, c.close_time, price,
            price * (1 + abs(rng.gauss(0, 0.003))),
            price * (1 - abs(rng.gauss(0, 0.003))), price,
            c.volume * rng.uniform(0.2, 4.0)))
    return out


def bar_index_of(candles, close_time):
    return (close_time - candles[0].close_time) // 60_000


def cut_points(candles, trades, n=4, offset=2):
    """Cuts placed just after each trade *opens*.

    Placement is what makes the test sharp.  A leak that peeks ``h`` bars ahead
    only shows up if the rewritten future lands inside a peek window whose
    verdict is still undecided.  Cutting a couple of bars after an entry does
    exactly that; cutting after the exit does not, because by then the peek has
    already been resolved by pre-cut candles.
    """
    out = []
    for t in trades[:n]:
        idx = bar_index_of(candles, t.entry_time) + offset
        if 100 < idx < len(candles) - 10:
            out.append(int(idx))
    return out


def run_bt(candles, cfg=None):
    bt = Backtester(cfg or Config())
    bt.run(candles)
    return bt


class TestPrefixDeterminism(unittest.TestCase):
    """Angle 1: a longer future must not rewrite the past."""

    @classmethod
    def setUpClass(cls):
        cls.candles = generate(60 * 24 * 24, seed=42)
        cls.cut = len(cls.candles) // 2

    def test_trades_before_cut_are_identical(self):
        full = run_bt(self.candles)
        part = run_bt(self.candles[:self.cut])

        cut_time = self.candles[self.cut - 1].close_time
        # A trade still open at the cut is force-closed in the short run, so
        # compare only trades that had already closed before the cut.
        full_closed = [t for t in fingerprint(full) if t[5] <= cut_time]
        part_closed = [t for t in fingerprint(part)
                       if t[5] <= cut_time and t[7] != "END_OF_DATA"]
        self.assertGreater(len(part_closed), 0, "no trades to compare")
        self.assertEqual(full_closed[:len(part_closed)], part_closed)

    def test_signals_before_cut_are_identical(self):
        def signals(candles):
            ctx, sig = SMCContext(Config()), SignalEngine(Config())
            out = []
            for c in candles:
                ctx.on_m1(c)
                for s in sig.on_bar(ctx):
                    out.append((s.created_at, s.setup_id, s.side.value,
                                round(s.entry, 6), round(s.stop, 6),
                                round(s.smc_score, 4)))
                    sig.mark_traded(s.setup_id)
            return out

        full = signals(self.candles)
        part = signals(self.candles[:self.cut])
        cut_time = self.candles[self.cut - 1].close_time
        self.assertGreater(len(part), 0, "no signals generated")
        self.assertEqual([s for s in full if s[0] <= cut_time], part)


class TestFutureMutation(unittest.TestCase):
    """Angle 2: rewrite the future, the past must not move."""

    @classmethod
    def setUpClass(cls):
        cls.candles = generate(60 * 24 * 10, seed=9)
        cls.base = run_bt(cls.candles)
        cls.cuts = cut_points(cls.candles, cls.base.trades)

    def test_setup_has_trades_and_cuts(self):
        self.assertGreater(len(self.base.trades), 0)
        self.assertGreater(len(self.cuts), 0, "no cut points to test")

    def test_mutated_future_does_not_change_past_decisions(self):
        for cut in self.cuts:
            with self.subTest(cut=cut):
                mutated = run_bt(mutate_future(self.candles, cut))
                cut_time = self.candles[cut - 1].close_time
                expected = entry_fingerprint(self.base, cut_time)
                actual = entry_fingerprint(mutated, cut_time)
                self.assertGreater(len(expected), 0)
                self.assertEqual(expected, actual)

    def test_closed_trades_before_cut_are_identical(self):
        base_fp = fingerprint(self.base)
        for cut in self.cuts:
            with self.subTest(cut=cut):
                mutated = run_bt(mutate_future(self.candles, cut))
                cut_time = self.candles[cut - 1].close_time
                expected = [t for t in base_fp if t[5] <= cut_time]
                actual = [t for t in fingerprint(mutated) if t[5] <= cut_time]
                self.assertEqual(expected, actual)

    def test_equity_curve_identical_before_cut(self):
        for cut in self.cuts:
            with self.subTest(cut=cut):
                mutated = run_bt(mutate_future(self.candles, cut))
                self.assertEqual(
                    [round(v, 8) for _, v in self.base.equity_curve[:cut]],
                    [round(v, 8) for _, v in mutated.equity_curve[:cut]])


class TestStructuralInvariants(unittest.TestCase):
    """Angle 3: the mechanics that make the above true."""

    def setUp(self):
        self.candles = generate(60 * 24 * 5, seed=11)

    def test_higher_timeframes_only_publish_closed_bars(self):
        book = MarketBook()
        for c in self.candles:
            book.push_m1(c)
            for tf in ("M5", "M15"):
                last = book.series[tf].last
                if last is None:
                    continue
                # The published bar must be entirely in the past.
                self.assertLessEqual(last.close_time, c.close_time)
                self.assertEqual(last.close_time - last.open_time, TF_MS[tf])

    def test_aggregated_bar_matches_its_m1_constituents(self):
        book = MarketBook()
        for c in self.candles[:600]:
            book.push_m1(c)
        for bar in book.m15:
            parts = [c for c in book.m1
                     if bar.open_time <= c.open_time < bar.close_time]
            self.assertEqual(len(parts), 15)
            self.assertEqual(bar.open, parts[0].open)
            self.assertEqual(bar.close, parts[-1].close)
            self.assertEqual(bar.high, max(p.high for p in parts))
            self.assertEqual(bar.low, min(p.low for p in parts))
            self.assertAlmostEqual(bar.volume, sum(p.volume for p in parts), 6)

    def test_swings_confirm_only_after_their_right_side_bars(self):
        ctx = SMCContext(Config())
        for c in self.candles:
            ctx.on_m1(c)
            now = ctx.now
            for tf in ("M15", "M5", "M1"):
                eng = ctx.views[tf].structure.internal_swings
                for s in eng.highs[-5:] + eng.lows[-5:]:
                    # Known now => confirmed now or earlier, and strictly after
                    # the pivot candle itself.
                    self.assertLessEqual(s.confirmed_at, now)
                    self.assertGreater(s.confirmed_at, s.time)

    def test_liquidity_levels_are_never_known_before_confirmation(self):
        ctx = SMCContext(Config())
        for c in self.candles:
            ctx.on_m1(c)
            for level in ctx.liquidity.levels.values():
                self.assertLessEqual(level.confirmed_at, ctx.now)

    def test_sweeps_reference_only_already_known_levels(self):
        ctx = SMCContext(Config())
        for c in self.candles:
            ctx.on_m1(c)
            for view in ctx.views.values():
                for s in view.sweeps.sweeps:
                    self.assertLessEqual(s.level.confirmed_at, s.time)

    def test_series_rejects_out_of_order_candles(self):
        book = MarketBook()
        c = self.candles[10]
        book.push_m1(c)
        with self.assertRaises(ValueError):
            book.push_m1(self.candles[5])


class TestExecutionCausality(unittest.TestCase):
    """A signal formed on bar t can never be filled on bar t."""

    def test_fill_is_never_on_the_signal_bar(self):
        cfg = Config()
        bt = Backtester(cfg)
        candles = generate(60 * 24 * 10, seed=21)

        signal_times = {}
        original_decide = bt._decide

        def spy(candle):
            before = bt.pending
            original_decide(candle)
            if bt.pending is not None and bt.pending is not before:
                signal_times[bt.pending.setup.setup_id] = candle.close_time

        bt._decide = spy
        bt.run(candles)
        self.assertGreater(len(bt.trades), 0, "no trades produced")
        for t in bt.trades:
            placed = signal_times.get(t.setup_id)
            if placed is None:
                continue
            # Entry timestamps are bar *open* times; the fill bar must open at
            # or after the close of the bar that produced the signal.
            self.assertGreaterEqual(t.entry_time, placed)

    def test_stop_is_assumed_before_target_within_one_bar(self):
        """A bar spanning both stop and TP must be resolved as a loss."""
        from smcbot.core.types import Setup, Side, Target, LiquidityLevel, LiquidityKind
        from smcbot.strategy.decision import Decision
        from smcbot.backtest.engine import PendingOrder

        cfg = Config()
        bt = Backtester(cfg)
        bt.risk.equity = 1000.0
        level = LiquidityLevel(110.0, LiquidityKind.INTERNAL, True, "M5", 5, 0, 0)
        setup = Setup(setup_id="x", created_at=0, side=Side.BUY,
                      state=None, entry=100.0, stop=99.0,
                      targets=[Target(102.0, level, 2.0, 2.0, 0.5)], rr=2.0,
                      smc_score=80.0, expires_at=10 ** 15)
        decision = Decision(True, "ok", setup, None, 0.005)
        bt.pending = PendingOrder(setup, decision, "MARKET", 100.0, 1.0, 0)

        # Bar 1 fills at the open; bar 2 engulfs both the stop and the target.
        bt._try_fill(Candle(0, 60_000, 100.0, 100.1, 99.9, 100.0, 1))
        self.assertIsNotNone(bt.position)
        bt._manage(Candle(60_000, 120_000, 100.0, 103.0, 98.0, 101.0, 1))
        self.assertIsNone(bt.position, "position should have been stopped out")
        self.assertEqual(bt.trades[-1].exit_reason, "STOP")
        self.assertLess(bt.trades[-1].pnl, 0)


class TestDetectorIsNotVacuous(unittest.TestCase):
    """Meta-test: plant a real leak and prove the mutation test catches it.

    Without this, "the look-ahead tests pass" could simply mean the tests are
    incapable of failing.  Here a deliberately cheating backtester peeks at the
    next two hours of candles before committing to an order.  Rewriting the
    future must then change its past trades -- and it does.
    """

    class PeekingBacktester(Backtester):
        def __init__(self, cfg, all_candles):
            super().__init__(cfg)
            self._all = all_candles
            self._i = -1

        def on_candle(self, candle):
            self._i += 1
            super().on_candle(candle)

        def _decide(self, candle):
            super()._decide(candle)
            if self.pending is None:
                return
            # THE LEAK: keep the order only if the future shows the target
            # being reached before the stop.  A decisive peek, unlike "price
            # ticks favourably at some point", which is almost always true and
            # would make this planted leak a no-op.
            future = self._all[self._i + 1:self._i + 240]
            setup = self.pending.setup
            if not future or not setup.targets:
                return
            long_ = setup.side.value == "BUY"
            tp, sl = setup.targets[0].price, setup.stop
            wins = False
            for c in future:
                if (c.low <= sl) if long_ else (c.high >= sl):
                    break
                if (c.high >= tp) if long_ else (c.low <= tp):
                    wins = True
                    break
            if not wins:
                self.pending = None

    def test_planted_leak_is_detected(self):
        candles = generate(60 * 24 * 10, seed=9)

        def peek(data):
            bt = self.PeekingBacktester(Config(), data)
            bt.run(data)
            return bt

        honest = run_bt(candles)
        cuts = cut_points(candles, honest.trades)
        self.assertGreater(len(cuts), 0)

        leaked = peek(candles)
        cuts = cut_points(candles, leaked.trades) + cuts
        detected = False
        for cut in cuts:
            mutated = peek(mutate_future(candles, cut))
            cut_time = candles[cut - 1].close_time
            if entry_fingerprint(leaked, cut_time) != entry_fingerprint(mutated, cut_time):
                detected = True
                break
        self.assertTrue(detected,
                        "the mutation test failed to notice a planted leak")



if __name__ == "__main__":
    unittest.main(verbosity=2)
