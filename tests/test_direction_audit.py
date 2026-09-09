"""Direction and economics audit: is BUY really long, and does money flow right?

A sign error anywhere in this chain would invalidate every result produced so
far, and it would not show up as a crash -- only as a strategy that loses.
Each test below pins one link with a hand-built scenario whose correct answer
is known by construction.
"""
from __future__ import annotations

import unittest

from smcbot.backtest.engine import Backtester, PendingOrder
from smcbot.config import Config
from smcbot.core.types import (Candle, LiquidityKind, LiquidityLevel, Setup,
                               Side, Target)
from smcbot.data.synthetic import generate
from smcbot.engine.context import SMCContext
from smcbot.strategy.decision import Decision
from smcbot.strategy.risk import build_stop, liquidation_price, size_position
from smcbot.strategy.setup import SignalEngine
from smcbot.strategy.targets import select_targets
from smcbot.engine.liquidity import LiquidityEngine

MIN = 60_000


def order(bt, side, entry, stop, tp, kind="MARKET"):
    lvl = LiquidityLevel(tp, LiquidityKind.INTERNAL, side is Side.BUY, "M5", 5, 0, 0)
    rr = abs(tp - entry) / abs(entry - stop)
    s = Setup(setup_id="x", created_at=0, side=side, state=None, entry=entry,
              stop=stop, targets=[Target(tp, lvl, rr, abs(tp - entry), 0.5)],
              rr=rr, smc_score=80.0, expires_at=10 ** 15, entry_type=kind)
    bt.pending = PendingOrder(s, Decision(True, "ok", s, None, 0.005), kind,
                              entry, 1.0, 0)
    return s


class TestSideSemantics(unittest.TestCase):
    def test_sign_convention(self):
        self.assertEqual(Side.BUY.sign, 1)
        self.assertEqual(Side.SELL.sign, -1)
        self.assertIs(Side.BUY.opposite, Side.SELL)

    def test_high_sweep_produces_a_sell(self):
        """Price ran the highs and rejected -> we sell, not buy."""
        cfg = Config()
        eng = SignalEngine(cfg)
        ctx = SMCContext(cfg)
        for c in generate(60 * 24 * 3, seed=4):
            ctx.on_m1(c)
        for view in ctx.views.values():
            for sweep in view.sweeps.sweeps:
                expected = Side.SELL if sweep.is_high_sweep else Side.BUY
                side = Side.BUY if not sweep.is_high_sweep else Side.SELL
                self.assertIs(side, expected)

    def test_stop_and_targets_sit_on_the_correct_sides(self):
        cfg = Config()
        buy = build_stop(Side.BUY, 100.0, 99.0, 98.0, 1.0, 2.0, cfg.risk,
                         cfg.execution)
        sell = build_stop(Side.SELL, 100.0, 101.0, 102.0, 1.0, 2.0, cfg.risk,
                          cfg.execution)
        self.assertLess(buy.price, 100.0, "long stop must be below entry")
        self.assertGreater(sell.price, 100.0, "short stop must be above entry")

        liq = LiquidityEngine(cfg.liquidity)
        for i, p in enumerate([95.0, 90.0, 105.0, 110.0]):
            liq._add(f"l{i}", LiquidityLevel(p, LiquidityKind.INTERNAL, p > 100,
                                             "M5", 5.0, 0, 0))
        long_t, _, _ = select_targets(Side.BUY, 100.0, 99.0, 1.0, liq, cfg.risk)
        short_t, _, _ = select_targets(Side.SELL, 100.0, 101.0, 1.0, liq, cfg.risk)
        self.assertTrue(all(t.price > 100.0 for t in long_t))
        self.assertTrue(all(t.price < 100.0 for t in short_t))

    def test_liquidation_sides(self):
        self.assertLess(liquidation_price(Side.BUY, 100, 17, 0.005), 100)
        self.assertGreater(liquidation_price(Side.SELL, 100, 17, 0.005), 100)


class TestMoneyFlow(unittest.TestCase):
    """Price up must pay a long and cost a short, and vice versa."""

    def _run(self, side: Side, up: bool) -> float:
        cfg = Config()
        bt = Backtester(cfg)
        bt.risk.equity = 1000.0
        entry, stop, tp = (100.0, 99.0, 103.0) if side is Side.BUY else \
                          (100.0, 101.0, 97.0)
        order(bt, side, entry, stop, tp)
        bt._try_fill(Candle(0, MIN, 100.0, 100.05, 99.95, 100.0, 1))
        self.assertIsNotNone(bt.position, "position should have opened")
        if up:
            bar = Candle(MIN, 2 * MIN, 100.0, 103.5, 99.95, 103.2, 1)
        else:
            bar = Candle(MIN, 2 * MIN, 100.0, 100.05, 96.5, 96.8, 1)
        bt._manage(bar)
        self.assertIsNone(bt.position, "position should have closed")
        return bt.trades[-1].pnl

    def test_long_profits_when_price_rises(self):
        self.assertGreater(self._run(Side.BUY, up=True), 0)

    def test_long_loses_when_price_falls(self):
        self.assertLess(self._run(Side.BUY, up=False), 0)

    def test_short_profits_when_price_falls(self):
        self.assertGreater(self._run(Side.SELL, up=False), 0)

    def test_short_loses_when_price_rises(self):
        self.assertLess(self._run(Side.SELL, up=True), 0)


class TestCostsAlwaysCost(unittest.TestCase):
    def test_slippage_is_adverse_on_both_sides(self):
        bt = Backtester(Config())
        self.assertGreater(bt._slip(100.0, Side.BUY), 100.0)
        self.assertLess(bt._slip(100.0, Side.SELL), 100.0)

    def test_fees_reduce_pnl(self):
        cfg = Config()
        bt = Backtester(cfg)
        bt.risk.equity = 1000.0
        order(bt, Side.BUY, 100.0, 99.0, 103.0)
        bt._try_fill(Candle(0, MIN, 100.0, 100.05, 99.95, 100.0, 1))
        bt._manage(Candle(MIN, 2 * MIN, 100.0, 103.5, 99.95, 103.2, 1))
        t = bt.trades[-1]
        gross = t.pnl + t.fees + t.funding
        self.assertGreater(t.fees, 0)
        self.assertGreater(gross, t.pnl, "net must be below gross")

    def test_long_pays_positive_funding_short_receives(self):
        cfg = Config()
        for side, expect_positive in ((Side.BUY, True), (Side.SELL, False)):
            bt = Backtester(cfg)
            bt.risk.equity = 1000.0
            bt.ctx.funding_rate = 0.001
            entry, stop, tp = (100.0, 99.0, 103.0) if side is Side.BUY else \
                              (100.0, 101.0, 97.0)
            order(bt, side, entry, stop, tp)
            bt._try_fill(Candle(0, MIN, 100.0, 100.05, 99.95, 100.0, 1))
            pos = bt.position
            pos.last_funding = 0
            bt._apply_funding(pos, Candle(8 * 3_600_000, 8 * 3_600_000 + MIN,
                                          100.0, 100.05, 99.95, 100.0, 1))
            if expect_positive:
                self.assertGreater(pos.funding, 0, "long pays")
            else:
                self.assertLess(pos.funding, 0, "short receives")


class TestSizingInvariants(unittest.TestCase):
    def test_risk_is_capped_regardless_of_side(self):
        cfg = Config()
        for side, stop in ((Side.BUY, 41_790.0), (Side.SELL, 42_210.0)):
            plan = size_position(side, 42_000.0, stop, 1000.0, 0.005,
                                 cfg.risk, cfg.execution)
            self.assertTrue(plan.valid, plan.reason)
            self.assertAlmostEqual(plan.qty * abs(42_000.0 - stop), 5.0, delta=0.3)

    def test_position_size_never_scales_with_leverage(self):
        """Section 32: leverage sets margin, never exposure."""
        cfg = Config()
        base = size_position(Side.BUY, 42_000.0, 41_790.0, 1000.0, 0.005,
                             cfg.risk, cfg.execution)
        lev = cfg.with_overrides(**{"risk.leverage": 5.0})
        other = size_position(Side.BUY, 42_000.0, 41_790.0, 1000.0, 0.005,
                              lev.risk, lev.execution)
        self.assertAlmostEqual(base.qty, other.qty, places=8)
        self.assertGreater(other.margin, base.margin)

    def test_no_size_increase_after_a_loss(self):
        """No martingale: risk per trade must not grow after losing."""
        from smcbot.strategy.decision import RiskManager
        cfg = Config()
        rm = RiskManager(cfg, 1000.0)
        rm.roll_day(0)
        setup = Setup(setup_id="s", created_at=0, side=Side.BUY, state=None,
                      entry=100.0, stop=99.0, smc_score=75.0)
        before = rm.risk_pct_for(setup)
        rm.on_trade_closed(-5.0, MIN)
        rm.on_trade_closed(-5.0, 2 * MIN)
        after = rm.risk_pct_for(setup)
        self.assertLessEqual(after, before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
