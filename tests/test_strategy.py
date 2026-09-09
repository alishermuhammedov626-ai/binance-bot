"""Scoring, targets, stops, sizing, the decision gate and backtest mechanics."""
from __future__ import annotations

import unittest

from smcbot.backtest.engine import Backtester, PendingOrder
from smcbot.backtest.metrics import compute_metrics, max_drawdown
from smcbot.backtest.montecarlo import monte_carlo
from smcbot.config import (Config, ExecutionConfig, LiquidityConfig, RiskConfig,
                           ScoreConfig)
from smcbot.core.types import (Candle, LiquidityKind, LiquidityLevel, Setup,
                               Side, Target, Trade)
from smcbot.engine.liquidity import LiquidityEngine
from smcbot.strategy.decision import Decision, DecisionEngine, RiskManager
from smcbot.strategy.risk import (build_stop, liquidation_price, round_step,
                                  round_tick, rr_of, size_position)
from smcbot.strategy.scoring import Evidence, classify, score
from smcbot.strategy.targets import select_targets

MIN = 60_000


class TestScoring(unittest.TestCase):
    def setUp(self):
        self.cfg = ScoreConfig()

    def test_empty_evidence_scores_zero(self):
        final, _ = score(Evidence(Side.BUY), self.cfg)
        self.assertEqual(final, 0.0)
        self.assertEqual(classify(final, self.cfg), "NO_TRADE")

    def test_perfect_setup_reaches_a_plus(self):
        ev = Evidence(
            Side.BUY, weekly_liquidity=True, liquidity_strength=10,
            liquidity_cluster=True, m15_external_aligned=True,
            m15_sweep_score=100, m15_internal_choch=True, m5_confirmation=True,
            m5_internal_choch=True, m5_bos=True, zone_kind="BREAKER",
            zone_quality=100, m1_confirmation=True, displacement_bonus=10,
            premium_discount_ok=True, rr=3.0)
        final, _ = score(ev, self.cfg)
        self.assertGreaterEqual(final, self.cfg.a_plus)
        self.assertEqual(classify(final, self.cfg), "A_PLUS")

    def test_liquidity_tiers_are_mutually_exclusive(self):
        """Weekly/daily/session must not stack in the denominator."""
        weekly = Evidence(Side.BUY, weekly_liquidity=True, liquidity_strength=10)
        session = Evidence(Side.BUY, session_liquidity=True, liquidity_strength=8)
        _, bw = score(weekly, self.cfg)
        _, bs = score(session, self.cfg)
        self.assertEqual(bw["liquidity_source"], self.cfg.weekly_liquidity)
        self.assertEqual(bs["liquidity_source"], self.cfg.session_liquidity)
        self.assertEqual(bw["_max"], bs["_max"])

    def test_classification_bands(self):
        c = self.cfg
        for value, expected in ((95, "A_PLUS"), (85, "HIGH_QUALITY"),
                                (72, "VALID"), (62, "WATCHLIST"), (30, "NO_TRADE")):
            self.assertEqual(classify(value, c), expected)


class TestStops(unittest.TestCase):
    def setUp(self):
        self.risk, self.exec = RiskConfig(), ExecutionConfig()

    def test_stop_sits_behind_the_swing_with_a_buffer(self):
        plan = build_stop(Side.BUY, 100.0, m1_swing=99.0, m5_swing=98.0,
                          atr_m1=1.0, atr_m5=2.0, cfg=self.risk,
                          execution=self.exec)
        self.assertTrue(plan.valid)
        self.assertEqual(plan.source, "M1_SWING")
        self.assertLess(plan.price, 99.0, "stop must be beyond the swing")

    def test_too_tight_m1_stop_falls_back_to_m5(self):
        plan = build_stop(Side.BUY, 100.0, m1_swing=99.99, m5_swing=97.0,
                          atr_m1=1.0, atr_m5=2.0, cfg=self.risk,
                          execution=self.exec)
        self.assertEqual(plan.source, "M5_SWING")

    def test_stop_on_the_wrong_side_is_rejected(self):
        plan = build_stop(Side.BUY, 100.0, m1_swing=105.0, m5_swing=None,
                          atr_m1=1.0, atr_m5=2.0, cfg=self.risk,
                          execution=self.exec)
        self.assertFalse(plan.valid)

    def test_sell_stop_is_above_entry(self):
        plan = build_stop(Side.SELL, 100.0, m1_swing=101.0, m5_swing=102.0,
                          atr_m1=1.0, atr_m5=2.0, cfg=self.risk,
                          execution=self.exec)
        self.assertTrue(plan.valid)
        self.assertGreater(plan.price, 101.0)


class TestSizing(unittest.TestCase):
    def setUp(self):
        self.risk, self.exec = RiskConfig(), ExecutionConfig()

    def test_size_comes_from_risk_and_stop_not_from_leverage(self):
        plan = size_position(Side.BUY, 42_000.0, 41_790.0, equity=1000.0,
                             risk_pct=0.005, cfg=self.risk, execution=self.exec)
        self.assertTrue(plan.valid, plan.reason)
        # qty * stop distance == risk in USDT (modulo the qty step)
        self.assertAlmostEqual(plan.qty * 210.0, 5.0, delta=0.25)
        self.assertAlmostEqual(plan.risk_usdt, 5.0)

    def test_doubling_the_stop_distance_halves_the_size(self):
        a = size_position(Side.BUY, 42_000.0, 41_790.0, 1000.0, 0.005,
                          self.risk, self.exec)
        b = size_position(Side.BUY, 42_000.0, 41_580.0, 1000.0, 0.005,
                          self.risk, self.exec)
        self.assertAlmostEqual(b.qty, a.qty / 2, delta=0.002)

    def test_stop_too_close_to_liquidation_is_refused(self):
        # A stop ~5.6% away with 17x leverage sits past the liquidation price.
        plan = size_position(Side.BUY, 42_000.0, 39_650.0, 1000.0, 0.005,
                             self.risk, self.exec)
        self.assertFalse(plan.valid)
        self.assertEqual(plan.reason, "liquidation_too_close")

    def test_liquidation_price_direction(self):
        long_liq = liquidation_price(Side.BUY, 100.0, 17, 0.005)
        short_liq = liquidation_price(Side.SELL, 100.0, 17, 0.005)
        self.assertLess(long_liq, 100.0)
        self.assertGreater(short_liq, 100.0)
        self.assertAlmostEqual(long_liq, 100 * (1 - (1 / 17 - 0.005)), places=6)

    def test_rounding_helpers(self):
        self.assertAlmostEqual(round_step(0.0037, 0.001), 0.003)
        self.assertAlmostEqual(round_tick(100.04, 0.1, -1), 100.0)
        self.assertAlmostEqual(round_tick(100.04, 0.1, +1), 100.1)

    def test_rr_of(self):
        self.assertAlmostEqual(rr_of(Side.BUY, 100, 99, 103), 3.0)
        self.assertAlmostEqual(rr_of(Side.SELL, 100, 101, 97), 3.0)


class TestTargets(unittest.TestCase):
    def _liq(self, prices):
        eng = LiquidityEngine(LiquidityConfig())
        for i, (price, tf, strength) in enumerate(prices):
            eng._add(f"l{i}", LiquidityLevel(price, LiquidityKind.INTERNAL, True,
                                             tf, strength, 0, 0))
        return eng

    def test_targets_follow_the_liquidity_hierarchy(self):
        liq = self._liq([(101.0, "M1", 3), (103.0, "M5", 5), (108.0, "M15", 7)])
        targets, rr, reason = select_targets(Side.BUY, 100.0, 99.0, atr=1.0,
                                             liq=liq, cfg=RiskConfig())
        self.assertEqual(reason, "ok")
        self.assertEqual([t.price for t in targets], [101.0, 103.0, 108.0])
        self.assertGreater(rr, 0)

    def test_no_target_means_no_trade(self):
        liq = self._liq([])
        targets, rr, reason = select_targets(Side.BUY, 100.0, 99.0, 1.0, liq,
                                             RiskConfig())
        self.assertEqual(reason, "no_liquidity_target")
        self.assertEqual(targets, [])

    def test_targets_too_close_fail_the_rr_gate(self):
        liq = self._liq([(100.5, "M1", 3)])
        _, _, reason = select_targets(Side.BUY, 100.0, 99.0, 1.0, liq, RiskConfig())
        self.assertEqual(reason, "rr_below_minimum")


class TestDecisionGate(unittest.TestCase):
    def _setup(self, score_=80.0, rr=2.0):
        level = LiquidityLevel(103.0, LiquidityKind.INTERNAL, True, "M5", 5, 0, 0)
        return Setup(setup_id="s", created_at=0, side=Side.BUY, state=None,
                     entry=100.0, stop=99.0,
                     targets=[Target(103.0, level, 3.0, 3.0, 0.5)],
                     rr=rr, smc_score=score_, expires_at=10 ** 15)

    def _ctx(self):
        """A context healthy enough to pass the market filters."""
        from smcbot.data.synthetic import generate
        from smcbot.engine.context import SMCContext
        ctx = SMCContext(Config())
        for c in generate(60 * 24 * 4, seed=2):
            ctx.on_m1(c)
        return ctx

    def setUp(self):
        self.cfg = Config()
        self.ctx = self._ctx()
        self.rm = RiskManager(self.cfg, 1000.0)
        self.rm.roll_day(self.ctx.now)
        self.engine = DecisionEngine(self.cfg, self.rm)

    def test_good_setup_is_approved(self):
        d = self.engine.evaluate(self._setup(), self.ctx, ml_probability=0.8)
        self.assertTrue(d.approved, d.reason)
        self.assertGreater(d.size.qty, 0)

    def test_open_position_blocks_new_trades(self):
        self.rm.open_positions = 1
        d = self.engine.evaluate(self._setup(), self.ctx, 0.8)
        self.assertFalse(d.approved)
        self.assertEqual(d.reason, "position_already_open")

    def test_low_rr_and_low_score_are_rejected(self):
        self.assertEqual(
            self.engine.evaluate(self._setup(rr=1.1), self.ctx, 0.8).reason,
            "rr_below_minimum")
        self.assertEqual(
            self.engine.evaluate(self._setup(score_=55.0), self.ctx, 0.8).reason,
            "smc_score_below_threshold")

    def test_ml_below_threshold_is_rejected_but_absent_model_is_not(self):
        self.assertEqual(self.engine.evaluate(self._setup(), self.ctx, 0.10).reason,
                         "ml_below_threshold")
        self.assertTrue(self.engine.evaluate(self._setup(), self.ctx, None).approved)

    def test_daily_loss_limit_stops_trading(self):
        self.rm.equity = self.rm.day_start_equity * 0.97      # -3%, limit is 2%
        self.rm.on_trade_closed(-1.0, self.ctx.now)
        d = self.engine.evaluate(self._setup(), self.ctx, 0.8)
        self.assertFalse(d.approved)
        self.assertEqual(d.reason, "daily_loss_limit")

    def test_cooldown_after_a_loss(self):
        self.rm.on_trade_closed(-5.0, self.ctx.now)
        self.rm.open_positions = 0
        d = self.engine.evaluate(self._setup(), self.ctx, 0.8)
        self.assertEqual(d.reason, "cooldown")

    def test_consecutive_losses_escalate_the_cooldown(self):
        now = self.ctx.now
        for _ in range(3):
            self.rm.on_trade_closed(-1.0, now)
        self.assertEqual(self.rm.consecutive_losses, 3)
        self.assertGreaterEqual(self.rm.cooldown_until - now,
                                60 * MIN)          # 3 losses -> >= 60 minutes

    def test_max_trades_per_day(self):
        self.rm.trades_today = self.cfg.risk.max_trades_per_day
        self.assertEqual(self.engine.evaluate(self._setup(), self.ctx, 0.8).reason,
                         "max_trades_per_day")

    def test_spread_and_volatility_filters(self):
        self.ctx.spread_bps = 999.0
        self.assertEqual(self.engine.evaluate(self._setup(), self.ctx, 0.8).reason,
                         "spread_too_wide")


class TestBacktestMechanics(unittest.TestCase):
    def _bt(self):
        bt = Backtester(Config())
        bt.risk.equity = 1000.0
        return bt

    def _order(self, bt, kind="MARKET", entry=100.0, stop=99.0, tp=103.0):
        level = LiquidityLevel(tp, LiquidityKind.INTERNAL, True, "M5", 5, 0, 0)
        setup = Setup(setup_id="x", created_at=0, side=Side.BUY, state=None,
                      entry=entry, stop=stop,
                      targets=[Target(tp, level, 3.0, 3.0, 0.5)],
                      rr=3.0, smc_score=80.0, expires_at=10 ** 15,
                      entry_type=kind)
        bt.pending = PendingOrder(setup, Decision(True, "ok", setup, None, 0.005),
                                  kind, entry, 1.0, 0)
        return setup

    def test_market_order_fills_at_next_bar_open_with_slippage(self):
        bt = self._bt()
        self._order(bt)
        bt._try_fill(Candle(0, MIN, 100.0, 100.5, 99.8, 100.2, 1))
        self.assertIsNotNone(bt.position)
        self.assertGreater(bt.position.entry, 100.0, "buy slippage is adverse")

    def test_limit_order_does_not_fill_without_a_touch(self):
        bt = self._bt()
        self._order(bt, kind="LIMIT", entry=98.0, stop=97.0)
        bt._try_fill(Candle(0, MIN, 100.0, 100.5, 99.0, 100.2, 1))
        self.assertIsNone(bt.position)
        self.assertIsNotNone(bt.pending)

    def test_limit_order_fills_on_touch(self):
        bt = self._bt()
        self._order(bt, kind="LIMIT", entry=98.0, stop=97.0)
        bt._try_fill(Candle(0, MIN, 100.0, 100.5, 97.5, 99.0, 1))
        self.assertIsNotNone(bt.position)
        self.assertAlmostEqual(bt.position.entry, 98.0)

    def test_fill_is_refused_when_price_ran_past_the_setup(self):
        """A stop on the wrong side of the fill must never open a position."""
        bt = self._bt()
        self._order(bt, kind="LIMIT", entry=98.0, stop=99.0)   # invalid for a buy
        bt._try_fill(Candle(0, MIN, 100.0, 100.5, 97.5, 99.0, 1))
        self.assertIsNone(bt.position)

    def test_limit_order_expires(self):
        bt = self._bt()
        self._order(bt, kind="LIMIT", entry=90.0, stop=89.0)
        for i in range(bt.cfg.execution.limit_order_ttl_bars + 1):
            bt._try_fill(Candle(i * MIN, (i + 1) * MIN, 100, 101, 99, 100, 1))
        self.assertIsNone(bt.pending)
        self.assertIsNone(bt.position)

    def test_partial_take_profit_and_breakeven_move(self):
        cfg = Config()
        bt = Backtester(cfg)
        bt.risk.equity = 1000.0
        level = LiquidityLevel(0, LiquidityKind.INTERNAL, True, "M5", 5, 0, 0)
        setup = Setup(setup_id="x", created_at=0, side=Side.BUY, state=None,
                      entry=100.0, stop=99.0,
                      targets=[Target(102.0, level, 2.0, 2.0, 0.5),
                               Target(104.0, level, 4.0, 4.0, 0.4),
                               Target(106.0, level, 6.0, 6.0, 0.3)],
                      rr=2.0, smc_score=80.0, expires_at=10 ** 15)
        bt.pending = PendingOrder(setup, Decision(True, "ok", setup, None, 0.005),
                                  "MARKET", 100.0, 1.0, 0)
        bt._try_fill(Candle(0, MIN, 100.0, 100.1, 99.9, 100.0, 1))
        qty0 = bt.position.qty
        bt._manage(Candle(MIN, 2 * MIN, 100.0, 102.5, 99.8, 102.0, 1))
        self.assertIn(0, bt.position.tp_hits)
        self.assertLess(bt.position.qty, qty0, "TP1 must reduce the position")
        self.assertTrue(bt.position.be_moved)
        self.assertGreaterEqual(bt.position.stop, bt.position.entry)

    def test_liquidation_is_checked_before_stop_and_target(self):
        bt = self._bt()
        self._order(bt)
        bt._try_fill(Candle(0, MIN, 100.0, 100.1, 99.9, 100.0, 1))
        bt.position.liq_price = 99.5      # forced above the stop for the test
        bt.position.stop = 99.0
        bt._manage(Candle(MIN, 2 * MIN, 100.0, 101.0, 98.0, 99.0, 1))
        self.assertEqual(bt.trades[-1].exit_reason, "LIQUIDATION")

    def test_funding_is_charged_at_the_eight_hour_boundary(self):
        bt = self._bt()
        self._order(bt)
        bt._try_fill(Candle(0, MIN, 100.0, 100.1, 99.9, 100.0, 1))
        pos = bt.position
        pos.last_funding = 0
        bt.ctx.funding_rate = 0.001
        before = pos.funding
        bt._apply_funding(pos, Candle(8 * 3_600_000, 8 * 3_600_000 + MIN,
                                      100.0, 100.1, 99.9, 100.0, 1))
        self.assertGreater(pos.funding, before, "a long pays positive funding")


class TestMetrics(unittest.TestCase):
    def _trade(self, pnl, r=1.0, equity=1000.0):
        t = Trade("s", "BTCUSDT", Side.BUY, 0, 100.0, 1.0, 99.0, [102.0],
                  80.0, 0.7, "BULLISH", "LONDON")
        t.pnl, t.r_multiple, t.equity_after = pnl, r, equity
        t.exit_time = MIN
        return t

    def test_profit_factor_and_expectancy(self):
        trades = [self._trade(10, 2.0), self._trade(-5, -1.0),
                  self._trade(10, 2.0), self._trade(-5, -1.0)]
        m = compute_metrics(trades, [(0, 1000.0), (MIN, 1010.0)], 1000.0)
        self.assertEqual(m["trades"], 4)
        self.assertAlmostEqual(m["win_rate"], 0.5)
        self.assertAlmostEqual(m["profit_factor"], 2.0)
        self.assertAlmostEqual(m["expectancy"], 2.5)
        self.assertTrue(m["expectancy_positive"])

    def test_high_win_rate_can_still_lose_money(self):
        """Section 58: win rate alone is not a criterion."""
        trades = [self._trade(1, 0.2) for _ in range(7)] + \
                 [self._trade(-10, -2.0) for _ in range(3)]
        m = compute_metrics(trades, [(0, 1000.0), (MIN, 977.0)], 1000.0)
        self.assertAlmostEqual(m["win_rate"], 0.7)
        self.assertLess(m["profit_factor"], 1.0)
        self.assertFalse(m["expectancy_positive"])

    def test_max_drawdown(self):
        dd, peak, trough = max_drawdown([(0, 100.0), (1, 120.0), (2, 90.0),
                                         (3, 130.0)])
        self.assertAlmostEqual(dd, 0.25)
        self.assertAlmostEqual(peak, 120.0)

    def test_no_trades_is_handled(self):
        self.assertEqual(compute_metrics([], [], 1000.0)["trades"], 0)


class TestMonteCarlo(unittest.TestCase):
    def test_losing_system_has_high_ruin_probability(self):
        trades = []
        equity = 1000.0
        for i in range(60):
            pnl = -50.0 if i % 3 else 30.0
            equity += pnl
            t = Trade("s", "B", Side.BUY, 0, 1, 1, 1, [1], 70, None, "", "")
            t.pnl, t.equity_after = pnl, equity
            trades.append(t)
        mc = monte_carlo(trades, 1000.0, runs=300, seed=1)
        self.assertGreater(mc.ruin_probability, 0.5)
        self.assertLess(mc.profitable_share, 0.2)

    def test_empty_trades(self):
        self.assertEqual(monte_carlo([], 1000.0, runs=10).runs, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
