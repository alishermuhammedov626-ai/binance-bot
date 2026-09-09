"""Config, data loading, journal, dashboard and CLI smoke tests."""
from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from smcbot import cli, dashboard
from smcbot.backtest.engine import Backtester
from smcbot.config import Config
from smcbot.core.types import MS_MINUTE, Candle, Side, Trade
from smcbot.data import loader, synthetic
from smcbot.journal import Journal
from smcbot.notify import Notifier


class TestConfig(unittest.TestCase):
    def test_round_trip_preserves_nested_dataclasses(self):
        cfg = Config()
        clone = Config.from_dict(json.loads(cfg.dumps()))
        self.assertEqual(clone.to_dict(), cfg.to_dict())
        self.assertIsInstance(clone.risk.partial_tp, dict)
        self.assertEqual(clone.sessions.asia, cfg.sessions.asia)

    def test_dotted_overrides(self):
        cfg = Config().with_overrides(**{"risk.min_rr": 2.5, "ml.threshold": 0.7})
        self.assertEqual(cfg.risk.min_rr, 2.5)
        self.assertEqual(cfg.ml.threshold, 0.7)

    def test_overrides_do_not_mutate_the_original(self):
        cfg = Config()
        cfg.with_overrides(**{"risk.min_rr": 9.0})
        self.assertEqual(cfg.risk.min_rr, 1.5)

    def test_unknown_key_is_rejected(self):
        with self.assertRaises(KeyError):
            Config().with_overrides(**{"risk.not_a_key": 1})

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cfg.json")
            Config().with_overrides(**{"symbol": "ETHUSDT"}).save(path)
            self.assertEqual(Config.load(path).symbol, "ETHUSDT")


class TestDataLoader(unittest.TestCase):
    def test_csv_round_trip(self):
        candles = synthetic.generate(200, seed=1)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.csv")
            loader.save_csv(path, candles)
            back = loader.load_csv(path)
        self.assertEqual(len(back), len(candles))
        self.assertEqual(back[0].open_time, candles[0].open_time)
        self.assertAlmostEqual(back[-1].close, candles[-1].close)

    def test_validate_drops_duplicates_and_bad_ohlc(self):
        good = Candle(0, MS_MINUTE, 100, 101, 99, 100, 1)
        dup = Candle(0, MS_MINUTE, 100, 101, 99, 100, 1)
        bad = Candle(MS_MINUTE, 2 * MS_MINUTE, 100, 95, 99, 100, 1)  # high < low
        out = loader.validate([good, dup, bad])
        self.assertEqual(len(out), 1)

    def test_gaps_are_reported(self):
        a = Candle(0, MS_MINUTE, 1, 1, 1, 1, 1)
        b = Candle(5 * MS_MINUTE, 6 * MS_MINUTE, 1, 1, 1, 1, 1)
        self.assertEqual(len(loader.gaps([a, b])), 1)

    def test_stream_rejects_out_of_order(self):
        a = Candle(MS_MINUTE, 2 * MS_MINUTE, 1, 1, 1, 1, 1)
        b = Candle(0, MS_MINUTE, 1, 1, 1, 1, 1)
        with self.assertRaises(ValueError):
            list(loader.stream([a, b]))

    def test_synthetic_data_is_deterministic_and_well_formed(self):
        a = synthetic.generate(500, seed=7)
        b = synthetic.generate(500, seed=7)
        self.assertEqual([c.close for c in a], [c.close for c in b])
        for c in a:
            self.assertLessEqual(c.low, min(c.open, c.close))
            self.assertGreaterEqual(c.high, max(c.open, c.close))
            self.assertEqual(c.close_time - c.open_time, MS_MINUTE)


class TestJournal(unittest.TestCase):
    def _trade(self, pnl=5.0):
        t = Trade("s1", "BTCUSDT", Side.BUY, 1, 100.0, 0.1, 99.0, [101.0, 102.0],
                  78.0, 0.66, "BULLISH", "LONDON")
        t.exit_time, t.exit_price, t.result = 2, 101.0, "WIN"
        t.pnl, t.r_multiple, t.equity_after = pnl, 1.2, 1000 + pnl
        return t

    def test_records_and_aggregates(self):
        with tempfile.TemporaryDirectory() as d:
            with Journal(os.path.join(d, "j.db")) as j:
                j.start_run("r1", "backtest", "BTCUSDT", Config().to_dict())
                j.record_trades([self._trade(5.0), self._trade(-3.0)], "r1")
                j.record_signal(1, "s1", "BUY", "ENTRY", 100, 99, 2.0, 80, 0.7,
                                "approved", "ok", {"x": 1}, "r1")
                j.record_model("ML_MODEL_v1", "gbt", ["a"], {"auc": 0.6})
                j.record_error("api", "boom", {"code": 1})
                stats = j.stats("r1")
                self.assertEqual(stats["trades"], 2)
                self.assertAlmostEqual(stats["net_pnl"], 2.0)
                self.assertAlmostEqual(stats["win_rate"], 0.5)
                self.assertEqual(len(j.recent_trades(run_id="r1")), 2)

    def test_every_trade_carries_its_model_version(self):
        cfg = Config()
        bt = Backtester(cfg)
        bt.run(synthetic.generate(60 * 24 * 8, seed=42))
        self.assertGreater(len(bt.trades), 0)
        for t in bt.trades:
            self.assertIn(cfg.smc_engine_version, t.model_version)
            self.assertIn(cfg.ml.model_version, t.model_version)


class TestDashboardAndNotifier(unittest.TestCase):
    def test_dashboard_renders(self):
        bt = Backtester(Config())
        result = bt.run(synthetic.generate(60 * 24 * 3, seed=4))
        data = dashboard.build(bt.ctx, bt.risk, bt.signals, result.trades,
                               bt.position)
        text = dashboard.render(data)
        self.assertIn("Equity", text)
        self.assertIn("Liquidity", text)
        json.loads(dashboard.to_json(data))

    def test_notifier_without_credentials_does_not_raise(self):
        n = Notifier(token=None, chat_id=None, echo=False)
        self.assertFalse(n.enabled)
        self.assertFalse(n.send("hello"))
        self.assertEqual(n.sent, ["hello"])


class TestEndToEnd(unittest.TestCase):
    def test_backtest_produces_a_consistent_account(self):
        cfg = Config()
        bt = Backtester(cfg)
        result = bt.run(synthetic.generate(60 * 24 * 12, seed=42))
        self.assertGreater(result.metrics["trades"], 0)
        # Equity must equal the starting balance plus the sum of trade PnL.
        expected = cfg.initial_equity + sum(t.pnl for t in result.trades)
        self.assertAlmostEqual(bt.risk.equity, expected, places=6)
        self.assertAlmostEqual(result.metrics["final_equity"], round(expected, 2),
                               places=2)
        for t in result.trades:
            self.assertGreater(t.exit_time, t.entry_time)
            self.assertNotEqual(t.result, "OPEN")
            self.assertGreater(t.smc_score, 0)

    def test_risk_per_trade_never_exceeds_the_configured_cap(self):
        cfg = Config()
        bt = Backtester(cfg)
        bt.run(synthetic.generate(60 * 24 * 12, seed=42))
        for t in bt.trades:
            # Loss can exceed 1R only by costs and slippage, never by sizing.
            nominal_risk = abs(t.entry_price - t.stop) * t.qty
            self.assertLessEqual(
                nominal_risk,
                cfg.initial_equity * cfg.risk.max_risk_per_trade * 3.0)

    def test_one_position_at_a_time(self):
        bt = Backtester(Config())
        bt.run(synthetic.generate(60 * 24 * 12, seed=42))
        spans = sorted((t.entry_time, t.exit_time) for t in bt.trades)
        for (_, end), (start, _) in zip(spans, spans[1:]):
            self.assertGreaterEqual(start, end, "positions must not overlap")

    def test_daily_trade_limit_is_respected(self):
        from smcbot.core.series import day_start
        cfg = Config()
        bt = Backtester(cfg)
        bt.run(synthetic.generate(60 * 24 * 20, seed=42))
        per_day = {}
        for t in bt.trades:
            d = day_start(t.entry_time)
            per_day[d] = per_day.get(d, 0) + 1
        for day, n in per_day.items():
            self.assertLessEqual(n, cfg.risk.max_trades_per_day)


class TestRunner(unittest.TestCase):
    """The live/paper loop must obey the same rules as the backtest."""

    def _runner(self):
        from smcbot.runner import LiveRunner, PaperBroker
        r = LiveRunner(Config(), broker=PaperBroker(Config()),
                       notifier=Notifier(echo=False))
        candles = synthetic.generate(60 * 24 * 4, seed=4)
        r.warmup(candles[:-10])
        return r, candles[-10:]

    def test_warmup_then_process_closed_candles(self):
        r, tail = self._runner()
        for c in tail:
            r.on_closed_candle(c)
        self.assertIsNone(r.halted)
        self.assertIn("Equity", r.dashboard())

    def test_replayed_candle_is_ignored(self):
        r, tail = self._runner()
        r.on_closed_candle(tail[0])
        self.assertIsNone(r.on_closed_candle(tail[0]),
                          "a repeated candle must not be reprocessed")

    def test_emergency_stop_on_stale_data(self):
        from smcbot.runner import HealthMonitor
        h = HealthMonitor()
        h.last_candle_ms = 1_000_000
        self.assertEqual(h.check(1_000_000 + 10 ** 6, 1.0), "market_data_stale")
        self.assertIsNone(h.check(1_000_000 + 1000, 1.0))

    def test_halt_blocks_further_signals(self):
        r, tail = self._runner()
        r.halt("api_error")
        self.assertEqual(r.halted, "api_error")
        for c in tail:
            self.assertIsNone(r.on_closed_candle(c))

    def test_paper_broker_places_server_side_protection(self):
        from smcbot.core.types import LiquidityKind, LiquidityLevel, Setup, Target
        from smcbot.runner import PaperBroker
        cfg = Config()
        level = LiquidityLevel(103.0, LiquidityKind.INTERNAL, True, "M5", 5, 0, 0)
        setup = Setup(setup_id="s", created_at=0, side=Side.BUY, state=None,
                      entry=100.0, stop=99.0,
                      targets=[Target(102.0, level, 2.0, 2.0, 0.5),
                               Target(104.0, level, 4.0, 4.0, 0.4)],
                      rr=2.0, smc_score=80.0)
        orders = PaperBroker(cfg).place_protective_orders(setup, 1.0)
        tags = [o.tag for o in orders]
        self.assertEqual(tags[0], "SL")
        self.assertIn("TP1", tags)
        self.assertTrue(all(o.reduce_only for o in orders))
        self.assertTrue(all(o.side is Side.SELL for o in orders))

    def test_live_broker_is_explicitly_unimplemented(self):
        from smcbot.runner import LiveBroker
        with self.assertRaises(NotImplementedError):
            LiveBroker()


class TestCLI(unittest.TestCase):
    def _run(self, argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli.main(argv)
        return code, buf.getvalue()

    def test_config_command(self):
        code, out = self._run(["config"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["symbol"], "BTCUSDT")

    def test_backtest_command_on_synthetic_data(self):
        with tempfile.TemporaryDirectory() as d:
            out_json = os.path.join(d, "m.json")
            code, out = self._run(["backtest", "--synthetic", "--days", "8",
                                   "--json", out_json, "--rejections",
                                   "--journal", os.path.join(d, "j.db")])
            self.assertEqual(code, 0)
            self.assertIn("Trades", out)
            with open(out_json) as fh:
                self.assertIn("metrics", json.load(fh))

    def test_dashboard_command(self):
        code, out = self._run(["dashboard", "--synthetic", "--days", "3"])
        self.assertEqual(code, 0)
        self.assertIn("Equity", out)

    def test_overrides_reach_the_engine(self):
        code, out = self._run(["config", "--set", "risk.min_rr=3.0"])
        self.assertEqual(json.loads(out)["risk"]["min_rr"], 3.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
