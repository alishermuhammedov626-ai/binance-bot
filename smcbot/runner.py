"""Sections 73-76, 90 -- the live / paper trading loop.

The runner shares :class:`SignalEngine`, :class:`DecisionEngine` and
:class:`RiskManager` with the backtester, so the rules cannot drift apart
between research and execution -- the single most common way a backtested
strategy stops resembling the deployed one.

Only *closed* candles are fed to the engines, exactly as in the backtest, so
the live bot sees precisely what the backtest saw.

Order routing is behind the :class:`Broker` interface.  ``PaperBroker`` is
complete and is what section 90 asks you to run for 2-4 weeks.  ``LiveBroker``
is deliberately not implemented: it needs exchange credentials and the
section-74 pre-flight checklist wired to a real API, and shipping a
half-working live executor would be worse than shipping none.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .config import Config
from .core.types import Candle, Fill, Order, Setup, Trade
from .dashboard import build as build_dashboard, render as render_dashboard
from .data import loader
from .engine.context import SMCContext
from .journal import Journal
from .ml.features import build_features
from .notify import Notifier
from .strategy.decision import DecisionEngine, RiskManager
from .strategy.setup import SignalEngine

MINUTE = 60_000


class Broker:
    """Minimal execution interface."""

    def place_entry(self, setup: Setup, qty: float) -> Optional[Fill]:
        raise NotImplementedError

    def place_protective_orders(self, setup: Setup, qty: float) -> List[Order]:
        raise NotImplementedError

    def close_position(self, price: float, reason: str) -> Optional[Fill]:
        raise NotImplementedError

    def sync(self, candle: Candle) -> List[str]:
        """Reconcile local state with the exchange; returns event tags."""
        return []


class PaperBroker(Broker):
    """Simulated execution against the live candle stream."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.position = None
        self.orders: List[Order] = []

    def place_entry(self, setup: Setup, qty: float) -> Optional[Fill]:
        slip = self.cfg.execution.slippage_bps / 10_000.0
        price = setup.entry * (1 + setup.side.sign * slip)
        fee = abs(qty * price) * self.cfg.execution.taker_fee
        return Fill(int(time.time() * 1000), price, qty, fee, "ENTRY")

    def place_protective_orders(self, setup: Setup, qty: float) -> List[Order]:
        """Section 75-76 -- stops and targets live on the exchange side."""
        orders = [Order(setup.side.opposite, "STOP", setup.stop, qty,
                        reduce_only=True, tag="SL")]
        portions = [self.cfg.risk.partial_tp[k] for k in ("tp1", "tp2", "tp3")]
        for i, target in enumerate(setup.targets[:3]):
            orders.append(Order(setup.side.opposite, "TAKE_PROFIT", target.price,
                                qty * portions[i], reduce_only=True,
                                tag=f"TP{i + 1}"))
        self.orders = orders
        return orders

    def close_position(self, price: float, reason: str) -> Optional[Fill]:
        return Fill(int(time.time() * 1000), price, 0.0, 0.0, reason)


class LiveBroker(Broker):                     # pragma: no cover
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Live order routing is not wired up. Implement place_entry / "
            "place_protective_orders against the exchange REST API, including "
            "the section-74 pre-flight checks (balance, margin, position, "
            "price, SL, TP, risk, min-qty) and server-side stops (section 75). "
            "Run PaperBroker for 2-4 weeks first (section 90)."
        )


@dataclass
class HealthMonitor:
    """Section 73 -- emergency stop conditions."""

    max_stale_seconds: int = 180
    max_consecutive_errors: int = 5
    max_spread_bps: float = 25.0
    last_candle_ms: int = 0
    errors: int = 0
    reasons: List[str] = field(default_factory=list)

    def check(self, now_ms: int, spread_bps: float) -> Optional[str]:
        if self.last_candle_ms and \
                (now_ms - self.last_candle_ms) > self.max_stale_seconds * 1000:
            return "market_data_stale"
        if self.errors >= self.max_consecutive_errors:
            return "repeated_execution_failure"
        if spread_bps > self.max_spread_bps:
            return "abnormal_spread"
        return None


class LiveRunner:
    """Poll closed M1 candles, run the pipeline, act through a broker."""

    def __init__(self, cfg: Config, broker: Optional[Broker] = None,
                 journal: Optional[Journal] = None,
                 notifier: Optional[Notifier] = None,
                 ml_predictor: Optional[Callable] = None,
                 run_id: str = ""):
        self.cfg = cfg
        self.ctx = SMCContext(cfg)
        self.signals = SignalEngine(cfg)
        self.risk = RiskManager(cfg, cfg.initial_equity)
        self.decider = DecisionEngine(cfg, self.risk)
        self.broker = broker or PaperBroker(cfg)
        self.journal = journal
        self.notifier = notifier or Notifier()
        self.ml_predictor = ml_predictor
        self.health = HealthMonitor()
        self.run_id = run_id or f"live-{int(time.time())}"
        self.trades: List[Trade] = []
        self.halted: Optional[str] = None
        self.last_open_time = 0

    # ------------------------------------------------------------------
    def warmup(self, candles: List[Candle]) -> None:
        """Replay history so the engines start with full context."""
        for c in candles:
            self.ctx.on_m1(c)
            self.last_open_time = max(self.last_open_time, c.open_time)
        self.risk.roll_day(self.ctx.now)

    def on_closed_candle(self, candle: Candle) -> Optional[Setup]:
        if candle.open_time <= self.last_open_time:
            return None                      # already processed
        self.last_open_time = candle.open_time
        self.health.last_candle_ms = candle.close_time
        self.risk.roll_day(candle.open_time)
        self.ctx.on_m1(candle)

        halt = self.health.check(candle.close_time, self.ctx.spread_bps)
        if halt:
            self.halt(halt)
            return None
        if self.halted:
            return None

        for setup in self.signals.on_bar(self.ctx):
            setup.features = build_features(self.ctx, setup)
            prob = self.ml_predictor(setup.features) if self.ml_predictor else None
            decision = self.decider.evaluate(setup, self.ctx, prob)
            if self.journal:
                self.journal.record_signal(
                    self.ctx.now, setup.setup_id, setup.side.value,
                    setup.state.value if setup.state else "", setup.entry,
                    setup.stop, setup.rr, setup.smc_score, prob,
                    "approved" if decision.approved else "rejected",
                    decision.reason, setup.score_breakdown, self.run_id)
            if not decision.approved:
                continue
            self._enter(setup, decision)
            return setup
        return None

    def _enter(self, setup: Setup, decision) -> None:
        qty = decision.size.qty
        fill = self.broker.place_entry(setup, qty)
        if fill is None:
            self.health.errors += 1
            return
        self.health.errors = 0
        self.broker.place_protective_orders(setup, qty)
        self.risk.open_positions += 1
        self.risk.trades_today += 1
        self.signals.mark_traded(setup.setup_id)
        self.notifier.trade_opened(setup, qty, decision.risk_pct,
                                   self.cfg.risk.leverage, self.cfg.symbol)

    def halt(self, reason: str) -> None:
        if self.halted:
            return
        self.halted = reason
        self.notifier.alert("EMERGENCY_STOP", reason)
        if self.journal:
            self.journal.record_error("emergency_stop", reason,
                                      {"run_id": self.run_id})

    def resume(self) -> None:
        self.halted = None
        self.health.errors = 0

    def dashboard(self) -> str:
        return render_dashboard(build_dashboard(
            self.ctx, self.risk, self.signals, self.trades, None))

    # ------------------------------------------------------------------
    def poll_forever(self, symbol: Optional[str] = None, interval: int = 20,
                     max_iterations: Optional[int] = None,
                     show_dashboard: bool = True) -> None:
        """Fetch closed M1 candles from the exchange and process them."""
        symbol = symbol or self.cfg.symbol
        seen = 0
        while max_iterations is None or seen < max_iterations:
            seen += 1
            try:
                recent = loader.fetch_binance(symbol, "1m", limit_total=10)
                now_ms = int(time.time() * 1000)
                for c in recent:
                    if c.close_time <= now_ms:      # closed candles only
                        self.on_closed_candle(c)
                if show_dashboard:
                    print(self.dashboard(), flush=True)
            except Exception as exc:               # pragma: no cover - network
                self.health.errors += 1
                if self.journal:
                    self.journal.record_error("poll", str(exc))
                if self.health.errors >= self.health.max_consecutive_errors:
                    self.halt("repeated_execution_failure")
            time.sleep(interval)
