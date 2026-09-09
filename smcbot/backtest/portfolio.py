"""Several symbols traded side by side out of one account.

Each symbol gets its own :class:`Backtester` -- its own context, signal engine
and per-symbol trade cap -- because a DOGE setup must not consume a PEPE slot.
What they share is the money: position sizing reads the portfolio equity, and a
portfolio-wide daily loss limit can stop every engine at once.

Timelines are merged by timestamp and each engine is only stepped on the bars
its own symbol actually has, so symbols with different listing dates or gaps
can be mixed without one of them silently dragging the others.

Equity propagation carries a one-bar lag: a trade closing on bar *t* in one
engine is reflected in the others' sizing from bar *t+1*. That is a deliberate
simplification of a portfolio effect and is documented rather than hidden.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from ..config import Config
from ..core.series import day_start
from ..core.types import Candle, Trade
from .engine import Backtester


@dataclass
class PortfolioResult:
    equity_curve: List[tuple]
    trades: List[Trade]                       # every symbol, chronological
    by_symbol: Dict[str, List[Trade]]
    metrics: dict
    per_symbol_metrics: Dict[str, dict]
    bars: int
    halted_days: int = 0
    rejections: Dict[str, int] = field(default_factory=dict)


class PortfolioBacktester:
    def __init__(self, cfg: Config, symbols: Sequence[str],
                 portfolio_daily_loss_limit: float = 0.02,
                 feature_builder: Optional[Callable] = None,
                 per_symbol: Optional[Dict[str, dict]] = None):
        """``per_symbol`` carries contract specs -- tick size, quantity step,
        minimum notional -- which differ by orders of magnitude between, say,
        DOGE and 1000PEPE. Sizing is wrong without them."""
        self.cfg = cfg
        self.symbols = list(symbols)
        self.limit = portfolio_daily_loss_limit
        self.equity = cfg.initial_equity
        self.per_symbol = per_symbol or {}
        self.engines: Dict[str, Backtester] = {}
        for sym in self.symbols:
            sym_cfg = cfg.with_overrides(**{"symbol": sym,
                                            **self.per_symbol.get(sym, {})})
            bt = Backtester(sym_cfg, feature_builder=feature_builder)
            bt.risk.equity = self.equity
            bt.risk.start_equity = self.equity
            self.engines[sym] = bt
        self.current_day: Optional[int] = None
        self.day_start_equity = self.equity
        self.halted_today = False
        self.halted_days = 0
        self.equity_curve: List[tuple] = []
        self.bars = 0

    # ------------------------------------------------------------------
    def run(self, data: Dict[str, List[Candle]],
            progress: Optional[Callable[[int], None]] = None) -> PortfolioResult:
        from .metrics import compute_metrics

        # One merged, ordered timeline; each symbol keeps its own cursor.
        stamps = sorted({c.open_time for candles in data.values() for c in candles})
        index: Dict[str, Dict[int, Candle]] = {
            sym: {c.open_time: c for c in candles} for sym, candles in data.items()
        }
        seen: Dict[str, int] = {sym: 0 for sym in self.symbols}

        for ts in stamps:
            self._roll_day(ts)
            for sym in self.symbols:
                candle = index.get(sym, {}).get(ts)
                if candle is None:
                    continue
                engine = self.engines[sym]
                if self.halted_today:
                    # Portfolio limit hit: manage what is open, open nothing new.
                    engine.risk.trading_disabled_until_day = day_start(ts)
                engine.on_candle(candle)

            # Collect whatever closed on this bar and settle it centrally.
            realised = 0.0
            for sym in self.symbols:
                trades = self.engines[sym].trades
                while seen[sym] < len(trades):
                    realised += trades[seen[sym]].pnl
                    seen[sym] += 1
            if realised:
                self.equity += realised
            for sym in self.symbols:
                self.engines[sym].risk.equity = self.equity

            if not self.halted_today and self.day_start_equity > 0:
                drawdown = (self.day_start_equity - self.equity) / self.day_start_equity
                if drawdown >= self.limit:
                    self.halted_today = True
                    self.halted_days += 1

            self.equity_curve.append((ts, round(self.mark_to_market(index, ts), 6)))
            self.bars += 1
            if progress and self.bars % 50_000 == 0:
                progress(self.bars)

        for engine in self.engines.values():
            engine._force_close_at_end()

        by_symbol = {sym: list(self.engines[sym].trades) for sym in self.symbols}
        all_trades = sorted((t for ts in by_symbol.values() for t in ts),
                            key=lambda t: t.entry_time)
        rejections: Dict[str, int] = {}
        for sym, engine in self.engines.items():
            for k, v in {**engine.signals.rejections, **engine.risk.rejections}.items():
                rejections[k] = rejections.get(k, 0) + v

        return PortfolioResult(
            equity_curve=self.equity_curve,
            trades=all_trades,
            by_symbol=by_symbol,
            metrics=compute_metrics(all_trades, self.equity_curve,
                                    self.cfg.initial_equity),
            per_symbol_metrics={
                sym: compute_metrics(ts, [], self.cfg.initial_equity)
                for sym, ts in by_symbol.items()
            },
            bars=self.bars,
            halted_days=self.halted_days,
            rejections=rejections,
        )

    # ------------------------------------------------------------------
    def _roll_day(self, ts: int) -> None:
        d = day_start(ts)
        if self.current_day is None:
            self.current_day = d
            self.day_start_equity = self.equity
            return
        if d != self.current_day:
            self.current_day = d
            self.day_start_equity = self.equity
            self.halted_today = False
            for engine in self.engines.values():
                engine.risk.trading_disabled_until_day = None

    def mark_to_market(self, index: Dict[str, Dict[int, Candle]], ts: int) -> float:
        eq = self.equity
        for sym, engine in self.engines.items():
            pos = engine.position
            if pos is None:
                continue
            candle = index.get(sym, {}).get(ts)
            price = candle.close if candle else pos.entry
            eq += pos.unrealised(price) + pos.realised - pos.fees - pos.funding
        return eq
