"""Sections 55-56, 74-76, 91 -- the event-driven backtester.

**Causality.**  Every bar is processed in this exact order:

1. fill orders that were *placed at the close of the previous bar*;
2. manage the open position against this bar's OHLC;
3. only now show this bar to the analysis context;
4. ask the strategy for a decision, which can place an order for the *next* bar.

Steps 1-2 happen before step 3, so the position manager never benefits from
knowing what the analysis engine will learn from the same bar, and step 4 can
never act on the bar it is currently inside.  A signal formed on the close of
bar *t* is filled no earlier than bar *t+1*.

**Pessimism.**  Intrabar path is unknowable from OHLC alone, so whenever a bar
could have hit both the stop and a take profit, the stop is assumed first.
Liquidation is checked before either.  Slippage, taker/maker fees and funding
are charged on every fill.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from ..config import Config
from ..core.types import (Candle, Fill, Setup, Side, Trade)
from ..engine.context import SMCContext
from ..strategy.decision import Decision, DecisionEngine, RiskManager
from ..strategy.risk import liquidation_price, round_step, round_tick
from ..strategy.setup import SignalEngine

HOUR_MS = 3_600_000


@dataclass
class PendingOrder:
    setup: Setup
    decision: Decision
    kind: str                 # MARKET | LIMIT
    price: float
    qty: float
    placed_at: int
    bars_alive: int = 0


@dataclass
class Position:
    setup: Setup
    side: Side
    entry: float
    qty: float
    initial_qty: float
    stop: float
    targets: List[float]
    portions: List[float]
    opened_at: int
    risk_per_unit: float
    liq_price: float
    risk_pct: float
    fees: float = 0.0
    funding: float = 0.0
    realised: float = 0.0
    tp_hits: List[int] = field(default_factory=list)
    mfe: float = 0.0
    mae: float = 0.0
    be_moved: bool = False
    trailed: bool = False
    fills: List[Fill] = field(default_factory=list)
    last_funding: int = 0
    exit_request: str = ""

    def unrealised(self, price: float) -> float:
        return (price - self.entry) * self.qty * self.side.sign

    def r_of(self, price: float) -> float:
        if self.risk_per_unit <= 0:
            return 0.0
        return (price - self.entry) * self.side.sign / self.risk_per_unit


class Backtester:
    def __init__(self, cfg: Config,
                 ml_predictor: Optional[Callable[[dict], float]] = None,
                 feature_builder: Optional[Callable] = None,
                 collect_rejected: bool = False):
        self.cfg = cfg
        self.ctx = SMCContext(cfg)
        self.signals = SignalEngine(cfg)
        self.risk = RiskManager(cfg, cfg.initial_equity)
        self.decider = DecisionEngine(cfg, self.risk)
        self.ml_predictor = ml_predictor
        self.feature_builder = feature_builder
        self.collect_rejected = collect_rejected

        self.position: Optional[Position] = None
        self.pending: Optional[PendingOrder] = None
        self.trades: List[Trade] = []
        self.rejected_setups: List[dict] = []
        self.equity_curve: List[tuple] = []
        self.bars = 0

    # ------------------------------------------------------------------
    def run(self, candles, progress: Optional[Callable[[int], None]] = None
            ) -> "BacktestResult":
        from .metrics import BacktestResult, compute_metrics
        for candle in candles:
            self.on_candle(candle)
            self.bars += 1
            if progress and self.bars % 20000 == 0:
                progress(self.bars)
        self._force_close_at_end()
        metrics = compute_metrics(self.trades, self.equity_curve,
                                  self.cfg.initial_equity)
        return BacktestResult(
            config=self.cfg, trades=self.trades, metrics=metrics,
            equity_curve=self.equity_curve, bars=self.bars,
            rejections={**self.signals.rejections, **self.risk.rejections},
            rejected_setups=self.rejected_setups,
        )

    # ------------------------------------------------------------------
    def on_candle(self, candle: Candle) -> None:
        self.risk.roll_day(candle.open_time)

        # 1. orders placed on the previous close
        self._try_fill(candle)
        # 2. manage the position inside this bar
        if self.position is not None:
            self._manage(candle)
        # 3. now the bar becomes visible to the analysis
        self.ctx.on_m1(candle)
        # 4. decide, for the next bar
        self._decide(candle)

        self.equity_curve.append((candle.close_time, self.mark_to_market(candle.close)))

    def mark_to_market(self, price: float) -> float:
        eq = self.risk.equity
        if self.position is not None:
            eq += self.position.unrealised(price) + self.position.realised
            eq -= self.position.fees + self.position.funding
        return round(eq, 6)

    # ---- execution ----------------------------------------------------
    def _slip(self, price: float, side: Side, adverse: bool = True) -> float:
        bps = self.cfg.execution.slippage_bps / 10_000.0
        direction = side.sign if adverse else -side.sign
        return price * (1.0 + direction * bps)

    def _fee(self, notional: float, maker: bool) -> float:
        rate = self.cfg.execution.maker_fee if maker else self.cfg.execution.taker_fee
        return abs(notional) * rate

    def _try_fill(self, candle: Candle) -> None:
        order = self.pending
        if order is None:
            return
        if self.position is not None:            # section 93 -- one position only
            self.pending = None
            return
        order.bars_alive += 1

        if order.kind == "MARKET":
            price = self._slip(candle.open, order.setup.side)
            self._open_position(order, price, candle.open_time, maker=False)
            self.pending = None
            return

        # LIMIT: only fills if this bar actually traded through the price.
        side = order.setup.side
        touched = (candle.low <= order.price) if side is Side.BUY else (candle.high >= order.price)
        if touched:
            # A gap through the limit fills at the (better) open, not the limit.
            if side is Side.BUY and candle.open < order.price:
                price = candle.open
            elif side is Side.SELL and candle.open > order.price:
                price = candle.open
            else:
                price = order.price
            self._open_position(order, price, candle.open_time, maker=True)
            self.pending = None
            return

        stale = order.bars_alive >= self.cfg.execution.limit_order_ttl_bars
        expired = candle.close_time > order.setup.expires_at
        if stale or expired:
            self.signals.reject("limit_order_expired")
            self.pending = None

    def _open_position(self, order: PendingOrder, price: float, ts: int,
                       maker: bool) -> None:
        cfg = self.cfg
        setup = order.setup
        side = setup.side
        # Re-derive risk from the *actual* fill so a slipped entry does not
        # silently take more risk than configured.
        risk_per_unit = abs(price - setup.stop)
        if risk_per_unit <= 0:
            self.signals.reject("fill_invalidated_stop")
            return
        qty = round_step((self.risk.equity * order.decision.risk_pct) / risk_per_unit,
                         cfg.execution.qty_step)
        qty = min(qty, order.qty) if order.qty > 0 else qty
        if qty < cfg.execution.min_qty or qty * price < cfg.execution.min_notional:
            self.signals.reject("fill_below_min_size")
            return
        # If price already ran past the stop or the first target, skip the trade.
        first_tp = setup.targets[0].price if setup.targets else None
        if side is Side.BUY and (price <= setup.stop or (first_tp and price >= first_tp)):
            self.signals.reject("fill_outside_setup")
            return
        if side is Side.SELL and (price >= setup.stop or (first_tp and price <= first_tp)):
            self.signals.reject("fill_outside_setup")
            return

        fee = self._fee(qty * price, maker)
        portions = self._portions(len(setup.targets))
        pos = Position(
            setup=setup, side=side, entry=price, qty=qty, initial_qty=qty,
            stop=setup.stop, targets=[t.price for t in setup.targets],
            portions=portions, opened_at=ts, risk_per_unit=risk_per_unit,
            liq_price=liquidation_price(side, price, cfg.risk.leverage,
                                        cfg.risk.maintenance_margin_rate),
            risk_pct=order.decision.risk_pct, fees=fee,
            last_funding=ts,
        )
        pos.fills.append(Fill(ts, price, qty, fee, "ENTRY"))
        self.position = pos
        self.risk.open_positions += 1
        self.risk.trades_today += 1
        self.risk.count_session_trade(self.ctx.session().value)
        self.signals.mark_traded(setup.setup_id)

    def _portions(self, n_targets: int) -> List[float]:
        p = self.cfg.risk.partial_tp
        weights = [p["tp1"], p["tp2"], p["tp3"]][:max(n_targets, 1)]
        total = sum(weights) or 1.0
        return [w / total for w in weights]

    # ---- position management ------------------------------------------
    def _manage(self, candle: Candle) -> None:
        pos = self.position
        assert pos is not None
        self._apply_funding(pos, candle)

        # Excursions, measured on the bar extremes (in R).
        best = candle.high if pos.side is Side.BUY else candle.low
        worst = candle.low if pos.side is Side.BUY else candle.high
        pos.mfe = max(pos.mfe, pos.r_of(best))
        pos.mae = min(pos.mae, pos.r_of(worst))

        # 1. Liquidation dominates everything (section 34).
        liq_hit = (candle.low <= pos.liq_price) if pos.side is Side.BUY \
            else (candle.high >= pos.liq_price)
        if liq_hit:
            self._close(pos, pos.liq_price, candle.close_time, "LIQUIDATION",
                        maker=False)
            return

        # 2. Stop before target -- we cannot see the intrabar path.
        stop_hit = (candle.low <= pos.stop) if pos.side is Side.BUY \
            else (candle.high >= pos.stop)
        if stop_hit:
            price = self._slip(pos.stop, pos.side)
            reason = ("BREAKEVEN" if pos.be_moved and not pos.trailed
                      else "TRAILING" if pos.trailed else "STOP")
            self._close(pos, price, candle.close_time, reason, maker=False)
            return

        # 3. Take profits, in order, within this bar.
        for i, tp in enumerate(pos.targets):
            if i in pos.tp_hits:
                continue
            hit = (candle.high >= tp) if pos.side is Side.BUY else (candle.low <= tp)
            if not hit:
                break                      # targets are ordered by distance
            self._take_profit(pos, i, tp, candle.close_time)
            if pos.qty <= 0:
                return

        # 4. Trailing (section 31).  Applied at the end of the bar, so a stop
        # raised on this bar can only be hit from the next one -- moving it up
        # on this bar's high and then testing this bar's low would assume an
        # intrabar order we cannot know.
        if self.cfg.risk.trailing_enabled:
            self._apply_trailing(pos)

        # 5. Strategy exit requested by the previous bar's analysis (section 95).
        if pos.exit_request:
            price = self._slip(candle.close, pos.side)
            self._close(pos, price, candle.close_time, pos.exit_request, maker=False)

    def _take_profit(self, pos: Position, index: int, price: float, ts: int) -> None:
        portion = pos.portions[index] if index < len(pos.portions) else 0.0
        qty = round_step(pos.initial_qty * portion, self.cfg.execution.qty_step)
        last = index == len(pos.targets) - 1
        if last and self.cfg.risk.trail_remainder and portion < 1.0:
            # Take the configured share and let the rest ride the trailing stop
            # instead of closing the position at the final target.
            qty = min(qty, pos.qty)
        elif last or qty > pos.qty:
            qty = pos.qty
        if qty <= 0:
            pos.tp_hits.append(index)
            return
        fee = self._fee(qty * price, maker=self.cfg.execution.use_server_side_stops)
        pnl = (price - pos.entry) * qty * pos.side.sign
        pos.realised += pnl
        pos.fees += fee
        pos.qty = round(pos.qty - qty, 10)
        pos.tp_hits.append(index)
        pos.fills.append(Fill(ts, price, qty, fee, f"TP{index + 1}"))

        if pos.qty <= 0:
            self._close(pos, price, ts, "TP_FINAL", maker=True, already_flat=True)
            return
        # Section 30 -- protect the runner after the first partial.
        if self.cfg.risk.move_to_be_after_tp1 and not pos.be_moved:
            offset = self.cfg.risk.be_offset_r * pos.risk_per_unit * pos.side.sign
            new_stop = pos.entry + offset
            if (pos.side is Side.BUY and new_stop > pos.stop) or \
               (pos.side is Side.SELL and new_stop < pos.stop):
                pos.stop = round_tick(new_stop, self.cfg.execution.tick_size)
                pos.be_moved = True

    def _apply_trailing(self, pos: Position) -> None:
        """Route to the configured trailing ladder."""
        risk = self.cfg.risk
        mode = risk.trailing_mode
        if mode == "LEGACY":
            if len(pos.tp_hits) >= risk.trailing_after_tp:
                self._trail(pos)
            return

        # R reached so far, measured on closed bars only.
        r = pos.mfe
        if mode == "A":
            be_at, trail_at = 0.25, 0.50
        elif mode == "B":
            be_at, trail_at = 0.50, 0.75
        elif mode == "C":
            be_at, trail_at = None, 0.75
        elif mode == "D":
            be_at, trail_at = None, 0.0
        else:
            return

        if be_at is not None and r >= be_at and not pos.be_moved:
            offset = risk.be_offset_r * pos.risk_per_unit * pos.side.sign
            new_stop = round_tick(pos.entry + offset, self.cfg.execution.tick_size)
            if self._improves(pos, new_stop):
                pos.stop = new_stop
                pos.be_moved = True
        if r >= trail_at:
            self._trail(pos, swing_tf=risk.trailing_swing_timeframe,
                        buffer_atr=risk.trailing_buffer_atr)

    @staticmethod
    def _improves(pos: Position, new_stop: float) -> bool:
        """A stop may only ever move in the position's favour."""
        return (new_stop > pos.stop) if pos.side is Side.BUY else (new_stop < pos.stop)

    def _trail(self, pos: Position, swing_tf: Optional[str] = None,
               buffer_atr: Optional[float] = None) -> None:
        """Trail behind the last *confirmed* swing of the chosen timeframe.

        The swing engine has not yet seen the current bar when this runs, so the
        anchor is a pivot that was already public -- no look-ahead.
        """
        tf = swing_tf or self.cfg.risk.trailing_timeframe
        eng = self.ctx.views[tf].structure.internal_swings
        anchor = eng.last_low() if pos.side is Side.BUY else eng.last_high()
        if anchor is None:
            return
        mult = (buffer_atr if buffer_atr is not None
                else self.cfg.risk.sl_atr_buffer)
        buffer_ = mult * (self.ctx.atr(tf) or 0.0)
        new_stop = (anchor.price - buffer_ if pos.side is Side.BUY
                    else anchor.price + buffer_)
        new_stop = round_tick(new_stop, self.cfg.execution.tick_size)
        if self._improves(pos, new_stop):
            pos.stop = new_stop
            pos.trailed = True

    def _apply_funding(self, pos: Position, candle: Candle) -> None:
        """Charge funding at each 8-hour boundary the position spans."""
        interval = self.cfg.execution.funding_interval_hours * HOUR_MS
        boundary = ((pos.last_funding // interval) + 1) * interval
        while boundary <= candle.open_time:
            notional = pos.qty * candle.open
            pos.funding += notional * self.ctx.funding_rate * pos.side.sign
            pos.last_funding = boundary
            boundary += interval

    def _close(self, pos: Position, price: float, ts: int, reason: str,
               maker: bool, already_flat: bool = False) -> None:
        if not already_flat and pos.qty > 0:
            fee = self._fee(pos.qty * price, maker)
            pnl = (price - pos.entry) * pos.qty * pos.side.sign
            pos.realised += pnl
            pos.fees += fee
            pos.fills.append(Fill(ts, price, pos.qty, fee, reason))
            pos.qty = 0.0

        # Round once, here, and use the same value for the journal and for the
        # account.  Rounding only the record would leave the equity curve and
        # the sum of trade PnL disagreeing by a few 1e-6 per trade.
        net = round(pos.realised - pos.fees - pos.funding, 6)
        setup = pos.setup
        r_mult = net / (pos.risk_per_unit * pos.initial_qty) \
            if pos.risk_per_unit * pos.initial_qty > 0 else 0.0
        trade = Trade(
            setup_id=setup.setup_id, symbol=self.cfg.symbol, side=pos.side,
            entry_time=pos.opened_at, entry_price=pos.entry, qty=pos.initial_qty,
            stop=setup.stop, targets=list(pos.targets),
            smc_score=setup.smc_score, ml_probability=setup.ml_probability,
            market_state=setup.meta.get("market_state", ""),
            session=setup.meta.get("session", ""),
            features=dict(setup.features),
            exit_time=ts, exit_price=price,
            result="WIN" if net > 0 else ("LOSS" if net < 0 else "BREAKEVEN"),
            pnl=net, fees=round(pos.fees, 6), funding=round(pos.funding, 6),
            mfe=round(pos.mfe, 4), mae=round(pos.mae, 4), r_multiple=round(r_mult, 4),
            tp_hits=list(pos.tp_hits), exit_reason=reason,
            holding_minutes=round((ts - pos.opened_at) / 60_000, 1),
            model_version=f"{self.cfg.smc_engine_version}|{self.cfg.ml.model_version}",
            fills=list(pos.fills),
            # The score breakdown rides along purely as a record: it is written
            # to the journal so a component-level audit is possible, and is
            # never read back by any decision.
            meta={**{k: v for k, v in setup.meta.items() if k != "evidence"},
                  "score_breakdown": dict(setup.score_breakdown)},
        )
        self.risk.on_trade_closed(net, ts)
        trade.equity_after = round(self.risk.equity, 6)
        self.trades.append(trade)
        self.position = None

    def _force_close_at_end(self) -> None:
        if self.position is None:
            return
        last = self.ctx.book.m1.last
        if last is None:
            return
        self._close(self.position, last.close, last.close_time, "END_OF_DATA",
                    maker=False)

    # ---- decisioning ---------------------------------------------------
    def _decide(self, candle: Candle) -> None:
        # Structure-based early exit (section 95): decided now, executed next bar.
        if self.position is not None:
            if self.cfg.risk.early_exit_on_invalidation and not self.position.exit_request:
                if self._invalidated(self.position):
                    self.position.exit_request = "STRATEGY_EXIT"
            return
        if self.pending is not None:
            return

        setups = self.signals.on_bar(self.ctx)
        for setup in setups:
            if self.feature_builder is not None:
                setup.features = self.feature_builder(self.ctx, setup)
            prob = None
            if self.ml_predictor is not None and setup.features:
                prob = self.ml_predictor(setup.features)
            decision = self.decider.evaluate(setup, self.ctx, prob)
            if not decision.approved:
                if self.collect_rejected:
                    self.rejected_setups.append(
                        {"time": self.ctx.now, "reason": decision.reason,
                         "score": setup.smc_score, "side": setup.side.value,
                         "features": dict(setup.features)})
                continue
            self.pending = PendingOrder(
                setup=setup, decision=decision, kind=setup.entry_type,
                price=setup.entry, qty=decision.size.qty if decision.size else 0.0,
                placed_at=self.ctx.now,
            )
            break

    def _invalidated(self, pos: Position) -> bool:
        """Opposite CHOCH/BOS on M5 against an open position."""
        for ev in self.ctx.views["M5"].last_events:
            if ev.bullish is not (pos.side is Side.BUY) and ev.time > pos.opened_at:
                return True
        return False
