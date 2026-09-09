"""Sections 33-38, 46, 87 -- the final decision gate and portfolio guard rails.

``SignalEngine`` says "here is a technically valid setup".  This module answers
"are we allowed to take it right now", which is a completely different
question: it owns the daily loss limit, cooldowns, trade counters, market-health
filters and the SMC+ML conjunction of section 87.

The same object is used by the backtest and by a live runner, so the rules can
never drift apart between them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..config import Config
from ..core.types import Setup
from ..core.series import day_start
from ..engine.context import SMCContext
from .risk import SizePlan, size_position

MINUTE = 60_000


@dataclass
class Decision:
    approved: bool
    reason: str
    setup: Optional[Setup] = None
    size: Optional[SizePlan] = None
    risk_pct: float = 0.0
    checks: Dict[str, bool] = field(default_factory=dict)


class RiskManager:
    """Mutable account-level state shared by every gate."""

    def __init__(self, cfg: Config, equity: float):
        self.cfg = cfg
        self.equity = equity
        self.start_equity = equity
        self.day_start_equity = equity
        self.current_day: Optional[int] = None
        self.trades_today = 0
        # session name -> trades opened in the current occurrence of it
        self.trades_this_session: Dict[str, int] = {}
        self.current_session: str = ""
        self.consecutive_losses = 0
        self.cooldown_until = 0
        self.trading_disabled_until_day: Optional[int] = None
        self.open_positions = 0
        self.rejections: Dict[str, int] = {}
        self.daily_pnl_history: List[tuple] = []

    # ------------------------------------------------------------------
    def roll_day(self, ts: int) -> None:
        d = day_start(ts)
        if self.current_day is None:
            self.current_day = d
            self.day_start_equity = self.equity
            return
        if d != self.current_day:
            self.daily_pnl_history.append(
                (self.current_day, self.equity - self.day_start_equity))
            self.current_day = d
            self.day_start_equity = self.equity
            self.trades_today = 0
            self.trades_this_session = {}
            if self.trading_disabled_until_day is not None and \
                    d > self.trading_disabled_until_day:
                self.trading_disabled_until_day = None

    @property
    def daily_drawdown(self) -> float:
        if self.day_start_equity <= 0:
            return 0.0
        return (self.day_start_equity - self.equity) / self.day_start_equity

    def on_trade_closed(self, pnl: float, ts: int) -> None:
        self.equity += pnl
        self.open_positions = max(0, self.open_positions - 1)
        risk = self.cfg.risk
        if pnl < 0:
            self.consecutive_losses += 1
            cooldown = risk.cooldown_after_loss_minutes
            for losses, minutes in sorted(
                    ((int(k), v) for k, v in risk.consecutive_loss_cooldowns.items()),
                    reverse=True):
                if self.consecutive_losses >= losses:
                    cooldown = max(cooldown, minutes)
                    break
            self.cooldown_until = ts + cooldown * MINUTE
        else:
            self.consecutive_losses = 0
            self.cooldown_until = ts + risk.cooldown_minutes * MINUTE
        # Section 35 -- stop for the rest of the UTC day.
        if self.daily_drawdown >= risk.daily_loss_limit:
            self.trading_disabled_until_day = day_start(ts)

    def note_session(self, session: str) -> None:
        """Reset the per-session counter when a new session begins."""
        if session != self.current_session:
            self.current_session = session
            self.trades_this_session[session] = 0

    def session_trades(self, session: str) -> int:
        return self.trades_this_session.get(session, 0)

    def count_session_trade(self, session: str) -> None:
        self.trades_this_session[session] = self.session_trades(session) + 1

    def reject(self, reason: str) -> None:
        self.rejections[reason] = self.rejections.get(reason, 0) + 1

    def risk_pct_for(self, setup: Setup) -> float:
        """Scale risk with conviction, inside the configured band (section 92)."""
        risk = self.cfg.risk
        lo, hi = risk.min_risk_per_trade, risk.max_risk_per_trade
        score = setup.smc_score
        if score >= self.cfg.score.a_plus:
            frac = 1.0
        elif score >= self.cfg.score.high_quality:
            frac = 0.7
        else:
            frac = 0.4
        return min(risk.risk_per_trade, lo + (hi - lo) * frac)


class DecisionEngine:
    """Section 87 -- every condition must hold; any single failure is NO TRADE."""

    def __init__(self, cfg: Config, risk: RiskManager):
        self.cfg = cfg
        self.risk = risk

    def market_ok(self, ctx: SMCContext) -> tuple:
        """Sections 46, 63-66 -- hard market-health filters."""
        f = self.cfg.filters
        if not ctx.ready():
            return False, "data_not_ready"
        vol = ctx.volatility_pct("M15")
        if vol < f.min_atr_pct:
            return False, "volatility_too_low"
        if vol > f.max_atr_pct:
            return False, "volatility_extreme"
        if ctx.spread_bps > f.max_spread_bps:
            return False, "spread_too_wide"
        if abs(ctx.funding_rate) > f.extreme_funding and vol > f.max_atr_pct * 0.6:
            return False, "extreme_funding_and_volatility"
        if f.news_filter:
            return False, "news_filter_no_feed"   # never assume "no news"
        return True, "ok"

    def evaluate(self, setup: Setup, ctx: SMCContext,
                 ml_probability: Optional[float]) -> Decision:
        cfg = self.cfg
        rm = self.risk
        checks: Dict[str, bool] = {}

        def fail(reason: str) -> Decision:
            rm.reject(reason)
            return Decision(False, reason, setup, None, 0.0, checks)

        # -- portfolio state (sections 33, 35-38, 93) -------------------
        checks["no_position"] = rm.open_positions < cfg.risk.max_open_positions
        if not checks["no_position"]:
            return fail("position_already_open")
        checks["daily_loss_limit"] = rm.trading_disabled_until_day is None
        if not checks["daily_loss_limit"]:
            return fail("daily_loss_limit")
        checks["cooldown"] = ctx.now >= rm.cooldown_until
        if not checks["cooldown"]:
            return fail("cooldown")
        checks["trade_limit"] = rm.trades_today < cfg.risk.max_trades_per_day
        if not checks["trade_limit"]:
            return fail("max_trades_per_day")

        # Section 37 as a per-session cap: an upper bound, never a quota.  A
        # session with no valid setup simply trades zero.
        session = ctx.session().value
        rm.note_session(session)
        if cfg.risk.max_trades_per_session > 0:
            checks["session_limit"] = (rm.session_trades(session)
                                       < cfg.risk.max_trades_per_session)
            if not checks["session_limit"]:
                return fail("max_trades_per_session")

        # -- market health (section 46) --------------------------------
        ok, why = self.market_ok(ctx)
        checks["market_ok"] = ok
        if not ok:
            return fail(why)

        # -- setup quality (sections 29, 45, 51) -----------------------
        checks["rr"] = setup.rr >= cfg.risk.min_rr
        if not checks["rr"]:
            return fail("rr_below_minimum")
        checks["smc_score"] = setup.smc_score >= cfg.score.valid
        if not checks["smc_score"]:
            return fail("smc_score_below_threshold")

        checks["ml"] = True
        if cfg.ml.enabled:
            if ml_probability is None:
                # No trained model yet: the SMC engine stands alone rather than
                # inventing a probability.
                checks["ml"] = True
            else:
                setup.ml_probability = ml_probability
                checks["ml"] = ml_probability >= cfg.ml.threshold
                if not checks["ml"]:
                    return fail("ml_below_threshold")

        # -- sizing and liquidation (sections 32, 34) ------------------
        risk_pct = rm.risk_pct_for(setup)
        plan = size_position(setup.side, setup.entry, setup.stop, rm.equity,
                             risk_pct, cfg.risk, cfg.execution)
        checks["size"] = plan.valid
        if not plan.valid:
            return fail(f"size:{plan.reason}")

        return Decision(True, "approved", setup, plan, risk_pct, checks)
