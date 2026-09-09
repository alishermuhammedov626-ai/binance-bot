"""Sections 57-59 -- performance measurement.

Win rate on its own is explicitly *not* the criterion (section 58): a 70% win
rate with oversized losses is a losing strategy.  The gate for production is
expectancy > 0 after costs (section 59), read together with profit factor and
drawdown.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

from ..core.types import Trade

DAY_MS = 86_400_000


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _stdev(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def max_drawdown(curve: List[tuple]) -> tuple:
    """Return ``(max_dd_fraction, peak_equity, trough_equity)``."""
    peak = -float("inf")
    max_dd = 0.0
    peak_at = trough = 0.0
    for _, eq in curve:
        peak = max(peak, eq)
        if peak > 0:
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd, peak_at, trough = dd, peak, eq
    return max_dd, peak_at, trough


def streaks(trades: List[Trade]) -> tuple:
    best = worst = cur_w = cur_l = 0
    for t in trades:
        if t.pnl > 0:
            cur_w += 1
            cur_l = 0
        elif t.pnl < 0:
            cur_l += 1
            cur_w = 0
        best = max(best, cur_w)
        worst = max(worst, cur_l)
    return best, worst


def compute_metrics(trades: List[Trade], curve: List[tuple],
                    initial_equity: float) -> Dict[str, float]:
    n = len(trades)
    if n == 0:
        return {"trades": 0, "note": "no trades"}

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl < 0]
    gross_win = sum(t.pnl for t in wins)
    gross_loss = -sum(t.pnl for t in losses)
    net = sum(t.pnl for t in trades)

    avg_win = _mean([t.pnl for t in wins])
    avg_loss = _mean([-t.pnl for t in losses])
    p_win = len(wins) / n
    p_loss = len(losses) / n
    # Section 59 -- costs are already inside t.pnl, so this is net expectancy.
    expectancy = p_win * avg_win - p_loss * avg_loss

    r_multiples = [t.r_multiple for t in trades]
    returns = []
    if curve:
        prev = curve[0][1]
        for _, eq in curve[1:]:
            if prev > 0:
                returns.append((eq - prev) / prev)
            prev = eq

    dd, peak, trough = max_drawdown(curve)
    best_streak, worst_streak = streaks(trades)

    # Annualisation from M1 bars (24/7 market).
    periods_per_year = 365 * 24 * 60
    mu, sd = _mean(returns), _stdev(returns)
    sharpe = (mu / sd * math.sqrt(periods_per_year)) if sd > 0 else 0.0
    downside = _stdev([r for r in returns if r < 0])
    sortino = (mu / downside * math.sqrt(periods_per_year)) if downside > 0 else 0.0

    span_days = ((curve[-1][0] - curve[0][0]) / DAY_MS) if len(curve) > 1 else 0.0
    total_return = net / initial_equity if initial_equity else 0.0
    cagr = ((1 + total_return) ** (365 / span_days) - 1) if span_days > 30 else 0.0
    calmar = (cagr / dd) if dd > 0 else 0.0

    return {
        "trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(p_win, 4),
        "loss_rate": round(p_loss, 4),
        "profit_factor": round(gross_win / gross_loss, 4) if gross_loss > 0
        else (float("inf") if gross_win > 0 else 0.0),
        "expectancy": round(expectancy, 4),
        "expectancy_r": round(_mean(r_multiples), 4),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "avg_rr": round(_mean([t.r_multiple for t in wins]), 4) if wins else 0.0,
        "net_pnl": round(net, 4),
        "total_return": round(total_return, 4),
        "cagr": round(cagr, 4),
        "max_drawdown": round(dd, 4),
        "peak_equity": round(peak, 2),
        "trough_equity": round(trough, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "consecutive_wins": best_streak,
        "consecutive_losses": worst_streak,
        "avg_mfe_r": round(_mean([t.mfe for t in trades]), 4),
        "avg_mae_r": round(_mean([t.mae for t in trades]), 4),
        "fees": round(sum(t.fees for t in trades), 4),
        "funding": round(sum(t.funding for t in trades), 4),
        "trades_per_day": round(n / span_days, 3) if span_days > 0 else 0.0,
        "span_days": round(span_days, 2),
        "final_equity": round(initial_equity + net, 2),
        "expectancy_positive": expectancy > 0,
    }


def breakdown(trades: List[Trade], key: str) -> Dict[str, dict]:
    """Per-session / per-state / per-side slices (sections 62, 88)."""
    groups: Dict[str, List[Trade]] = {}
    for t in trades:
        k = str(getattr(t, key, None) or t.meta.get(key, "UNKNOWN"))
        groups.setdefault(k, []).append(t)
    out = {}
    for k, ts in sorted(groups.items()):
        wins = [t for t in ts if t.pnl > 0]
        gl = -sum(t.pnl for t in ts if t.pnl < 0)
        out[k] = {
            "trades": len(ts),
            "win_rate": round(len(wins) / len(ts), 4),
            "net_pnl": round(sum(t.pnl for t in ts), 4),
            "expectancy_r": round(_mean([t.r_multiple for t in ts]), 4),
            "profit_factor": round(sum(t.pnl for t in wins) / gl, 3) if gl > 0 else None,
        }
    return out


@dataclass
class BacktestResult:
    config: object
    trades: List[Trade]
    metrics: Dict[str, float]
    equity_curve: List[tuple]
    bars: int
    rejections: Dict[str, int] = field(default_factory=dict)
    rejected_setups: List[dict] = field(default_factory=list)

    def summary(self) -> str:
        m = self.metrics
        if not m or m.get("trades", 0) == 0:
            return f"No trades over {self.bars} bars."
        lines = [
            f"Trades           : {m['trades']}  ({m['trades_per_day']}/day over "
            f"{m['span_days']}d)",
            f"Win rate         : {m['win_rate']:.2%}  "
            f"({m['wins']}W / {m['losses']}L)",
            f"Profit factor    : {m['profit_factor']}",
            f"Expectancy       : {m['expectancy']:.4f} USDT  "
            f"({m['expectancy_r']:.3f} R/trade)",
            f"Net PnL          : {m['net_pnl']:.2f}  "
            f"({m['total_return']:.2%})   final equity {m['final_equity']}",
            f"Max drawdown     : {m['max_drawdown']:.2%}",
            f"Sharpe / Sortino : {m['sharpe']} / {m['sortino']}   Calmar {m['calmar']}",
            f"Avg win / loss   : {m['avg_win']:.2f} / {m['avg_loss']:.2f}",
            f"Streaks          : {m['consecutive_wins']}W / {m['consecutive_losses']}L",
            f"MFE / MAE (R)    : {m['avg_mfe_r']} / {m['avg_mae_r']}",
            f"Costs            : fees {m['fees']:.2f}, funding {m['funding']:.2f}",
            f"Production gate  : expectancy {'> 0 OK' if m['expectancy_positive'] else '<= 0 REJECT'}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "metrics": self.metrics,
            "bars": self.bars,
            "by_session": breakdown(self.trades, "session"),
            "by_market_state": breakdown(self.trades, "market_state"),
            "by_side": {k: v for k, v in breakdown(self.trades, "side").items()},
            "top_rejections": dict(sorted(self.rejections.items(),
                                          key=lambda kv: -kv[1])[:15]),
        }
