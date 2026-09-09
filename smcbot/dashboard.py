"""Section 98 -- the state dashboard.

Renders the live context and account state as text or JSON.  It reads the same
objects the strategy reads, so what is shown is exactly what the bot sees.
"""
from __future__ import annotations

import datetime as dt
import json
from typing import Dict, List, Optional

from .backtest.metrics import compute_metrics
from .core.types import Trade
from .engine.context import SMCContext
from .strategy.decision import RiskManager
from .strategy.setup import SignalEngine


def _ts(ms: Optional[int]) -> str:
    if not ms:
        return "-"
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S")


def build(ctx: SMCContext, risk: RiskManager, signals: SignalEngine,
          trades: List[Trade], position=None) -> Dict[str, object]:
    snap = ctx.snapshot()
    today = [t for t in trades if risk.current_day is not None
             and t.entry_time >= risk.current_day]
    metrics = compute_metrics(trades, [], risk.start_equity) if trades else {}
    best = max((c for c in signals.candidates), key=lambda c: c.state.value,
               default=None)
    return {
        "time": _ts(snap["time"]),
        "price": snap["price"],
        "market_state": snap["market_state"],
        "session": snap["session"],
        "structure": {"m15": snap["m15"], "m5": snap["m5"], "m1": snap["m1"]},
        "atr": snap["atr"],
        "liquidity": {
            "alive": snap["liquidity_alive"],
            "above": snap["liquidity_above"],
            "below": snap["liquidity_below"],
            "last_sweep": _ts(snap["last_sweep"]),
        },
        "candidates": [
            {"id": c.setup_id, "side": c.side.value, "state": c.state.value,
             "kind": c.kind, "expires": _ts(c.expires_at)}
            for c in signals.candidates
        ],
        "position": None if position is None else {
            "side": position.side.value, "entry": position.entry,
            "qty": position.qty, "stop": position.stop,
            "targets": position.targets, "tp_hits": position.tp_hits,
            "unrealised": round(position.unrealised(snap["price"]), 4),
            "r": round(position.r_of(snap["price"]), 3),
        },
        "account": {
            "equity": round(risk.equity, 2),
            "daily_pnl": round(risk.equity - risk.day_start_equity, 2),
            "daily_drawdown": round(risk.daily_drawdown, 4),
            "trades_today": risk.trades_today,
            "consecutive_losses": risk.consecutive_losses,
            "cooldown_until": _ts(risk.cooldown_until),
            "trading_disabled": risk.trading_disabled_until_day is not None,
        },
        "performance": {
            "trades": metrics.get("trades", 0),
            "win_rate": metrics.get("win_rate", 0.0),
            "profit_factor": metrics.get("profit_factor", 0.0),
            "expectancy_r": metrics.get("expectancy_r", 0.0),
            "max_drawdown": metrics.get("max_drawdown", 0.0),
            "trades_today": len(today),
        },
    }


def render(data: Dict[str, object]) -> str:
    a, p = data["account"], data["performance"]
    lines = [
        "=" * 62,
        f" {data['time']} UTC   price {data['price']}   {data['market_state']}"
        f" / {data['session']}",
        "=" * 62,
        f" M15 ext {data['structure']['m15']['external']:8s} int "
        f"{data['structure']['m15']['internal']:8s} zone "
        f"{data['structure']['m15']['zone']}",
        f" M5  int {data['structure']['m5']['internal']:8s} "
        f"M1 int {data['structure']['m1']['internal']}",
        f" Range   {data['structure']['m15']['range_low']} .. "
        f"{data['structure']['m15']['range_high']}",
        f" ATR     {data['atr']}",
        f" Liquidity: {data['liquidity']['alive']} alive "
        f"({data['liquidity']['below']} below / {data['liquidity']['above']} above)"
        f"   last sweep {data['liquidity']['last_sweep']}",
        "-" * 62,
    ]
    if data["candidates"]:
        lines.append(" Active setups:")
        for c in data["candidates"]:
            lines.append(f"   {c['side']:4s} {c['state']:22s} {c['kind']:15s}"
                         f" exp {c['expires']}")
    else:
        lines.append(" Active setups: none")
    pos = data["position"]
    if pos:
        lines += ["-" * 62,
                  f" POSITION {pos['side']} {pos['qty']} @ {pos['entry']}  "
                  f"SL {pos['stop']}  TP {pos['targets']}  "
                  f"uPnL {pos['unrealised']:+.2f} ({pos['r']:+.2f}R)"]
    lines += [
        "-" * 62,
        f" Equity {a['equity']:.2f}   day {a['daily_pnl']:+.2f} "
        f"({a['daily_drawdown']:.2%})   trades today {a['trades_today']}",
        f" Losses in a row {a['consecutive_losses']}   cooldown until "
        f"{a['cooldown_until']}   disabled: {a['trading_disabled']}",
        f" Total {p['trades']} trades  win {p['win_rate']:.2%}  "
        f"PF {p['profit_factor']}  exp {p['expectancy_r']}R  "
        f"maxDD {p['max_drawdown']:.2%}",
        "=" * 62,
    ]
    return "\n".join(lines)


def to_json(data: Dict[str, object]) -> str:
    return json.dumps(data, indent=2, default=str)
