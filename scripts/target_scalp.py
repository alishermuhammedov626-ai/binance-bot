"""TP fixed at 1%, four sessions, and an honest look at what win rate buys.

    python scripts/target_scalp.py --csv data/BTCUSDT_1m.csv

THE ARITHMETIC THIS SCRIPT EXISTS TO SHOW
-----------------------------------------
With the target fixed, the win rate is not something a strategy chooses -- it
is set almost entirely by the stop. On a driftless market the chance of
touching the target before the stop is SL / (TP + SL), so:

    TP 1.0%, SL 2.33%  ->  70% win rate, and expectancy exactly zero
    TP 1.0%, SL 1.00%  ->  50% win rate, and expectancy exactly zero

Every stop distance has its own break-even win rate. Beating it is what an
edge means; reaching 70% by widening the stop is not an edge, it is
relabelling the same zero. So each row below prints the win rate the signal
*would need* next to the win rate it *delivered*, and the gap between them is
the only number that matters.

Nothing here is tuned toward a target. The stop is swept, the result is
reported, and the out-of-sample column is the verdict.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from collections import defaultdict
from typing import Dict, List, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smcbot.backtest.engine import Backtester
from smcbot.config import Config
from smcbot.core.types import Side, Trade
from smcbot.data import loader, synthetic

TP_PCT = 1.0
SL_GRID = [0.50, 0.65, 0.80, 1.00, 1.25, 1.50, 2.00, 2.33]

# Four Tashkent blocks -> UTC (UTC+5), so a day holds at most four slots.
#   ASIA     00:00-08:00 local -> 19-03 UTC
#   LONDON   08:00-14:00 local -> 03-09 UTC
#   NEW YORK 14:00-20:00 local -> 09-15 UTC
#   LATE     20:00-24:00 local -> 15-19 UTC
BASE = {
    "sessions.asia": (19, 3), "sessions.london": (3, 9),
    "sessions.new_york": (9, 15), "sessions.late": (15, 19),
    "risk.allowed_sessions": ["ASIA", "LONDON", "NEW_YORK", "LATE"],
    "risk.max_trades_per_session": 1,
    "risk.max_trades_per_day": 4,

    "risk.tp_mode": "PERCENT",
    "risk.tp_percent_levels": [TP_PCT],
    "risk.partial_tp": {"tp1": 1.0, "tp2": 0.0, "tp3": 0.0},
    "risk.move_to_be_after_tp1": False,
    "risk.trailing_enabled": False,
    "risk.early_exit_on_invalidation": False,

    "risk.stop_mode": "ATR_CLAMPED",
    "risk.sl_atr_multiplier": 1.5,
    "risk.sl_atr_period_timeframe": "M5",

    "risk.leverage": 17.0,
    "risk.risk_per_trade": 0.005,
    "risk.min_risk_per_trade": 0.0025,
    "risk.max_risk_per_trade": 0.005,
    "risk.consecutive_loss_cooldowns": {"2": 120, "3": 180, "5": 360},
    "risk.min_rr": 0.3,          # RR is fixed by TP/SL here, not a filter
    "risk.min_first_target_rr": 0.3,   # a scalp target below 1R is the point
}


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def gross_pnl(t: Trade) -> float:
    return t.pnl + t.fees + t.funding


def risk_usdt(t: Trade) -> float:
    return abs(t.entry_price - t.stop) * t.qty


def net_r(t: Trade) -> float:
    r = risk_usdt(t)
    return t.pnl / r if r > 0 else 0.0


def fee_r(t: Trade) -> float:
    r = risk_usdt(t)
    return t.fees / r if r > 0 else 0.0


def max_dd(curve: Sequence[tuple]) -> float:
    peak, worst = -float("inf"), 0.0
    for _, eq in curve:
        peak = max(peak, eq)
        if peak > 0:
            worst = max(worst, (peak - eq) / peak)
    return worst


def random_walk_win(tp: float, sl: float) -> float:
    """Chance of touching TP before SL with no drift."""
    return sl / (tp + sl)


def breakeven_win(tp: float, sl: float, fee_r_value: float) -> float:
    """Win rate needed for zero net expectancy, costs included.

    p*(TP/SL) - (1-p)*1 - fee_r = 0
    """
    rr = tp / sl
    return (1.0 + fee_r_value) / (1.0 + rr)


def summarise(trades: List[Trade], curve: List[tuple], days: float) -> dict:
    if not trades:
        return {"n": 0}
    nets = [net_r(t) for t in trades]
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl < 0]
    gw = sum(t.pnl for t in wins)
    gl = -sum(t.pnl for t in losses)
    return {
        "n": len(trades),
        "per_day": len(trades) / days,
        "win": len(wins) / len(trades),
        "gross_win": sum(1 for t in trades if gross_pnl(t) > 0) / len(trades),
        "exp": mean(nets),
        "gross_exp": mean([gross_pnl(t) / risk_usdt(t) for t in trades
                           if risk_usdt(t) > 0]),
        "pf": (gw / gl) if gl > 0 else float("inf"),
        "fee": mean([fee_r(t) for t in trades]),
        "dd": max_dd(curve),
        "net": sum(t.pnl for t in trades),
        "sl_pct": mean([100 * abs(t.entry_price - t.stop) / t.entry_price
                        for t in trades]),
        "hold": mean([t.holding_minutes for t in trades]),
    }


HEAD = (f"{'SL%':>6s} {'RR':>5s} {'n':>5s} {'t/day':>6s} {'kerak':>7s} "
        f"{'o^lchan':>8s} {'farq':>7s} {'expR':>7s} {'PF':>6s} {'fee/R':>6s} "
        f"{'maxDD':>7s} {'net':>9s}")


def line(sl: float, s: dict) -> str:
    if not s.get("n"):
        return f"{sl:6.2f} {'-':>5s} {'savdo yo^q':>5s}"
    need = breakeven_win(TP_PCT, s["sl_pct"], s["fee"])
    pf = " inf" if s["pf"] == float("inf") else f"{s['pf']:6.3f}"
    return (f"{s['sl_pct']:6.2f} {TP_PCT / s['sl_pct']:5.2f} {s['n']:5d} "
            f"{s['per_day']:6.2f} {need * 100:6.1f}% {s['win'] * 100:7.1f}% "
            f"{(s['win'] - need) * 100:+6.1f}% {s['exp']:+7.3f} {pf} "
            f"{s['fee']:6.3f} {s['dd'] * 100:6.1f}% {s['net']:+9.2f}")


def show_theory() -> None:
    print("=" * 96)
    print(f"NAZARIYA: TP {TP_PCT}% qat'iy, SL o'zgaradi")
    print("=" * 96)
    print(f"{'SL%':>6s} {'RR':>6s} {'random-walk win':>17s} "
          f"{'fee/R (taxmin)':>15s} {'nolga chiqish uchun kerak':>26s}")
    print("-" * 96)
    for sl in SL_GRID:
        rr = TP_PCT / sl
        fee = 2 * 0.0005 / (sl / 100.0)      # two taker fills
        need = breakeven_win(TP_PCT, sl, fee)
        print(f"{sl:6.2f} {rr:6.2f} {random_walk_win(TP_PCT, sl) * 100:16.1f}% "
              f"{fee:15.3f} {need * 100:25.1f}%")
    print("\nO'qish: SL kengaysa win rate ko'tariladi, lekin kerakli chegara")
    print("ham birga ko'tariladi. 70% ni SL orqali 'olish' mumkin, edge orqali")
    print("emas. Ahamiyatga ega yagona ustun -- o'lchangan minus kerakli.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--days", type=int, default=180)
    args = ap.parse_args()

    if args.csv:
        print(f"[data] {args.csv}")
        candles = loader.load_csv(args.csv)
    else:
        candles = synthetic.generate(args.days * 1440, seed=42)
    n = len(candles)
    print(f"[data] {n} candles ({n / 1440:.0f} kun)\n")

    show_theory()

    a, b = int(n * 0.6), int(n * 0.8)
    parts = [("TRAIN", candles[:a]), ("VALIDATION", candles[a:b]),
             ("OOS TEST", candles[b:])]

    results: Dict[str, Dict[float, dict]] = {}
    for label, chunk in parts:
        days = len(chunk) / 1440
        print(f"\n{'=' * 96}\n{label}  ({days:.0f} kun)  --  TP {TP_PCT}%\n{'=' * 96}")
        print(HEAD)
        print("-" * 96)
        results[label] = {}
        for sl in SL_GRID:
            cfg = Config().with_overrides(**{
                **BASE, "risk.sl_min_pct": sl, "risk.sl_max_pct": sl})
            t0 = time.time()
            bt = Backtester(cfg)
            res = bt.run(chunk)
            s = summarise(res.trades, res.equity_curve, days)
            results[label][sl] = s
            out = line(sl, s)
            if not s.get("n"):
                # Say *why* rather than leaving a silent blank row.
                top = sorted(res.rejections.items(), key=lambda kv: -kv[1])[:2]
                out += "   sabab: " + ", ".join(f"{k}={v}" for k, v in top)
            print(out)
            sys.stdout.flush()

    print(f"\n{'=' * 96}\nXULOSA -- OOS TEST\n{'=' * 96}")
    oos = results["OOS TEST"]
    usable = [(sl, s) for sl, s in oos.items() if s.get("n", 0) >= 20]
    if not usable:
        print("OOS davrda hech bir variant 20 savdo chegarasidan o'tmadi.")
        return 0

    hit70 = [(sl, s) for sl, s in usable if s["win"] >= 0.70]
    positive = [(sl, s) for sl, s in usable if s["exp"] > 0]
    beats = [(sl, s) for sl, s in usable
             if s["win"] > breakeven_win(TP_PCT, s["sl_pct"], s["fee"])]

    print(f"70%+ win rate: "
          f"{', '.join(f'SL {sl}%' for sl, _ in hit70) if hit70 else 'yo^q'}")
    print(f"musbat expectancy: "
          f"{', '.join(f'SL {sl}%' for sl, _ in positive) if positive else 'yo^q'}")
    print(f"kerakli chegaradan yuqori: "
          f"{', '.join(f'SL {sl}%' for sl, _ in beats) if beats else 'yo^q'}")

    both = [(sl, s) for sl, s in usable if s["win"] >= 0.70 and s["exp"] > 0]
    print(f"\nIkkalasi ham (70%+ VA musbat expectancy): "
          f"{', '.join(f'SL {sl}%' for sl, _ in both) if both else 'YO^Q'}")
    if not both:
        print("\nBu shuni bildiradi: TP 1% da 70% win rate va musbat kutilma")
        print("bir vaqtda chiqmadi. 70% ni SL kengaytirib olish mumkin, lekin")
        print("u paytda kerakli chegara ham 73%+ ga ko'tariladi.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
