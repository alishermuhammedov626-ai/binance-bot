"""JARVIS SCALP -- the specified rules, run exactly as written.

    python scripts/jarvis_scalp.py --csv data/BTCUSDT_1m.csv

No parameter search. The configuration below is a direct transcription of the
rules; nothing is tuned to make the result look better. Where the rules left a
detail open, the most conservative reading was taken and is listed under
"INTERPRETATIONS" in the output so the choice is visible rather than buried.

The data is split TRAIN / VALIDATION / OOS. Because nothing is being selected,
all three are reported side by side and the OOS column is the one that counts.
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

# Tashkent is UTC+5, and SessionConfig speaks UTC.
#   ASIA     00:00-08:00 local -> 19:00-03:00 UTC (wraps midnight)
#   LONDON   10:00-17:00 local -> 05:00-12:00 UTC
#   NEW YORK 17:00-23:00 local -> 12:00-18:00 UTC
JARVIS = {
    "sessions.asia": (19, 3),
    "sessions.london": (5, 12),
    "sessions.new_york": (12, 18),

    # Section 11 -- ATR(14) x 1.5, clamped to 0.30%-0.50%, from the fill price.
    "risk.stop_mode": "ATR_CLAMPED",
    "risk.sl_atr_multiplier": 1.5,
    "risk.sl_min_pct": 0.30,
    "risk.sl_max_pct": 0.50,
    "risk.sl_atr_period_timeframe": "M5",

    # Section 12 -- TP1 0.70%, TP2 1.00%, partial.
    "risk.tp_mode": "PERCENT",
    "risk.tp_percent_levels": [0.70, 1.00],
    "risk.partial_tp": {"tp1": 0.5, "tp2": 0.5, "tp3": 0.0},

    # Section 13 -- trail the remainder after TP1, on M1 structure, one way only.
    "risk.trailing_enabled": True,
    "risk.trailing_mode": "LEGACY",
    "risk.trailing_after_tp": 1,
    "risk.trailing_timeframe": "M1",
    "risk.move_to_be_after_tp1": False,
    # The rules enumerate the exits as TP1 / TP2 / trailing / SL, so the
    # inherited structure-invalidation exit is switched off.
    "risk.early_exit_on_invalidation": False,

    # Section 14 -- 17x, risk 0.25-0.50% of equity.
    "risk.leverage": 17.0,
    "risk.risk_per_trade": 0.005,
    "risk.min_risk_per_trade": 0.0025,
    "risk.max_risk_per_trade": 0.005,

    # Section 15 -- one trade per session, three sessions a day.
    "risk.max_trades_per_session": 1,
    "risk.max_trades_per_day": 3,
    # Hours outside the three named sessions are not traded at all.
    "risk.allowed_sessions": ["ASIA", "LONDON", "NEW_YORK"],

    # Section 16 -- two stops in a row, then cooldown.
    "risk.consecutive_loss_cooldowns": {"2": 120, "3": 180, "5": 360},
}

INTERPRETATIONS = [
    "Sessiyalar Toshkent vaqti (UTC+5) dan UTC ga o'girildi: ASIA 19-03 UTC, "
    "LONDON 05-12 UTC, NEW YORK 12-18 UTC.",
    "Qoidalarda 3 ta sessiya nomlangan. 08:00-10:00 va 23:00-00:00 (Toshkent) "
    "oraliqlari hech qaysi sessiyaga kirmaydi -- eng konservativ talqin: u "
    "yerda savdo yo'q, ya'ni kuniga maksimum 3 ta.",
    "TP1 da yopiladigan ulush ko'rsatilmagan. Neytral 50/50 olindi: TP1 da "
    "50%, qolgan 50% TP2 yoki trailing bilan chiqadi.",
    "ATR(14) qaysi timeframe'da ekani ko'rsatilmagan. Setup timeframe'i M5 "
    "olindi. Eslatma: BTC da M5 ATR14 x1.5 ko'pincha 0.30% dan kichik, "
    "shuning uchun MIN clamp tez-tez bog'laydi -- hisobotda ko'rinadi.",
    "'Dual-path SL verification': ATR yo'li va foiz chegaralari alohida "
    "hisoblanadi, natija ikkisining kesishmasi; qaysi yo'l bog'laganini "
    "stop_source qayd qiladi (ATR / MIN / MAX).",
    "Cooldown uzunligi ko'rsatilmagan; 2 ketma-ket SL uchun 120 daqiqa olindi.",
    "Entry mavjud koddagi valid execution usulida: signal bar yopilgach, "
    "keyingi bar ochilishida to'ldiriladi.",
    "Qoidalarda chiqish turlari TP1 / TP2 / trailing / SL deb sanalgan. "
    "Mavjud botdagi 'structure invalidation' erta chiqishi ro'yxatda yo'q, "
    "shuning uchun o'chirildi.",
    "Kuniga maksimum 3 ta savdo (3 ta sessiya x 1). Qoidalarda 'kuniga 3-4' "
    "deyilgan; 4-chi faqat 4-sessiya blogi bo'lganda mumkin edi, u yo'q.",
]


# ------------------------------------------------------------------ helpers
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


def streaks(trades: Sequence[Trade]) -> tuple:
    best = worst = w = l = 0
    for t in trades:
        if t.pnl > 0:
            w, l = w + 1, 0
        elif t.pnl < 0:
            l, w = l + 1, 0
        best, worst = max(best, w), max(worst, l)
    return best, worst


def max_dd(curve: Sequence[tuple]) -> float:
    peak, worst = -float("inf"), 0.0
    for _, eq in curve:
        peak = max(peak, eq)
        if peak > 0:
            worst = max(worst, (peak - eq) / peak)
    return worst


def report(name: str, trades: List[Trade], curve: List[tuple],
           days: float, equity0: float) -> dict:
    n = len(trades)
    if n == 0:
        print(f"\n### {name}: savdo yo'q ({days:.0f} kun)")
        return {"n": 0}
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl < 0]
    nets = [net_r(t) for t in trades]
    gw = sum(t.pnl for t in wins)
    gl = -sum(t.pnl for t in losses)
    best, worst = streaks(trades)
    exits: Dict[str, int] = defaultdict(int)
    for t in trades:
        exits[t.exit_reason or "?"] += 1
    hold = [t.holding_minutes for t in trades]
    per_session: Dict[str, List[Trade]] = defaultdict(list)
    for t in trades:
        per_session[t.session or "?"].append(t)
    per_day: Dict[int, int] = defaultdict(int)
    for t in trades:
        per_day[t.entry_time // 86_400_000] += 1

    def wr(ts: List[Trade]) -> str:
        return (f"{100 * sum(1 for x in ts if x.pnl > 0) / len(ts):.1f}% "
                f"({len(ts)})") if ts else "-"

    rows = [
        ("1  Total trades", f"{n}"),
        ("2  Trades/day", f"{n / days:.2f}"),
        ("3  Trades/session", ", ".join(
            f"{k}:{len(v)}" for k, v in sorted(per_session.items()))),
        ("4  Win rate", f"{100 * len(wins) / n:.1f}%"),
        ("5  Loss rate", f"{100 * len(losses) / n:.1f}%"),
        ("6  Average win", f"{mean([t.pnl for t in wins]):+.3f} USDT "
                           f"({mean([net_r(t) for t in wins]):+.3f} R)"),
        ("7  Average loss", f"{mean([t.pnl for t in losses]):+.3f} USDT "
                            f"({mean([net_r(t) for t in losses]):+.3f} R)"),
        ("8  Expectancy", f"{mean([t.pnl for t in trades]):+.3f} USDT "
                          f"({mean(nets):+.3f} R)"),
        ("9  Profit factor", f"{gw / gl:.3f}" if gl > 0 else "inf"),
        ("10 Gross PnL", f"{sum(gross_pnl(t) for t in trades):+.2f} USDT"),
        ("11 Commission", f"{sum(t.fees for t in trades):.2f} USDT"),
        ("12 Funding", f"{sum(t.funding for t in trades):+.2f} USDT"),
        ("13 Net PnL", f"{sum(t.pnl for t in trades):+.2f} USDT"),
        ("14 Fee/R", f"{mean([fee_r(t) for t in trades]):.3f}"),
        ("15 Max drawdown", f"{100 * max_dd(curve):.2f}%"),
        ("16 Max win streak", f"{best}"),
        ("17 Max loss streak", f"{worst}"),
        ("18 TP1 hit %", f"{100 * sum(1 for t in trades if 0 in t.tp_hits) / n:.1f}%"),
        ("19 TP2 hit %", f"{100 * sum(1 for t in trades if 1 in t.tp_hits) / n:.1f}%"),
        ("20 Trailing exit %", f"{100 * exits['TRAILING'] / n:.1f}%"),
        ("21 SL exit %", f"{100 * exits['STOP'] / n:.1f}%"),
        ("22 Avg holding", f"{mean(hold):.1f} min"),
        ("23 Median holding", f"{statistics.median(hold):.1f} min"),
        ("24 BUY win rate", wr([t for t in trades if t.side is Side.BUY])),
        ("25 SELL win rate", wr([t for t in trades if t.side is Side.SELL])),
        ("26 ASIA win rate", wr(per_session.get("ASIA", []))),
        ("27 LONDON win rate", wr(per_session.get("LONDON", []))),
        ("28 NEW YORK win rate", wr(per_session.get("NEW_YORK", []))),
    ]
    print(f"\n{'=' * 74}\n### {name}  ({days:.0f} kun)\n{'=' * 74}")
    for k, v in rows:
        print(f"{k:24s} {v}")
    print(f"{'   max trades in a day':24s} {max(per_day.values())}")
    print(f"{'   sessions with a trade':24s} "
          f"{100 * n / max(days * 3, 1):.1f}% of available session slots")
    print(f"{'   ending balance':24s} "
          f"{equity0 + sum(t.pnl for t in trades):.2f} USDT")
    print(f"{'   exit mix':24s} {dict(exits)}")
    stop_src: Dict[str, int] = defaultdict(int)
    for t in trades:
        stop_src[t.meta.get("stop_source", "?")] += 1
    print(f"{'   SL path (dual-path)':24s} {dict(stop_src)}")
    return {
        "n": n, "win": len(wins) / n, "exp_r": mean(nets),
        "pf": (gw / gl) if gl > 0 else float("inf"),
        "net": sum(t.pnl for t in trades), "fees": sum(t.fees for t in trades),
        "fee_r": mean([fee_r(t) for t in trades]), "dd": max_dd(curve),
        "per_day": n / days,
    }


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
    print(f"[data] {n} candles ({n / 1440:.0f} kun)")

    print("\n" + "=" * 74)
    print("INTERPRETATIONS (qoidalarda ochiq qolgan joylar)")
    print("=" * 74)
    for i, note in enumerate(INTERPRETATIONS, 1):
        print(f"{i}. {note}")

    cfg = Config().with_overrides(**JARVIS)
    a, b = int(n * 0.6), int(n * 0.8)
    parts = [("TRAIN", candles[:a]), ("VALIDATION", candles[a:b]),
             ("OOS TEST", candles[b:])]

    summaries = {}
    for label, chunk in parts:
        t0 = time.time()
        res = Backtester(cfg).run(chunk)
        summaries[label] = report(f"JARVIS SCALP -- {label}", res.trades,
                                  res.equity_curve, len(chunk) / 1440,
                                  cfg.initial_equity)
        print(f"   [{time.time() - t0:.0f}s]")
        sys.stdout.flush()

    print("\n" + "=" * 74)
    print("BASELINE vs JARVIS SCALP (bir xil OOS davrda)")
    print("=" * 74)
    oos = parts[-1][1]
    base_res = Backtester(Config()).run(oos)
    base = report("BASELINE (hozirgi bot) -- OOS TEST", base_res.trades,
                  base_res.equity_curve, len(oos) / 1440,
                  Config().initial_equity)

    head = (f"{'metrika':22s} {'BASELINE':>14s} {'JARVIS SCALP':>14s}")
    print(f"\n{head}\n{'-' * len(head)}")
    js = summaries["OOS TEST"]
    for key, label, fmt in (
            ("n", "Trades", "{:d}"), ("per_day", "Trades/day", "{:.2f}"),
            ("win", "Win rate", "{:.1%}"), ("exp_r", "Expectancy R", "{:+.3f}"),
            ("pf", "Profit factor", "{:.3f}"), ("net", "Net PnL", "{:+.2f}"),
            ("fees", "Commission", "{:.2f}"), ("fee_r", "Fee/R", "{:.3f}"),
            ("dd", "Max DD", "{:.1%}")):
        bv = base.get(key)
        jv = js.get(key)
        bs = fmt.format(bv) if bv is not None else "-"
        jsv = fmt.format(jv) if jv is not None else "-"
        print(f"{label:22s} {bs:>14s} {jsv:>14s}")

    print("\n" + "=" * 74)
    print("MAQSAD: WIN RATE >= 70% (OOS)")
    print("=" * 74)
    if js.get("n"):
        print(f"OOS win rate  : {js['win']:.1%}   "
              f"({'YETDI' if js['win'] >= 0.70 else 'YETMADI'})")
        print(f"OOS expectancy: {js['exp_r']:+.3f} R  "
              f"({'musbat' if js['exp_r'] > 0 else 'manfiy'})")
        print(f"OOS PF        : {js['pf']:.3f}")
        print("\nHech bir parametr natijani chiroyliroq ko'rsatish uchun")
        print("o'zgartirilmadi. Yuqoridagi konfiguratsiya qoidalarning")
        print("to'g'ridan-to'g'ri ko'chirmasi.")
    else:
        print("OOS davrda savdo bo'lmadi.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
