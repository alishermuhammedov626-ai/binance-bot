"""Does the SMC entry beat a coin flip with the same geometry?

    python scripts/random_control.py --csv data/BTCUSDT_1m.csv --sl 0.8 --tp 1.0

Every result so far is consistent with one uncomfortable possibility: that the
entry adds nothing, and the win rate is set entirely by the TP/SL ratio. This
settles it. Random entries are taken on random bars -- same target, same stop,
same slippage and commission, same three-a-day cap -- and repeated over many
seeds to build a distribution. The SMC engine's result is then placed in that
distribution.

If the SMC number sits inside the random cloud, the setup logic is not adding
measurable value at this geometry. That is a finding about the entry, not about
the exits: costs and stop placement were already fixed and are held identical
on both sides here.
"""
from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smcbot.config import Config
from smcbot.core.types import Candle
from smcbot.data import loader, synthetic

DAY_MS = 86_400_000


def simulate(candles: Sequence[Candle], entries: Sequence[Tuple[int, int]],
             tp_pct: float, sl_pct: float, taker: float, slip_bps: float,
             max_hold: int) -> List[float]:
    """Walk each entry forward. Returns net R per trade.

    Same pessimism as the backtester: a bar spanning both levels is a stop,
    because the intrabar order is unknowable.
    """
    slip = slip_bps / 10_000.0
    out: List[float] = []
    for idx, direction in entries:
        if idx + 1 >= len(candles):
            continue
        fill = candles[idx + 1].open * (1 + direction * slip)
        tp = fill * (1 + direction * tp_pct / 100.0)
        sl = fill * (1 - direction * sl_pct / 100.0)
        risk = abs(fill - sl)
        if risk <= 0:
            continue
        # Commission in R: entry plus exit, both taker, on notional = risk/sl_pct.
        fee_r = 2 * taker / (sl_pct / 100.0)
        result = None
        for j in range(idx + 1, min(idx + 1 + max_hold, len(candles))):
            c = candles[j]
            hit_sl = (c.low <= sl) if direction > 0 else (c.high >= sl)
            hit_tp = (c.high >= tp) if direction > 0 else (c.low <= tp)
            if hit_sl:
                result = -1.0
                break
            if hit_tp:
                result = (tp - fill) * direction / risk
                break
        if result is None:
            last = candles[min(idx + max_hold, len(candles) - 1)]
            result = (last.close - fill) * direction / risk
        out.append(result - fee_r)
    return out


def random_entries(candles: Sequence[Candle], per_day: int, rng: random.Random,
                   warmup: int = 300) -> List[Tuple[int, int]]:
    """Up to ``per_day`` entries a day, at random bars, random direction."""
    by_day: Dict[int, List[int]] = {}
    for i, c in enumerate(candles):
        if i < warmup:
            continue
        by_day.setdefault(c.open_time // DAY_MS, []).append(i)
    entries: List[Tuple[int, int]] = []
    for day, idxs in by_day.items():
        if len(idxs) < per_day:
            continue
        for i in rng.sample(idxs, per_day):
            entries.append((i, rng.choice((1, -1))))
    entries.sort()
    return entries


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--tp", type=float, default=1.0)
    ap.add_argument("--sl", type=float, default=0.8)
    ap.add_argument("--per-day", type=int, default=3)
    ap.add_argument("--runs", type=int, default=400)
    ap.add_argument("--split", type=float, default=0.8,
                    help="use the tail as the comparison block")
    ap.add_argument("--smc-win", type=float,
                    help="measured SMC win rate in %% on the same block")
    ap.add_argument("--smc-exp", type=float,
                    help="measured SMC net expectancy in R on the same block")
    args = ap.parse_args()

    if args.csv:
        candles = loader.load_csv(args.csv)
    else:
        candles = synthetic.generate(args.days * 1440, seed=42)
    block = candles[int(len(candles) * args.split):]
    days = len(block) / 1440
    print(f"[data] {len(block)} candles ({days:.0f} kun) "
          f"-- TP {args.tp}%, SL {args.sl}%, kuniga {args.per_day} ta")

    cfg = Config()
    taker = cfg.execution.taker_fee
    slip = cfg.execution.slippage_bps
    fee_r = 2 * taker / (args.sl / 100.0)
    rr = args.tp / args.sl
    print(f"[cost] fee/R = {fee_r:.3f},  RR = {rr:.2f},  "
          f"nolga chiqish uchun kerak = {(1 + fee_r) / (1 + rr) * 100:.1f}%")

    wins: List[float] = []
    exps: List[float] = []
    for seed in range(args.runs):
        rng = random.Random(seed)
        rs = simulate(block, random_entries(block, args.per_day, rng),
                      args.tp, args.sl, taker, slip, max_hold=60 * 12)
        if len(rs) < 20:
            continue
        wins.append(100 * sum(1 for r in rs if r > 0) / len(rs))
        exps.append(sum(rs) / len(rs))

    if not wins:
        print("yetarli tasodifiy savdo chiqmadi")
        return 1

    def pct(xs: List[float], q: float) -> float:
        s = sorted(xs)
        return s[min(int(len(s) * q), len(s) - 1)]

    print(f"\nTASODIFIY KIRISH ({len(wins)} ta takrorlash, har birida "
          f"~{args.per_day * days:.0f} savdo)")
    print(f"  win rate   o'rtacha {statistics.mean(wins):5.1f}%   "
          f"p05 {pct(wins, .05):5.1f}%   p95 {pct(wins, .95):5.1f}%")
    print(f"  expectancy o'rtacha {statistics.mean(exps):+.3f}R   "
          f"p05 {pct(exps, .05):+.3f}R  p95 {pct(exps, .95):+.3f}R")

    if args.smc_win is not None:
        above = sum(1 for w in wins if w >= args.smc_win)
        print(f"\nSMC win rate {args.smc_win:.1f}% -- tasodifiy natijalarning "
              f"{100 * above / len(wins):.1f}% i shundan yuqori")
        print(f"  => p = {(above + 1) / (len(wins) + 1):.4f}")
    if args.smc_exp is not None:
        above = sum(1 for e in exps if e >= args.smc_exp)
        print(f"SMC expectancy {args.smc_exp:+.3f}R -- tasodifiy natijalarning "
              f"{100 * above / len(exps):.1f}% i shundan yuqori")
        print(f"  => p = {(above + 1) / (len(exps) + 1):.4f}")
        print("\nO'qish: p kichik (masalan <0.05) bo'lsa SMC kirish tasodifdan")
        print("ustun. p katta bo'lsa -- shu geometriyada kirish mantig'i")
        print("o'lchanadigan qiymat qo'shmayapti.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
