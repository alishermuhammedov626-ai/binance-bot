"""Judge every filter on evidence: KEEP, REMOVE, or SOFT.

    python scripts/filter_ablation.py --csv data/BTCUSDT_1m.csv

Section 33 forbids both over- and under-filtering and asks that each filter be
tested rather than assumed. So each one is switched off in turn and the result
compared against the baseline that has them all on:

    KEEP   -- switching it off makes expectancy meaningfully worse.
              The filter is doing real work.
    REMOVE -- switching it off makes expectancy meaningfully better.
              The filter is discarding setups that were fine.
    SOFT   -- no meaningful difference. It is not earning its veto; the
              information belongs in the ML score, where a weak signal can be
              outvoted instead of ending the setup.

"Meaningful" is judged on out-of-sample expectancy with a minimum trade count,
not on win rate: a filter that raises win rate while cutting expectancy is
making the strategy worse in the only way that pays.

Each filter is removed on its own. Interactions are not explored -- removing
two at once can differ from removing each -- and that limit is real.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smcbot.backtest.engine import Backtester
from smcbot.config import Config
from smcbot.core.types import Trade
from smcbot.data import loader, synthetic

MIN_TRADES = 30
MATERIAL = 0.05          # R per trade that counts as a real difference

# label -> overrides that switch the filter OFF
FILTERS: List[Tuple[str, dict]] = [
    ("entry chase protection", {"filters.max_chase_atr": 99.0}),
    ("counter-trend minor liquidity", {"filters.allow_counter_trend_minor": True}),
    ("M1 confirmation (hard)", {"filters.require_m1_confirmation": False}),
    ("SMC score >= 70", {"score.valid": 0.0}),
    ("minimum RR", {"risk.min_rr": 0.1, "risk.min_first_target_rr": 0.1}),
    ("range midpoint block", {"structure.range_midpoint_block": 0.0}),
    ("volatility band", {"filters.min_atr_pct": 0.0, "filters.max_atr_pct": 1.0}),
    ("spread limit", {"filters.max_spread_bps": 9999.0}),
    ("duplicate-signal dedupe", {"setup.dedupe_atr": 0.001}),
    ("setup expiry (TTL)", {"setup.m15_ttl_minutes": 600}),
    ("stop-too-wide guard", {"risk.max_sl_atr": 99.0}),
    ("strategy exit on invalidation", {"risk.early_exit_on_invalidation": False}),
]


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def risk_usdt(t: Trade) -> float:
    return abs(t.entry_price - t.stop) * t.qty


def net_r(t: Trade) -> float:
    r = risk_usdt(t)
    return t.pnl / r if r > 0 else 0.0


def summarise(trades: List[Trade]) -> dict:
    if not trades:
        return {"n": 0}
    nets = [net_r(t) for t in trades]
    wins = [t for t in trades if t.pnl > 0]
    gw = sum(t.pnl for t in wins)
    gl = -sum(t.pnl for t in trades if t.pnl < 0)
    return {"n": len(trades), "win": len(wins) / len(trades),
            "exp": mean(nets), "med": statistics.median(nets),
            "pf": (gw / gl) if gl > 0 else float("inf"),
            "net": sum(t.pnl for t in trades)}


HEAD = (f"{'filtr o^chirilganda':34s} {'n':>5s} {'d_n':>6s} {'win%':>6s} "
        f"{'d_win':>7s} {'expR':>7s} {'d_exp':>7s} {'PF':>6s} {'hukm':>8s}")


def verdict(base: dict, alt: dict) -> str:
    if alt.get("n", 0) < MIN_TRADES:
        return "n kichik"
    d = alt["exp"] - base["exp"]
    if d <= -MATERIAL:
        return "KEEP"
    if d >= MATERIAL:
        return "REMOVE"
    return "SOFT"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--split", type=float, default=0.8,
                    help="tail share used as the out-of-sample block")
    ap.add_argument("--config", help="JSON overrides applied to every run")
    args = ap.parse_args()

    if args.csv:
        candles = loader.load_csv(args.csv)
    else:
        candles = synthetic.generate(args.days * 1440, seed=42)

    base_over: dict = {}
    if args.config and os.path.exists(args.config):
        import json
        with open(args.config, encoding="utf-8") as fh:
            base_over = json.load(fh).get("overrides", {})
        print(f"[cfg] asos: {base_over}")

    cut = int(len(candles) * args.split)
    blocks = [("SELECTION", candles[:cut]), ("OUT-OF-SAMPLE", candles[cut:])]

    for label, chunk in blocks:
        days = len(chunk) / 1440
        print(f"\n{'=' * 104}\n{label}  ({days:.0f} kun)\n{'=' * 104}")
        base_cfg = Config().with_overrides(**base_over) if base_over else Config()
        t0 = time.time()
        base = summarise(Backtester(base_cfg).run(chunk).trades)
        if not base.get("n"):
            print("baseline savdo bermadi")
            continue
        print(HEAD)
        print("-" * 104)
        pf = " inf" if base["pf"] == float("inf") else f"{base['pf']:6.3f}"
        print(f"{'BASELINE (hammasi yoqilgan)':34s} {base['n']:5d} {'':>6s} "
              f"{base['win'] * 100:5.1f}% {'':>7s} {base['exp']:+7.3f} {'':>7s} {pf}")
        print("-" * 104)

        for name, off in FILTERS:
            cfg = Config().with_overrides(**{**base_over, **off})
            alt = summarise(Backtester(cfg).run(chunk).trades)
            if not alt.get("n"):
                print(f"{name:34s} {'savdo yo^q':>5s}")
                continue
            pf = " inf" if alt["pf"] == float("inf") else f"{alt['pf']:6.3f}"
            print(f"{name:34s} {alt['n']:5d} {alt['n'] - base['n']:+6d} "
                  f"{alt['win'] * 100:5.1f}% {(alt['win'] - base['win']) * 100:+6.1f}% "
                  f"{alt['exp']:+7.3f} {alt['exp'] - base['exp']:+7.3f} {pf} "
                  f"{verdict(base, alt):>8s}")
            sys.stdout.flush()
        print(f"\n[{time.time() - t0:.0f}s]")

    print(f"\n{'=' * 104}")
    print("O'QISH")
    print("=" * 104)
    print("KEEP   -- filtrni o'chirish natijani yomonlashtirdi; u ish qilyapti.")
    print("REMOVE -- o'chirish yaxshiladi; filtr yaxshi setuplarni tashlayapti.")
    print("SOFT   -- farq sezilmadi; veto huquqini oqlamayapti, uni ML")
    print("          xususiyatiga aylantirish mumkin.")
    print(f"\nChegara: |d_exp| >= {MATERIAL} R va n >= {MIN_TRADES}.")
    print("Hukm SELECTION emas, OUT-OF-SAMPLE ustunidan olinadi.")
    print("Har filtr alohida o'chirildi -- ikkitasini birga o'chirish")
    print("boshqacha natija berishi mumkin, bu tekshirilmadi.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
