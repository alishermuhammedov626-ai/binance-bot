"""Test named hypotheses about why a backtest lost money.

This is not a parameter search.  Each variant below is a *structural* claim
about the strategy that can be right or wrong, and every result is printed --
including the ones that make things worse.  Picking the best row out of a wide
sweep and calling it the strategy is exactly the overfitting section 60 of the
specification forbids.

    python scripts/diagnose.py --csv data/BTCUSDT_1m.csv

The cost columns are the point.  ``fee_R`` is the round-trip commission
expressed in units of the trade's own risk: if it approaches 0.5, costs alone
demand roughly a 15-20 point higher win rate to break even, and no amount of
signal quality will save the strategy.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smcbot.backtest.engine import Backtester
from smcbot.config import Config
from smcbot.data import loader, synthetic

# Each entry: label -> (hypothesis, config overrides)
VARIANTS: List[tuple] = [
    ("baseline", "the shipped defaults", {}),
    ("wider stop (M5)", "tight M1 stops inflate notional, so fees eat the edge",
     {"risk.min_sl_atr": 1.5}),
    ("wider stop + no BE", "and the breakeven move truncates the winners",
     {"risk.min_sl_atr": 1.5, "risk.move_to_be_after_tp1": False}),
    ("single TP", "partial exits cap wins while losses stay full size",
     {"risk.partial_tp": {"tp1": 1.0, "tp2": 0.0, "tp3": 0.0}}),
    ("wider stop + single TP", "both of the above together",
     {"risk.min_sl_atr": 1.5,
      "risk.partial_tp": {"tp1": 1.0, "tp2": 0.0, "tp3": 0.0},
      "risk.move_to_be_after_tp1": False}),
    ("A+ setups only", "the score threshold is too permissive",
     {"score.valid": 85.0}),
    ("zero costs", "control: how much of the loss is purely commission?",
     {"execution.taker_fee": 0.0, "execution.maker_fee": 0.0,
      "execution.slippage_bps": 0.0}),
]


def fee_in_r(result) -> float:
    """Round-trip commission per trade, in units of that trade's risk."""
    rs = []
    for t in result.trades:
        risk = abs(t.entry_price - t.stop) * t.qty
        if risk > 0:
            rs.append(t.fees / risk)
    return sum(rs) / len(rs) if rs else 0.0


def run(cfg: Config, candles) -> dict:
    result = Backtester(cfg).run(candles)
    m = result.metrics
    if not m.get("trades"):
        return {"trades": 0}
    return {
        "trades": m["trades"],
        "win_rate": m["win_rate"],
        "profit_factor": m["profit_factor"],
        "expectancy_r": m["expectancy_r"],
        "net_pnl": m["net_pnl"],
        "max_dd": m["max_drawdown"],
        "fees": m["fees"],
        "fee_R": fee_in_r(result),
        "avg_mfe": m["avg_mfe_r"],
        "avg_mae": m["avg_mae_r"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.csv:
        print(f"[data] {args.csv}")
        candles = loader.load_csv(args.csv)
    else:
        candles = synthetic.generate(args.days * 1440, seed=args.seed)
    print(f"[data] {len(candles)} candles\n")

    header = (f"{'variant':24s} {'trades':>7s} {'win%':>6s} {'PF':>6s} "
              f"{'expR':>7s} {'net':>9s} {'maxDD':>7s} {'fee/R':>6s} "
              f"{'MFE':>6s} {'MAE':>6s}")
    print(header)
    print("-" * len(header))

    for label, hypothesis, overrides in VARIANTS:
        cfg = Config().with_overrides(**overrides) if overrides else Config()
        t0 = time.time()
        r = run(cfg, candles)
        if not r["trades"]:
            print(f"{label:24s} {'no trades':>7s}")
            continue
        print(f"{label:24s} {r['trades']:7d} {r['win_rate']*100:5.1f}% "
              f"{r['profit_factor']:6.2f} {r['expectancy_r']:7.3f} "
              f"{r['net_pnl']:9.2f} {r['max_dd']*100:6.1f}% {r['fee_R']:6.3f} "
              f"{r['avg_mfe']:6.2f} {r['avg_mae']:6.2f}"
              f"   [{time.time() - t0:.0f}s]")
        sys.stdout.flush()

    print("\nHypotheses:")
    for label, hypothesis, _ in VARIANTS:
        print(f"  {label:24s} {hypothesis}")
    print("\nRead 'zero costs' first: if it is profitable and the others are")
    print("not, the signal has some merit and the cost structure is the")
    print("problem. If it loses too, the entries themselves are wrong and no")
    print("fee tuning will rescue them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
