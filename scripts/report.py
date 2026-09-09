"""Full forensic report on a backtest run. Changes nothing, explains everything.

    python scripts/report.py --csv data/BTCUSDT_1m.csv --db diag.db

Runs the backtest once with the ML feature vector attached to every trade (so
entry-quality slices are available), writes the SQLite journal for your own
queries, and prints:

  1. exit reasons          count, share, PnL, average and median R
  2. TP1 lifecycle         reached / then breakeven / then loss / ran to target
  3. cost structure        gross vs net, commission against loss and volume
  4. MFE-MAE              how far winners and losers travelled before resolving
  5. entry quality         score, side, setup type, session, confirmations
  6. verdict               signal vs exit mechanism vs cost, decided by the numbers

Nothing here tunes a parameter. It only measures the run you already have.
"""
from __future__ import annotations

import argparse
import json
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
from smcbot.journal import Journal
from smcbot.ml.features import build_features


# ----------------------------------------------------------------- helpers
def pct(a: float, b: float) -> str:
    return f"{100.0 * a / b:5.1f}%" if b else "    -"


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def median(xs: Sequence[float]) -> float:
    return statistics.median(xs) if xs else 0.0


def head(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def table(rows: List[tuple], headers: Sequence[str]) -> None:
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows
              else len(str(h)) for i, h in enumerate(headers)]
    print("  ".join(str(h).ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))


def gross_pnl(t: Trade) -> float:
    """PnL before commission and funding."""
    return t.pnl + t.fees + t.funding


def volume_of(t: Trade) -> float:
    """Total notional traded by this position, entry plus every exit."""
    if t.fills:
        return sum(abs(f.qty * f.price) for f in t.fills)
    return abs(t.qty * t.entry_price) * 2.0


def risk_usdt(t: Trade) -> float:
    return abs(t.entry_price - t.stop) * t.qty


# ----------------------------------------------------------------- sections
def section_exits(trades: List[Trade]) -> Dict[str, list]:
    head("1. EXIT REASON")
    groups: Dict[str, List[Trade]] = defaultdict(list)
    for t in trades:
        groups[t.exit_reason or "UNKNOWN"].append(t)

    rows = []
    for reason, ts in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        rs = [t.r_multiple for t in ts]
        rows.append((
            reason, len(ts), pct(len(ts), len(trades)),
            f"{sum(t.pnl for t in ts):9.2f}",
            f"{mean([t.pnl for t in ts]):8.3f}",
            f"{mean(rs):7.3f}", f"{median(rs):7.3f}",
        ))
    table(rows, ["reason", "count", "share", "total_pnl", "avg_pnl",
                 "avg_R", "med_R"])
    return groups


def section_tp1(trades: List[Trade]) -> dict:
    head("2. TP1 LIFECYCLE")
    reached = [t for t in trades if 0 in t.tp_hits]
    missed = [t for t in trades if 0 not in t.tp_hits]

    after_be = [t for t in reached if t.exit_reason == "BREAKEVEN"]
    after_stop = [t for t in reached if t.exit_reason == "STOP"]
    to_final = [t for t in reached if t.exit_reason == "TP_FINAL"]
    other = [t for t in reached
             if t.exit_reason not in ("BREAKEVEN", "STOP", "TP_FINAL")]
    reached_loss = [t for t in reached if t.pnl < 0]

    print(f"TP1 ga yetdi          : {len(reached):4d}  ({pct(len(reached), len(trades))})")
    print(f"  -> BREAKEVEN da     : {len(after_be):4d}  "
          f"avg {mean([t.r_multiple for t in after_be]):+.3f}R")
    print(f"  -> STOP da          : {len(after_stop):4d}  "
          f"avg {mean([t.r_multiple for t in after_stop]):+.3f}R")
    print(f"  -> TP_FINAL da      : {len(to_final):4d}  "
          f"avg {mean([t.r_multiple for t in to_final]):+.3f}R")
    print(f"  -> boshqa           : {len(other):4d}")
    print()
    print(f"TP1 ga yetib LOSS     : {len(reached_loss):4d}  "
          f"({pct(len(reached_loss), max(len(reached), 1))} TP1 ga yetganlardan)")
    print(f"  ulardan avg PnL     : {mean([t.pnl for t in reached_loss]):+.3f} USDT")
    print(f"  ulardan avg fee     : {mean([t.fees for t in reached_loss]):.3f} USDT")
    print(f"  ulardan avg gross   : {mean([gross_pnl(t) for t in reached_loss]):+.3f} USDT")
    print()
    print(f"TP1 ga yetmadi        : {len(missed):4d}  ({pct(len(missed), len(trades))})")
    print(f"  avg R               : {mean([t.r_multiple for t in missed]):+.3f}")

    depth = defaultdict(int)
    for t in trades:
        depth[len(t.tp_hits)] += 1
    print("\nnechta TP darajasiga yetgan:")
    for k in sorted(depth):
        print(f"  {k} ta TP: {depth[k]:4d}  ({pct(depth[k], len(trades))})")
    return {"reached": reached, "reached_loss": reached_loss, "missed": missed,
            "after_be": after_be, "to_final": to_final}


def section_costs(trades: List[Trade], equity0: float) -> dict:
    head("3. COST STRUCTURE")
    gross = sum(gross_pnl(t) for t in trades)
    fees = sum(t.fees for t in trades)
    funding = sum(t.funding for t in trades)
    net = sum(t.pnl for t in trades)
    volume = sum(volume_of(t) for t in trades)
    gross_losers = [gross_pnl(t) for t in trades if gross_pnl(t) < 0]
    gross_loss = -sum(gross_losers)
    gross_win = sum(g for g in (gross_pnl(t) for t in trades) if g > 0)
    fee_r = [t.fees / risk_usdt(t) for t in trades if risk_usdt(t) > 0]
    stop_pct = [abs(t.entry_price - t.stop) / t.entry_price for t in trades]

    print(f"gross PnL (fee/funding'siz) : {gross:10.2f} USDT")
    print(f"  gross yutuq               : {gross_win:10.2f}")
    print(f"  gross yutqazish           : {-gross_loss:10.2f}")
    print(f"  gross profit factor       : "
          f"{gross_win / gross_loss:10.3f}" if gross_loss else "n/a")
    print(f"total commission            : {fees:10.2f} USDT")
    print(f"total funding               : {funding:10.2f} USDT")
    print(f"net PnL                     : {net:10.2f} USDT  "
          f"({100 * net / equity0:+.1f}% of start)")
    print()
    print(f"commission / gross loss     : {pct(fees, gross_loss)}")
    print(f"commission / |net loss|     : {pct(fees, abs(net)) if net else '-'}")
    print(f"total traded volume         : {volume:12.0f} USDT")
    print(f"commission / volume         : {10000 * fees / volume:6.2f} bps"
          if volume else "")
    print(f"volume / equity (turnover)  : {volume / equity0:10.1f}x")
    print()
    print(f"fee per trade, in R         : avg {mean(fee_r):.3f}   "
          f"median {median(fee_r):.3f}   max {max(fee_r):.3f}")
    print(f"stop distance, % of price   : avg {100 * mean(stop_pct):.4f}%   "
          f"median {100 * median(stop_pct):.4f}%")
    print(f"  -> 0.001 / stop% predicts fee/R = "
          f"{0.001 / mean(stop_pct):.3f} (round trip, taker both sides)")
    return {"gross": gross, "fees": fees, "net": net, "fee_r": mean(fee_r),
            "gross_win": gross_win, "gross_loss": gross_loss,
            "stop_pct": mean(stop_pct)}


def section_excursion(trades: List[Trade]) -> dict:
    head("4. MFE / MAE")
    winners = [t for t in trades if t.pnl > 0]
    losers = [t for t in trades if t.pnl < 0]

    print(f"winners ({len(winners)}): avg MFE {mean([t.mfe for t in winners]):+.3f}R   "
          f"avg MAE {mean([t.mae for t in winners]):+.3f}R")
    print(f"losers  ({len(losers)}): avg MFE {mean([t.mfe for t in losers]):+.3f}R   "
          f"avg MAE {mean([t.mae for t in losers]):+.3f}R")
    print()
    print("loserlar qancha foydaga chiqib keyin yo'qotgan:")
    for level in (0.5, 1.0, 1.5, 2.0):
        n = sum(1 for t in losers if t.mfe >= level)
        print(f"  +{level:.1f}R ga yetgan : {n:4d}  ({pct(n, max(len(losers), 1))} loserlardan)")
    print()
    print("winnerlar qancha zararga tushib keyin qaytgan:")
    for level in (0.5, 1.0):
        n = sum(1 for t in winners if t.mae <= -level)
        print(f"  -{level:.1f}R ko'rgan  : {n:4d}  ({pct(n, max(len(winners), 1))} winnerlardan)")

    gross_w = [t for t in trades if gross_pnl(t) > 0]
    print(f"\nkomissiyasiz yutuq bo'lardi : {len(gross_w)} "
          f"({pct(len(gross_w), len(trades))})  "
          f"-- haqiqiy yutuq {len(winners)} ({pct(len(winners), len(trades))})")
    print(f"komissiya tufayli yutuqdan yutqazishga o'tgan: "
          f"{len(gross_w) - len(winners)} ta")
    return {"winners": winners, "losers": losers, "gross_winners": gross_w}


def slice_by(trades: List[Trade], key, label: str, limit: int = 20) -> None:
    groups: Dict[str, List[Trade]] = defaultdict(list)
    for t in trades:
        groups[str(key(t))].append(t)
    rows = []
    for k, ts in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:limit]:
        wins = [t for t in ts if t.pnl > 0]
        rows.append((k, len(ts), pct(len(wins), len(ts)),
                     f"{mean([t.r_multiple for t in ts]):+7.3f}",
                     f"{sum(t.pnl for t in ts):9.2f}",
                     f"{mean([gross_pnl(t) for t in ts]):+8.3f}"))
    print(f"\n{label}")
    table(rows, ["value", "count", "win%", "avg_R", "net_pnl", "avg_gross"])


def section_entry_quality(trades: List[Trade]) -> None:
    head("5. ENTRY QUALITY")

    def bucket(t: Trade) -> str:
        s = t.smc_score
        for lo in (90, 85, 80, 75, 70):
            if s >= lo:
                return f"{lo}-{lo + 5}"
        return "<70"

    slice_by(trades, bucket, "SMC score:")

    with_ml = [t for t in trades if t.ml_probability is not None]
    if with_ml:
        slice_by(with_ml, lambda t: f"{t.ml_probability:.1f}", "ML probability:")
    else:
        print("\nML probability: bu runda model ishlatilmagan (SMC-only). "
              "ML taqsimoti uchun `pipeline` kerak.")

    slice_by(trades, lambda t: t.side.value, "LONG vs SHORT:")
    slice_by(trades, lambda t: t.meta.get("kind", "?"), "Setup type:")
    slice_by(trades, lambda t: t.market_state or "?", "Market state:")
    slice_by(trades, lambda t: t.session or "?", "Session:")
    slice_by(trades, lambda t: t.meta.get("zone_kind", "?"), "Zone kind:")
    slice_by(trades, lambda t: t.meta.get("sweep_tf", "?"), "Sweep timeframe:")
    slice_by(trades, lambda t: t.meta.get("stop_source", "?"), "Stop source:")
    slice_by(trades, lambda t: t.meta.get("classification", "?"), "Classification:")

    feat = [t for t in trades if t.features]
    if not feat:
        print("\nConfirmation kombinatsiyalari: feature vektor yo'q.")
        return

    # The raw *_choch / *_bos flags say "has one ever happened", which is almost
    # always true and collapses every trade into one bucket.  Freshness is the
    # informative part, so the combination is built from the age features using
    # the same windows the setup engine considers valid.
    fresh_within = {"m15": 20, "m5": 24, "m1": 12}

    def combo(t: Trade) -> str:
        f = t.features

        def mark(tf: str) -> str:
            limit = fresh_within[tf]
            c = "C" if f.get(f"{tf}_choch_age", 999) <= limit else ""
            b = "B" if f.get(f"{tf}_bos_age", 999) <= limit else ""
            return (c + b) or "-"

        return f"M15:{mark('m15'):2s} M5:{mark('m5'):2s} M1:{mark('m1'):2s}"

    slice_by(feat, combo,
             "M15/M5/M1 confirmation, yangi bo'lganlari "
             "(C=CHOCH, B=BOS, -=eskirgan):")
    slice_by(feat, lambda t: f"M1 CHOCH age <= {'12' if t.features.get('m1_choch_age', 999) <= 12 else '>12'}",
             "M1 CHOCH yangiligi:")


def section_verdict(trades: List[Trade], costs: dict, tp1: dict, exc: dict) -> None:
    head("6. VERDICT")
    n = len(trades)
    gross_r = mean([(gross_pnl(t) / risk_usdt(t)) for t in trades
                    if risk_usdt(t) > 0])
    net_r = mean([t.r_multiple for t in trades])
    gross_pf = costs["gross_win"] / costs["gross_loss"] if costs["gross_loss"] else 0

    print(f"A) SIGNAL  -- xarajatsiz natija")
    print(f"   gross expectancy : {gross_r:+.3f} R/trade")
    print(f"   gross PF         : {gross_pf:.3f}")
    verdict_a = ("edge YO'Q (gross PF <= 1)" if gross_pf <= 1.0
                 else "edge BOR (gross PF > 1)")
    print(f"   => {verdict_a}")

    print(f"\nB) EXIT MEXANIZMI")
    reached = len(tp1["reached"])
    lost_after_tp1 = len(tp1["reached_loss"])
    print(f"   TP1 ga yetgan    : {reached} ({pct(reached, n)})")
    print(f"   shundan minusda  : {lost_after_tp1} "
          f"({pct(lost_after_tp1, max(reached, 1))})")
    flipped = len(exc["gross_winners"]) - len(exc["winners"])
    print(f"   komissiya yutuqni yutqazishga aylantirgan: {flipped} ta "
          f"({pct(flipped, n)})")
    verdict_b = ("MUAMMO: TP1 ga yetganlarning yarmidan ko'pi minusda tugagan"
                 if reached and lost_after_tp1 / reached > 0.5
                 else "exit mexanizmi asosiy sabab emas")
    print(f"   => {verdict_b}")

    print(f"\nC) XARAJAT")
    print(f"   fee/R            : {costs['fee_r']:.3f}")
    print(f"   commission/gross loss : {pct(costs['fees'], costs['gross_loss'])}")
    print(f"   net - gross      : {net_r - gross_r:+.3f} R/trade xarajatga ketgan")
    verdict_c = ("KRITIK (fee/R > 0.5)" if costs["fee_r"] > 0.5
                 else "sezilarli" if costs["fee_r"] > 0.2 else "normal")
    print(f"   => {verdict_c}")

    print(f"\nUMUMIY: net {net_r:+.3f} R/trade = gross {gross_r:+.3f} "
          f"+ xarajat {net_r - gross_r:+.3f}")


# ----------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--db", default="diag.db")
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = Config.load(args.config) if args.config else Config()
    if args.csv:
        print(f"[data] {args.csv}")
        candles = loader.load_csv(args.csv)
    else:
        candles = synthetic.generate(args.days * 1440, seed=42)
    print(f"[data] {len(candles)} candles")

    t0 = time.time()
    # feature_builder (without a predictor) attaches the full feature vector to
    # every trade, which is what the confirmation-combination slice needs.
    bt = Backtester(cfg, feature_builder=build_features)
    result = bt.run(candles)
    print(f"[run ] {time.time() - t0:.0f}s, {len(result.trades)} trades\n")
    print(result.summary())

    if os.path.exists(args.db):
        os.remove(args.db)
    run_id = f"diag-{int(time.time())}"
    with Journal(args.db) as j:
        j.start_run(run_id, "diagnostic", cfg.symbol, cfg.to_dict())
        j.record_trades(result.trades, run_id)
        j.finish_run(run_id, result.metrics)
    print(f"\n[journal] {args.db} ({run_id})")

    trades = result.trades
    if not trades:
        print("savdo yo'q")
        return 1

    section_exits(trades)
    tp1 = section_tp1(trades)
    costs = section_costs(trades, cfg.initial_equity)
    exc = section_excursion(trades)
    section_entry_quality(trades)
    section_verdict(trades, costs, tp1, exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
