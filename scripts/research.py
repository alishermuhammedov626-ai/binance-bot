"""Staged parameter research with an untouched out-of-sample period.

    python scripts/research.py --csv data/BTCUSDT_1m.csv --stage all

WHAT THIS IS
------------
A sequential (greedy) search, not a grid.  A full grid over stops, targets,
partials, entries, filters and ML thresholds is ~1500 backtests; at ~140s each
that is 60 hours.  Stages run in the order the diagnosis implies -- cost first,
because at fee/R ~ 1.0 no signal choice can pay -- and each stage inherits the
winner of the one before.

WHAT PROTECTS IT FROM OVERFITTING
---------------------------------
* The series is split chronologically.  ``--split 0.6`` means the first 60% is
  the SELECTION period and the last 40% is the TEST period.  Every stage runs
  on the selection period only.  The test period is touched exactly once, at
  the end, by --stage final.
* Within the selection period each variant's trades are also cut into folds,
  and a variant that only works in one fold is flagged.  Consistency across
  folds is reported next to the headline number.
* The selection criterion is net expectancy in R with a minimum trade count --
  never win rate alone.  A variant that reaches a high win rate by taking six
  trades is rejected by the count gate.
* The final result carries a bootstrap confidence interval and a permutation
  test against the baseline.

WHAT IT IS NOT
--------------
Greedy search can miss interactions, and reusing the selection period across
stages still spends statistical power.  The test period is the only honest
number; treat every selection-period figure as a hypothesis.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smcbot.backtest.engine import Backtester
from smcbot.config import Config
from smcbot.core.types import Candle, Trade
from smcbot.data import loader, synthetic
from smcbot.ml.features import build_features

MIN_TRADES = 40          # a variant with fewer is not evidence


# ------------------------------------------------------------------ metrics
def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def gross_pnl(t: Trade) -> float:
    return t.pnl + t.fees + t.funding


def risk_usdt(t: Trade) -> float:
    return abs(t.entry_price - t.stop) * t.qty


def net_r(t: Trade) -> float:
    r = risk_usdt(t)
    return t.pnl / r if r > 0 else 0.0


def gross_r(t: Trade) -> float:
    r = risk_usdt(t)
    return gross_pnl(t) / r if r > 0 else 0.0


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


def summarise(trades: Sequence[Trade], curve: Sequence[tuple]) -> dict:
    if not trades:
        return {"n": 0}
    nets = [net_r(t) for t in trades]
    wins = [r for r in nets if r > 0]
    losses = [-r for r in nets if r < 0]
    gross_win = sum(r for r in nets if r > 0)
    gross_loss = -sum(r for r in nets if r < 0)
    return {
        "n": len(trades),
        "win": len(wins) / len(trades),
        "avg_win": mean(wins),
        "avg_loss": mean(losses),
        "exp": mean(nets),
        "gross_exp": mean([gross_r(t) for t in trades]),
        "pf": (gross_win / gross_loss) if gross_loss > 0
        else (float("inf") if gross_win > 0 else 0.0),
        "dd": max_dd(curve) if curve else 0.0,
        "fee": mean([fee_r(t) for t in trades]),
        "sl_pct": mean([100 * abs(t.entry_price - t.stop) / t.entry_price
                        for t in trades]),
        "med_r": statistics.median(nets),
        "tp_rate": sum(1 for t in trades if t.tp_hits) / len(trades),
        "trail_rate": sum(1 for t in trades if t.exit_reason == "TRAILING") / len(trades),
        "be_rate": sum(1 for t in trades if t.exit_reason == "BREAKEVEN") / len(trades),
        "hold": mean([t.holding_minutes for t in trades]),
        "per_day": len(trades) / max(
            (max(t.entry_time for t in trades)
             - min(t.entry_time for t in trades)) / 86_400_000, 1e-9),
    }


ROW_HEAD = (f"{'variant':34s} {'n':>5s} {'t/day':>6s} {'win%':>6s} {'avgW':>6s} "
            f"{'avgL':>6s} {'expR':>7s} {'medR':>6s} {'PF':>6s} {'maxDD':>7s} "
            f"{'fee/R':>6s} {'grossR':>7s} {'TP%':>5s} {'trl%':>5s} {'BE%':>5s} "
            f"{'hold':>5s}")


def row(label: str, s: dict, extra: str = "") -> str:
    if not s.get("n"):
        return f"{label:34s} {'savdo yo^q':>5s}"
    p = " inf" if s["pf"] == float("inf") else f"{s['pf']:6.3f}"
    return (f"{label:34s} {s['n']:5d} {s['per_day']:6.2f} {s['win'] * 100:5.1f}% "
            f"{s['avg_win']:6.2f} {s['avg_loss']:6.2f} {s['exp']:+7.3f} "
            f"{s['med_r']:+6.2f} {p} {s['dd'] * 100:6.1f}% {s['fee']:6.3f} "
            f"{s['gross_exp']:+7.3f} {s['tp_rate'] * 100:4.0f}% "
            f"{s['trail_rate'] * 100:4.0f}% {s['be_rate'] * 100:4.0f}% "
            f"{s['hold']:5.0f}{extra}")


# ------------------------------------------------------------------ stats
def bootstrap_ci(values: Sequence[float], runs: int = 5000, seed: int = 3,
                 alpha: float = 0.05) -> Tuple[float, float]:
    if len(values) < 5:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = sorted(mean([values[rng.randrange(n)] for _ in range(n)])
                   for _ in range(runs))
    lo = means[int(runs * alpha / 2)]
    hi = means[int(runs * (1 - alpha / 2))]
    return lo, hi


def permutation_p(a: Sequence[float], b: Sequence[float], runs: int = 10000,
                  seed: int = 7) -> float:
    if not a or not b:
        return 1.0
    rng = random.Random(seed)
    pool = list(a) + list(b)
    obs = abs(mean(a) - mean(b))
    na = len(a)
    hits = sum(1 for _ in range(runs)
               if (rng.shuffle(pool) or abs(mean(pool[:na]) - mean(pool[na:]))) >= obs)
    return (hits + 1) / (runs + 1)


def fold_consistency(trades: Sequence[Trade], folds: int = 4) -> str:
    """Sign of expectancy in each equal-sized chronological slice."""
    if len(trades) < folds * 8:
        return " folds:n/a"
    ts = sorted(trades, key=lambda t: t.entry_time)
    size = len(ts) // folds
    marks = []
    for i in range(folds):
        chunk = ts[i * size:(i + 1) * size] if i < folds - 1 else ts[(folds - 1) * size:]
        marks.append("+" if mean([net_r(t) for t in chunk]) > 0 else "-")
    return " folds:" + "".join(marks)


# ------------------------------------------------------------------ runner
class Runner:
    def __init__(self, candles: List[Candle], base: Config, use_ml: bool = False):
        self.candles = candles
        self.base = base
        self.use_ml = use_ml
        self.cache: Dict[str, tuple] = {}

    def run(self, overrides: dict) -> tuple:
        key = json.dumps(overrides, sort_keys=True, default=str)
        if key in self.cache:
            return self.cache[key]
        cfg = self.base.with_overrides(**overrides) if overrides else self.base
        bt = Backtester(cfg, feature_builder=build_features)
        res = bt.run(self.candles)
        out = (res.trades, res.equity_curve, cfg)
        self.cache[key] = out
        return out

    def evaluate(self, overrides: dict) -> dict:
        trades, curve, _ = self.run(overrides)
        return summarise(trades, curve)


def pick_best(runner: Runner, stage: str, variants: List[Tuple[str, dict]],
              inherited: dict, min_trades: int = MIN_TRADES) -> Tuple[str, dict]:
    print(f"\n{'=' * 110}\n{stage}\n{'=' * 110}")
    print(ROW_HEAD)
    print("-" * 110)
    scored = []
    for label, ov in variants:
        merged = {**inherited, **ov}
        trades, curve, _ = runner.run(merged)
        s = summarise(trades, curve)
        note = fold_consistency(trades) if trades else ""
        flag = "" if s.get("n", 0) >= min_trades else "   (n kichik -- tanlanmaydi)"
        print(row(label, s, note + flag))
        sys.stdout.flush()
        if s.get("n", 0) >= min_trades:
            scored.append((s["exp"], label, ov, s))
    if not scored:
        print(f"\n=> hech bir variant {min_trades} savdo chegarasidan o'tmadi; "
              f"o'zgartirilmaydi")
        return "(o'zgarishsiz)", {}
    scored.sort(key=lambda x: -x[0])
    best_exp, best_label, best_ov, best_s = scored[0]
    print(f"\n=> tanlandi: {best_label}  (net expR {best_exp:+.3f}, "
          f"n={best_s['n']})")
    return best_label, best_ov


# ------------------------------------------------------------------ stages
def stage_stop(runner: Runner, inherited: dict) -> dict:
    variants = []
    for mode in ("M1_SWING", "M5_SWING", "SWEEP_EXTREME"):
        for buf in (0.05, 0.10, 0.15, 0.20):
            variants.append((f"stop={mode} buffer={buf}",
                             {"risk.stop_mode": mode, "risk.sl_atr_buffer": buf}))
    _, ov = pick_best(runner, "1-BOSQICH: STOP JOYLASHUVI (xarajat darvozasi)",
                      variants, inherited)
    return ov


def stage_target(runner: Runner, inherited: dict) -> dict:
    variants = [("tp=LIQUIDITY (hozirgi)", {"risk.tp_mode": "LIQUIDITY"})]
    for m in (0.75, 0.90, 1.00, 1.15, 1.35, 1.50, 2.00):
        variants.append((f"tp=ATR x{m}",
                         {"risk.tp_mode": "ATR",
                          "risk.tp_atr_multiples": [m, m * 1.8, m * 2.6]}))
    _, ov = pick_best(runner, "2-BOSQICH: TARGET", variants, inherited)
    return ov


def stage_partials(runner: Runner, inherited: dict) -> dict:
    variants = [
        ("partial 30/30/40 + BE", {"risk.partial_tp": {"tp1": .3, "tp2": .3, "tp3": .4},
                                   "risk.move_to_be_after_tp1": True}),
        ("partial 30/30/40 no BE", {"risk.partial_tp": {"tp1": .3, "tp2": .3, "tp3": .4},
                                    "risk.move_to_be_after_tp1": False}),
        ("partial 50/50", {"risk.partial_tp": {"tp1": .5, "tp2": .5, "tp3": .0},
                           "risk.move_to_be_after_tp1": False}),
        ("partial 70/30", {"risk.partial_tp": {"tp1": .7, "tp2": .3, "tp3": .0},
                           "risk.move_to_be_after_tp1": False}),
        ("single TP 100%", {"risk.partial_tp": {"tp1": 1.0, "tp2": .0, "tp3": .0},
                            "risk.move_to_be_after_tp1": False}),
        ("single TP + trailing off", {"risk.partial_tp": {"tp1": 1.0, "tp2": .0, "tp3": .0},
                                      "risk.move_to_be_after_tp1": False,
                                      "risk.trailing_enabled": False}),
    ]
    _, ov = pick_best(runner, "3-BOSQICH: PARTIAL TP / BE / TRAILING",
                      variants, inherited)
    return ov


def stage_entry(runner: Runner, inherited: dict) -> dict:
    variants = [(f"chase<={c} ATR", {"filters.max_chase_atr": c})
                for c in (0.25, 0.35, 0.50, 0.75, 1.00)]
    _, ov = pick_best(runner, "4-BOSQICH: ENTRY CHASE", variants, inherited)
    return ov


def stage_filters(runner: Runner, inherited: dict) -> dict:
    variants = [("fee filtri yo'q", {"filters.max_fee_r": 0.0})]
    for f in (0.25, 0.35, 0.50, 0.75):
        variants.append((f"fee/R <= {f}", {"filters.max_fee_r": f}))
    for sc in (70.0, 75.0, 80.0):
        variants.append((f"SMC score >= {sc:.0f}", {"score.valid": sc}))
    _, ov = pick_best(runner, "5-BOSQICH: NO-TRADE FILTRLAR", variants, inherited)
    return ov


STAGES = {"stop": stage_stop, "target": stage_target, "partials": stage_partials,
          "entry": stage_entry, "filters": stage_filters}

# ---------------------------------------------------------- scalp stages
def stage_scalp_tp(runner: Runner, inherited: dict) -> dict:
    """Short fixed targets against ATR targets against the liquidity ladder."""
    variants = [("tp=LIQUIDITY (baseline)", {"risk.tp_mode": "LIQUIDITY"})]
    for pct in (0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
        variants.append((f"tp={pct}% (single)",
                         {"risk.tp_mode": "PERCENT",
                          "risk.tp_percent_levels": [pct],
                          "risk.partial_tp": {"tp1": 1.0, "tp2": .0, "tp3": .0}}))
    for m in (0.50, 0.75, 1.00, 1.25, 1.50):
        variants.append((f"tp=ATR x{m} (single)",
                         {"risk.tp_mode": "ATR",
                          "risk.tp_atr_multiples": [m],
                          "risk.partial_tp": {"tp1": 1.0, "tp2": .0, "tp3": .0}}))
    _, ov = pick_best(runner, "SCALP 1: TARGET MASOFASI", variants, inherited)
    return ov


def stage_scalp_stop(runner: Runner, inherited: dict) -> dict:
    variants = []
    for mode in ("M1_SWING", "M5_SWING", "SWEEP_EXTREME"):
        for buf in (0.05, 0.10, 0.15, 0.20):
            variants.append((f"SL={mode} buf={buf}",
                             {"risk.stop_mode": mode, "risk.sl_atr_buffer": buf}))
    _, ov = pick_best(runner, "SCALP 2: INITIAL STOP", variants, inherited)
    return ov


def stage_scalp_trailing(runner: Runner, inherited: dict) -> dict:
    variants = [("trail=LEGACY", {"risk.trailing_mode": "LEGACY"}),
                ("trail=off", {"risk.trailing_enabled": False})]
    for mode in ("A", "B", "C", "D"):
        for buf in (0.05, 0.10, 0.15, 0.20):
            variants.append((f"trail={mode} buf={buf}",
                             {"risk.trailing_enabled": True,
                              "risk.trailing_mode": mode,
                              "risk.trailing_buffer_atr": buf}))
    _, ov = pick_best(runner, "SCALP 3: TRAILING LADDER + BUFFER",
                      variants, inherited)
    return ov


def stage_scalp_partials(runner: Runner, inherited: dict) -> dict:
    """B/C/D take a share at TP1 and let the remainder ride the trail."""
    variants = [
        ("A: 100% TP1", {"risk.partial_tp": {"tp1": 1.0, "tp2": .0, "tp3": .0},
                         "risk.trail_remainder": False}),
        ("B: 50% TP1 + 50% trail", {"risk.partial_tp": {"tp1": .5, "tp2": .0, "tp3": .0},
                                    "risk.trail_remainder": True}),
        ("C: 70% TP1 + 30% trail", {"risk.partial_tp": {"tp1": .7, "tp2": .0, "tp3": .0},
                                    "risk.trail_remainder": True}),
        ("D: 30% TP1 + 70% trail", {"risk.partial_tp": {"tp1": .3, "tp2": .0, "tp3": .0},
                                    "risk.trail_remainder": True}),
    ]
    _, ov = pick_best(runner, "SCALP 4: PARTIAL TP / QOLDIQ", variants, inherited)
    return ov


def stage_scalp_gates(runner: Runner, inherited: dict) -> dict:
    variants = [("gate: yo'q", {"filters.max_fee_r": 0.0, "risk.min_rr": 1.5})]
    for f in (0.50, 0.75):
        variants.append((f"fee/R <= {f}", {"filters.max_fee_r": f}))
    for rr in (1.25, 1.50, 1.75, 2.00):
        variants.append((f"min RR {rr}", {"risk.min_rr": rr}))
    for c in (0.25, 0.35, 0.50, 0.75):
        variants.append((f"chase <= {c} ATR", {"filters.max_chase_atr": c}))
    _, ov = pick_best(runner, "SCALP 5: NO-TRADE FILTRLAR", variants, inherited)
    return ov


def stage_scalp_session(runner: Runner, inherited: dict) -> dict:
    variants = [("session cap: yo'q", {"risk.max_trades_per_session": 0}),
                ("max 1 trade/session", {"risk.max_trades_per_session": 1}),
                ("max 2 trade/session", {"risk.max_trades_per_session": 2})]
    _, ov = pick_best(runner, "SCALP 6: SESSION LIMITI", variants, inherited,
                      min_trades=25)
    return ov


def stage_scalp_risk(runner: Runner, inherited: dict) -> dict:
    """Leverage must change margin only; expectancy in R must not move."""
    variants = []
    for risk_pct in (0.0025, 0.0035, 0.0050):
        for lev in (17.0, 20.0, 25.0):
            variants.append((f"risk {risk_pct * 100:.2f}% lev {lev:.0f}x",
                             {"risk.risk_per_trade": risk_pct,
                              "risk.max_risk_per_trade": risk_pct,
                              "risk.min_risk_per_trade": risk_pct,
                              "risk.leverage": lev}))
    _, ov = pick_best(runner, "SCALP 7: RISK % VA LEVERAGE", variants, inherited)
    return ov


SCALP_STAGES = {
    "scalp_tp": stage_scalp_tp, "scalp_stop": stage_scalp_stop,
    "scalp_trailing": stage_scalp_trailing, "scalp_partials": stage_scalp_partials,
    "scalp_gates": stage_scalp_gates, "scalp_session": stage_scalp_session,
    "scalp_risk": stage_scalp_risk,
}
SCALP_ORDER = ["scalp_stop", "scalp_tp", "scalp_trailing", "scalp_partials",
               "scalp_gates", "scalp_session", "scalp_risk"]



# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--days", type=int, default=200)
    ap.add_argument("--split", type=float, default=0.6,
                    help="share of the series used for selection")
    ap.add_argument("--stage", default="all",
                    help="all | final | one stage name")
    ap.add_argument("--preset", default="default", choices=["default", "scalp"],
                    help="scalp = short fixed targets, trailing ladders, "
                         "session cap")
    ap.add_argument("--config", help="inherited overrides from earlier stages (JSON)")
    ap.add_argument("--out", default="research_state.json")
    args = ap.parse_args()

    if args.csv:
        print(f"[data] {args.csv}")
        candles = loader.load_csv(args.csv)
    else:
        candles = synthetic.generate(args.days * 1440, seed=42)
    cut = int(len(candles) * args.split)
    selection, test = candles[:cut], candles[cut:]
    print(f"[data] {len(candles)} candles -> selection {len(selection)} "
          f"({len(selection) / 1440:.0f}d), TEST {len(test)} "
          f"({len(test) / 1440:.0f}d, teginilmaydi)")

    inherited: dict = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, encoding="utf-8") as fh:
            inherited = json.load(fh).get("overrides", {})
        print(f"[cfg ] meros: {inherited}")

    base = Config()
    if args.stage == "final":
        print("\n" + "=" * 110)
        print("YAKUNIY: TEGINILMAGAN OUT-OF-SAMPLE DAVR")
        print("=" * 110)
        sel_runner = Runner(selection, base)
        test_runner = Runner(test, base)
        print(ROW_HEAD)
        print("-" * 110)
        base_sel = sel_runner.evaluate({})
        base_test = test_runner.evaluate({})
        print(row("BASELINE / selection", base_sel))
        print(row("BASELINE / TEST", base_test))
        chosen_sel = sel_runner.evaluate(inherited)
        chosen_trades, chosen_curve, cfg = test_runner.run(inherited)
        chosen_test = summarise(chosen_trades, chosen_curve)
        print(row("TANLANGAN / selection", chosen_sel))
        print(row("TANLANGAN / TEST", chosen_test))

        if chosen_test.get("n", 0) >= 10:
            nets = [net_r(t) for t in chosen_trades]
            lo, hi = bootstrap_ci(nets)
            base_trades, _, _ = test_runner.run({})
            p = permutation_p(nets, [net_r(t) for t in base_trades])
            print(f"\nTEST davri statistikasi:")
            print(f"  n                  : {len(nets)}")
            print(f"  net expectancy     : {mean(nets):+.3f} R")
            print(f"  bootstrap 95% CI   : [{lo:+.3f}, {hi:+.3f}]")
            print(f"  CI nolni o'z ichiga oladimi: "
                  f"{'HA -- edge isbotlanmagan' if lo <= 0 <= hi else 'YO^Q'}")
            print(f"  baseline'ga qarshi permutation p = {p:.4f}")
            print(f"  win rate           : {chosen_test['win'] * 100:.1f}%")
            print(f"  70% maqsadiga      : "
                  f"{'YETDI' if chosen_test['win'] >= 0.70 else 'YETMADI'}")
        print("\nEslatma: bu davr faqat bir marta ishlatildi. Agar endi shu")
        print("natijaga qarab parametr o'zgartirilsa, u ham selection davriga")
        print("aylanadi va yangi teginilmagan data kerak bo'ladi.")
        return 0

    runner = Runner(selection, base)
    print("\nBASELINE (selection davri):")
    print(ROW_HEAD)
    print("-" * 110)
    print(row("baseline", runner.evaluate({})))

    table = {**STAGES, **SCALP_STAGES}
    default_order = SCALP_ORDER if args.preset == "scalp" else [
        "stop", "target", "partials", "entry", "filters"]
    order = default_order if args.stage == "all" else [args.stage]
    for name in order:
        if name not in table:
            print(f"noma'lum bosqich: {name}")
            return 1
        t0 = time.time()
        ov = table[name](runner, inherited)
        inherited = {**inherited, **ov}
        print(f"[{name}] {time.time() - t0:.0f}s; "
              f"jamlangan konfiguratsiya: {json.dumps(inherited, default=str)}")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"overrides": inherited}, fh, indent=2, default=str)
    print(f"\n[saqlandi] {args.out}")
    print("Keyingi qadam:")
    print(f"  python3 scripts/research.py --csv <csv> --config {args.out} --stage final")
    return 0


if __name__ == "__main__":
    sys.exit(main())
