"""Which score components actually predict the outcome, and which invert it?

    python scripts/score_audit.py --csv data/BTCUSDT_1m.csv

Everything is measured on **gross** R -- PnL before commission and funding,
divided by the trade's own risk. Commission is a property of the stop distance,
not of the setup's quality, so judging components on net R would just re-measure
the cost problem and hide whatever signal the components carry.

Changes nothing: no threshold, weight, stop, target or cost model is touched.
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smcbot.backtest.engine import Backtester
from smcbot.config import Config, ScoreConfig
from smcbot.core.types import Trade
from smcbot.data import loader, synthetic
from smcbot.ml.features import build_features


# ------------------------------------------------------------------ basics
def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def median(xs: Sequence[float]) -> float:
    return statistics.median(xs) if xs else 0.0


def gross_pnl(t: Trade) -> float:
    return t.pnl + t.fees + t.funding


def risk_usdt(t: Trade) -> float:
    return abs(t.entry_price - t.stop) * t.qty


def gross_r(t: Trade) -> float:
    r = risk_usdt(t)
    return gross_pnl(t) / r if r > 0 else 0.0


def profit_factor(rs: Sequence[float]) -> float:
    win = sum(r for r in rs if r > 0)
    loss = -sum(r for r in rs if r < 0)
    if loss <= 0:
        return float("inf") if win > 0 else 0.0
    return win / loss


def stats(trades: Sequence[Trade]) -> dict:
    rs = [gross_r(t) for t in trades]
    wins = sum(1 for r in rs if r > 0)
    return {"n": len(trades),
            "win": wins / len(trades) if trades else 0.0,
            "avg": mean(rs), "med": median(rs), "pf": profit_factor(rs)}


def fmt(s: dict) -> str:
    pf = "inf" if s["pf"] == float("inf") else f"{s['pf']:5.3f}"
    return (f"{s['n']:5d}  {s['win'] * 100:5.1f}%  {s['avg']:+7.3f}  "
            f"{s['med']:+7.3f}  {pf:>6s}")


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Rank correlation; robust to the non-linear shape of R distributions."""
    n = len(xs)
    if n < 3:
        return 0.0

    def ranks(vs):
        order = sorted(range(n), key=lambda i: vs[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def head(title: str) -> None:
    print(f"\n{'=' * 92}\n{title}\n{'=' * 92}")


# ------------------------------------------------------------------ formula
def show_formula(cfg: ScoreConfig, sample: Optional[Trade]) -> None:
    head("0. SCORE QANDAY HISOBLANADI")
    print("smcbot/strategy/scoring.py -> score(evidence, cfg)\n")
    print("Har komponent ball oladi, keyin hammasi maksimal yig'indiga bo'linadi:")
    print("    FINAL = 100 * sum(points) / sum(max_points)\n")
    rows = [
        ("liquidity_source", cfg.weekly_liquidity,
         "weekly=10 / daily=10 / session=8 / aks holda strength*10/10"),
        ("liquidity_cluster", cfg.liquidity_cluster, "bir joyda >=2 level"),
        ("m15_external_structure", cfg.m15_external_structure, "trend bilan mos"),
        ("m15_sweep", cfg.m15_sweep, "sweep_score/100 ga proporsional"),
        ("m15_internal_choch", cfg.m15_internal_choch, "M15 CHOCH bor"),
        ("m5_confirmation", cfg.m5_confirmation, "M5 shift bor"),
        ("m5_internal_choch", cfg.m5_internal_choch, "M5 CHOCH bor"),
        ("m5_bos", cfg.m5_bos, "M5 BOS bor"),
        ("zone", cfg.zone + cfg.breaker_bonus,
         f"FVG/OB: {cfg.zone}*quality; BREAKER: +{cfg.breaker_bonus}"),
        ("m1_confirmation", cfg.m1_confirmation, "M1 shift bor"),
        ("displacement", cfg.displacement_max, "0/5/10 (weak/medium/strong)"),
        ("premium_discount", cfg.premium_discount, "BUY discountda / SELL premiumda"),
        ("rr", cfg.rr_bonus, "RR>=2.5 -> 5 ; >=2.0 -> 3 ; >=1.5 -> 1.5"),
    ]
    print(f"{'component':24s} {'max':>5s}  shart")
    print("-" * 92)
    total = 0.0
    for name, mx, note in rows:
        total += mx
        print(f"{name:24s} {mx:5.1f}  {note}")
    print("-" * 92)
    print(f"{'JAMI max':24s} {total:5.1f}\n")

    if sample is None:
        return
    b = sample.meta.get("score_breakdown") or {}
    print(f"Misol (bitta haqiqiy savdo, score {sample.smc_score}):")
    for k in sorted(b):
        if k.startswith("_"):
            continue
        print(f"  {k:24s} {b[k]:6.3f}")
    print(f"  {'_raw (yig%sindi)' % chr(39):24s} {b.get('_raw', 0):6.3f}")
    print(f"  {'_max':24s} {b.get('_max', 0):6.3f}")
    print(f"  {'_final = 100*raw/max':24s} {b.get('_final', 0):6.3f}")


# ------------------------------------------------------------------ sections
COMPONENTS: List[Tuple[str, Callable[[Trade], bool]]] = []


def build_component_tests() -> None:
    """Each test answers 'was this component present in this setup?'"""
    def bd(t: Trade, key: str) -> float:
        return (t.meta.get("score_breakdown") or {}).get(key, 0.0)

    def feat(t: Trade, key: str, default: float = 0.0) -> float:
        return t.features.get(key, default) if t.features else default

    COMPONENTS.extend([
        # points-based (straight from the score breakdown)
        ("rr_bonus", lambda t: bd(t, "rr") > 0),
        ("rr_bonus_full (RR>=2.5)", lambda t: bd(t, "rr") >= 5.0),
        ("liquidity_cluster", lambda t: bd(t, "liquidity_cluster") > 0),
        ("m15_external_aligned", lambda t: bd(t, "m15_external_structure") > 0),
        ("m15_internal_choch", lambda t: bd(t, "m15_internal_choch") > 0),
        ("m5_confirmation", lambda t: bd(t, "m5_confirmation") > 0),
        ("m5_internal_choch", lambda t: bd(t, "m5_internal_choch") > 0),
        ("m5_bos", lambda t: bd(t, "m5_bos") > 0),
        ("m1_confirmation", lambda t: bd(t, "m1_confirmation") > 0),
        ("premium_discount", lambda t: bd(t, "premium_discount") > 0),
        ("displacement (any)", lambda t: bd(t, "displacement") > 0),
        ("displacement STRONG", lambda t: bd(t, "displacement") >= 9.0),
        # liquidity tiers (breakdown collapses them; features keep them apart)
        ("weekly_liquidity", lambda t: feat(t, "liquidity_weekly") > 0),
        ("daily_liquidity", lambda t: feat(t, "liquidity_daily") > 0),
        ("session_liquidity", lambda t: feat(t, "liquidity_session") > 0),
        # sweep origin
        ("M15 sweep", lambda t: t.meta.get("sweep_tf") == "M15"),
        ("M5 sweep", lambda t: t.meta.get("sweep_tf") == "M5"),
        ("sweep_score >= 70", lambda t: feat(t, "sweep_score") >= 70),
        # zone kind
        ("zone = FVG", lambda t: t.meta.get("zone_kind") == "FVG"),
        ("zone = OB", lambda t: t.meta.get("zone_kind") == "OB"),
        ("zone = BREAKER", lambda t: t.meta.get("zone_kind") == "BREAKER"),
        # freshness of structure (age features, not the 'ever happened' flags)
        ("M15 CHOCH fresh (<=20 bar)", lambda t: feat(t, "m15_choch_age", 999) <= 20),
        ("M15 BOS fresh (<=20 bar)", lambda t: feat(t, "m15_bos_age", 999) <= 20),
        ("M5 CHOCH fresh (<=24)", lambda t: feat(t, "m5_choch_age", 999) <= 24),
        ("M5 BOS fresh (<=24)", lambda t: feat(t, "m5_bos_age", 999) <= 24),
        ("M1 CHOCH fresh (<=12)", lambda t: feat(t, "m1_choch_age", 999) <= 12),
        ("M1 BOS fresh (<=12)", lambda t: feat(t, "m1_bos_age", 999) <= 12),
        # entry mechanics
        ("LIMIT entry", lambda t: feat(t, "entry_is_limit") > 0),
        ("stop from M1 swing", lambda t: t.meta.get("stop_source") == "M1_SWING"),
    ])


def section_components(trades: List[Trade]) -> List[tuple]:
    head("1. HAR KOMPONENT: BOR vs YO'Q (gross R bo'yicha)")
    base = stats(trades)
    print(f"{'BAZA (hammasi)':32s} {'n':>5s}  {'win%':>6s}  {'avgR':>7s}  "
          f"{'medR':>7s}  {'PF':>6s}")
    print(f"{'':32s} {fmt(base)}")
    print("\n" + "-" * 92)
    print(f"{'component':32s} {'n':>5s}  {'win%':>6s}  {'avgR':>7s}  "
          f"{'medR':>7s}  {'PF':>6s}   delta_avgR")
    print("-" * 92)

    rows = []
    for name, test in COMPONENTS:
        with_ = [t for t in trades if test(t)]
        without = [t for t in trades if not test(t)]
        if not with_ or not without:
            print(f"{name:32s}  -- hamma yoki hech biri ({len(with_)}/{len(trades)})")
            continue
        sw, so = stats(with_), stats(without)
        delta = sw["avg"] - so["avg"]
        print(f"{name + '  [BOR]':32s} {fmt(sw)}   {delta:+7.3f}")
        print(f"{name + '  [YOQ]':32s} {fmt(so)}")
        rows.append((name, sw, so, delta))
    return rows


def section_buckets(trades: List[Trade]) -> None:
    head("2. SCORE -> NATIJA")

    def show(width: int, title: str) -> None:
        print(f"\n{title}")
        print(f"{'bucket':>10s} {'n':>5s}  {'win%':>6s}  {'avgR':>7s}  "
              f"{'medR':>7s}  {'PF':>6s}")
        groups: Dict[int, List[Trade]] = defaultdict(list)
        for t in trades:
            groups[int(t.smc_score // width) * width].append(t)
        for k in sorted(groups):
            g = groups[k]
            if len(g) < 3:
                print(f"{k:>7d}-{k + width:<3d} {len(g):5d}  (namuna kichik)")
                continue
            print(f"{k:>7d}-{k + width:<3d} {fmt(stats(g))}")

    show(5, "5 ballik bucket:")
    show(1, "1 ballik bucket:")

    scores = [t.smc_score for t in trades]
    grs = [gross_r(t) for t in trades]
    print(f"\nSpearman(score, gross R) = {spearman(scores, grs):+.4f}")
    print("  (musbat = ball foydali, manfiy = ball teskari ishlaydi)")


def section_rr_hypothesis(trades: List[Trade]) -> None:
    head("3. GIPOTEZA: BALL RR NI MUKOFOTLAYDI, RR ESA EHTIMOLNI PASAYTIRADI")
    have = [t for t in trades if t.features]
    if not have:
        print("feature vektor yo'q")
        return

    print("\nSetup RR -> natija:")
    print(f"{'RR':>10s} {'n':>5s}  {'win%':>6s}  {'avgR':>7s}  {'medR':>7s}  {'PF':>6s}")
    groups: Dict[str, List[Trade]] = defaultdict(list)
    for t in have:
        rr = t.features.get("rr", 0.0)
        key = ("1.5-2.0" if rr < 2.0 else "2.0-2.5" if rr < 2.5 else
               "2.5-4.0" if rr < 4.0 else "4.0+")
        groups[key].append(t)
    for k in ("1.5-2.0", "2.0-2.5", "2.5-4.0", "4.0+"):
        if k in groups and len(groups[k]) >= 3:
            print(f"{k:>10s} {fmt(stats(groups[k]))}")

    rr_vals = [t.features.get("rr", 0.0) for t in have]
    wins = [1.0 if gross_r(t) > 0 else 0.0 for t in have]
    grs = [gross_r(t) for t in have]
    print(f"\nSpearman(RR, win)      = {spearman(rr_vals, wins):+.4f}")
    print(f"Spearman(RR, gross R)  = {spearman(rr_vals, grs):+.4f}")

    print("\nTP1 masofasi (ATR) -> natija:")
    d = [(t.features.get("tp1_distance_atr", 0.0), t) for t in have]
    d.sort(key=lambda x: x[0])
    q = max(len(d) // 4, 1)
    for i, label in enumerate(["eng yaqin 25%", "25-50%", "50-75%", "eng uzoq 25%"]):
        chunk = [t for _, t in d[i * q:(i + 1) * q]] if i < 3 else [t for _, t in d[3 * q:]]
        if len(chunk) >= 3:
            print(f"{label:>14s} {fmt(stats(chunk))}")

    print("\nBall va RR bog'liqligi:")
    sc = [t.smc_score for t in have]
    print(f"Spearman(score, RR)            = {spearman(sc, rr_vals):+.4f}")
    print(f"Spearman(score, TP1 masofasi)  = "
          f"{spearman(sc, [t.features.get('tp1_distance_atr', 0.0) for t in have]):+.4f}")
    print(f"Spearman(score, SL masofasi)   = "
          f"{spearman(sc, [t.features.get('sl_distance_atr', 0.0) for t in have]):+.4f}")


def section_verdict(rows: List[tuple], trades: List[Trade]) -> None:
    head("4. XULOSA: A) FOYDALI  B) ZARARLI  C) NEYTRAL")
    ranked = sorted(rows, key=lambda r: -r[3])
    # Only trust a split that has a reasonable number of trades on both sides.
    solid = [r for r in ranked if r[1]["n"] >= 25 and r[2]["n"] >= 25]

    def block(title: str, items: List[tuple]) -> None:
        print(f"\n{title}")
        if not items:
            print("  (yo'q)")
            return
        print(f"{'component':32s} {'bor_n':>6s} {'bor_avgR':>9s} "
              f"{'yoq_avgR':>9s} {'delta':>8s} {'bor_PF':>7s}")
        for name, sw, so, delta in items:
            pf = "inf" if sw["pf"] == float("inf") else f"{sw['pf']:.3f}"
            print(f"{name:32s} {sw['n']:6d} {sw['avg']:+9.3f} "
                  f"{so['avg']:+9.3f} {delta:+8.3f} {pf:>7s}")

    block("A) FOYDALI (delta >= +0.10R, ikkala tomonda >=25 savdo):",
          [r for r in solid if r[3] >= 0.10])
    block("B) ZARARLI (delta <= -0.10R):",
          [r for r in solid if r[3] <= -0.10])
    block("C) NEYTRAL (|delta| < 0.10R):",
          [r for r in solid if abs(r[3]) < 0.10])

    small = [r for r in ranked if r not in solid]
    if small:
        print(f"\nNamuna kichik bo'lgani uchun hukm qilinmadi "
              f"({len(small)} ta): " + ", ".join(r[0] for r in small))


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--days", type=int, default=120)
    args = ap.parse_args()

    cfg = Config()
    if args.csv:
        print(f"[data] {args.csv}")
        candles = loader.load_csv(args.csv)
    else:
        candles = synthetic.generate(args.days * 1440, seed=42)
    print(f"[data] {len(candles)} candles")

    t0 = time.time()
    result = Backtester(cfg, feature_builder=build_features).run(candles)
    trades = result.trades
    print(f"[run ] {time.time() - t0:.0f}s, {len(trades)} trades")
    if len(trades) < 20:
        print("juda kam savdo, tahlil ma'nosiz")
        return 1

    print(f"\nBAZA: gross avg R = {mean([gross_r(t) for t in trades]):+.4f}, "
          f"gross PF = {profit_factor([gross_r(t) for t in trades]):.3f}, "
          f"gross win% = "
          f"{100 * sum(1 for t in trades if gross_r(t) > 0) / len(trades):.1f}%")

    show_formula(cfg.score, trades[0])
    build_component_tests()
    rows = section_components(trades)
    section_buckets(trades)
    section_rr_hypothesis(trades)
    section_verdict(rows, trades)
    return 0


if __name__ == "__main__":
    sys.exit(main())
