"""Why do M15-sweep setups lose while M5-sweep setups do not?

    python scripts/sweep_audit.py --csv data/BTCUSDT_1m.csv

The previous audit found the largest single split in the whole strategy:
M15 sweeps -0.155 gross R over 432 trades, M5 sweeps +0.181 over 128.  That
is either the most useful thing we know or a multiple-comparison artefact, so
this script does two jobs: it slices both groups across every dimension that
might explain the gap, and it runs a permutation test to say how likely the
gap is under chance alone.

Everything is gross R -- before commission -- because commission follows the
stop distance rather than the setup.  Nothing is modified.
"""
from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smcbot.backtest.engine import Backtester
from smcbot.config import Config
from smcbot.core.types import Trade
from smcbot.data import loader, synthetic
from smcbot.ml.features import build_features

MIN_N = 25          # below this a split is reported but never called an edge


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


def fee_r(t: Trade) -> float:
    r = risk_usdt(t)
    return t.fees / r if r > 0 else 0.0


def pf(rs: Sequence[float]) -> float:
    win = sum(r for r in rs if r > 0)
    loss = -sum(r for r in rs if r < 0)
    return (win / loss) if loss > 0 else (float("inf") if win > 0 else 0.0)


def row(trades: Sequence[Trade]) -> dict:
    rs = [gross_r(t) for t in trades]
    return {
        "n": len(trades),
        "win": sum(1 for r in rs if r > 0) / len(trades) if trades else 0.0,
        "avg": mean(rs), "med": median(rs), "pf": pf(rs),
        "rr": mean([t.features.get("rr", 0.0) for t in trades if t.features]),
        "sl": mean([100 * abs(t.entry_price - t.stop) / t.entry_price
                    for t in trades]),
        "fee": mean([fee_r(t) for t in trades]),
    }


def line(label: str, s: dict, width: int = 30) -> str:
    p = "  inf" if s["pf"] == float("inf") else f"{s['pf']:5.3f}"
    return (f"{label:<{width}s} {s['n']:5d} {s['win'] * 100:5.1f}% "
            f"{s['avg']:+7.3f} {s['med']:+7.3f} {p:>6s} "
            f"{s['rr']:5.2f} {s['sl']:6.3f}% {s['fee']:5.3f}")


HEADER = (f"{'':30s} {'n':>5s} {'win%':>6s} {'avgR':>7s} {'medR':>7s} "
          f"{'PF':>6s} {'RR':>5s} {'SL%':>7s} {'fee/R':>5s}")


def head(t: str) -> None:
    print(f"\n{'=' * 96}\n{t}\n{'=' * 96}")


def compare(groups: Dict[str, List[Trade]], title: str,
            min_show: int = 1) -> None:
    print(f"\n{title}")
    print(HEADER)
    print("-" * 96)
    for k, ts in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(ts) < min_show:
            continue
        tag = "" if len(ts) >= MIN_N else "  (kichik)"
        print(line(str(k), row(ts)) + tag)


def split_by(trades: Sequence[Trade], key: Callable[[Trade], str]
             ) -> Dict[str, List[Trade]]:
    out: Dict[str, List[Trade]] = defaultdict(list)
    for t in trades:
        out[str(key(t))].append(t)
    return out


def permutation_test(a: Sequence[float], b: Sequence[float],
                     runs: int = 20000, seed: int = 5) -> float:
    """Two-sided p-value for the difference in means, labels reshuffled.

    Makes no distributional assumption, which matters because R is skewed.
    """
    rng = random.Random(seed)
    pool = list(a) + list(b)
    observed = abs(mean(a) - mean(b))
    na = len(a)
    hits = 0
    for _ in range(runs):
        rng.shuffle(pool)
        if abs(mean(pool[:na]) - mean(pool[na:])) >= observed:
            hits += 1
    return (hits + 1) / (runs + 1)


# ------------------------------------------------------------------ sections
def section_overview(m15: List[Trade], m5: List[Trade], all_t: List[Trade]) -> None:
    head("1. UMUMIY: M15 sweep vs M5 sweep")
    print(HEADER)
    print("-" * 96)
    print(line("HAMMASI", row(all_t)))
    print(line("M15 sweep", row(m15)))
    print(line("M5 sweep", row(m5)))
    d = mean([gross_r(t) for t in m5]) - mean([gross_r(t) for t in m15])
    print(f"\ndelta (M5 - M15) = {d:+.3f} R")

    p = permutation_test([gross_r(t) for t in m15], [gross_r(t) for t in m5])
    print(f"permutation test p = {p:.4f}  (20000 ta aralashtirish)")
    print("  29 ta komponent sinalgan edi; Bonferroni chegarasi ~ 0.05/29 = 0.0017")
    if p < 0.0017:
        print("  => ko'p taqqoslashni hisobga olganda ham SIGNIFIKANT")
    elif p < 0.05:
        print("  => yakka holda signifikant, lekin ko'p taqqoslash chegarasidan o'tmadi")
    else:
        print("  => signifikant EMAS; tasodif bilan izohlanishi mumkin")


def section_slices(m15: List[Trade], m5: List[Trade]) -> None:
    head("2. HAR IKKI GURUH, BIR XIL KESIMLARDA")
    print("\nEslatma: sweep yo'nalishi va savdo yo'nalishi ayni narsa -- kodda")
    print("BUY faqat low-sweep'dan, SELL faqat high-sweep'dan tug'iladi.")

    def feat(t: Trade, k: str, d: float = 0.0) -> float:
        return t.features.get(k, d) if t.features else d

    dims = [
        ("side", lambda t: t.side.value),
        ("setup kind", lambda t: t.meta.get("kind", "?")),
        ("sweep level kind", lambda t: t.meta.get("sweep_level_kind", "?")),
        ("sweep score", lambda t: ("<50" if t.meta.get("sweep_score", 0) < 50
                                   else "50-70" if t.meta.get("sweep_score", 0) < 70
                                   else "70-85" if t.meta.get("sweep_score", 0) < 85
                                   else "85+")),
        ("M15 external aligned",
         lambda t: "aligned" if (t.meta.get("score_breakdown") or {})
         .get("m15_external_structure", 0) > 0 else "counter"),
        ("M15 CHOCH fresh", lambda t: "yes" if feat(t, "m15_choch_age", 999) <= 20 else "no"),
        ("M5 CHOCH fresh", lambda t: "yes" if feat(t, "m5_choch_age", 999) <= 24 else "no"),
        ("M5 BOS fresh", lambda t: "yes" if feat(t, "m5_bos_age", 999) <= 24 else "no"),
        ("zone kind", lambda t: t.meta.get("zone_kind", "?")),
        ("premium/discount ok",
         lambda t: "yes" if (t.meta.get("score_breakdown") or {})
         .get("premium_discount", 0) > 0 else "no"),
        ("RR bucket", lambda t: ("1.5-2" if feat(t, "rr") < 2 else
                                 "2-2.5" if feat(t, "rr") < 2.5 else
                                 "2.5-4" if feat(t, "rr") < 4 else "4+")),
        ("market state", lambda t: t.market_state or "?"),
        ("session", lambda t: t.session or "?"),
    ]
    for label, key in dims:
        print(f"\n--- {label} ---")
        print(HEADER)
        print("-" * 96)
        for tag, group in (("M15", m15), ("M5 ", m5)):
            for k, ts in sorted(split_by(group, key).items(),
                                key=lambda kv: -len(kv[1])):
                mark = "" if len(ts) >= MIN_N else "  (kichik)"
                print(line(f"{tag} | {k}", row(ts)) + mark)


def section_chain(m15: List[Trade], m5: List[Trade]) -> None:
    head("3. ZANJIR: M15 sweep -> M5 conf -> M1 entry   vs   M5 sweep -> M1 entry")
    print(HEADER)
    print("-" * 96)
    print(line("M15 sweep zanjiri", row(m15)))
    print(line("M5 sweep zanjiri", row(m5)))

    def feat(t: Trade, k: str, d: float = 0.0) -> float:
        return t.features.get(k, d) if t.features else d

    print("\nZanjirdagi bo'g'inlar sifati (o'rtacha):")
    fields = [
        ("sweep score", lambda t: t.meta.get("sweep_score", 0.0)),
        ("sweep penetration (ATR)", lambda t: t.meta.get("sweep_penetration_atr", 0.0)),
        ("sweep wick ratio", lambda t: t.meta.get("sweep_wick_ratio", 0.0)),
        ("sweep return bars", lambda t: t.meta.get("sweep_return_bars", 0.0)),
        ("sweep volume ratio", lambda t: t.meta.get("sweep_volume_ratio", 0.0)),
        ("sweep -> entry (min)", lambda t: t.meta.get("sweep_age_min", 0.0)),
        ("sweep -> entry (ATR)", lambda t: t.meta.get("sweep_distance_atr", 0.0)),
        ("M5 CHOCH age (bar)", lambda t: min(feat(t, "m5_choch_age", 999), 200)),
        ("M5 BOS age (bar)", lambda t: min(feat(t, "m5_bos_age", 999), 200)),
        ("M1 CHOCH age (bar)", lambda t: min(feat(t, "m1_choch_age", 999), 200)),
        ("TP1 distance (ATR)", lambda t: feat(t, "tp1_distance_atr")),
        ("SL distance (ATR)", lambda t: feat(t, "sl_distance_atr")),
        ("RR", lambda t: feat(t, "rr")),
        ("zone quality", lambda t: feat(t, "zone_quality")),
        ("MFE (R)", lambda t: t.mfe),
        ("MAE (R)", lambda t: t.mae),
    ]
    print(f"{'':30s} {'M15':>10s} {'M5':>10s} {'farq':>10s}")
    print("-" * 64)
    for label, fn in fields:
        a, b = mean([fn(t) for t in m15]), mean([fn(t) for t in m5])
        print(f"{label:30s} {a:10.3f} {b:10.3f} {b - a:+10.3f}")


def section_m15_diagnosis(m15: List[Trade]) -> None:
    head("4. M15 SWEEP NIMA UCHUN YOMON? -- alohida gipotezalar")

    def feat(t: Trade, k: str, d: float = 0.0) -> float:
        return t.features.get(k, d) if t.features else d

    def quartiles(label: str, fn: Callable[[Trade], float]) -> None:
        vals = sorted(((fn(t), t) for t in m15), key=lambda x: x[0])
        q = max(len(vals) // 4, 1)
        print(f"\n{label}")
        print(HEADER)
        print("-" * 96)
        for i, name in enumerate(["Q1 (eng kichik)", "Q2", "Q3", "Q4 (eng katta)"]):
            chunk = ([t for _, t in vals[i * q:(i + 1) * q]] if i < 3
                     else [t for _, t in vals[3 * q:]])
            if chunk:
                lo, hi = fn(chunk[0]), fn(chunk[-1])
                print(line(f"{name} [{lo:.2f}..{hi:.2f}]", row(chunk)))

    print("\n4a. Sweep juda erta / juda kech olinganmi?")
    quartiles("sweep -> entry vaqti (minut):",
              lambda t: t.meta.get("sweep_age_min", 0.0))

    print("\n4b. Entry sweepdan juda uzoqmi?")
    quartiles("sweep extremumidan entry masofasi (ATR):",
              lambda t: t.meta.get("sweep_distance_atr", 0.0))

    print("\n4c. Target juda uzoqmi?")
    quartiles("TP1 masofasi (ATR):", lambda t: feat(t, "tp1_distance_atr"))

    print("\n4d. Sweep aslida oddiy liquidity take (continuation ichida)mi?")
    groups = split_by(m15, lambda t: f"{t.meta.get('kind', '?')} / "
                      f"{'aligned' if (t.meta.get('score_breakdown') or {}).get('m15_external_structure', 0) > 0 else 'counter'}")
    compare(groups, "setup kind + external structure:")

    print("\n4e. M5 confirmation sifati (M15 sweepdan keyin):")
    quartiles("M5 CHOCH yoshi (bar, kichik = yangi):",
              lambda t: min(feat(t, "m5_choch_age", 999), 200))

    print("\n4f. Qarama-qarshi structure bormi?")
    def opposed(t: Trade) -> str:
        m15t = feat(t, "m15_internal_trend")
        m5t = feat(t, "m5_internal_trend")
        # features are already signed by trade direction: +1 = with us
        if m15t > 0 and m5t > 0:
            return "M15+ M5+ (ikkalasi biz tomonda)"
        if m15t > 0 >= m5t:
            return "M15+ M5- (M5 qarshi)"
        if m5t > 0 >= m15t:
            return "M15- M5+ (M15 qarshi)"
        return "M15- M5- (ikkalasi qarshi)"
    compare(split_by(m15, opposed), "internal trend mosligi:")


def section_m5_combos(m5: List[Trade], m15: List[Trade]) -> None:
    head("5. M5 SWEEP USTUNLIGI BOSHQA KOMPONENTLAR BILAN BOG'LIQMI?")

    def feat(t: Trade, k: str, d: float = 0.0) -> float:
        return t.features.get(k, d) if t.features else d

    def bd(t: Trade, k: str) -> float:
        return (t.meta.get("score_breakdown") or {}).get(k, 0.0)

    combos = [
        ("premium_discount", lambda t: bd(t, "premium_discount") > 0),
        ("M5 CHOCH fresh", lambda t: feat(t, "m5_choch_age", 999) <= 24),
        ("zone = FVG", lambda t: t.meta.get("zone_kind") == "FVG"),
        ("M1 BOS fresh", lambda t: feat(t, "m1_bos_age", 999) <= 12),
        ("m5_internal_choch (ball)", lambda t: bd(t, "m5_internal_choch") > 0),
        ("REVERSAL", lambda t: t.meta.get("kind") == "REVERSAL"),
        ("CONTINUATION", lambda t: t.meta.get("kind") == "CONTINUATION"),
    ]
    print(HEADER)
    print("-" * 96)
    print(line("M5 sweep (baza)", row(m5)))
    print(line("M15 sweep (baza)", row(m15)))
    print("-" * 96)
    for label, test in combos:
        for tag, group in (("M5 ", m5), ("M15", m15)):
            sub = [t for t in group if test(t)]
            if not sub:
                continue
            mark = "" if len(sub) >= MIN_N else "  (kichik)"
            print(line(f"{tag} + {label}", row(sub)) + mark)
        print("-" * 96)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--min-group", type=int, default=20,
                    help="refuse to analyse a group smaller than this")
    args = ap.parse_args()

    cfg = Config()
    if args.csv:
        print(f"[data] {args.csv}")
        candles = loader.load_csv(args.csv)
    else:
        candles = synthetic.generate(args.days * 1440, seed=42)
    print(f"[data] {len(candles)} candles")

    t0 = time.time()
    trades = Backtester(cfg, feature_builder=build_features).run(candles).trades
    print(f"[run ] {time.time() - t0:.0f}s, {len(trades)} trades")

    m15 = [t for t in trades if t.meta.get("sweep_tf") == "M15"]
    m5 = [t for t in trades if t.meta.get("sweep_tf") == "M5"]
    if len(m15) < args.min_group or len(m5) < args.min_group:
        print(f"juda kam: M15={len(m15)}, M5={len(m5)}")
        return 1

    section_overview(m15, m5, trades)
    section_slices(m15, m5)
    section_chain(m15, m5)
    section_m15_diagnosis(m15)
    section_m5_combos(m5, m15)

    head("6. KO'P TAQQOSLASH HAQIDA OGOHLANTIRISH")
    print("Bu skript ~60 ta kesim chiqaradi. p=0.05 da tasodifan ~3 tasi")
    print("'signifikant' ko'rinadi. Qoidalar:")
    print(f"  - n < {MIN_N} bo'lgan hech narsa edge emas, faqat kuzatuv;")
    print("  - 1-bo'limdagi permutation p asosiy hukm, qolgani izoh;")
    print("  - bitta yil, bitta coin -- takrorlanmaguncha bu gipoteza.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
