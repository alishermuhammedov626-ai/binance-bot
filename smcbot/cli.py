"""Command line interface.

    python -m smcbot fetch       --symbol BTCUSDT --days 365 --out data/btc.csv
    python -m smcbot backtest    --csv data/btc.csv
    python -m smcbot dataset     --csv data/btc.csv --out data/samples.json
    python -m smcbot walkforward --csv data/btc.csv
    python -m smcbot train       --csv data/btc.csv --out models/ml_v1.json
    python -m smcbot pipeline    --csv data/btc.csv --journal runs/smcbot.db
    python -m smcbot dashboard   --synthetic --days 20

Add ``--synthetic --days N`` to any command to run on generated data with no
network access.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional

from .backtest.engine import Backtester
from .backtest.metrics import breakdown
from .backtest.montecarlo import monte_carlo
from .config import Config
from .core.types import Candle
from .data import loader, synthetic
from .journal import Journal
from .ml.dataset import build_dataset, dataset_summary, labelled
from .ml.features import build_features
from .ml.walkforward import WalkForwardPredictor, train_final, walk_forward


# ---------------------------------------------------------------- helpers
def load_config(path: Optional[str], overrides: Optional[List[str]]) -> Config:
    cfg = Config.load(path) if path else Config()
    if overrides:
        kv = {}
        for item in overrides:
            key, _, value = item.partition("=")
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = value
            kv[key.strip()] = parsed
        cfg = cfg.with_overrides(**kv)
    return cfg


def load_candles(args) -> List[Candle]:
    if getattr(args, "synthetic", False) or not getattr(args, "csv", None):
        minutes = int(getattr(args, "days", 60)) * 24 * 60
        print(f"[data] synthetic: {args.days} days ({minutes} M1 candles), "
              f"seed {args.seed}")
        return synthetic.generate(minutes, seed=args.seed)
    print(f"[data] loading {args.csv}")
    candles = loader.load_csv(args.csv, limit=getattr(args, "limit", None))
    holes = loader.gaps(candles)
    print(f"[data] {len(candles)} candles, {len(holes)} gaps")
    return candles


def progress(n: int) -> None:
    print(f"      ... {n} bars", flush=True)


def _split(candles: List[Candle], ratio: float) -> tuple:
    cut = int(len(candles) * ratio)
    return candles[:cut], candles[cut:]


# ---------------------------------------------------------------- commands
def cmd_fetch(args) -> int:
    """Download M1 candles, streaming to disk so a long job can be resumed."""
    import datetime as dt

    out = args.out
    start = args.start
    if args.resume:
        last = loader.last_candle_time(out)
        if last is not None:
            start = str(last + 60_000)
            print(f"[fetch] resuming {out} from {_fmt(last)}")
    if start is None:
        start = (dt.datetime.now(dt.timezone.utc) -
                 dt.timedelta(days=args.days)).strftime("%Y-%m-%d")
    if not args.resume and os.path.exists(out):
        os.remove(out)

    print(f"[fetch] {args.symbol} {args.interval} from {start} -> {out}")
    written = [0]

    def sink(batch):
        written[0] += loader.append_csv(out, batch)

    def show(total, last_open):
        print(f"      {total:>8d} candles, at {_fmt(last_open)}", flush=True)

    try:
        loader.fetch_binance(args.symbol, args.interval, start, args.end,
                             limit_total=args.max_candles,
                             sleep_between=args.sleep,
                             progress=show, on_batch=sink)
    except KeyboardInterrupt:
        print(f"\n[fetch] interrupted -- {written[0]} candles saved; "
              f"rerun with --resume to continue")
        return 1
    except Exception as exc:
        print(f"[fetch] FAILED after {written[0]} candles: {exc}")
        print("        rerun with --resume to continue from where it stopped")
        return 1

    candles = loader.load_csv(out)
    holes = loader.gaps(candles)
    print(f"[fetch] {len(candles)} candles saved, {len(holes)} gaps")
    if holes:
        print(f"        first gap: {_fmt(holes[0][0])} -> {_fmt(holes[0][1])}")
    return 0


def _fmt(ms: int) -> str:
    import datetime as dt
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime(
        "%Y-%m-%d %H:%M")


def cmd_backtest(args) -> int:
    cfg = load_config(args.config, args.set)
    candles = load_candles(args)
    predictor = None
    if args.ml:
        predictor = _walkforward_predictor(cfg, candles)
    t0 = time.time()
    bt = Backtester(cfg, ml_predictor=predictor,
                    feature_builder=build_features if predictor else None)
    result = bt.run(candles, progress=progress if args.verbose else None)
    print(f"\n[backtest] {len(candles)} bars in {time.time() - t0:.1f}s")
    print(result.summary())
    if args.breakdown:
        for key in ("session", "market_state", "kind"):
            print(f"\nBy {key}:")
            for k, v in breakdown(result.trades, key).items():
                print(f"  {k:16s} {v}")
    if args.rejections:
        print("\nTop rejection reasons:")
        for k, v in sorted(result.rejections.items(), key=lambda kv: -kv[1])[:20]:
            print(f"  {v:8d}  {k}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(result.to_dict(), fh, indent=2, default=str)
        print(f"\n[backtest] wrote {args.json}")
    if args.journal:
        run_id = f"bt-{int(time.time())}"
        with Journal(args.journal) as j:
            j.start_run(run_id, "backtest", cfg.symbol, cfg.to_dict())
            j.record_trades(result.trades, run_id)
            j.finish_run(run_id, result.metrics)
        print(f"[journal] {len(result.trades)} trades -> {args.journal} ({run_id})")
    return 0


def cmd_dataset(args) -> int:
    cfg = load_config(args.config, args.set)
    candles = load_candles(args)
    samples = build_dataset(cfg, candles, progress=progress if args.verbose else None)
    summary = dataset_summary(samples)
    print(json.dumps(summary, indent=2))
    if args.out:
        rows = [{"time": s.time, "side": s.side, "label": s.label,
                 "outcome": s.outcome, "r_multiple": s.r_multiple,
                 "smc_score": s.smc_score, "mfe": s.mfe, "mae": s.mae,
                 "features": s.features} for s in samples]
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
        print(f"[dataset] wrote {len(rows)} samples -> {args.out}")
    return 0


def cmd_walkforward(args) -> int:
    cfg = load_config(args.config, args.set)
    candles = load_candles(args)
    samples = build_dataset(cfg, candles, progress=progress if args.verbose else None)
    print(json.dumps(dataset_summary(samples), indent=2))
    wf = walk_forward(cfg, samples)
    if wf is None:
        print("\n[walkforward] not enough labelled setups "
              f"(need >= {cfg.ml.min_train_samples}); use more data.")
        return 1
    print("\n" + wf.summary())
    return 0


def cmd_train(args) -> int:
    cfg = load_config(args.config, args.set)
    candles = load_candles(args)
    samples = build_dataset(cfg, candles, progress=progress if args.verbose else None)
    wf = walk_forward(cfg, samples)
    if wf is not None:
        print(wf.summary())
        print()
    trained = train_final(cfg, samples)
    if trained is None:
        print(f"[train] not enough labelled samples "
              f"({len(labelled(samples))} < {cfg.ml.min_train_samples})")
        return 1
    model, names = trained
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    if hasattr(model, "save"):
        model.save(args.out)
        print(f"[train] saved {cfg.ml.model_version} -> {args.out} "
              f"({len(names)} features)")
    if args.journal:
        with Journal(args.journal) as j:
            j.record_model(cfg.ml.model_version, "gbt", names,
                           {"oos_auc": wf.oos_auc} if wf else {}, args.out)
    return 0


def _walkforward_predictor(cfg: Config, candles: List[Candle]
                           ) -> Optional[WalkForwardPredictor]:
    """Leak-free ML for a backtest: a fold's model only scores later bars."""
    print("[ml] building dataset for walk-forward models ...")
    samples = build_dataset(cfg, candles)
    print("[ml] " + json.dumps(dataset_summary(samples)))
    wf = walk_forward(cfg, samples)
    if wf is None:
        print("[ml] not enough labelled setups -- running SMC-only")
        return None
    print(wf.summary())
    return WalkForwardPredictor(wf, cfg.ml.embargo_minutes * 60_000)


def cmd_pipeline(args) -> int:
    """Section 91: backtest -> out-of-sample -> walk-forward -> Monte Carlo."""
    cfg = load_config(args.config, args.set)
    candles = load_candles(args)
    in_sample, out_sample = _split(candles, args.split)
    print(f"[pipeline] in-sample {len(in_sample)} bars / "
          f"out-of-sample {len(out_sample)} bars")

    print("\n=== 1. IN-SAMPLE BACKTEST (SMC only) ===")
    r_in = Backtester(cfg).run(in_sample)
    print(r_in.summary())

    print("\n=== 2. OUT-OF-SAMPLE BACKTEST (SMC only) ===")
    r_out = Backtester(cfg).run(out_sample)
    print(r_out.summary())

    print("\n=== 3. WALK-FORWARD ML VALIDATION ===")
    samples = build_dataset(cfg, candles)
    print(json.dumps(dataset_summary(samples), indent=2))
    wf = walk_forward(cfg, samples)
    predictor = None
    if wf is None:
        print("[ml] not enough labelled setups for walk-forward")
    else:
        print(wf.summary())
        predictor = WalkForwardPredictor(wf, cfg.ml.embargo_minutes * 60_000)

    print("\n=== 4. FULL BACKTEST WITH LEAK-FREE ML FILTER ===")
    r_ml = Backtester(cfg, ml_predictor=predictor,
                      feature_builder=build_features if predictor else None)
    result_ml = r_ml.run(candles)
    print(result_ml.summary())
    if predictor is not None:
        print(f"[ml] {predictor.calls} scored, "
              f"{predictor.no_model_calls} before the first model was available")

    print("\n=== 5. MONTE CARLO ===")
    mc = monte_carlo(result_ml.trades or r_out.trades, cfg.initial_equity,
                     runs=args.mc_runs)
    print(mc.summary())

    print("\n=== VERDICT (section 59/91) ===")
    m = result_ml.metrics
    checks = {
        "trades >= 100": m.get("trades", 0) >= 100,
        "expectancy > 0": bool(m.get("expectancy_positive")),
        "profit factor > 1.2": (m.get("profit_factor") or 0) > 1.2,
        "max drawdown < 25%": (m.get("max_drawdown") or 1) < 0.25,
        "out-of-sample expectancy > 0": bool(
            r_out.metrics.get("expectancy_positive")),
        "walk-forward AUC > 0.55": bool(wf and wf.oos_auc > 0.55),
        "risk of ruin < 5%": mc.ruin_probability < 0.05,
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    ready = all(checks.values())
    print(f"\n  => {'READY for paper trading' if ready else 'NOT ready for live money'}"
          " (section 90/91 requires paper trading before capital either way)")

    if args.journal:
        run_id = f"pipeline-{int(time.time())}"
        with Journal(args.journal) as j:
            j.start_run(run_id, "pipeline", cfg.symbol, cfg.to_dict())
            j.record_trades(result_ml.trades, run_id)
            j.finish_run(run_id, {"in_sample": r_in.metrics,
                                  "out_of_sample": r_out.metrics,
                                  "with_ml": result_ml.metrics,
                                  "walk_forward_auc": wf.oos_auc if wf else None,
                                  "monte_carlo_ruin": mc.ruin_probability,
                                  "checks": checks})
        print(f"[journal] -> {args.journal} ({run_id})")
    return 0 if ready else 2


def cmd_dashboard(args) -> int:
    from . import dashboard as dash
    cfg = load_config(args.config, args.set)
    candles = load_candles(args)
    bt = Backtester(cfg)
    result = bt.run(candles)
    data = dash.build(bt.ctx, bt.risk, bt.signals, result.trades, bt.position)
    print(dash.to_json(data) if args.json else dash.render(data))
    return 0


def cmd_montecarlo(args) -> int:
    cfg = load_config(args.config, args.set)
    candles = load_candles(args)
    result = Backtester(cfg).run(candles)
    print(result.summary())
    print()
    print(monte_carlo(result.trades, cfg.initial_equity,
                      runs=args.mc_runs).summary())
    return 0


def cmd_config(args) -> int:
    cfg = load_config(args.config, args.set)
    if args.out:
        cfg.save(args.out)
        print(f"[config] wrote {args.out}")
    else:
        print(cfg.dumps())
    return 0


# ---------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("smcbot", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, data: bool = True):
        sp.add_argument("--config", help="JSON config file")
        sp.add_argument("--set", nargs="*", metavar="key.path=value",
                        help="config overrides, e.g. risk.min_rr=2.0")
        sp.add_argument("-v", "--verbose", action="store_true")
        if data:
            sp.add_argument("--csv", help="M1 candles CSV")
            sp.add_argument("--synthetic", action="store_true")
            sp.add_argument("--days", type=int, default=60)
            sp.add_argument("--seed", type=int, default=42)
            sp.add_argument("--limit", type=int)

    sp = sub.add_parser("fetch", help="download Binance M1 klines")
    sp.add_argument("--symbol", default="BTCUSDT")
    sp.add_argument("--interval", default="1m")
    sp.add_argument("--days", type=int, default=365)
    sp.add_argument("--start")
    sp.add_argument("--end")
    sp.add_argument("--out", default="data/candles.csv")
    sp.add_argument("--resume", action="store_true",
                    help="continue an interrupted download instead of restarting")
    sp.add_argument("--sleep", type=float, default=0.25,
                    help="seconds between requests (rate-limit pacing)")
    sp.add_argument("--max-candles", type=int, default=2_000_000)
    sp.set_defaults(func=cmd_fetch)

    sp = sub.add_parser("backtest", help="run the backtest")
    common(sp)
    sp.add_argument("--ml", action="store_true",
                    help="enable the leak-free walk-forward ML filter")
    sp.add_argument("--breakdown", action="store_true")
    sp.add_argument("--rejections", action="store_true")
    sp.add_argument("--json", help="write metrics JSON here")
    sp.add_argument("--journal", help="SQLite journal path")
    sp.set_defaults(func=cmd_backtest)

    sp = sub.add_parser("dataset", help="build the labelled ML dataset")
    common(sp)
    sp.add_argument("--out")
    sp.set_defaults(func=cmd_dataset)

    sp = sub.add_parser("walkforward", help="time-series ML validation")
    common(sp)
    sp.set_defaults(func=cmd_walkforward)

    sp = sub.add_parser("train", help="train and save the production model")
    common(sp)
    sp.add_argument("--out", default="models/ml_v1.json")
    sp.add_argument("--journal")
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("pipeline", help="backtest -> OOS -> walk-forward -> MC")
    common(sp)
    sp.add_argument("--split", type=float, default=0.7)
    sp.add_argument("--mc-runs", type=int, default=2000)
    sp.add_argument("--journal")
    sp.set_defaults(func=cmd_pipeline)

    sp = sub.add_parser("montecarlo", help="Monte Carlo on backtest trades")
    common(sp)
    sp.add_argument("--mc-runs", type=int, default=2000)
    sp.set_defaults(func=cmd_montecarlo)

    sp = sub.add_parser("dashboard", help="replay and print the dashboard")
    common(sp)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_dashboard)

    sp = sub.add_parser("config", help="print or write the default config")
    common(sp, data=False)
    sp.add_argument("--out")
    sp.set_defaults(func=cmd_config)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
