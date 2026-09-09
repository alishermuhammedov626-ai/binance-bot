"""Six memecoin perpetuals traded independently out of one account.

    # 1. data (each coin separately, resumable)
    python scripts/memecoin_portfolio.py --fetch --days 365
    # 2. backtest
    python scripts/memecoin_portfolio.py --data-dir data/meme

Each coin runs its own SMC engine with its own three-trades-a-day cap, so a
DOGE setup never consumes a PEPE slot. The account is shared: sizing reads
portfolio equity and a portfolio-wide daily loss limit stops every engine at
once. Results are reported per coin and for the portfolio, then the coins are
ranked -- coins that fail are shown, not hidden.

TRAIN / VALIDATION / OOS are split chronologically and nothing is selected on
the OOS block.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from typing import Dict, List, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smcbot.backtest.portfolio import PortfolioBacktester
from smcbot.config import Config
from smcbot.core.types import Side, Trade
from smcbot.data import loader
from smcbot.data.specs import resolve, spec_for

COINS = ["DOGE", "PEPE", "SHIB", "PUMP", "BONK", "WIF"]

# The strategy under test. TP1 1.00%, TP2 1.30%, structure stop clamped to
# 0.30-0.60%, trailing after +0.5R on M1 swings, three trades a day per coin.
MODEL = {
    "risk.tp_mode": "PERCENT",
    "risk.tp_percent_levels": [1.00, 1.30],
    "risk.partial_tp": {"tp1": 0.5, "tp2": 0.5, "tp3": 0.0},
    "risk.stop_mode": "ATR_CLAMPED",
    "risk.sl_atr_multiplier": 1.5,
    "risk.sl_atr_period_timeframe": "M5",
    "risk.sl_min_pct": 0.30,
    "risk.sl_max_pct": 0.60,
    "risk.trailing_enabled": True,
    "risk.trailing_mode": "STEPS",
    "risk.trailing_steps": [[0.5, "TRAIL", 0.0]],
    "risk.trailing_swing_timeframe": "M1",
    "risk.trailing_buffer_atr": 0.10,
    "risk.move_to_be_after_tp1": False,
    "risk.early_exit_on_invalidation": False,
    "risk.max_trades_per_day": 3,          # per coin
    "risk.max_trades_per_session": 0,      # the cap is daily, not per session
    "risk.leverage": 17.0,
    "risk.risk_per_trade": 0.005,
    "risk.min_risk_per_trade": 0.0025,
    "risk.max_risk_per_trade": 0.005,
    "risk.min_rr": 0.3,
    "risk.min_first_target_rr": 0.3,
    "risk.consecutive_loss_cooldowns": {"2": 120, "3": 180, "5": 360},
    # Memecoins move and spread more than majors.
    "execution.slippage_bps": 4.0,
    "filters.max_atr_pct": 0.06,           # extreme pump/dump -> no trade
    "filters.max_spread_bps": 12.0,
}
PORTFOLIO_DAILY_LOSS = 0.02


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


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


def loss_streak(trades: Sequence[Trade]) -> int:
    worst = cur = 0
    for t in trades:
        cur = cur + 1 if t.pnl < 0 else 0
        worst = max(worst, cur)
    return worst


def coin_stats(trades: List[Trade], days: float) -> dict:
    if not trades:
        return {"n": 0}
    nets = [net_r(t) for t in trades]
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl < 0]
    gw = sum(t.pnl for t in wins)
    gl = -sum(t.pnl for t in losses)
    return {
        "n": len(trades), "per_day": len(trades) / days,
        "win": len(wins) / len(trades),
        "avg_win": mean([net_r(t) for t in wins]),
        "avg_loss": mean([-net_r(t) for t in losses]),
        "exp": mean(nets), "pf": (gw / gl) if gl > 0 else float("inf"),
        "net": sum(t.pnl for t in trades),
        "fees": sum(t.fees for t in trades),
        "funding": sum(t.funding for t in trades),
        "fee_r": mean([fee_r(t) for t in trades]),
        "streak": loss_streak(trades),
        "mfe": mean([t.mfe for t in trades]), "mae": mean([t.mae for t in trades]),
        "avg_tp": mean([100 * abs(t.targets[0] - t.entry_price) / t.entry_price
                        for t in trades if t.targets]),
        "avg_sl": mean([100 * abs(t.entry_price - t.stop) / t.entry_price
                        for t in trades]),
        "hold": mean([t.holding_minutes for t in trades]),
    }


HEAD = (f"{'coin':14s} {'n':>5s} {'/day':>5s} {'win%':>6s} {'avgW':>6s} "
        f"{'avgL':>6s} {'expR':>7s} {'PF':>6s} {'net':>9s} {'fee/R':>6s} "
        f"{'TP%':>5s} {'SL%':>5s} {'strk':>4s} {'MFE':>5s} {'MAE':>6s}")


def row(name: str, s: dict) -> str:
    if not s.get("n"):
        return f"{name:14s} {'savdo yo^q':>5s}"
    pf = " inf" if s["pf"] == float("inf") else f"{s['pf']:6.3f}"
    return (f"{name:14s} {s['n']:5d} {s['per_day']:5.2f} {s['win'] * 100:5.1f}% "
            f"{s['avg_win']:6.2f} {s['avg_loss']:6.2f} {s['exp']:+7.3f} {pf} "
            f"{s['net']:+9.2f} {s['fee_r']:6.3f} {s['avg_tp']:5.2f} "
            f"{s['avg_sl']:5.2f} {s['streak']:4d} {s['mfe']:5.2f} {s['mae']:6.2f}")


def do_fetch(args) -> int:
    os.makedirs(args.data_dir, exist_ok=True)
    for coin in COINS:
        symbol = resolve(coin)
        out = os.path.join(args.data_dir, f"{symbol}_1m.csv")
        print(f"\n[fetch] {coin} -> {symbol} -> {out}")
        start = None
        if args.resume:
            last = loader.last_candle_time(out)
            if last:
                start = str(last + 60_000)
        if start is None:
            import datetime as dt
            start = (dt.datetime.now(dt.timezone.utc)
                     - dt.timedelta(days=args.days)).strftime("%Y-%m-%d")
        written = [0]
        try:
            loader.fetch_binance(
                symbol, "1m", start, None, sleep_between=0.3,
                on_batch=lambda b: written.__setitem__(
                    0, written[0] + loader.append_csv(out, b)),
                progress=lambda n, t: print(f"      {n} candles", flush=True))
            print(f"[fetch] {symbol}: {written[0]} candles")
        except Exception as exc:
            print(f"[fetch] {symbol} FAILED: {exc}")
            print("        Agar symbol mavjud bo'lmasa, uni ro'yxatdan chiqaring "
                  "yoki to'g'ri nomini bering.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--data-dir", default="data/meme")
    ap.add_argument("--coins", nargs="*", default=COINS)
    args = ap.parse_args()

    if args.fetch:
        return do_fetch(args)

    data: Dict[str, List] = {}
    per_symbol: Dict[str, dict] = {}
    for coin in args.coins:
        symbol = resolve(coin)
        path = os.path.join(args.data_dir, f"{symbol}_1m.csv")
        if not os.path.exists(path):
            print(f"[data] {symbol}: {path} yo'q -- o'tkazib yuborildi")
            continue
        candles = loader.load_csv(path)
        if len(candles) < 20_000:
            print(f"[data] {symbol}: faqat {len(candles)} candle -- juda kam, "
                  f"o'tkazib yuborildi")
            continue
        data[symbol] = candles
        per_symbol[symbol] = spec_for(symbol)
        span = (candles[-1].close_time - candles[0].open_time) / 86_400_000
        print(f"[data] {symbol}: {len(candles)} candles ({span:.0f} kun), "
              f"{len(loader.gaps(candles))} bo'shliq")
    if not data:
        print("hech qanday data topilmadi; avval --fetch bilan yuklab oling")
        return 1

    # Chronological split across the union of all timelines.
    lo = min(c[0].open_time for c in data.values())
    hi = max(c[-1].close_time for c in data.values())
    a = lo + int((hi - lo) * 0.6)
    b = lo + int((hi - lo) * 0.8)
    blocks = [("TRAIN", lo, a), ("VALIDATION", a, b), ("OOS TEST", b, hi)]

    cfg = Config().with_overrides(**MODEL)
    oos_per_coin: Dict[str, dict] = {}

    for label, start, end in blocks:
        chunk = {s: [c for c in cs if start <= c.open_time < end]
                 for s, cs in data.items()}
        chunk = {s: cs for s, cs in chunk.items() if len(cs) > 5_000}
        if not chunk:
            continue
        days = (end - start) / 86_400_000
        print(f"\n{'=' * 118}\n{label}  ({days:.0f} kun, {len(chunk)} coin)\n{'=' * 118}")
        t0 = time.time()
        pb = PortfolioBacktester(cfg, list(chunk), PORTFOLIO_DAILY_LOSS,
                                 per_symbol=per_symbol)
        res = pb.run(chunk)
        print(HEAD)
        print("-" * 118)
        for sym in sorted(chunk):
            s = coin_stats(res.by_symbol.get(sym, []), days)
            print(row(sym, s))
            if label == "OOS TEST":
                oos_per_coin[sym] = s
        print("-" * 118)
        port = coin_stats(res.trades, days)
        print(row("PORTFEL", port))
        print(f"\n  portfel max DD      : {100 * max_dd(res.equity_curve):.2f}%")
        print(f"  yakuniy equity      : {pb.equity:.2f} USDT "
              f"({cfg.initial_equity} dan)")
        print(f"  kunlik limit tegdi  : {res.halted_days} kun")
        print(f"  jami komissiya      : {port.get('fees', 0):.2f} USDT")
        print(f"  jami funding        : {port.get('funding', 0):+.2f} USDT")
        print(f"  [{time.time() - t0:.0f}s]")
        sys.stdout.flush()

    if not oos_per_coin:
        return 0
    print(f"\n{'=' * 118}\nRANKING -- OOS TEST\n{'=' * 118}")
    usable = {s: v for s, v in oos_per_coin.items() if v.get("n", 0) >= 20}
    small = {s: v for s, v in oos_per_coin.items() if 0 < v.get("n", 0) < 20}
    if usable:
        print(f"{'#':>2s} {'coin':14s} {'win%':>6s} {'expR':>7s} {'PF':>6s} "
              f"{'n':>5s} {'fee/R':>6s}")
        for i, (sym, s) in enumerate(
                sorted(usable.items(), key=lambda kv: -kv[1]["win"]), 1):
            pf = " inf" if s["pf"] == float("inf") else f"{s['pf']:6.3f}"
            print(f"{i:2d} {sym:14s} {s['win'] * 100:5.1f}% {s['exp']:+7.3f} {pf} "
                  f"{s['n']:5d} {s['fee_r']:6.3f}")
    if small:
        print(f"\nNamuna kichik (n<20), reyting qilinmadi: "
              + ", ".join(f"{s} (n={v['n']})" for s, v in small.items()))

    good = {s: v for s, v in usable.items() if v["win"] >= 0.70 and v["exp"] > 0}
    print(f"\n70%+ win rate VA musbat expectancy: "
          f"{', '.join(good) if good else 'HECH QAYSI COIN'}")
    if not good:
        print("Bu yashirilmayapti: OOS da hech bir coin ikkala shartni birga")
        print("bajarmadi. Yuqoridagi jadval haqiqiy natija.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
