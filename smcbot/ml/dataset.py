"""Sections 49, 53-54 -- training data construction.

Two ideas are kept strictly apart:

* **Features** are built live, inside the replay, from closed candles only.
* **Labels** are built afterwards by walking the *future* candles to see whether
  the target or the stop came first.  Using future data for the label is the
  definition of supervised learning; using it for a feature would be leakage.
  The two never touch: :func:`collect_setups` produces the features and stops,
  and only then does :func:`label_setups` look forward.

Setups are collected *without* portfolio constraints (no one-position rule, no
daily limits).  Otherwise the training set would only contain the setups the
bot happened to have room for, which is a selection bias that makes the model
learn the risk manager's schedule rather than the market.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from ..config import Config
from ..core.types import Candle, Setup, Side
from ..engine.context import SMCContext
from ..strategy.setup import SignalEngine
from .features import build_features

MAX_HOLD_BARS = 60 * 12          # 12h -- beyond this the idea has gone stale


@dataclass
class Sample:
    setup_id: str
    time: int
    bar_index: int
    side: str
    entry: float
    stop: float
    targets: List[float]
    entry_type: str
    smc_score: float
    features: Dict[str, float]
    meta: Dict[str, object] = field(default_factory=dict)
    # ---- filled by the labeller ----
    label: Optional[int] = None          # 1 = TP1 before SL
    filled: bool = False
    fill_price: float = 0.0
    fill_bar: int = 0
    exit_bar: int = 0
    exit_price: float = 0.0
    r_multiple: float = 0.0
    mfe: float = 0.0
    mae: float = 0.0
    tp_hits: List[int] = field(default_factory=list)
    outcome: str = ""                    # TP1 | STOP | TIMEOUT | NO_FILL

    def row(self) -> Dict[str, float]:
        return self.features


def collect_setups(cfg: Config, candles: Sequence[Candle],
                   progress: Optional[Callable[[int], None]] = None
                   ) -> List[Sample]:
    """Replay the market and record every setup that reaches ENTRY."""
    ctx = SMCContext(cfg)
    sig = SignalEngine(cfg)
    out: List[Sample] = []
    for i, candle in enumerate(candles):
        ctx.on_m1(candle)
        for setup in sig.on_bar(ctx):
            feats = build_features(ctx, setup)
            out.append(Sample(
                setup_id=setup.setup_id, time=ctx.now, bar_index=i,
                side=setup.side.value, entry=setup.entry, stop=setup.stop,
                targets=[t.price for t in setup.targets],
                entry_type=setup.entry_type, smc_score=setup.smc_score,
                features=feats,
                meta={k: v for k, v in setup.meta.items() if k != "evidence"},
            ))
            # De-duplicate exactly as the live bot does (section 70).
            sig.mark_traded(setup.setup_id)
        if progress and i and i % 20000 == 0:
            progress(i)
    return out


def label_setups(samples: List[Sample], candles: Sequence[Candle],
                 cfg: Config, max_hold_bars: int = MAX_HOLD_BARS) -> List[Sample]:
    """Walk forward from each setup and record what actually happened.

    Intrabar order is unknowable, so a bar that spans both the stop and a target
    is resolved as a stop -- the same pessimism the backtester uses, which keeps
    labels and live results consistent.
    """
    slip = cfg.execution.slippage_bps / 10_000.0
    ttl = cfg.execution.limit_order_ttl_bars

    for s in samples:
        side = Side(s.side)
        sign = side.sign
        start = s.bar_index + 1              # never the signal bar itself
        fill_price = None
        fill_bar = 0

        for j in range(start, min(start + ttl + 1, len(candles))):
            c = candles[j]
            if s.entry_type == "MARKET":
                fill_price = c.open * (1 + sign * slip)
                fill_bar = j
                break
            touched = (c.low <= s.entry) if side is Side.BUY else (c.high >= s.entry)
            if touched:
                if side is Side.BUY:
                    fill_price = min(c.open, s.entry)
                else:
                    fill_price = max(c.open, s.entry)
                fill_bar = j
                break

        if fill_price is None:
            s.outcome = "NO_FILL"
            s.label = None
            continue

        s.filled = True
        s.fill_price = fill_price
        s.fill_bar = fill_bar
        risk = abs(fill_price - s.stop)
        if risk <= 0:
            s.outcome = "NO_FILL"
            s.label = None
            continue

        mfe = mae = 0.0
        resolved = False
        for j in range(fill_bar, min(fill_bar + max_hold_bars, len(candles))):
            c = candles[j]
            best = c.high if side is Side.BUY else c.low
            worst = c.low if side is Side.BUY else c.high
            mfe = max(mfe, (best - fill_price) * sign / risk)
            mae = min(mae, (worst - fill_price) * sign / risk)

            stop_hit = (c.low <= s.stop) if side is Side.BUY else (c.high >= s.stop)
            if stop_hit:
                s.outcome, s.label = "STOP", 0
                s.exit_bar, s.exit_price = j, s.stop
                s.r_multiple = -1.0
                resolved = True
                break
            for k, tp in enumerate(s.targets):
                hit = (c.high >= tp) if side is Side.BUY else (c.low <= tp)
                if hit and k not in s.tp_hits:
                    s.tp_hits.append(k)
            if 0 in s.tp_hits:
                s.outcome, s.label = "TP1", 1
                s.exit_bar, s.exit_price = j, s.targets[0]
                s.r_multiple = (s.targets[0] - fill_price) * sign / risk
                resolved = True
                break

        if not resolved:
            last = candles[min(fill_bar + max_hold_bars, len(candles)) - 1]
            s.outcome = "TIMEOUT"
            s.exit_bar = min(fill_bar + max_hold_bars, len(candles)) - 1
            s.exit_price = last.close
            s.r_multiple = (last.close - fill_price) * sign / risk
            # A trade that never reached its first target is not a win.
            s.label = 1 if s.r_multiple > 0 else 0
        s.mfe = round(mfe, 4)
        s.mae = round(mae, 4)
    return samples


def build_dataset(cfg: Config, candles: Sequence[Candle],
                  progress: Optional[Callable[[int], None]] = None) -> List[Sample]:
    samples = collect_setups(cfg, candles, progress)
    return label_setups(samples, candles, cfg)


def labelled(samples: Sequence[Sample]) -> List[Sample]:
    out = [s for s in samples if s.label is not None]
    out.sort(key=lambda s: s.time)        # chronological -- required by section 53
    return out


def dataset_summary(samples: Sequence[Sample]) -> dict:
    lab = labelled(samples)
    n = len(lab)
    wins = sum(1 for s in lab if s.label == 1)
    outcomes: Dict[str, int] = {}
    for s in samples:
        outcomes[s.outcome] = outcomes.get(s.outcome, 0) + 1
    return {
        "collected": len(samples),
        "labelled": n,
        "win_rate": round(wins / n, 4) if n else 0.0,
        "outcomes": outcomes,
        "avg_r": round(sum(s.r_multiple for s in lab) / n, 4) if n else 0.0,
        "first": lab[0].time if lab else None,
        "last": lab[-1].time if lab else None,
    }
