"""Sections 53-54 -- time-series validation, walk-forward, and leak-free scoring.

Random train/test shuffling is explicitly forbidden: it lets the model learn
from tomorrow to trade today.  Folds here are contiguous and ordered, with an
**embargo** gap between train and test so a setup that was still open when the
training window ended cannot influence the test window.

:class:`WalkForwardPredictor` is what makes an ML-enabled backtest honest.  It
holds one model per fold and, when asked for a probability at time *t*, serves
the model whose training data ended before *t* -- never the one that has seen
*t*.  Before the first fold it returns ``None``, and the decision engine then
falls back to the SMC score alone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..config import Config
from .dataset import Sample, labelled
from .features import feature_names, vectorise, vectorise_all
from .model import make_model

MINUTE = 60_000


# ---------------------------------------------------------------- metrics
def roc_auc(y: Sequence[int], p: Sequence[float]) -> float:
    pairs = sorted(zip(p, y))
    pos = sum(y)
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return 0.5
    # Rank-sum (Mann-Whitney) with average ranks for ties.
    ranks: List[float] = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rank_sum = sum(r for r, (_, yi) in zip(ranks, pairs) if yi == 1)
    return round((rank_sum - pos * (pos + 1) / 2.0) / (pos * neg), 4)


def log_loss(y: Sequence[int], p: Sequence[float]) -> float:
    eps = 1e-12
    return round(-sum(yi * math.log(max(pi, eps)) + (1 - yi) * math.log(max(1 - pi, eps))
                      for yi, pi in zip(y, p)) / max(len(y), 1), 4)


def brier(y: Sequence[int], p: Sequence[float]) -> float:
    return round(sum((pi - yi) ** 2 for yi, pi in zip(y, p)) / max(len(y), 1), 4)


def threshold_report(y: Sequence[int], p: Sequence[float],
                     thresholds: Sequence[float] = (0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7)
                     ) -> Dict[str, dict]:
    out = {}
    base = sum(y) / len(y) if y else 0.0
    for t in thresholds:
        taken = [(yi, pi) for yi, pi in zip(y, p) if pi >= t]
        n = len(taken)
        wins = sum(yi for yi, _ in taken)
        out[f"{t:.2f}"] = {
            "taken": n,
            "share": round(n / len(y), 4) if y else 0.0,
            "win_rate": round(wins / n, 4) if n else 0.0,
            "lift": round((wins / n) - base, 4) if n else 0.0,
        }
    return out


# ---------------------------------------------------------------- folds
def time_series_folds(n: int, n_folds: int, embargo: int = 0
                      ) -> List[Tuple[List[int], List[int]]]:
    """Expanding-window folds: train on [0, k), test on the next block."""
    folds: List[Tuple[List[int], List[int]]] = []
    if n_folds < 1 or n < n_folds + 1:
        return folds
    block = n // (n_folds + 1)
    if block == 0:
        return folds
    for k in range(1, n_folds + 1):
        train_end = block * k
        test_end = block * (k + 1) if k < n_folds else n
        train = list(range(0, max(train_end - embargo, 0)))
        test = list(range(train_end, test_end))
        if len(train) >= 2 and test:
            folds.append((train, test))
    return folds


@dataclass
class FoldResult:
    index: int
    train_size: int
    test_size: int
    train_end_time: int
    test_start_time: int
    test_end_time: int
    auc: float
    accuracy: float
    log_loss: float
    brier: float
    base_rate: float
    model: object = None
    predictions: List[Tuple[int, float, int]] = field(default_factory=list)  # ts, p, y


@dataclass
class WalkForwardResult:
    folds: List[FoldResult]
    feature_names: List[str]
    oos_auc: float
    oos_log_loss: float
    oos_brier: float
    oos_accuracy: float
    base_rate: float
    thresholds: Dict[str, dict]
    importances: Dict[str, float]

    def summary(self) -> str:
        lines = [f"Walk-forward folds : {len(self.folds)}",
                 f"Out-of-sample AUC  : {self.oos_auc}  "
                 f"(0.5 = no skill, base rate {self.base_rate})",
                 f"OOS log loss/Brier : {self.oos_log_loss} / {self.oos_brier}",
                 f"OOS accuracy       : {self.oos_accuracy}"]
        for f in self.folds:
            lines.append(f"  fold {f.index}: train {f.train_size:4d} "
                         f"test {f.test_size:4d}  AUC {f.auc}  acc {f.accuracy}")
        lines.append("Threshold sweep (out-of-sample):")
        for t, r in self.thresholds.items():
            lines.append(f"  p>={t}: taken {r['taken']:4d} ({r['share']:.0%})  "
                         f"win {r['win_rate']:.2%}  lift {r['lift']:+.4f}")
        top = list(self.importances.items())[:10]
        if top:
            lines.append("Top features: " +
                         ", ".join(f"{k}={v}" for k, v in top))
        return "\n".join(lines)


def walk_forward(cfg: Config, samples: Sequence[Sample]
                 ) -> Optional[WalkForwardResult]:
    """Train/evaluate in strict chronological order (section 53)."""
    data = labelled(samples)
    if len(data) < max(cfg.ml.min_train_samples, 20):
        return None
    names = feature_names([s.features for s in data])
    X = vectorise_all([s.features for s in data], names)
    y = [int(s.label) for s in data]

    folds = time_series_folds(len(data), cfg.ml.n_folds, cfg.ml.embargo_samples)
    if not folds:
        return None

    results: List[FoldResult] = []
    all_y: List[int] = []
    all_p: List[float] = []
    importances: Dict[str, float] = {}

    for i, (train_idx, test_idx) in enumerate(folds, start=1):
        ytr = [y[j] for j in train_idx]
        if len(set(ytr)) < 2:
            continue                       # a single-class window teaches nothing
        model = make_model(cfg.ml)
        model.fit([X[j] for j in train_idx], ytr, names)
        probs = [model.predict_proba(X[j]) for j in test_idx]
        yte = [y[j] for j in test_idx]
        acc = sum(int((p >= 0.5) == bool(t)) for p, t in zip(probs, yte)) / len(yte)
        results.append(FoldResult(
            index=i, train_size=len(train_idx), test_size=len(test_idx),
            train_end_time=data[train_idx[-1]].time,
            test_start_time=data[test_idx[0]].time,
            test_end_time=data[test_idx[-1]].time,
            auc=roc_auc(yte, probs), accuracy=round(acc, 4),
            log_loss=log_loss(yte, probs), brier=brier(yte, probs),
            base_rate=round(sum(yte) / len(yte), 4), model=model,
            predictions=[(data[j].time, p, y[j]) for j, p in zip(test_idx, probs)],
        ))
        all_y.extend(yte)
        all_p.extend(probs)
        for k, v in (model.importances() or {}).items():
            importances[k] = importances.get(k, 0.0) + v

    if not results:
        return None
    total = sum(importances.values()) or 1.0
    importances = dict(sorted(((k, round(v / total, 5))
                               for k, v in importances.items()),
                              key=lambda kv: -kv[1]))
    acc = sum(int((p >= 0.5) == bool(t)) for p, t in zip(all_p, all_y)) / len(all_y)
    return WalkForwardResult(
        folds=results, feature_names=names,
        oos_auc=roc_auc(all_y, all_p), oos_log_loss=log_loss(all_y, all_p),
        oos_brier=brier(all_y, all_p), oos_accuracy=round(acc, 4),
        base_rate=round(sum(all_y) / len(all_y), 4),
        thresholds=threshold_report(all_y, all_p), importances=importances,
    )


class WalkForwardPredictor:
    """Serves, for any timestamp, only a model that was trained before it.

    This is what allows an ML-enabled backtest to remain out-of-sample: at bar
    *t* the bot is scored by the model it could actually have had at bar *t*.
    """

    def __init__(self, result: WalkForwardResult, embargo_ms: int = 0):
        self.names = result.feature_names
        self.embargo_ms = embargo_ms
        # (usable_from, model), ordered; a fold's model is usable only after the
        # end of its own training data (plus embargo).
        self.segments: List[Tuple[int, object]] = sorted(
            (f.train_end_time + embargo_ms, f.model) for f in result.folds)
        self.no_model_calls = 0
        self.calls = 0

    def model_for(self, ts: int) -> Optional[object]:
        chosen = None
        for usable_from, model in self.segments:
            if usable_from <= ts:
                chosen = model
            else:
                break
        return chosen

    def __call__(self, features: Dict[str, float]) -> Optional[float]:
        self.calls += 1
        ts = int(features.get("_time", 0))
        model = self.model_for(ts)
        if model is None:
            self.no_model_calls += 1
            return None
        return model.predict_proba(vectorise(features, self.names))


def train_final(cfg: Config, samples: Sequence[Sample]) -> Optional[tuple]:
    """Train one production model on all labelled history (section 89)."""
    data = labelled(samples)
    if len(data) < cfg.ml.min_train_samples:
        return None
    names = feature_names([s.features for s in data])
    X = vectorise_all([s.features for s in data], names)
    y = [int(s.label) for s in data]
    if len(set(y)) < 2:
        return None
    model = make_model(cfg.ml)
    model.fit(X, y, names)
    return model, names
