"""Section 50 -- the quality filter model.

A histogram gradient-boosted tree ensemble with logistic loss and Newton leaf
values (the XGBoost formulation), implemented in pure Python so the bot has no
hard dependency.  If LightGBM or scikit-learn happen to be installed,
:func:`make_model` prefers them -- the interface is identical either way.

Tree models are the right first choice here (section 50): the SMC features are
tabular, mixed-scale and full of interactions, and there is nowhere near enough
data for a neural network.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

MAX_BINS = 32


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class _Node:
    feature: int = -1
    threshold_bin: int = -1
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None
    value: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return self.left is None

    def to_dict(self) -> dict:
        if self.is_leaf:
            # Not rounded: a saved model must reproduce the live model exactly.
            return {"v": self.value}
        return {"f": self.feature, "t": self.threshold_bin,
                "l": self.left.to_dict(), "r": self.right.to_dict()}

    @staticmethod
    def from_dict(d: dict) -> "_Node":
        if "v" in d:
            return _Node(value=d["v"])
        return _Node(feature=d["f"], threshold_bin=d["t"],
                     left=_Node.from_dict(d["l"]), right=_Node.from_dict(d["r"]))


class GradientBoostingClassifier:
    """Binary classifier: P(win | features)."""

    def __init__(self, n_estimators: int = 120, learning_rate: float = 0.06,
                 max_depth: int = 3, min_samples_leaf: int = 8,
                 subsample: float = 0.85, l2: float = 1.0, seed: int = 7):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.l2 = l2
        self.seed = seed
        self.trees: List[_Node] = []
        self.base_score = 0.0
        self.bin_edges: List[List[float]] = []
        self.feature_names: List[str] = []
        self.gain: Dict[int, float] = {}
        self.n_features = 0
        self.fitted = False

    # ---- binning ------------------------------------------------------
    def _fit_bins(self, X: Sequence[Sequence[float]]) -> None:
        self.bin_edges = []
        for j in range(self.n_features):
            col = sorted({row[j] for row in X})
            if len(col) <= MAX_BINS:
                edges = col[1:]                      # split points between values
            else:
                step = len(col) / MAX_BINS
                edges = [col[min(int(step * (k + 1)), len(col) - 1)]
                         for k in range(MAX_BINS - 1)]
                edges = sorted(set(edges))
            self.bin_edges.append(edges)

    def _bin_row(self, row: Sequence[float]) -> List[int]:
        out = []
        for j, edges in enumerate(self.bin_edges):
            v = row[j] if j < len(row) else 0.0
            lo, hi = 0, len(edges)
            while lo < hi:                            # bisect_left
                mid = (lo + hi) // 2
                if edges[mid] <= v:
                    lo = mid + 1
                else:
                    hi = mid
            out.append(lo)
        return out

    # ---- training -----------------------------------------------------
    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int],
            feature_names: Optional[List[str]] = None) -> "GradientBoostingClassifier":
        n = len(X)
        if n == 0:
            raise ValueError("empty training set")
        self.n_features = len(X[0])
        self.feature_names = feature_names or [f"f{i}" for i in range(self.n_features)]
        self._fit_bins(X)
        Xb = [self._bin_row(r) for r in X]

        pos = sum(1 for v in y if v == 1)
        rate = min(max(pos / n, 1e-6), 1 - 1e-6)
        self.base_score = math.log(rate / (1 - rate))
        scores = [self.base_score] * n
        rng = random.Random(self.seed)
        self.trees = []
        self.gain = {}

        for _ in range(self.n_estimators):
            probs = [sigmoid(s) for s in scores]
            grad = [p - yi for p, yi in zip(probs, y)]        # dL/dscore
            hess = [max(p * (1 - p), 1e-6) for p in probs]

            idx = list(range(n))
            if 0 < self.subsample < 1.0:
                k = max(self.min_samples_leaf * 2, int(n * self.subsample))
                idx = rng.sample(idx, min(k, n))

            tree = self._build(Xb, grad, hess, idx, depth=0)
            self.trees.append(tree)
            for i in range(n):
                scores[i] += self.learning_rate * self._predict_node(tree, Xb[i])
        self.fitted = True
        return self

    def _build(self, Xb, grad, hess, idx, depth: int) -> _Node:
        G = sum(grad[i] for i in idx)
        H = sum(hess[i] for i in idx)
        leaf_value = -G / (H + self.l2)
        if depth >= self.max_depth or len(idx) < 2 * self.min_samples_leaf:
            return _Node(value=leaf_value)

        parent_score = (G * G) / (H + self.l2)
        best = (0.0, -1, -1)          # gain, feature, bin threshold
        for j in range(self.n_features):
            nbins = len(self.bin_edges[j]) + 1
            if nbins < 2:
                continue
            gh = [[0.0, 0.0, 0] for _ in range(nbins)]
            for i in idx:
                b = Xb[i][j]
                cell = gh[b]
                cell[0] += grad[i]
                cell[1] += hess[i]
                cell[2] += 1
            gl = hl = 0.0
            cl = 0
            for b in range(nbins - 1):
                gl += gh[b][0]
                hl += gh[b][1]
                cl += gh[b][2]
                cr = len(idx) - cl
                if cl < self.min_samples_leaf or cr < self.min_samples_leaf:
                    continue
                gr, hr = G - gl, H - hl
                gain = (gl * gl) / (hl + self.l2) + (gr * gr) / (hr + self.l2) \
                    - parent_score
                if gain > best[0]:
                    best = (gain, j, b)

        if best[1] < 0 or best[0] <= 1e-9:
            return _Node(value=leaf_value)

        gain, j, b = best
        self.gain[j] = self.gain.get(j, 0.0) + gain
        left = [i for i in idx if Xb[i][j] <= b]
        right = [i for i in idx if Xb[i][j] > b]
        if not left or not right:
            return _Node(value=leaf_value)
        return _Node(feature=j, threshold_bin=b,
                     left=self._build(Xb, grad, hess, left, depth + 1),
                     right=self._build(Xb, grad, hess, right, depth + 1))

    @staticmethod
    def _predict_node(node: _Node, row_bins: Sequence[int]) -> float:
        while not node.is_leaf:
            node = node.left if row_bins[node.feature] <= node.threshold_bin else node.right
        return node.value

    # ---- inference ----------------------------------------------------
    def decision_function(self, row: Sequence[float]) -> float:
        bins = self._bin_row(row)
        score = self.base_score
        for tree in self.trees:
            score += self.learning_rate * self._predict_node(tree, bins)
        return score

    def predict_proba(self, row: Sequence[float]) -> float:
        if not self.fitted:
            return 0.5
        return sigmoid(self.decision_function(row))

    def predict_proba_all(self, X: Sequence[Sequence[float]]) -> List[float]:
        return [self.predict_proba(r) for r in X]

    def importances(self) -> Dict[str, float]:
        total = sum(self.gain.values()) or 1.0
        out = {self.feature_names[j]: round(g / total, 5)
               for j, g in self.gain.items() if j < len(self.feature_names)}
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    # ---- persistence ---------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "kind": "gbt",
            "n_estimators": self.n_estimators, "learning_rate": self.learning_rate,
            "max_depth": self.max_depth, "min_samples_leaf": self.min_samples_leaf,
            "subsample": self.subsample, "l2": self.l2, "seed": self.seed,
            "base_score": self.base_score, "bin_edges": self.bin_edges,
            "feature_names": self.feature_names, "n_features": self.n_features,
            "trees": [t.to_dict() for t in self.trees],
            "gain": {str(k): v for k, v in self.gain.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GradientBoostingClassifier":
        m = cls(d["n_estimators"], d["learning_rate"], d["max_depth"],
                d["min_samples_leaf"], d["subsample"], d.get("l2", 1.0), d["seed"])
        m.base_score = d["base_score"]
        m.bin_edges = d["bin_edges"]
        m.feature_names = d["feature_names"]
        m.n_features = d["n_features"]
        m.trees = [_Node.from_dict(t) for t in d["trees"]]
        m.gain = {int(k): v for k, v in d.get("gain", {}).items()}
        m.fitted = True
        return m

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh)

    @classmethod
    def load(cls, path: str) -> "GradientBoostingClassifier":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def make_model(cfg) -> GradientBoostingClassifier:
    """Factory honouring :class:`MLConfig`.

    LightGBM / scikit-learn are used when available; otherwise the built-in
    implementation runs.  Both expose ``fit`` / ``predict_proba(row)``.
    """
    try:                                       # pragma: no cover - optional dep
        import lightgbm  # noqa: F401
        from .backends import LightGBMModel
        return LightGBMModel(cfg)
    except Exception:
        pass
    try:                                       # pragma: no cover - optional dep
        import sklearn  # noqa: F401
        from .backends import SklearnModel
        return SklearnModel(cfg)
    except Exception:
        pass
    return GradientBoostingClassifier(
        n_estimators=cfg.n_estimators, learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth, min_samples_leaf=cfg.min_samples_leaf,
        subsample=cfg.subsample, seed=cfg.seed,
    )
