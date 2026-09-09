"""Optional LightGBM / scikit-learn wrappers with the built-in model's API."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence


class _Base:
    def __init__(self, cfg):
        self.cfg = cfg
        self.feature_names: List[str] = []
        self.model = None
        self.fitted = False

    def importances(self) -> Dict[str, float]:
        try:
            raw = list(self.model.feature_importances_)
        except Exception:
            return {}
        total = sum(raw) or 1.0
        out = {n: round(v / total, 5) for n, v in zip(self.feature_names, raw)}
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def predict_proba_all(self, X) -> List[float]:
        return [float(p) for p in self.model.predict_proba(list(X))[:, 1]]

    def predict_proba(self, row: Sequence[float]) -> float:
        if not self.fitted:
            return 0.5
        return self.predict_proba_all([row])[0]


class SklearnModel(_Base):                     # pragma: no cover - optional dep
    def fit(self, X, y, feature_names: Optional[List[str]] = None):
        from sklearn.ensemble import GradientBoostingClassifier as SKGB
        self.feature_names = feature_names or []
        self.model = SKGB(
            n_estimators=self.cfg.n_estimators, learning_rate=self.cfg.learning_rate,
            max_depth=self.cfg.max_depth, min_samples_leaf=self.cfg.min_samples_leaf,
            subsample=self.cfg.subsample, random_state=self.cfg.seed)
        self.model.fit(list(X), list(y))
        self.fitted = True
        return self


class LightGBMModel(_Base):                    # pragma: no cover - optional dep
    def fit(self, X, y, feature_names: Optional[List[str]] = None):
        import lightgbm as lgb
        self.feature_names = feature_names or []
        self.model = lgb.LGBMClassifier(
            n_estimators=self.cfg.n_estimators, learning_rate=self.cfg.learning_rate,
            max_depth=self.cfg.max_depth, min_child_samples=self.cfg.min_samples_leaf,
            subsample=self.cfg.subsample, random_state=self.cfg.seed, verbose=-1)
        self.model.fit(list(X), list(y))
        self.fitted = True
        return self
