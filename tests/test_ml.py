"""ML tests: model correctness, chronological validation, and leak-free scoring."""
from __future__ import annotations

import json
import random
import unittest

from smcbot.config import Config, MLConfig
from smcbot.ml.dataset import Sample, label_setups, labelled
from smcbot.ml.features import feature_names, vectorise
from smcbot.ml.model import GradientBoostingClassifier, sigmoid
from smcbot.ml.walkforward import (FoldResult, WalkForwardPredictor, brier,
                                   log_loss, roc_auc, threshold_report,
                                   time_series_folds)
from smcbot.core.types import Candle

MIN = 60_000


class TestModel(unittest.TestCase):
    def _data(self, n=500, seed=0):
        rng = random.Random(seed)
        X, y = [], []
        for _ in range(n):
            a, b, c, noise = (rng.uniform(0, 1) for _ in range(4))
            X.append([a, b, c, noise])
            y.append(1 if a + b - c + rng.gauss(0, 0.1) > 0.8 else 0)
        return X, y

    def test_model_learns_a_separable_signal(self):
        X, y = self._data()
        model = GradientBoostingClassifier(n_estimators=60, seed=1)
        model.fit(X[:400], y[:400], ["a", "b", "c", "noise"])
        correct = sum(int((model.predict_proba(x) >= 0.5) == bool(t))
                      for x, t in zip(X[400:], y[400:]))
        self.assertGreater(correct / 100, 0.75)

    def test_informative_features_outrank_noise(self):
        X, y = self._data()
        model = GradientBoostingClassifier(n_estimators=60, seed=1)
        model.fit(X, y, ["a", "b", "c", "noise"])
        imp = model.importances()
        self.assertLess(imp.get("noise", 0.0), imp.get("a", 0.0))

    def test_untrained_model_is_neutral(self):
        self.assertEqual(GradientBoostingClassifier().predict_proba([1, 2, 3]), 0.5)

    def test_probabilities_stay_in_range(self):
        X, y = self._data(200, seed=3)
        model = GradientBoostingClassifier(n_estimators=40, seed=2).fit(X, y)
        for x in X:
            p = model.predict_proba(x)
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_serialisation_is_exact(self):
        X, y = self._data(200, seed=4)
        model = GradientBoostingClassifier(n_estimators=30, seed=5).fit(X, y)
        clone = GradientBoostingClassifier.from_dict(
            json.loads(json.dumps(model.to_dict())))
        for x in X[:50]:
            self.assertEqual(model.predict_proba(x), clone.predict_proba(x))

    def test_single_class_training_is_degenerate_but_safe(self):
        X = [[random.random() for _ in range(3)] for _ in range(50)]
        model = GradientBoostingClassifier(n_estimators=10, seed=1).fit(X, [1] * 50)
        self.assertGreater(model.predict_proba(X[0]), 0.9)

    def test_sigmoid_is_stable_at_extremes(self):
        self.assertAlmostEqual(sigmoid(0), 0.5)
        self.assertAlmostEqual(sigmoid(1000), 1.0)
        self.assertAlmostEqual(sigmoid(-1000), 0.0)


class TestValidationMetrics(unittest.TestCase):
    def test_auc(self):
        self.assertEqual(roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(roc_auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]), 0.0)
        self.assertEqual(roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]), 0.5)
        self.assertEqual(roc_auc([1, 1], [0.5, 0.6]), 0.5)     # single class

    def test_log_loss_and_brier(self):
        self.assertLess(log_loss([1, 0], [0.9, 0.1]), log_loss([1, 0], [0.5, 0.5]))
        self.assertAlmostEqual(brier([1, 0], [1.0, 0.0]), 0.0)

    def test_threshold_report_lift(self):
        y = [1, 1, 1, 0, 0, 0]
        p = [0.9, 0.8, 0.7, 0.2, 0.1, 0.05]
        rep = threshold_report(y, p, thresholds=(0.5,))
        self.assertEqual(rep["0.50"]["taken"], 3)
        self.assertAlmostEqual(rep["0.50"]["win_rate"], 1.0)


class TestChronologicalSplitting(unittest.TestCase):
    """Section 53: folds must be ordered in time, never shuffled."""

    def test_folds_are_contiguous_and_ordered(self):
        folds = time_series_folds(100, 4, embargo=0)
        self.assertEqual(len(folds), 4)
        for train, test in folds:
            self.assertEqual(train, list(range(len(train))))
            self.assertEqual(test, list(range(test[0], test[-1] + 1)))
            self.assertLess(max(train), min(test),
                            "training data must precede test data")

    def test_embargo_removes_the_training_tail(self):
        no_gap = time_series_folds(100, 4, embargo=0)
        gapped = time_series_folds(100, 4, embargo=5)
        for (tr_a, te_a), (tr_b, te_b) in zip(no_gap, gapped):
            self.assertEqual(te_a, te_b)
            self.assertEqual(len(tr_b), len(tr_a) - 5)
            self.assertLess(max(tr_b), min(te_b) - 1)

    def test_too_little_data_yields_no_folds(self):
        self.assertEqual(time_series_folds(3, 4, 0), [])


class TestWalkForwardPredictor(unittest.TestCase):
    """A fold's model must never score a bar it was trained on."""

    class FakeModel:
        def __init__(self, tag):
            self.tag = tag

        def predict_proba(self, row):
            return self.tag

    def _result(self):
        class R:
            feature_names = ["a"]
            folds = [
                FoldResult(1, 10, 10, train_end_time=1_000, test_start_time=2_000,
                           test_end_time=3_000, auc=0.6, accuracy=0.6,
                           log_loss=0.6, brier=0.2, base_rate=0.5,
                           model=TestWalkForwardPredictor.FakeModel(0.11)),
                FoldResult(2, 20, 10, train_end_time=5_000, test_start_time=6_000,
                           test_end_time=7_000, auc=0.6, accuracy=0.6,
                           log_loss=0.6, brier=0.2, base_rate=0.5,
                           model=TestWalkForwardPredictor.FakeModel(0.22)),
            ]
        return R()

    def test_no_model_before_the_first_training_window_ends(self):
        p = WalkForwardPredictor(self._result(), embargo_ms=0)
        self.assertIsNone(p({"_time": 500, "a": 1.0}))
        self.assertEqual(p.no_model_calls, 1)

    def test_uses_the_latest_model_trained_strictly_before_the_bar(self):
        p = WalkForwardPredictor(self._result(), embargo_ms=0)
        self.assertEqual(p({"_time": 1_500, "a": 1.0}), 0.11)
        self.assertEqual(p({"_time": 4_000, "a": 1.0}), 0.11)
        self.assertEqual(p({"_time": 6_000, "a": 1.0}), 0.22)

    def test_embargo_delays_model_availability(self):
        p = WalkForwardPredictor(self._result(), embargo_ms=2_000)
        self.assertIsNone(p({"_time": 1_500, "a": 1.0}))
        self.assertEqual(p({"_time": 3_500, "a": 1.0}), 0.11)


class TestFeatureHygiene(unittest.TestCase):
    def test_metadata_keys_are_excluded_from_the_model_input(self):
        rows = [{"_time": 1, "_side": 1, "a": 2.0, "b": 3.0}]
        names = feature_names(rows)
        self.assertEqual(names, ["a", "b"])
        self.assertEqual(vectorise(rows[0], names), [2.0, 3.0])

    def test_missing_features_default_to_zero(self):
        self.assertEqual(vectorise({"a": 1.0}, ["a", "b"]), [1.0, 0.0])


class TestLabelling(unittest.TestCase):
    """Section 49: TP before SL = WIN, resolved pessimistically."""

    def _candles(self, rows):
        return [Candle(i * MIN, (i + 1) * MIN, o, h, l, c, 1.0)
                for i, (o, h, l, c) in enumerate(rows)]

    def _sample(self, **kw):
        base = dict(setup_id="s", time=0, bar_index=0, side="BUY", entry=100.0,
                    stop=99.0, targets=[102.0], entry_type="MARKET",
                    smc_score=80.0, features={"_time": 0, "x": 1.0})
        base.update(kw)
        return Sample(**base)

    def test_target_first_is_a_win(self):
        candles = self._candles([(100, 100, 100, 100), (100, 100.5, 99.5, 100),
                                 (100, 103, 99.8, 102.5)])
        s = label_setups([self._sample()], candles, Config())[0]
        self.assertEqual(s.outcome, "TP1")
        self.assertEqual(s.label, 1)

    def test_stop_first_is_a_loss(self):
        candles = self._candles([(100, 100, 100, 100), (100, 100.5, 99.5, 100),
                                 (100, 100.5, 98.0, 98.5)])
        s = label_setups([self._sample()], candles, Config())[0]
        self.assertEqual(s.outcome, "STOP")
        self.assertEqual(s.label, 0)

    def test_bar_hitting_both_is_labelled_a_loss(self):
        """Same pessimism as the backtester -- otherwise labels flatter live results."""
        candles = self._candles([(100, 100, 100, 100), (100, 100.5, 99.5, 100),
                                 (100, 105, 97, 101)])
        s = label_setups([self._sample()], candles, Config())[0]
        self.assertEqual(s.outcome, "STOP")
        self.assertEqual(s.label, 0)

    def test_unfilled_limit_order_is_not_labelled(self):
        candles = self._candles([(100, 100.2, 99.9, 100)] * 30)
        s = label_setups([self._sample(entry_type="LIMIT", entry=90.0,
                                       stop=89.0)], candles, Config())[0]
        self.assertEqual(s.outcome, "NO_FILL")
        self.assertIsNone(s.label)
        self.assertEqual(labelled([s]), [])

    def test_timeout_is_labelled_by_sign(self):
        candles = self._candles([(100, 100.2, 99.8, 100)] * 40)
        s = label_setups([self._sample()], candles, Config(), max_hold_bars=20)[0]
        self.assertEqual(s.outcome, "TIMEOUT")
        self.assertIn(s.label, (0, 1))

    def test_labels_are_sorted_chronologically(self):
        a = self._sample(time=200)
        b = self._sample(time=100)
        a.label = b.label = 1
        self.assertEqual([s.time for s in labelled([a, b])], [100, 200])


if __name__ == "__main__":
    unittest.main(verbosity=2)
