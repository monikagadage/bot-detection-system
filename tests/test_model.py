"""Logistic regression: it learns, it round-trips, feedback moves it."""

from __future__ import annotations

from botshield.features import FEATURE_NAMES
from botshield.model import LogisticRegression


def test_learns_a_linearly_separable_split():
    # x0 high -> bot, x0 low -> human; other features are noise-free zeros.
    X = [[0.0] * 3 for _ in range(20)] + [[5.0, 0.0, 0.0] for _ in range(20)]
    y = [0] * 20 + [1] * 20
    model = LogisticRegression(3).fit(X, y, epochs=300)
    preds = [1 if model.predict_proba(x) >= 0.5 else 0 for x in X]
    assert preds == y


def test_seed_model_fits_seed_data(system):
    preds = [1 if system.model.predict_proba(x) >= 0.5 else 0 for x in system.seed_X]
    acc = sum(p == t for p, t in zip(preds, system.seed_y)) / len(system.seed_y)
    assert acc > 0.9


def test_json_round_trip_preserves_predictions(system):
    clone = LogisticRegression.from_json(system.model.to_json())
    for x in system.seed_X[:50]:
        assert abs(clone.predict_proba(x) - system.model.predict_proba(x)) < 1e-12


def test_partial_fit_moves_probability_toward_label(fresh_model):
    x = [0.0] * len(FEATURE_NAMES)
    x[FEATURE_NAMES.index("ua_bot_token")] = 1.0
    before = fresh_model.predict_proba(x)
    for _ in range(20):
        fresh_model.partial_fit(x, 1)
    after = fresh_model.predict_proba(x)
    assert after > before


def test_weights_report_is_sorted_by_magnitude(system):
    report = system.model.weights_report(FEATURE_NAMES)
    magnitudes = [abs(w) for _name, w in report]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_fit_is_deterministic_given_data():
    X = [[1.0, 2.0], [3.0, 0.5], [0.2, 4.0], [5.0, 1.0]]
    y = [0, 1, 0, 1]
    a = LogisticRegression(2).fit(X, y, epochs=100)
    b = LogisticRegression(2).fit(X, y, epochs=100)
    assert a.w == b.w and a.b == b.b
