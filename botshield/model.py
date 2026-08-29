"""
The learned scorer — logistic regression, from scratch, no numpy.

Logistic regression is the simplest useful classifier: it learns one weight
per feature plus a bias, adds them up (a linear score `z`), and squashes the
result through the sigmoid into a probability in (0, 1).

    z = b + w0*x0 + w1*x1 + ... + wn*xn
    p(bot) = sigmoid(z) = 1 / (1 + e^-z)

Training nudges the weights to make `p` closer to the true label (0 = human,
1 = bot) for every example, using gradient descent. The gradient of the
log-loss for one example turns out to be beautifully simple:

    error = p - y
    dL/dw_j = error * x_j
    dL/db   = error

Real bot-detection models are usually gradient-boosted trees (XGBoost /
LightGBM) or deep nets, and are trained on a cluster over billions of
labeled requests. The serving idea is the same: fixed weights in, a score
out, in well under a millisecond.

Features are standardized (subtract mean, divide by std) before training so
that `req_60s` (range ~0..40) doesn't dwarf `ua_missing` (0 or 1). The
scaler's mean/std are learned once from the seed set and then frozen — the
same numbers must be used at serving time.
"""

from __future__ import annotations

import json
import math
import random


class LogisticRegression:
    def __init__(self, n_features: int) -> None:
        self.n = n_features
        self.w = [0.0] * n_features
        self.b = 0.0
        self.mean = [0.0] * n_features
        self.std = [1.0] * n_features

    # ---- scaling --------------------------------------------------------

    def fit_scaler(self, X: list[list[float]]) -> None:
        rows = len(X)
        for j in range(self.n):
            col = [row[j] for row in X]
            m = sum(col) / rows
            var = sum((v - m) ** 2 for v in col) / rows
            self.mean[j] = m
            self.std[j] = math.sqrt(var) if var > 1e-12 else 1.0

    def _scale(self, x: list[float]) -> list[float]:
        return [(x[j] - self.mean[j]) / self.std[j] for j in range(self.n)]

    # ---- prediction ----------------------------------------------------

    @staticmethod
    def _sigmoid(z: float) -> float:
        if z <= -35.0:
            return 0.0
        if z >= 35.0:
            return 1.0
        return 1.0 / (1.0 + math.exp(-z))

    def _proba_scaled(self, xs: list[float]) -> float:
        z = self.b + sum(self.w[j] * xs[j] for j in range(self.n))
        return self._sigmoid(z)

    def predict_proba(self, x: list[float]) -> float:
        """p(bot) for one raw (unscaled) feature vector."""
        return self._proba_scaled(self._scale(x))

    # ---- training -----------------------------------------------------

    def fit(
        self,
        X: list[list[float]],
        y: list[int],
        epochs: int = 400,
        lr: float = 0.3,
        l2: float = 1e-3,
        fit_scaler: bool = True,
        seed: int = 0,
    ) -> "LogisticRegression":
        """Batch gradient descent over the whole training set."""
        if fit_scaler:
            self.fit_scaler(X)
        Xs = [self._scale(row) for row in X]
        m = len(Xs)
        if m == 0:
            return self
        # Reset weights so repeated calls are deterministic given the data.
        self.w = [0.0] * self.n
        self.b = 0.0
        _ = random.Random(seed)  # kept for API symmetry; batch GD needs no shuffle
        for _epoch in range(epochs):
            gw = [0.0] * self.n
            gb = 0.0
            for i in range(m):
                err = self._proba_scaled(Xs[i]) - y[i]
                xi = Xs[i]
                for j in range(self.n):
                    gw[j] += err * xi[j]
                gb += err
            for j in range(self.n):
                self.w[j] -= lr * (gw[j] / m + l2 * self.w[j])
            self.b -= lr * (gb / m)
        return self

    def partial_fit(self, x: list[float], label: int, lr: float = 0.05) -> None:
        """One online gradient step from a single labeled example.

        Used when a piece of feedback arrives and we want the model to react
        immediately, before the next full retrain. The scaler is NOT updated.
        """
        xs = self._scale(x)
        err = self._proba_scaled(xs) - label
        for j in range(self.n):
            self.w[j] -= lr * err * xs[j]
        self.b -= lr * err

    # ---- introspection / persistence --------------------------------

    def weights_report(self, names: list[str]) -> list[tuple[str, float]]:
        """Feature weights, largest magnitude first — what the model leans on."""
        return sorted(zip(names, self.w), key=lambda t: -abs(t[1]))

    def to_json(self) -> str:
        return json.dumps(
            {"n": self.n, "w": self.w, "b": self.b, "mean": self.mean, "std": self.std}
        )

    @classmethod
    def from_json(cls, text: str) -> "LogisticRegression":
        d = json.loads(text)
        m = cls(d["n"])
        m.w, m.b, m.mean, m.std = d["w"], d["b"], d["mean"], d["std"]
        return m

    def load_from(self, other: "LogisticRegression") -> None:
        """Copy another model's parameters into this one, in place.

        The pipeline holds a single model object that everything references;
        retraining builds a fresh model and then swaps its numbers in here so
        in-flight requests never see a half-updated model.
        """
        self.n = other.n
        self.w = list(other.w)
        self.b = other.b
        self.mean = list(other.mean)
        self.std = list(other.std)
