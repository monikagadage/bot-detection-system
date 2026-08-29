"""
The pipeline — wires the stages together and owns the feedback loop.

    check()               synchronous: event -> decision (the hot path)
    on_challenge_solved() a client passed a challenge (weak "human" signal)
    on_feedback()         an operator (or an automated labeler) says
                          "request X was actually a bot / a human"
    retrain()             fold every collected label into the training set
                          and refit the model
"""

from __future__ import annotations

import itertools

from . import rules
from .decision import ALLOW, decide
from .features import extract, vector
from .model import LogisticRegression


class Pipeline:
    def __init__(self, store, model: LogisticRegression) -> None:
        self.store = store
        self.model = model
        self._ids = itertools.count(1)

    # ---- hot path -----------------------------------------------------

    def check(self, event):
        """Evaluate one request. Returns (request_id, Decision, features)."""
        self.store.record(event.ip, event.ts, event.path)
        features = extract(event, self.store)
        hard_reason, rule_score, hits = rules.evaluate(event, features, self.store)
        decision = decide(event, features, hard_reason, rule_score, hits, self.model, self.store)

        request_id = f"req-{next(self._ids)}"
        self.store.log_decision(request_id, event.ip, features, decision.action)

        # A filled honeypot is ground truth we get for free — label it now.
        if features["honeypot_filled"]:
            self.store.add_label(features, 1, "honeypot")

        return request_id, decision, features

    # ---- feedback loop --------------------------------------------

    def on_challenge_solved(self, token: str):
        """Record that a challenge was passed.

        Passing a JS proof-of-work is only a weak human signal (headless
        browsers pass them too), so this labels the request as human with a
        low-trust source rather than treating it as certain.
        """
        challenge = self.store.solve_challenge(token)
        return challenge

    def on_feedback(self, request_id: str, label: int, source: str = "operator") -> bool:
        """An external label for a past request. Stores it and nudges the
        model online so the effect is immediate (a full retrain catches up
        later)."""
        rec = self.store.get_decision(request_id)
        if rec is None:
            return False
        self.store.add_label(rec["features"], label, source)
        self.model.partial_fit(vector(rec["features"]), label)
        return True

    def retrain(self, seed_X: list[list[float]], seed_y: list[int]) -> dict:
        """Refit the model on the seed data plus everything in the label store."""
        labels = self.store.labels()
        X = list(seed_X) + [vector(f) for (f, _l, _s) in labels]
        y = list(seed_y) + [lbl for (_f, lbl, _s) in labels]

        fresh = LogisticRegression(len(seed_X[0]) if seed_X else len(vector(labels[0][0])))
        fresh.fit(X, y)
        self.model.load_from(fresh)  # atomic-ish swap into the live model

        return {
            "seed_examples": len(seed_X),
            "collected_labels": len(labels),
            "total_examples": len(X),
        }
