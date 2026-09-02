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
from .decision import decide
from .features import event_target_key, extract, vector
from .model import LogisticRegression
from .routes import route_of


class Pipeline:
    def __init__(self, store, model: LogisticRegression, model_path: str | None = None) -> None:
        self.store = store
        self.model = model
        self.model_path = model_path
        self._ids = itertools.count(1)

    def save_model(self) -> None:
        """Persist the current model weights, if a path was configured."""
        if self.model_path:
            with open(self.model_path, "w") as fh:
                fh.write(self.model.to_json())

    # ---- hot path -----------------------------------------------------

    def check(self, event):
        """Evaluate one request. Returns (request_id, Decision, features)."""
        self.store.record(event.ip, event.ts, event.path)
        self.store.record_target(event_target_key(event), event.ts, event.ip)
        features = extract(event, self.store)
        hard_reason, rule_score, hits = rules.evaluate(event, features, self.store)
        decision = decide(event, features, hard_reason, rule_score, hits, self.model, self.store)

        request_id = f"req-{next(self._ids)}"
        self.store.log_decision(request_id, event.ip, features, decision,
                                route=route_of(event.path))

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
        self.save_model()

        return {
            "seed_examples": len(seed_X),
            "collected_labels": len(labels),
            "total_examples": len(X),
        }
