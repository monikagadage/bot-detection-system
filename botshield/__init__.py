"""
botshield — a tiny, readable bot-detection system.

Not production code. Every piece favors being understandable end-to-end over
being fast or complete; DESIGN.md (repo root) explains how the real thing
(Cloudflare Bot Management, DataDome, HUMAN/PerimeterX, Akamai) differs and
why.

The pipeline, in order:

    request event
        -> store.record         (sliding-window counters, the "Redis" stand-in)
        -> features.extract     (event + window state  -> a numeric feature row)
        -> rules.evaluate       (deterministic, explainable checks)
        -> model.predict_proba  (a learned score, p(bot))
        -> decision.decide      (combine -> ALLOW / CHALLENGE / BLOCK)
        -> store.log_decision   (so feedback can refer back to it)

and asynchronously:

    challenge outcomes + honeypot hits + operator /feedback
        -> store.add_label
        -> pipeline.retrain     (fold new labels into the training set, refit)
"""

__all__ = [
    "events",
    "store",
    "features",
    "rules",
    "model",
    "decision",
    "pipeline",
    "synth",
]
