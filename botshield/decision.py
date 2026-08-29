"""
The decision / policy layer — turn scores into an action.

Actions, from least to most aggressive:
  ALLOW      let the request through
  CHALLENGE  make the client prove it's a browser (JS proof-of-work, CAPTCHA)
             before continuing — cheap for a human, expensive for a bot farm
  BLOCK      reject outright

Splitting "not sure" into CHALLENGE instead of forcing ALLOW-or-BLOCK is the
single most important design choice in bot management: it keeps false
positives from turning into blocked customers while still stopping bots that
can't solve the challenge.

The combined score is a weighted blend of the rules score and the model
score. A hard rule short-circuits straight to BLOCK.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .features import vector
from .rules import ESCALATE_RULES

ALLOW = "ALLOW"
CHALLENGE = "CHALLENGE"
BLOCK = "BLOCK"

# How much to trust each source, and where the action thresholds sit.
# These are the knobs an operator tunes against the false-positive rate.
RULE_WEIGHT = 0.4
MODEL_WEIGHT = 0.6
CHALLENGE_AT = 0.45
BLOCK_AT = 0.80


@dataclass
class Decision:
    action: str
    score: float
    model_proba: float
    rule_score: float
    reasons: list[str] = field(default_factory=list)
    challenge_token: str | None = None


def decide(event, features, hard_reason, rule_score, rule_hits, model, store) -> Decision:
    model_p = model.predict_proba(vector(features))

    if hard_reason is not None:
        return Decision(BLOCK, 1.0, model_p, rule_score, [hard_reason])

    combined = RULE_WEIGHT * rule_score + MODEL_WEIGHT * model_p
    reasons = [f"{h.name}: {h.reason}" for h in rule_hits]
    reasons.append(f"model p(bot) = {model_p:.2f}")

    # An escalation rule (e.g. distributed_attack) forces at least a challenge
    # even when the blended score alone would have allowed the request.
    escalated = any(h.name in ESCALATE_RULES for h in rule_hits)
    if escalated and combined < CHALLENGE_AT:
        combined = CHALLENGE_AT
        reasons.append("escalated to CHALLENGE by a high-confidence rule")

    if combined >= BLOCK_AT:
        return Decision(BLOCK, combined, model_p, rule_score, reasons)
    if combined >= CHALLENGE_AT:
        token = store.issue_challenge(event.ip, event.ts)
        return Decision(CHALLENGE, combined, model_p, rule_score, reasons, token)
    return Decision(ALLOW, combined, model_p, rule_score, reasons)
