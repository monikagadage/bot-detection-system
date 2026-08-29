"""
The rules engine — deterministic, human-written checks.

Why have this at all when there's an ML model? Three reasons real systems
keep a rules layer next to the model:

  1. Explainability. "Blocked: the hidden honeypot field was filled in" is
     something a support agent can act on. "Blocked: model score 0.91" is not.
  2. Speed of response. When a new attack starts at 3am you ship a rule in
     minutes; retraining and rolling out a model takes hours to days.
  3. Hard guarantees. Some signals are near-certain (honeypot filled,
     50 requests in 10 seconds). Those should block outright, not be
     averaged against a model that might disagree.

`evaluate` returns:
  - hard_reason: a string if a hard rule fired (caller should BLOCK), else None
  - soft_score:  0.0..1.0, the weighted fraction of soft rules that fired
  - hits:        the individual RuleHit objects, for logging / explanation
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RuleHit:
    name: str
    weight: float
    reason: str


def evaluate(event, features: dict, store) -> tuple[str | None, float, list[RuleHit]]:
    # ---- hard rules: any one of these => block, no averaging ---------------
    if store.is_blocked(event.ip):
        return "ip is on the blocklist", 1.0, [RuleHit("blocklist", 1.0, "IP on blocklist")]
    if features["honeypot_filled"]:
        return (
            "honeypot form field was filled in",
            1.0,
            [RuleHit("honeypot", 1.0, "hidden trap field was submitted with a value")],
        )
    if features["req_10s"] > 50:
        return (
            "impossible request rate",
            1.0,
            [RuleHit("velocity_hard", 1.0, f"{int(features['req_10s'])} requests in 10s")],
        )

    # ---- soft rules: each contributes its weight if it fires --------------
    # (name, condition, weight, human-readable reason)
    soft = [
        ("ua_missing",          features["ua_missing"] == 1.0,        2.0, "no User-Agent header"),
        ("ua_bot_token",        features["ua_bot_token"] == 1.0,      3.0, "User-Agent names a script or crawler"),
        ("accept_missing",      features["accept_missing"] == 1.0,    1.0, "no Accept header"),
        ("accept_lang_missing", features["accept_lang_missing"] == 1.0, 1.0, "no Accept-Language header"),
        ("cookie_missing",      features["cookie_missing"] == 1.0,    1.0, "no session cookie"),
        ("high_velocity",       features["req_60s"] > 20,             2.0, "more than 20 requests in 60s"),
        ("path_scan",           features["distinct_paths_60s"] > 15,  2.0, "hit more than 15 distinct paths in 60s"),
        ("robotic_timing",      features["req_60s"] >= 6 and features["iat_cv"] < 0.15,
                                                                     3.0, "near-constant gap between requests"),
        ("datacenter_ip",       features["asn_datacenter"] == 1.0,    1.5, "request from a hosting/datacenter network"),
        ("bad_reputation",      features["ip_reputation"] > 0.6,      2.5, "IP has poor historical reputation"),
    ]

    hits = [RuleHit(name, weight, reason) for (name, fired, weight, reason) in soft if fired]
    total_weight = sum(weight for (_n, _f, weight, _r) in soft)
    soft_score = (sum(h.weight for h in hits) / total_weight) if total_weight else 0.0
    return None, soft_score, hits
