"""
Feature extraction — turn a request event (plus the window state in the
store) into a fixed list of numbers the rules and the model can both read.

Keeping this in one place matters: the rules engine, the ML model, and the
training pipeline must all see the *same* features computed the *same* way,
or a model trained offline behaves differently online ("training/serving
skew" — one of the classic ways an ML system quietly breaks).
"""

from __future__ import annotations

from .events import BOT_UA_TOKENS, RequestEvent

# Order matters — the model stores one weight per position.
FEATURE_NAMES: list[str] = [
    "req_10s",             # requests from this IP in the last 10s
    "req_60s",             # requests from this IP in the last 60s
    "distinct_paths_60s",  # how many different URLs it hit in 60s
    "iat_mean",            # mean gap between its requests, seconds
    "iat_cv",              # gap variability (std/mean); low = robotic timing
    "ua_missing",          # 1 if no User-Agent at all
    "ua_bot_token",        # 1 if the UA names a script/tool/crawler
    "accept_missing",      # 1 if no Accept header (browsers always send one)
    "accept_lang_missing", # 1 if no Accept-Language header
    "cookie_missing",      # 1 if no cookie (no established session)
    "honeypot_filled",     # 1 if the hidden trap field was filled in
    "ip_reputation",       # 0..1 from the reputation feed
    "asn_datacenter",      # 1 if the IP is on a hosting/cloud network
]


def extract(event: RequestEvent, store) -> dict:
    """Compute every feature in FEATURE_NAMES for this event."""
    now = event.ts
    w10 = store.window(event.ip, now, 10)
    w60 = store.window(event.ip, now, 60)

    # Inter-arrival times: the gaps between consecutive requests from this IP.
    # Humans are bursty and irregular; a loop with time.sleep(2) is not.
    times = sorted(t for (t, _p) in w60)
    gaps = [b - a for a, b in zip(times, times[1:])]
    if gaps:
        mean_gap = sum(gaps) / len(gaps)
        var = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
        std_gap = var ** 0.5
        iat_mean = mean_gap
        iat_cv = (std_gap / mean_gap) if mean_gap > 1e-9 else 0.0
    else:
        # Not enough requests yet to say anything — use neutral-ish values
        # (a long gap, high variability) so a first request looks human.
        iat_mean = 30.0
        iat_cv = 1.0

    ua = event.user_agent.lower()

    return {
        "req_10s": float(len(w10)),
        "req_60s": float(len(w60)),
        "distinct_paths_60s": float(len({p for (_t, p) in w60})),
        "iat_mean": float(iat_mean),
        "iat_cv": float(iat_cv),
        "ua_missing": 1.0 if not ua.strip() else 0.0,
        "ua_bot_token": 1.0 if any(tok in ua for tok in BOT_UA_TOKENS) else 0.0,
        "accept_missing": 1.0 if not event.accept.strip() else 0.0,
        "accept_lang_missing": 1.0 if not event.accept_language.strip() else 0.0,
        "cookie_missing": 1.0 if not event.cookie.strip() else 0.0,
        "honeypot_filled": 1.0 if event.honeypot_value.strip() else 0.0,
        "ip_reputation": float(store.reputation(event.ip)),
        "asn_datacenter": 1.0 if event.asn_type == "datacenter" else 0.0,
    }


def vector(features: dict) -> list[float]:
    """The feature dict as a plain list, in FEATURE_NAMES order."""
    return [features[name] for name in FEATURE_NAMES]
