"""
Synthetic traffic — labeled human and bot sessions, so the whole thing can
be trained and evaluated without a real website.

A *session* is one client (usually one IP) making a burst of requests. Each
archetype below has a distinct fingerprint across headers, timing, velocity,
and which network it comes from. `build_dataset` replays sessions through a
throwaway Store so the window-based features (velocity, timing regularity)
come out realistic.

The "mimic" archetype is deliberately left OUT of the seed training set: it
copies a real browser's headers and comes from a residential proxy, so the
seed model — which leans on header features — waves it through. Feeding a
few labeled mimic sessions back in via /feedback and retraining is what
teaches the model to also weigh timing regularity. That's the feedback loop,
made visible.
"""

from __future__ import annotations

import random

from .events import RequestEvent

_REAL_UAS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile Safari/604.1",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
]
_SCRIPT_UAS = [
    "python-requests/2.31.0",
    "curl/8.4.0",
    "Go-http-client/2.0",
    "Scrapy/2.11 (+https://scrapy.org)",
    "",  # some scripts send no UA at all
]
_CONTENT_PATHS = [f"/article/{i}" for i in range(1, 40)] + [
    "/", "/search", "/about", "/pricing", "/login", "/account", "/cart",
]


def _human(rng: random.Random, ts0: float) -> tuple[list[RequestEvent], int]:
    ip = f"73.{rng.randint(1, 254)}.{rng.randint(1, 254)}.{rng.randint(1, 254)}"
    ua = rng.choice(_REAL_UAS)
    has_cookie = rng.random() < 0.85
    n = rng.randint(2, 8)
    ts = ts0
    events = []
    for _ in range(n):
        events.append(RequestEvent(
            ip=ip, path=rng.choice(_CONTENT_PATHS), user_agent=ua,
            accept="text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            accept_language="en-US,en;q=0.9",
            cookie=("session=%d" % rng.randint(1, 9999)) if has_cookie else "",
            asn_type="residential", ts=ts,
        ))
        ts += min(40.0, max(1.5, rng.expovariate(1 / 12.0)))  # bursty, irregular
    return events, 0


def _naive_scraper(rng: random.Random, ts0: float) -> tuple[list[RequestEvent], int]:
    ip = f"185.{rng.randint(1, 254)}.{rng.randint(1, 254)}.{rng.randint(1, 254)}"
    ua = rng.choice(_SCRIPT_UAS)
    target = rng.choice(_CONTENT_PATHS)
    n = rng.randint(15, 40)
    ts = ts0
    events = []
    for _ in range(n):
        events.append(RequestEvent(
            ip=ip, path=target, user_agent=ua, accept="", accept_language="",
            cookie="", asn_type="datacenter", ts=ts,
        ))
        ts += max(0.3, rng.gauss(1.8, 0.5))  # tight, only mildly noisy
    return events, 1


def _crawler(rng: random.Random, ts0: float) -> tuple[list[RequestEvent], int]:
    ip = f"185.{rng.randint(1, 254)}.{rng.randint(1, 254)}.{rng.randint(1, 254)}"
    n = rng.randint(20, 45)
    ts = ts0
    events = []
    for _ in range(n):
        events.append(RequestEvent(
            ip=ip, path=rng.choice(_CONTENT_PATHS), user_agent="Scrapy/2.11 (+https://scrapy.org)",
            accept="text/html", accept_language="", cookie="",
            asn_type="datacenter", ts=ts,
        ))
        ts += max(0.4, rng.gauss(1.4, 0.2))  # very regular
    return events, 1


def _form_spammer(rng: random.Random, ts0: float) -> tuple[list[RequestEvent], int]:
    ip = f"45.{rng.randint(1, 254)}.{rng.randint(1, 254)}.{rng.randint(1, 254)}"
    n = rng.randint(4, 12)
    ts = ts0
    events = []
    for _ in range(n):
        fills_trap = rng.random() < 0.7
        events.append(RequestEvent(
            ip=ip, method="POST", path=rng.choice(["/signup", "/comment", "/contact"]),
            user_agent=rng.choice(_REAL_UAS),  # spammers often spoof a real UA
            accept="text/html", accept_language="en-US,en;q=0.9", cookie="",
            honeypot_value="http://spam.example" if fills_trap else "",
            asn_type=rng.choice(["datacenter", "residential"]), ts=ts,
        ))
        ts += max(1.0, rng.gauss(6.0, 2.0))
    return events, 1


def _credential_stuffer(rng: random.Random, ts0: float) -> tuple[list[RequestEvent], int]:
    # Each attempt comes from a DIFFERENT IP (a proxy pool), so per-IP
    # velocity stays low. This one is meant to be hard to catch here — the
    # notes explain that you'd aggregate per-target (/login) instead.
    n = rng.randint(3, 8)
    ts = ts0
    events = []
    for _ in range(n):
        ip = f"185.{rng.randint(1, 254)}.{rng.randint(1, 254)}.{rng.randint(1, 254)}"
        events.append(RequestEvent(
            ip=ip, method="POST", path="/login", user_agent=rng.choice(_SCRIPT_UAS),
            accept="", accept_language="", cookie="", asn_type="datacenter", ts=ts,
        ))
        ts += max(0.5, rng.gauss(3.0, 1.0))
    return events, 1


def _mimic(rng: random.Random, ts0: float) -> tuple[list[RequestEvent], int]:
    # Sophisticated scraper: real browser UA, full headers, a cookie, a
    # residential-proxy IP (clean reputation). Only its behaviour betrays it
    # — it crawls many pages at an almost perfectly constant cadence.
    ip = f"73.{rng.randint(1, 254)}.{rng.randint(1, 254)}.{rng.randint(1, 254)}"
    ua = rng.choice(_REAL_UAS)
    n = rng.randint(16, 28)
    ts = ts0
    events = []
    for _ in range(n):
        events.append(RequestEvent(
            ip=ip, path=rng.choice(_CONTENT_PATHS), user_agent=ua,
            accept="text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            accept_language="en-US,en;q=0.9",
            cookie="session=%d" % rng.randint(1, 9999),
            asn_type="residential", ts=ts,
        ))
        ts += max(0.5, rng.gauss(3.5, 0.12))  # metronome-steady
    return events, 1


SEED_ARCHETYPES = {
    "human": _human,
    "naive_scraper": _naive_scraper,
    "crawler": _crawler,
    "form_spammer": _form_spammer,
    "credential_stuffer": _credential_stuffer,
}
# "mimic" is intentionally excluded from the seed mix.
ALL_ARCHETYPES = {**SEED_ARCHETYPES, "mimic": _mimic}


def generate_sessions(
    counts: dict[str, int],
    seed: int = 0,
    ts_start: float = 1_000_000.0,
) -> list[tuple[str, list[RequestEvent], int]]:
    """Return [(archetype_name, events, label), ...] for the given per-archetype counts."""
    rng = random.Random(seed)
    sessions: list[tuple[str, list[RequestEvent], int]] = []
    ts = ts_start
    plan = [(name, i) for name, c in counts.items() for i in range(c)]
    rng.shuffle(plan)
    for name, _i in plan:
        events, label = ALL_ARCHETYPES[name](rng, ts)
        sessions.append((name, events, label))
        ts += rng.uniform(120.0, 400.0)  # sessions spread out over time
    return sessions


def seed_session_counts() -> dict[str, int]:
    return {
        "human": 120,
        "naive_scraper": 40,
        "crawler": 30,
        "form_spammer": 25,
        "credential_stuffer": 20,
    }


def build_dataset(
    sessions: list[tuple[str, list[RequestEvent], int]],
) -> tuple[list[list[float]], list[int]]:
    """Replay sessions through a throwaway Store and collect (features, label)
    for every request."""
    from .features import extract, vector
    from .store import Store

    store = Store()
    X: list[list[float]] = []
    y: list[int] = []
    for _name, events, label in sessions:
        for ev in events:
            store.record(ev.ip, ev.ts, ev.path)
            X.append(vector(extract(ev, store)))
            y.append(label)
    return X, y
