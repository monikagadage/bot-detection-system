"""
Self-check — run this to confirm every piece works together.

    python3 selfcheck.py

Prints [PASS] / [FAIL] per check and exits non-zero if anything fails.
"""

from __future__ import annotations

import sys

from botshield.bootstrap import build_system
from botshield.decision import ALLOW, BLOCK, CHALLENGE
from botshield.events import RequestEvent
from botshield.features import FEATURE_NAMES, extract
from botshield.store import Store
from botshield.synth import ALL_ARCHETYPES, build_dataset, generate_sessions

_failures = 0


def check(label: str, ok: bool, hint: str = "") -> None:
    global _failures
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + ("" if ok else f"  <-- {hint}"))
    if not ok:
        _failures += 1


# ---- features ---------------------------------------------------------
store = Store()
ev = RequestEvent(ip="9.9.9.9", path="/", user_agent="", accept="", ts=1000.0)
store.record(ev.ip, ev.ts, ev.path)
f = extract(ev, store)
check("every feature name is produced", set(f) == set(FEATURE_NAMES),
      "extract() must return exactly the keys in FEATURE_NAMES")
check("missing UA -> ua_missing = 1", f["ua_missing"] == 1.0, "empty user_agent")
check("first request -> req_60s = 1", f["req_60s"] == 1.0, "the current event counts")

# velocity + timing features respond to a burst
store2 = Store()
for i in range(10):
    e = RequestEvent(ip="8.8.8.8", path="/x", ts=1000.0 + i * 2.0)  # every 2s exactly
    store2.record(e.ip, e.ts, e.path)
fb = extract(e, store2)
check("burst -> req_60s counts them", fb["req_60s"] == 10.0)
check("constant interval -> iat_cv near 0", fb["iat_cv"] < 0.01,
      "std/mean of a constant series is 0")

# ---- rules ---------------------------------------------------------
from botshield import rules  # noqa: E402

hp = RequestEvent(ip="7.7.7.7", honeypot_value="x", ts=1.0)
store3 = Store()
store3.record(hp.ip, hp.ts, hp.path)
hard, score, hits = rules.evaluate(hp, extract(hp, store3), store3)
check("honeypot fill is a hard rule", hard is not None and "honeypot" in hard.lower())

clean = RequestEvent(ip="73.1.1.1", user_agent="Mozilla/5.0 Safari/605",
                     accept="text/html", accept_language="en", cookie="s=1", ts=1.0)
store4 = Store()
store4.record(clean.ip, clean.ts, clean.path)
hard4, score4, _ = rules.evaluate(clean, extract(clean, store4), store4)
check("a clean request trips no hard rule", hard4 is None)
check("a clean request has a low soft score", score4 < 0.1, f"got {score4:.3f}")

# ---- model separates the seed classes ------------------------------
sys_ = build_system(seed=1)
X, y = sys_.seed_X, sys_.seed_y
preds = [1 if sys_.model.predict_proba(x) >= 0.5 else 0 for x in X]
acc = sum(p == t for p, t in zip(preds, y)) / len(y)
check("seed model fits the seed data (acc > 0.9)", acc > 0.9, f"accuracy {acc:.3f}")

# ---- end-to-end decisions on held-out traffic ---------------------
def verdicts(mix, seed):
    s = build_system(seed=0)
    out = []
    for name, events, label in generate_sessions(mix, seed=seed, ts_start=5_000_000.0):
        last = None
        for e in events:
            _rid, d, _f = s.pipeline.check(e)
            last = d
        out.append((name, label, last.action))
    return out

v = verdicts({"human": 40, "naive_scraper": 15, "crawler": 12}, seed=7)
human_allow = sum(1 for n, l, a in v if l == 0 and a == ALLOW) / max(1, sum(1 for _n, l, _a in v if l == 0))
bot_flag = sum(1 for n, l, a in v if l == 1 and a in (CHALLENGE, BLOCK)) / max(1, sum(1 for _n, l, _a in v if l == 1))
check("most humans are ALLOWed", human_allow > 0.85, f"{human_allow:.2f}")
check("obvious bots are flagged", bot_flag > 0.9, f"{bot_flag:.2f}")

# ---- feedback loop improves the 'mimic' archetype ---------------
sys2 = build_system(seed=0)
mimic_mix = {"mimic": 30}
before = 0
all_rids = []
for name, events, label in generate_sessions(mimic_mix, seed=3, ts_start=6_000_000.0):
    last_d = None
    for e in events:
        rid, last_d, _f = sys2.pipeline.check(e)
        all_rids.append(rid)
    if last_d.action in (CHALLENGE, BLOCK):
        before += 1

# an operator labels the whole abusive session, not just its last request
for rid in all_rids:
    sys2.pipeline.on_feedback(rid, 1, "review")
sys2.pipeline.retrain(sys2.seed_X, sys2.seed_y)

after = 0
for name, events, label in generate_sessions(mimic_mix, seed=4, ts_start=7_000_000.0):
    last_d = None
    for e in events:
        _rid, last_d, _f = sys2.pipeline.check(e)
    if last_d.action in (CHALLENGE, BLOCK):
        after += 1

check("feedback + retrain catches more 'mimic' bots",
      after > before, f"before={before}/30  after={after}/30")

print()
if _failures:
    print(f"{_failures} check(s) failed.")
    sys.exit(1)
print("All checks passed.")
