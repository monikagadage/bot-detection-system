"""
Self-check — run this to confirm every piece works together.

    python3 selfcheck.py

Prints [PASS] / [FAIL] per check and exits non-zero if anything fails.
The seed model is trained once and reused across checks to keep this fast.
"""

from __future__ import annotations

import os
import random
import sys
import tempfile
from collections import defaultdict

from botshield import rules
from botshield.bootstrap import build_system
from botshield.decision import ALLOW, BLOCK, CHALLENGE
from botshield.events import RequestEvent
from botshield.features import FEATURE_NAMES, extract
from botshield.model import LogisticRegression
from botshield.pipeline import Pipeline
from botshield.sketch import CountMinSketch
from botshield.store import Store
from botshield.synth import generate_sessions

_failures = 0


def check(label: str, ok: bool, hint: str = "") -> None:
    global _failures
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + ("" if ok else f"  <-- {hint}"))
    if not ok:
        _failures += 1


BASE = build_system(seed=0)   # trained once; reused read-only below


def fresh_pipeline(**store_kwargs) -> Pipeline:
    """A pipeline with an empty store but the already-trained seed model."""
    return Pipeline(Store(**store_kwargs), BASE.model)


def copy_model() -> LogisticRegression:
    """A detached copy of the seed model, for checks that retrain."""
    return LogisticRegression.from_json(BASE.model.to_json())


def run_sessions(pipeline: Pipeline, sessions):
    """Push every event; return [(archetype, label, last_action, rids)]."""
    out = []
    for name, events, label in sessions:
        rids, last = [], None
        for ev in events:
            rid, decision, _f = pipeline.check(ev)
            rids.append(rid)
            last = decision
        out.append((name, label, last.action, rids))
    return out


# ---- features ---------------------------------------------------------
store = Store()
ev = RequestEvent(ip="9.9.9.9", path="/", user_agent="", accept="", ts=1000.0)
store.record(ev.ip, ev.ts, ev.path)
f = extract(ev, store)
check("every feature name is produced", set(f) == set(FEATURE_NAMES),
      "extract() must return exactly the keys in FEATURE_NAMES")
check("missing UA -> ua_missing = 1", f["ua_missing"] == 1.0, "empty user_agent")
check("first request -> req_60s = 1", f["req_60s"] == 1.0, "the current event counts")

store2 = Store()
for i in range(10):
    e = RequestEvent(ip="8.8.8.8", path="/x", ts=1000.0 + i * 2.0)  # every 2s exactly
    store2.record(e.ip, e.ts, e.path)
fb = extract(e, store2)
check("burst -> req_60s counts them", fb["req_60s"] == 10.0)
check("constant interval -> iat_cv near 0", fb["iat_cv"] < 0.01,
      "std/mean of a constant series is 0")

# ---- rules ----------------------------------------------------------
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

# ---- model separates the seed classes -----------------------------
preds = [1 if BASE.model.predict_proba(x) >= 0.5 else 0 for x in BASE.seed_X]
acc = sum(p == t for p, t in zip(preds, BASE.seed_y)) / len(BASE.seed_y)
check("seed model fits the seed data (acc > 0.9)", acc > 0.9, f"accuracy {acc:.3f}")

# ---- end-to-end decisions on held-out traffic --------------------
v = run_sessions(fresh_pipeline(), generate_sessions(
    {"human": 40, "naive_scraper": 15, "crawler": 12}, seed=7, ts_start=5_000_000.0))
humans = [r for r in v if r[1] == 0]
bots = [r for r in v if r[1] == 1]
human_allow = sum(a == ALLOW for _n, _l, a, _r in humans) / len(humans)
bot_flag = sum(a in (CHALLENGE, BLOCK) for _n, _l, a, _r in bots) / len(bots)
check("most humans are ALLOWed", human_allow > 0.85, f"{human_allow:.2f}")
check("obvious bots are flagged", bot_flag > 0.9, f"{bot_flag:.2f}")

# ---- feedback loop improves the 'mimic' archetype ---------------
mimic_pipe = Pipeline(Store(), copy_model())
before_rows = run_sessions(mimic_pipe, generate_sessions(
    {"mimic": 30}, seed=3, ts_start=6_000_000.0))
before = sum(a in (CHALLENGE, BLOCK) for _n, _l, a, _r in before_rows)
for _n, _l, _a, rids in before_rows:          # operator labels whole sessions
    for rid in rids:
        mimic_pipe.on_feedback(rid, 1, "review")
mimic_pipe.retrain(BASE.seed_X, BASE.seed_y)
after_rows = run_sessions(mimic_pipe, generate_sessions(
    {"mimic": 30}, seed=4, ts_start=7_000_000.0))
after = sum(a in (CHALLENGE, BLOCK) for _n, _l, a, _r in after_rows)
check("feedback + retrain catches more 'mimic' bots",
      after > before, f"before={before}/30  after={after}/30")

# ---- per-target: a distributed attack is caught though each IP is clean
campaign = generate_sessions({"stuffing_campaign": 1}, seed=55, ts_start=9_000_000.0)[0][1]

solo_actions = set()
for ev in campaign[:20]:
    _rid, d, _f = fresh_pipeline().check(ev)   # this IP, no swarm around it
    solo_actions.add(d.action)
check("a lone credential-stuffing request looks clean (ALLOW in isolation)",
      solo_actions == {ALLOW}, f"got {solo_actions}")

swarm = defaultdict(int)
swarm_pipe = fresh_pipeline()
for ev in campaign:
    _rid, d, _f = swarm_pipe.check(ev)
    swarm[d.action] += 1
flagged = swarm[CHALLENGE] + swarm[BLOCK]
check("the same requests as a swarm get flagged (per-target window)",
      flagged >= 0.6 * len(campaign), f"{flagged}/{len(campaign)}: {dict(swarm)}")

hv = run_sessions(fresh_pipeline(), generate_sessions(
    {"human": 40}, seed=71, ts_start=10_000_000.0))
ok_humans = sum(a == ALLOW for _n, _l, a, _r in hv)
check("normal users are unaffected by the per-target rules",
      ok_humans >= 38, f"{ok_humans}/40 allowed")

# ---- persistence: labels + decisions + model survive a restart -----
with tempfile.TemporaryDirectory() as _tmp:
    db = os.path.join(_tmp, "s.db")
    mp = os.path.join(_tmp, "m.json")
    with open(mp, "w") as fh:                  # pre-seed so build_system loads, not trains
        fh.write(BASE.model.to_json())

    s_a = build_system(seed=0, persist_path=db, model_path=mp)
    rid, _d, _f = s_a.pipeline.check(
        RequestEvent(ip="185.9.9.9", path="/x", user_agent="curl/8", ts=1.0))
    s_a.pipeline.on_feedback(rid, 1, "review")
    labels_a = len(s_a.store.labels())

    s_b = build_system(seed=0, persist_path=db, model_path=mp)   # "restart"
    check("labels reload from SQLite after restart",
          len(s_b.store.labels()) == labels_a and labels_a >= 1,
          f"{len(s_b.store.labels())} vs {labels_a}")
    check("the decision log reload lets old request_ids resolve",
          s_b.store.get_decision(rid) is not None, f"{rid} missing after restart")
    check("model.json is written", os.path.exists(mp))

# ---- count-min sketch: close estimates, bounded memory ------------
cms = CountMinSketch(width=2048, depth=4)
truth: dict[str, int] = {}
rng = random.Random(0)
for _ in range(20000):
    k = f"ip-{rng.randint(0, 1500)}"
    cms.add(k)
    truth[k] = truth.get(k, 0) + 1
heavy = max(truth, key=truth.get)
check("sketch never under-counts", all(cms.estimate(k) >= v for k, v in truth.items()))
check("sketch estimate is tight for a heavy hitter",
      abs(cms.estimate(heavy) - truth[heavy]) <= 0.02 * cms.total(),
      f"est {cms.estimate(heavy)} vs true {truth[heavy]}")

sk = fresh_pipeline(counter="sketch")
for i in range(9000):                          # a flood of distinct IPs
    sk.check(RequestEvent(ip=f"9.{i // 65536 % 256}.{i // 256 % 256}.{i % 256}",
                          path="/login", method="POST", ts=1000.0 + i * 0.001))
st = sk.store.stats()
check("sketch mode caps tracked IPs under a flood of distinct IPs",
      st["tracked_ips"] <= 4096, f"tracked_ips = {st['tracked_ips']}")
check("sketch mode still detects the flood on /login",
      sk.store.target_rate("/login|residential", 1000.0 + 9000 * 0.001, 60) > 2000,
      "target_rate should still see thousands of hits")

print()
if _failures:
    print(f"{_failures} check(s) failed.")
    sys.exit(1)
print("All checks passed.")
