"""End-to-end pipeline behaviour and the feedback loop."""

from __future__ import annotations

import os
import tempfile

from botshield.bootstrap import build_system
from botshield.decision import ALLOW, BLOCK, CHALLENGE
from botshield.events import RequestEvent
from botshield.pipeline import Pipeline
from botshield.store import Store
from botshield.synth import generate_sessions


def _run(pipeline, sessions):
    rows = []
    for name, events, label in sessions:
        rids, last = [], None
        for ev in events:
            rid, decision, _f = pipeline.check(ev)
            rids.append(rid)
            last = decision
        rows.append((name, label, last.action, rids))
    return rows


def test_plausible_human_is_allowed(pipeline, make_event):
    _rid, decision, _f = pipeline.check(make_event())
    assert decision.action == ALLOW


def test_crude_scraper_is_flagged(pipeline):
    ev = RequestEvent(ip="185.10.20.30", path="/article/3",
                      user_agent="python-requests/2.31.0", ts=1_000_000.0)
    _rid, decision, _f = pipeline.check(ev)
    assert decision.action in (CHALLENGE, BLOCK)


def test_honeypot_submission_is_blocked_and_self_labeled(pipeline):
    ev = RequestEvent(ip="12.1.1.1", method="POST", path="/signup",
                      user_agent="Mozilla/5.0", honeypot_value="http://spam.example",
                      ts=1_000_000.0)
    _rid, decision, _f = pipeline.check(ev)
    assert decision.action == BLOCK
    # A filled honeypot is free ground truth — the pipeline records it.
    assert any(src == "honeypot" and lbl == 1 for _f, lbl, src in pipeline.store.labels())


def test_held_out_traffic_separates_humans_from_bots(system):
    pipe = Pipeline(Store(), system.model)
    rows = _run(pipe, generate_sessions(
        {"human": 40, "naive_scraper": 15, "crawler": 12}, seed=7, ts_start=5_000_000.0))
    humans = [r for r in rows if r[1] == 0]
    bots = [r for r in rows if r[1] == 1]
    human_allow = sum(a == ALLOW for *_x, a, _r in humans) / len(humans)
    bot_flag = sum(a in (CHALLENGE, BLOCK) for *_x, a, _r in bots) / len(bots)
    assert human_allow > 0.85
    assert bot_flag > 0.9


def test_feedback_then_retrain_catches_more_mimics(system):
    from botshield.model import LogisticRegression

    model = LogisticRegression.from_json(system.model.to_json())
    pipe = Pipeline(Store(), model)

    before_rows = _run(pipe, generate_sessions({"mimic": 30}, seed=3, ts_start=6_000_000.0))
    before = sum(a in (CHALLENGE, BLOCK) for *_x, a, _r in before_rows)

    for *_x, rids in before_rows:
        for rid in rids:
            assert pipe.on_feedback(rid, 1, "review")
    pipe.retrain(system.seed_X, system.seed_y)

    after_rows = _run(pipe, generate_sessions({"mimic": 30}, seed=4, ts_start=7_000_000.0))
    after = sum(a in (CHALLENGE, BLOCK) for *_x, a, _r in after_rows)
    assert after > before


def test_on_feedback_rejects_unknown_request_id(pipeline):
    assert pipeline.on_feedback("req-does-not-exist", 1) is False


def test_lone_stuffing_request_is_clean_but_the_swarm_is_flagged(system):
    campaign = generate_sessions(
        {"stuffing_campaign": 1}, seed=55, ts_start=9_000_000.0)[0][1]

    # Each request judged on its own, with no swarm around it.
    solo_actions = {
        Pipeline(Store(), system.model).check(ev)[1].action for ev in campaign[:20]
    }
    assert solo_actions == {ALLOW}

    swarm = Pipeline(Store(), system.model)
    flagged = sum(swarm.check(ev)[1].action in (CHALLENGE, BLOCK) for ev in campaign)
    assert flagged >= 0.6 * len(campaign)


def test_state_survives_a_restart():
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "s.db")
        mp = os.path.join(tmp, "m.json")

        first = build_system(seed=0, persist_path=db, model_path=mp)
        rid, _d, _f = first.pipeline.check(
            RequestEvent(ip="185.9.9.9", path="/x", user_agent="curl/8", ts=1.0))
        first.pipeline.on_feedback(rid, 1, "review")
        labels_before = len(first.store.labels())

        second = build_system(seed=0, persist_path=db, model_path=mp)  # "restart"
        assert len(second.store.labels()) == labels_before >= 1
        assert second.store.get_decision(rid) is not None
        assert os.path.exists(mp)
