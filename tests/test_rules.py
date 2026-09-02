"""The deterministic rules engine — hard blocks, soft scoring, escalation."""

from __future__ import annotations

from botshield import rules
from botshield.events import RequestEvent
from botshield.features import event_target_key, extract
from botshield.routes import route_of


def _evaluate(store, event):
    store.record(event.ip, event.ts, event.path)
    store.record_target(event_target_key(event), event.ts, event.ip)
    return rules.evaluate(event, extract(event, store), store)


def test_honeypot_is_a_hard_block(store):
    ev = RequestEvent(ip="7.7.7.7", honeypot_value="x", ts=1.0)
    hard, score, hits = _evaluate(store, ev)
    assert hard is not None
    assert "honeypot" in hard.lower()
    assert score == 1.0


def test_blocklisted_ip_is_a_hard_block(store):
    store.block("6.6.6.6")
    ev = RequestEvent(ip="6.6.6.6", user_agent="Mozilla/5.0", accept="text/html",
                      accept_language="en", cookie="s=1", ts=1.0)
    hard, _score, _hits = _evaluate(store, ev)
    assert hard is not None and "blocklist" in hard.lower()


def test_impossible_request_rate_is_a_hard_block(store):
    ip = "5.5.5.5"
    for i in range(60):
        store.record(ip, 1000.0 + i * 0.1, "/x")  # 60 requests in 6s
        store.record_target("/x|residential", 1000.0 + i * 0.1, ip)
    ev = RequestEvent(ip=ip, path="/x", ts=1006.0)
    hard, _score, _hits = rules.evaluate(ev, extract(ev, store), store)
    assert hard is not None


def test_clean_request_trips_no_rule(store):
    ev = RequestEvent(ip="73.1.1.1", user_agent="Mozilla/5.0 Safari/605",
                      accept="text/html", accept_language="en", cookie="s=1", ts=1.0)
    hard, score, hits = _evaluate(store, ev)
    assert hard is None
    assert score < 0.1
    assert hits == []


def test_scripted_client_scores_but_does_not_hard_block(store):
    ev = RequestEvent(ip="73.2.2.2", user_agent="python-requests/2.31.0", ts=1.0)
    hard, score, hits = _evaluate(store, ev)
    assert hard is None
    assert score > 0.0
    assert any(h.name == "ua_bot_token" for h in hits)


def test_soft_score_saturates_at_one(store):
    # Pile on many independent signals; the score is capped, not summed past 1.
    ev = RequestEvent(ip="185.1.2.3", user_agent="curl/8", accept="", accept_language="",
                      cookie="", asn_type="datacenter", ts=1.0)
    _hard, score, _hits = _evaluate(store, ev)
    assert 0.0 <= score <= 1.0


def test_distributed_attack_on_sensitive_route_escalates(store):
    # 20 distinct datacenter IPs hammer /login -> the swarm rules fire.
    now = 1000.0
    for i in range(40):
        ip = f"45.0.0.{i}"
        ev = RequestEvent(ip=ip, path="/login", method="POST", asn_type="datacenter", ts=now)
        store.record(ip, now, "/login")
        store.record_target(event_target_key(ev), now, ip)
    ev = RequestEvent(ip="45.0.0.99", path="/login", method="POST",
                      asn_type="datacenter", ts=now)
    _hard, _score, hits = rules.evaluate(ev, extract(ev, store), store)
    names = {h.name for h in hits}
    assert "distributed_attack" in names
    assert "distributed_attack_sensitive" in names
    assert route_of(ev.path) in rules.SENSITIVE_ROUTES
