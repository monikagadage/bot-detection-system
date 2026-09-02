"""Feature extraction — the contract the rules and the model both depend on."""

from __future__ import annotations

from botshield.events import RequestEvent
from botshield.features import FEATURE_NAMES, extract, vector


def _record_and_extract(store, event):
    store.record(event.ip, event.ts, event.path)
    return extract(event, store)


def test_extract_returns_exactly_the_declared_features(store, make_event):
    f = _record_and_extract(store, make_event())
    assert set(f) == set(FEATURE_NAMES)


def test_vector_is_in_feature_name_order(store, make_event):
    f = _record_and_extract(store, make_event())
    assert vector(f) == [f[name] for name in FEATURE_NAMES]


def test_missing_headers_flagged(store):
    ev = RequestEvent(ip="9.9.9.9", user_agent="", accept="", accept_language="",
                      cookie="", ts=1000.0)
    f = _record_and_extract(store, ev)
    assert f["ua_missing"] == 1.0
    assert f["accept_missing"] == 1.0
    assert f["accept_lang_missing"] == 1.0
    assert f["cookie_missing"] == 1.0


def test_bot_user_agent_token_detected(store, make_event):
    f = _record_and_extract(store, make_event(user_agent="python-requests/2.31.0"))
    assert f["ua_bot_token"] == 1.0
    assert f["ua_missing"] == 0.0


def test_honeypot_value_sets_feature(store, make_event):
    f = _record_and_extract(store, make_event(honeypot_value="http://spam.example"))
    assert f["honeypot_filled"] == 1.0


def test_constant_interval_gives_near_zero_cv(store):
    ip = "8.8.8.8"
    ev = None
    for i in range(10):
        ev = RequestEvent(ip=ip, path="/x", ts=1000.0 + i * 2.0)  # exactly every 2s
        store.record(ev.ip, ev.ts, ev.path)
    f = extract(ev, store)
    assert f["req_60s"] == 10.0
    assert f["iat_cv"] < 0.01
    assert abs(f["iat_mean"] - 2.0) < 1e-6


def test_first_request_looks_human_on_timing(store, make_event):
    # Not enough history to judge cadence -> neutral, human-ish defaults.
    f = _record_and_extract(store, make_event())
    assert f["req_60s"] == 1.0
    assert f["iat_cv"] == 1.0


def test_distinct_paths_counts_unique_urls(store):
    ip = "7.7.7.7"
    ev = None
    for i in range(12):
        ev = RequestEvent(ip=ip, path=f"/p/{i}", ts=1000.0 + i)
        store.record(ev.ip, ev.ts, ev.path)
    f = extract(ev, store)
    assert f["distinct_paths_60s"] == 12.0


def test_datacenter_asn_flagged(store, make_event):
    f = _record_and_extract(store, make_event(asn_type="datacenter"))
    assert f["asn_datacenter"] == 1.0


def test_reputation_feed_reflected_in_features(store):
    ev = RequestEvent(ip="185.10.20.30", path="/", ts=1000.0)  # a "bad" prefix
    f = _record_and_extract(store, ev)
    assert f["ip_reputation"] > 0.5
