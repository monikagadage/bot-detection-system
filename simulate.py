"""
End-to-end demo: start the service, fire mixed human + bot traffic at it over
real HTTP, score the results, then show the feedback loop closing the gap on
the one bot archetype the seed model misses.

    python3 simulate.py

No arguments, no network access, ~a few seconds. It:

  1. starts server.py in a background thread
  2. PHASE 1 — sends a batch of traffic, prints a confusion matrix and a
     per-archetype breakdown of ALLOW / CHALLENGE / BLOCK
  3. PHASE 2 — feeds labels back (POST /feedback) for the bots that slipped
     through, then POST /retrain
  4. PHASE 3 — sends a fresh batch (new IPs) and prints the matrix again

Watch the "mimic" row: human-looking headers, robotic timing. The seed model
lets it through; after feedback + retrain it gets challenged/blocked.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from collections import defaultdict
from http.server import ThreadingHTTPServer

import server as srv
from botshield.synth import generate_sessions

BASE = "http://127.0.0.1:8599"


def _post(path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read())


def _get(path: str) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=5) as resp:
        return json.loads(resp.read())


def _start_server() -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer(("127.0.0.1", 8599), srv._make_handler())
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    for _ in range(50):
        try:
            _get("/")
            return httpd
        except OSError:
            time.sleep(0.05)
    raise RuntimeError("server did not come up")


def _send_batch(sessions) -> list[dict]:
    """Send every session; return one verdict record per session (using the
    decision on its LAST request, which has the most window context)."""
    results = []
    for archetype, events, label in sessions:
        last = None
        request_ids = []
        for ev in events:
            last = _post("/check", {
                "ip": ev.ip, "method": ev.method, "path": ev.path,
                "user_agent": ev.user_agent, "accept": ev.accept,
                "accept_language": ev.accept_language, "cookie": ev.cookie,
                "honeypot_value": ev.honeypot_value, "asn_type": ev.asn_type,
                "ts": ev.ts,
            })
            request_ids.append(last["request_id"])
        results.append({"archetype": archetype, "label": label,
                        "action": last["action"], "request_id": last["request_id"],
                        "request_ids": request_ids})
    return results


def _predicted_bot(action: str) -> bool:
    return action in ("BLOCK", "CHALLENGE")


def _report(title: str, results: list[dict]) -> None:
    tp = sum(1 for r in results if r["label"] == 1 and _predicted_bot(r["action"]))
    fn = sum(1 for r in results if r["label"] == 1 and not _predicted_bot(r["action"]))
    tn = sum(1 for r in results if r["label"] == 0 and not _predicted_bot(r["action"]))
    fp = sum(1 for r in results if r["label"] == 0 and _predicted_bot(r["action"]))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    print(f"\n=== {title} ===")
    print(f"  bots caught (recall):     {recall:5.1%}   ({tp}/{tp + fn})")
    print(f"  precision on 'bot' calls:  {precision:5.1%}   ({tp}/{tp + fp} flagged were bots)")
    print(f"  humans wrongly flagged:    {fp}/{tn + fp}")

    by = defaultdict(lambda: defaultdict(int))
    for r in results:
        by[r["archetype"]][r["action"]] += 1
    print(f"  {'archetype':<20}{'ALLOW':>7}{'CHALLENGE':>11}{'BLOCK':>7}")
    for archetype in sorted(by):
        row = by[archetype]
        print(f"  {archetype:<20}{row['ALLOW']:>7}{row['CHALLENGE']:>11}{row['BLOCK']:>7}")


def main() -> None:
    httpd = _start_server()
    print(f"service up on {BASE}; seed model on {len(srv.SYS.seed_X)} labeled requests")

    live_mix = {"human": 60, "naive_scraper": 20, "crawler": 15,
                "form_spammer": 12, "credential_stuffer": 10, "mimic": 15}

    # ---- PHASE 1 -------------------------------------------------------
    batch1 = generate_sessions(live_mix, seed=101, ts_start=2_000_000.0)
    results1 = _send_batch(batch1)
    _report("PHASE 1 — seed model, no feedback yet", results1)

    # ---- PHASE 2: feedback loop -------------------------------------
    fed_sessions = 0
    fed_labels = 0
    for r in results1:
        # An operator reviews sessions that were ALLOWed but were really a bot
        # and labels the whole session. (In production: challenge failures,
        # abuse reports, and a review queue all feed this same endpoint.)
        if r["label"] == 1 and not _predicted_bot(r["action"]):
            for rid in r["request_ids"]:
                _post("/feedback", {"request_id": rid, "label": 1, "source": "review"})
                fed_labels += 1
            fed_sessions += 1
        # Confirm the humans it got right too, so retraining doesn't drift
        # toward calling everything a bot.
        if r["label"] == 0 and r["action"] == "ALLOW" and fed_sessions and r["request_id"][-1] in "02468":
            for rid in r["request_ids"]:
                _post("/feedback", {"request_id": rid, "label": 0, "source": "review"})
                fed_labels += 1
    print(f"\n[operator labeled {fed_sessions} missed-bot sessions "
          f"({fed_labels} request labels total) + a sample of confirmed humans]")

    retrain_summary = _post("/retrain", {})
    print("[retrained]", json.dumps({k: retrain_summary[k] for k in
          ("seed_examples", "collected_labels", "total_examples")}))
    print("  top model features now:", ", ".join(
        f"{n}={w}" for n, w in retrain_summary["model_top_features"]))

    # ---- PHASE 3: fresh traffic, retrained model ------------------
    batch2 = generate_sessions(live_mix, seed=202, ts_start=3_000_000.0)
    results2 = _send_batch(batch2)
    _report("PHASE 3 — after feedback + retrain (fresh traffic)", results2)

    print("\n" + json.dumps(_get("/stats"), indent=2))
    httpd.shutdown()


if __name__ == "__main__":
    main()
