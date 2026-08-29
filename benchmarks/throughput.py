"""
Request throughput — in-process, and over real HTTP through the threaded
server.

    python3 benchmarks/throughput.py

The HTTP number is far lower than the in-process one: that gap is the
socket, HTTP parsing, JSON encode/decode, and the GIL serializing the
threaded handlers — i.e. the cost of the transport, not the detector.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

import _bench  # noqa: F401  (sys.path)

import server as srv
from botshield.bootstrap import build_system
from botshield.synth import generate_sessions

N = 20_000


def event_dicts():
    sessions = generate_sessions(
        {"human": 300, "naive_scraper": 40, "crawler": 30, "mimic": 30,
         "stuffing_campaign": 6, "form_spammer": 30, "credential_stuffer": 20},
        seed=11, ts_start=2_000_000.0)
    evs = [ev for _n, e, _l in sessions for ev in e]
    evs.sort(key=lambda e: e.ts)
    return [
        {"ip": e.ip, "method": e.method, "path": e.path, "user_agent": e.user_agent,
         "accept": e.accept, "accept_language": e.accept_language, "cookie": e.cookie,
         "honeypot_value": e.honeypot_value, "asn_type": e.asn_type, "ts": e.ts}
        for e in evs
    ]


def bench_in_process(events) -> float:
    sysm = build_system(seed=0)
    from botshield.events import RequestEvent
    objs = [RequestEvent.from_json(d) for d in events]
    for o in objs[:2000]:
        sysm.pipeline.check(o)
    t0 = time.perf_counter()
    for o in objs:
        sysm.pipeline.check(o)
    return len(objs) / (time.perf_counter() - t0)


def bench_http(events) -> float:
    httpd = ThreadingHTTPServer(("127.0.0.1", 8611), srv._make_handler())
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = "http://127.0.0.1:8611"

    def post(d):
        req = urllib.request.Request(base + "/check", data=json.dumps(d).encode(),
                                     method="POST")
        with urllib.request.urlopen(req, timeout=5) as r:
            r.read()

    for d in events[:1000]:
        post(d)
    t0 = time.perf_counter()
    for d in events:
        post(d)
    rps = len(events) / (time.perf_counter() - t0)
    httpd.shutdown()
    return rps


def main() -> None:
    events = event_dicts()[:N]
    print(f"{len(events):,} requests\n")
    print(f"  in-process pipeline.check():  {bench_in_process(events):>10,.0f} req/s")
    print(f"  over HTTP (threaded server):  {bench_http(events):>10,.0f} req/s")


if __name__ == "__main__":
    main()
