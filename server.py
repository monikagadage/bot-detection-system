"""
The bot-detection service — a real HTTP server, standard library only.

Run it:

    python3 server.py            # listens on 127.0.0.1:8500

Then talk to it:

    # a plausible human request -> ALLOW
    curl -s localhost:8500/check -d '{
      "ip":"73.5.9.2","path":"/article/3",
      "user_agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/605.1.15",
      "accept":"text/html","accept_language":"en-US","cookie":"session=42"}'

    # a crude scraper -> BLOCK
    curl -s localhost:8500/check -d '{
      "ip":"185.10.20.30","path":"/article/3","user_agent":"python-requests/2.31.0"}'

    curl -s localhost:8500/stats

Routes:
    GET  /                                             -> the live dashboard (HTML)
    POST /check      body = a request event (JSON)   -> decision
    POST /challenge  body = {"token": "..."}          -> {"solved": bool}
    POST /feedback   body = {"request_id":"...", "label":0|1}
    POST /retrain    body = {}                         -> retrain on collected labels
    POST /replay     body = {}                         -> run demo traffic through it
    GET  /stats                                        -> counters + top model features
    GET  /recent?n=50                                  -> latest decisions (feed)
    GET  /timeseries                                   -> decisions bucketed by time
    GET  /api                                          -> the JSON route list

Set BOTSHIELD_DATA=./data to persist labels, the decision log, and the
learned model across restarts (SQLite + model.json under that directory).
Set BOTSHIELD_COUNTER=sketch to estimate request rates from a fixed-size
count-min sketch instead of exact per-IP history.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from botshield.bootstrap import build_system
from botshield.events import RequestEvent
from botshield.features import FEATURE_NAMES
from botshield.synth import generate_sessions

_DASHBOARD = os.path.join(os.path.dirname(__file__), "dashboard", "index.html")

_DATA_DIR = os.environ.get("BOTSHIELD_DATA")
_COUNTER = os.environ.get("BOTSHIELD_COUNTER", "exact")
if _DATA_DIR:
    SYS = build_system(
        persist_path=os.path.join(_DATA_DIR, "botshield.db"),
        model_path=os.path.join(_DATA_DIR, "model.json"),
        counter=_COUNTER,
    )
else:
    SYS = build_system(counter=_COUNTER)

_replay_lock = threading.Lock()
_replaying = False


def _replay_traffic() -> None:
    """Push a demo traffic mix through the live pipeline at wall-clock time so
    the dashboard animates. Runs in a background thread."""
    global _replaying
    with _replay_lock:
        if _replaying:
            return
        _replaying = True
    try:
        mix = {"human": 40, "naive_scraper": 12, "crawler": 8, "form_spammer": 8,
               "credential_stuffer": 6, "stuffing_campaign": 3, "mimic": 8}
        sessions = generate_sessions(mix, seed=int(time.time()) % 9999)
        # Replay session by session, keeping each session's *internal* timing
        # (compressed to <= ~2.5s) so behavioural features stay meaningful and
        # unrelated sessions don't all pile into the same instant.
        for _name, events, _label in sessions:
            if not events:
                continue
            t0 = events[0].ts
            span = (events[-1].ts - t0) or 1.0
            scale = min(1.0, 2.5 / span)
            base = time.time()
            for ev in events:
                ev.ts = base + (ev.ts - t0) * scale
                SYS.pipeline.check(ev)
                time.sleep(0.012)
            time.sleep(0.12)
    finally:
        with _replay_lock:
            _replaying = False


def _make_handler():
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):  # keep the console quiet
            pass

        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj, indent=2).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, code: int, text: str) -> None:
            body = text.encode()
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(n) if n else b""
            return json.loads(raw) if raw.strip() else {}

        def _model_features(self) -> list[list]:
            return [[name, round(w, 3)]
                    for name, w in SYS.model.weights_report(FEATURE_NAMES)[:8]]

        def do_GET(self):
            route, _, query = self.path.partition("?")
            route = route.rstrip("/") or "/"
            params = urllib.parse.parse_qs(query)

            if route == "/":
                try:
                    with open(_DASHBOARD, encoding="utf-8") as fh:
                        self._send_html(200, fh.read())
                except OSError:
                    self._send(200, {"service": "botshield", "note": "dashboard file missing"})
            elif route == "/api":
                self._send(200, {"service": "botshield", "routes":
                                 ["POST /check", "POST /challenge", "POST /feedback",
                                  "POST /retrain", "POST /replay", "GET /stats",
                                  "GET /recent", "GET /timeseries"]})
            elif route == "/stats":
                stats = SYS.store.stats()
                stats["seed_examples"] = len(SYS.seed_X)
                stats["model_top_features"] = self._model_features()
                stats["replaying"] = _replaying
                self._send(200, stats)
            elif route == "/recent":
                n = int(params.get("n", ["50"])[0])
                self._send(200, {"recent": SYS.store.recent(n)})
            elif route == "/timeseries":
                self._send(200, {"series": SYS.store.timeseries()})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            try:
                body = self._body()
            except (ValueError, json.JSONDecodeError) as exc:
                self._send(400, {"error": f"invalid JSON body: {exc}"})
                return

            if self.path == "/check":
                _rid, decision, _feat = SYS.pipeline.check(RequestEvent.from_json(body))
                request_id = _rid
                self._send(200, {
                    "request_id": request_id,
                    "action": decision.action,
                    "score": round(decision.score, 3),
                    "model_proba": round(decision.model_proba, 3),
                    "rule_score": round(decision.rule_score, 3),
                    "reasons": decision.reasons,
                    "challenge_token": decision.challenge_token,
                })
            elif self.path == "/challenge":
                challenge = SYS.pipeline.on_challenge_solved(str(body.get("token", "")))
                self._send(200, {"solved": challenge is not None})
            elif self.path == "/feedback":
                ok = SYS.pipeline.on_feedback(
                    str(body.get("request_id", "")),
                    int(body.get("label", 0)),
                    str(body.get("source", "operator")),
                )
                self._send(200 if ok else 404,
                           {"ok": ok, "error": None if ok else "unknown request_id"})
            elif self.path == "/retrain":
                summary = SYS.pipeline.retrain(SYS.seed_X, SYS.seed_y)
                summary["model_top_features"] = self._model_features()
                self._send(200, summary)
            elif self.path == "/replay":
                was_running = _replaying
                threading.Thread(target=_replay_traffic, daemon=True).start()
                self._send(200, {"started": not was_running})
            else:
                self._send(404, {"error": "not found"})

    return Handler


def main(host: str = "127.0.0.1", port: int = 8500) -> None:
    server = ThreadingHTTPServer((host, port), _make_handler())
    print(f"botshield listening on http://{host}:{port}  (Ctrl-C to stop)")
    print(f"dashboard: http://{host}:{port}/   —   seed model on "
          f"{len(SYS.seed_X)} labeled requests")
    if _DATA_DIR:
        print(f"persisting to {_DATA_DIR}/ "
              f"({len(SYS.store.labels())} labels loaded from disk)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
