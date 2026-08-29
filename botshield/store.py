"""
The state store — everything the detector needs to remember between requests.

In production this is split across systems:
  - sliding-window request counters  -> Redis (or a per-edge in-memory store)
  - IP / ASN reputation              -> a feed refreshed from threat intel
  - labels for training              -> a data warehouse / feature store
  - challenge tokens                 -> a short-TTL cache
  - decision log                     -> a message queue into a data lake

Here it is one in-memory object behind a lock. The lock matters: the HTTP
server is threaded, so several requests touch this at once.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

# IPs whose first two octets match one of these are treated as having poor
# reputation (score 0.8). Stands in for a real reputation feed. 185.* and
# 45.* are ranges that, in the traffic simulator, only bots use.
_STATIC_BAD_PREFIXES = ("185.", "45.", "193.")

# How long to keep per-IP request history. Feature windows are <= 60s, so
# 120s of retention is plenty and keeps memory bounded.
_RETENTION_S = 120.0


class Store:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: dict[str, deque] = defaultdict(deque)   # ip -> deque[(ts, path)]
        self._targets: dict[str, deque] = defaultdict(deque)  # target_key -> deque[(ts, ip)]
        self._labels: list[tuple[dict, int, str]] = []        # (features, label, source)
        self._challenges: dict[str, dict] = {}                # token -> {ip, issued, solved}
        self._decisions: dict[str, dict] = {}                 # request_id -> {ip, features, action}
        self._blocklist: set[str] = set()
        self._reputation: dict[str, float] = {}              # dynamic overrides

    # ---- sliding-window request history -------------------------------------

    def record(self, ip: str, ts: float, path: str) -> None:
        """Append a request and drop anything older than the retention window."""
        with self._lock:
            dq = self._events[ip]
            dq.append((ts, path))
            cutoff = ts - _RETENTION_S
            while dq and dq[0][0] < cutoff:
                dq.popleft()

    def window(self, ip: str, now: float, seconds: float) -> list[tuple[float, str]]:
        """Every (ts, path) from this IP in the last `seconds` seconds."""
        lo = now - seconds
        with self._lock:
            return [(t, p) for (t, p) in self._events.get(ip, ()) if t >= lo]

    # ---- per-target (per-endpoint) request history -------------------------
    # Keyed by route+ASN instead of by IP, so a coordinated attack spread
    # across thousands of quiet IPs still shows up as one loud endpoint.

    def record_target(self, target_key: str, ts: float, ip: str) -> None:
        with self._lock:
            dq = self._targets[target_key]
            dq.append((ts, ip))
            cutoff = ts - _RETENTION_S
            while dq and dq[0][0] < cutoff:
                dq.popleft()

    def target_window(self, target_key: str, now: float, seconds: float) -> list[tuple[float, str]]:
        """Every (ts, ip) hitting this route+ASN in the last `seconds` seconds."""
        lo = now - seconds
        with self._lock:
            return [(t, ip) for (t, ip) in self._targets.get(target_key, ()) if t >= lo]

    # ---- reputation / blocklist -------------------------------------------

    def reputation(self, ip: str) -> float:
        """0.0 (clean) .. 1.0 (known bad). Dynamic overrides beat the static feed."""
        with self._lock:
            if ip in self._reputation:
                return self._reputation[ip]
        return 0.8 if ip.startswith(_STATIC_BAD_PREFIXES) else 0.0

    def set_reputation(self, ip: str, score: float) -> None:
        with self._lock:
            self._reputation[ip] = max(0.0, min(1.0, score))

    def is_blocked(self, ip: str) -> bool:
        with self._lock:
            return ip in self._blocklist

    def block(self, ip: str) -> None:
        with self._lock:
            self._blocklist.add(ip)

    # ---- labels (for training) ------------------------------------------

    def add_label(self, features: dict, label: int, source: str) -> None:
        with self._lock:
            self._labels.append((dict(features), int(label), source))

    def labels(self) -> list[tuple[dict, int, str]]:
        with self._lock:
            return list(self._labels)

    # ---- challenge tokens ------------------------------------------------

    def issue_challenge(self, ip: str, ts: float) -> str:
        with self._lock:
            token = f"chal-{len(self._challenges)}-{int(ts * 1000) % 1_000_000}"
            self._challenges[token] = {"ip": ip, "issued": ts, "solved": False}
            return token

    def solve_challenge(self, token: str) -> dict | None:
        with self._lock:
            c = self._challenges.get(token)
            if c is None:
                return None
            c["solved"] = True
            return dict(c)

    # ---- decision log --------------------------------------------------

    def log_decision(self, request_id: str, ip: str, features: dict, action: str) -> None:
        with self._lock:
            self._decisions[request_id] = {
                "ip": ip,
                "features": dict(features),
                "action": action,
            }

    def get_decision(self, request_id: str) -> dict | None:
        with self._lock:
            rec = self._decisions.get(request_id)
            return dict(rec) if rec is not None else None

    def stats(self) -> dict:
        with self._lock:
            actions: dict[str, int] = defaultdict(int)
            for rec in self._decisions.values():
                actions[rec["action"]] += 1
            label_sources: dict[str, int] = defaultdict(int)
            for _f, _l, src in self._labels:
                label_sources[src] += 1
            return {
                "tracked_ips": len(self._events),
                "tracked_targets": len(self._targets),
                "decisions": len(self._decisions),
                "decisions_by_action": dict(actions),
                "labels": len(self._labels),
                "labels_by_source": dict(label_sources),
                "challenges_issued": len(self._challenges),
                "challenges_solved": sum(1 for c in self._challenges.values() if c["solved"]),
                "blocklist_size": len(self._blocklist),
            }
