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

Pass `persist_path` to mirror the labels, decision log, and challenge tokens
to a SQLite file and reload them on startup — see `persist.py`. The
in-memory structures stay the read path; SQLite is just write-through
durability.

Pass `counter="sketch"` to estimate request *rates* from a fixed-size
count-min sketch instead of exact per-IP history (`sketch.py`). Under a
high-cardinality attack (millions of distinct IPs) the exact store grows
without bound; the sketch does not. In sketch mode the per-IP / per-target
history deques are also capped, so `distinct_paths` and timing features are
computed from a bounded recent tail.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, defaultdict, deque

from .persist import Db
from .sketch import CountMinSketch

# IPs whose first two octets match one of these are treated as having poor
# reputation (score 0.8). Stands in for a real reputation feed. 185.* and
# 45.* are ranges that, in the traffic simulator, only bots use.
_STATIC_BAD_PREFIXES = ("185.", "45.", "193.")

# How long to keep per-IP request history. Feature windows are <= 60s, so
# 120s of retention is plenty and keeps memory bounded.
_RETENTION_S = 120.0

# Sketch mode: time-bucket width for rate estimation, and caps on the
# bounded structures.
_BUCKET_S = 5.0
_SKETCH_DEQUE_CAP = 64      # recent events kept per IP / per target
_SKETCH_MAX_KEYS = 4096     # distinct IPs / targets kept before LRU eviction


class Store:
    def __init__(self, persist_path: str | None = None, counter: str = "exact") -> None:
        if counter not in ("exact", "sketch"):
            raise ValueError("counter must be 'exact' or 'sketch'")
        self._lock = threading.Lock()
        self.counter = counter
        self._sketch_mode = counter == "sketch"

        maxlen = _SKETCH_DEQUE_CAP if self._sketch_mode else None
        self._events: dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))
        self._targets: dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))
        self._labels: list[tuple[dict, int, str]] = []        # (features, label, source)
        self._challenges: dict[str, dict] = {}                # token -> {ip, issued, solved}
        self._decisions: dict[str, dict] = {}                 # request_id -> {ip, features, action}
        self._blocklist: set[str] = set()
        self._reputation: dict[str, float] = {}              # dynamic overrides

        # Sketch mode: fixed-size rate estimators (key = "<id>#<bucket>"), and
        # LRU order so the history maps can be trimmed to _SKETCH_MAX_KEYS.
        self._ip_sketch = CountMinSketch() if self._sketch_mode else None
        self._target_sketch = CountMinSketch() if self._sketch_mode else None
        self._ip_lru: OrderedDict[str, None] = OrderedDict()
        self._target_lru: OrderedDict[str, None] = OrderedDict()

        # Optional SQLite durability. Reload prior state so /feedback still
        # resolves old request_ids and the model can be retrained from disk.
        self._db = Db(persist_path) if persist_path else None
        if self._db is not None:
            self._labels = self._db.load_labels()
            self._decisions = self._db.load_decisions()
            self._challenges = self._db.load_challenges()

    @staticmethod
    def _buckets(now: float, seconds: float) -> range:
        return range(int((now - seconds) // _BUCKET_S), int(now // _BUCKET_S) + 1)

    def _touch_lru(self, lru: "OrderedDict[str, None]", key: str, history: dict) -> None:
        """Record recent use of `key`; evict the least-recently-used history
        entry once the map exceeds the cap (sketch mode only)."""
        lru[key] = None
        lru.move_to_end(key)
        while len(lru) > _SKETCH_MAX_KEYS:
            old, _ = lru.popitem(last=False)
            history.pop(old, None)

    # ---- sliding-window request history -------------------------------------

    def record(self, ip: str, ts: float, path: str) -> None:
        """Append a request and drop anything older than the retention window."""
        with self._lock:
            dq = self._events[ip]
            dq.append((ts, path))
            if dq.maxlen is None:
                cutoff = ts - _RETENTION_S
                while dq and dq[0][0] < cutoff:
                    dq.popleft()
            if self._sketch_mode:
                self._ip_sketch.add(f"{ip}#{int(ts // _BUCKET_S)}")
                self._touch_lru(self._ip_lru, ip, self._events)

    def window(self, ip: str, now: float, seconds: float) -> list[tuple[float, str]]:
        """Every (ts, path) from this IP in the last `seconds` seconds.
        In sketch mode this is the bounded recent tail, not the full history."""
        lo = now - seconds
        with self._lock:
            return [(t, p) for (t, p) in self._events.get(ip, ()) if t >= lo]

    def request_rate(self, ip: str, now: float, seconds: float) -> float:
        """How many requests this IP made in the last `seconds` seconds.
        Exact from history, or estimated from the count-min sketch."""
        if not self._sketch_mode:
            return float(len(self.window(ip, now, seconds)))
        with self._lock:
            return float(sum(self._ip_sketch.estimate(f"{ip}#{b}")
                             for b in self._buckets(now, seconds)))

    # ---- per-target (per-endpoint) request history -------------------------
    # Keyed by route+ASN instead of by IP, so a coordinated attack spread
    # across thousands of quiet IPs still shows up as one loud endpoint.

    def record_target(self, target_key: str, ts: float, ip: str) -> None:
        with self._lock:
            dq = self._targets[target_key]
            dq.append((ts, ip))
            if dq.maxlen is None:
                cutoff = ts - _RETENTION_S
                while dq and dq[0][0] < cutoff:
                    dq.popleft()
            if self._sketch_mode:
                self._target_sketch.add(f"{target_key}#{int(ts // _BUCKET_S)}")
                self._touch_lru(self._target_lru, target_key, self._targets)

    def target_window(self, target_key: str, now: float, seconds: float) -> list[tuple[float, str]]:
        """Every (ts, ip) hitting this route+ASN in the last `seconds` seconds.
        In sketch mode this is the bounded recent tail, not the full history."""
        lo = now - seconds
        with self._lock:
            return [(t, ip) for (t, ip) in self._targets.get(target_key, ()) if t >= lo]

    def target_rate(self, target_key: str, now: float, seconds: float) -> float:
        """Total requests to this route+ASN in the last `seconds` seconds."""
        if not self._sketch_mode:
            return float(len(self.target_window(target_key, now, seconds)))
        with self._lock:
            return float(sum(self._target_sketch.estimate(f"{target_key}#{b}")
                             for b in self._buckets(now, seconds)))

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
            if self._db is not None:
                self._db.add_label(features, int(label), source, time.time())

    def labels(self) -> list[tuple[dict, int, str]]:
        with self._lock:
            return list(self._labels)

    # ---- challenge tokens ------------------------------------------------

    def issue_challenge(self, ip: str, ts: float) -> str:
        with self._lock:
            token = f"chal-{len(self._challenges)}-{int(ts * 1000) % 1_000_000}"
            self._challenges[token] = {"ip": ip, "issued": ts, "solved": False}
            if self._db is not None:
                self._db.put_challenge(token, ip, ts, False)
            return token

    def solve_challenge(self, token: str) -> dict | None:
        with self._lock:
            c = self._challenges.get(token)
            if c is None:
                return None
            c["solved"] = True
            if self._db is not None:
                self._db.put_challenge(token, c["ip"], c["issued"], True)
            return dict(c)

    # ---- decision log --------------------------------------------------

    def log_decision(self, request_id: str, ip: str, features: dict, action: str) -> None:
        with self._lock:
            self._decisions[request_id] = {
                "ip": ip,
                "features": dict(features),
                "action": action,
            }
            if self._db is not None:
                self._db.put_decision(request_id, ip, features, action, time.time())

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
                "counter_mode": self.counter,
                "tracked_ips": len(self._events),
                "tracked_targets": len(self._targets),
                "sketch_bytes": (self._ip_sketch.memory_bytes() if self._sketch_mode else 0),
                "decisions": len(self._decisions),
                "decisions_by_action": dict(actions),
                "labels": len(self._labels),
                "labels_by_source": dict(label_sources),
                "challenges_issued": len(self._challenges),
                "challenges_solved": sum(1 for c in self._challenges.values() if c["solved"]),
                "blocklist_size": len(self._blocklist),
                "persistent": self._db is not None,
            }
