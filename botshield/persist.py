"""
Persistence — an optional SQLite mirror for the parts of the store that must
survive a restart: the labels the model is trained on, the decision log
(so /feedback can still refer to a request_id after a redeploy), and issued
challenge tokens.

The in-memory structures in `Store` stay the source of truth for *reads*
(they're fast and already there); this just writes through to disk and
reloads on startup. The learned model is persisted separately as JSON — see
`bootstrap.build_system` / `Pipeline.retrain`.

Everything here runs while the caller holds `Store._lock`, so the sqlite
connection needs `check_same_thread=False` and no locking of its own.
"""

from __future__ import annotations

import json
import os
import sqlite3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL, source TEXT, label INTEGER, features_json TEXT
);
CREATE TABLE IF NOT EXISTS decisions (
    request_id TEXT PRIMARY KEY,
    ts REAL, ip TEXT, action TEXT, features_json TEXT
);
CREATE TABLE IF NOT EXISTS challenges (
    token TEXT PRIMARY KEY,
    ip TEXT, issued REAL, solved INTEGER
);
"""


class Db:
    def __init__(self, path: str) -> None:
        self.path = path
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ---- writes (caller holds Store._lock) --------------------------------

    def add_label(self, features: dict, label: int, source: str, ts: float) -> None:
        self._conn.execute(
            "INSERT INTO labels (ts, source, label, features_json) VALUES (?, ?, ?, ?)",
            (ts, source, int(label), json.dumps(features)),
        )
        self._conn.commit()

    def put_decision(self, request_id: str, ip: str, features: dict, action: str, ts: float) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO decisions (request_id, ts, ip, action, features_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (request_id, ts, ip, action, json.dumps(features)),
        )
        self._conn.commit()

    def put_challenge(self, token: str, ip: str, issued: float, solved: bool) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO challenges (token, ip, issued, solved) VALUES (?, ?, ?, ?)",
            (token, ip, issued, 1 if solved else 0),
        )
        self._conn.commit()

    # ---- load (startup, single-threaded) -------------------------------

    def load_labels(self) -> list[tuple[dict, int, str]]:
        rows = self._conn.execute(
            "SELECT features_json, label, source FROM labels ORDER BY id"
        ).fetchall()
        return [(json.loads(fj), int(lbl), src) for (fj, lbl, src) in rows]

    def load_decisions(self) -> dict[str, dict]:
        rows = self._conn.execute(
            "SELECT request_id, ip, features_json, action FROM decisions"
        ).fetchall()
        return {rid: {"ip": ip, "features": json.loads(fj), "action": act}
                for (rid, ip, fj, act) in rows}

    def load_challenges(self) -> dict[str, dict]:
        rows = self._conn.execute(
            "SELECT token, ip, issued, solved FROM challenges"
        ).fetchall()
        return {tok: {"ip": ip, "issued": iss, "solved": bool(sv)}
                for (tok, ip, iss, sv) in rows}
