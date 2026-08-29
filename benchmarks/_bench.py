"""Shared helpers for the benchmark scripts. Importing this also puts the
repo root on sys.path so `import botshield...` works when a script is run as
`python3 benchmarks/whatever.py`."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def percentiles(samples_s: list[float], ps=(50, 90, 99, 99.9)) -> dict[str, float]:
    """Given durations in seconds, return {'p50': microseconds, ...}."""
    ordered = sorted(samples_s)
    out = {}
    for p in ps:
        idx = min(len(ordered) - 1, int(round(p / 100 * (len(ordered) - 1))))
        out[f"p{p:g}"] = ordered[idx] * 1e6
    return out


def flatten(sessions) -> list:
    """generate_sessions(...) -> a flat, time-ordered list of RequestEvents."""
    events = [ev for _n, evs, _l in sessions for ev in evs]
    events.sort(key=lambda e: e.ts)
    return events


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.0f} TB"
