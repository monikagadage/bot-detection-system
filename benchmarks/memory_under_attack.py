"""
Exact per-IP history vs. the count-min sketch, under a flood of distinct IPs.

Feeds the same attack (N unique IPs, one request each) into an exact `Store`
and a `counter="sketch"` `Store`, measuring peak tracked memory with
`tracemalloc` at each step. The exact store grows with the number of
attackers; the sketch stays flat.

    python3 benchmarks/memory_under_attack.py
"""

from __future__ import annotations

import tracemalloc

from _bench import human_bytes

from botshield.events import RequestEvent
from botshield.features import event_target_key
from botshield.store import Store

STEPS = [10_000, 50_000, 100_000, 250_000, 500_000]


def feed(store: Store, n: int, start: int) -> None:
    for i in range(start, start + n):
        ev = RequestEvent(
            ip=f"{10 + i // 16_777_216}.{i // 65536 % 256}.{i // 256 % 256}.{i % 256}",
            path="/login", method="POST", user_agent="python-requests/2.31",
            asn_type="datacenter", ts=1000.0 + i * 0.0005,
        )
        store.record(ev.ip, ev.ts, ev.path)
        store.record_target(event_target_key(ev), ev.ts, ev.ip)


def measure(counter: str) -> list[tuple[int, int]]:
    tracemalloc.start()
    store = Store(counter=counter)
    fed = 0
    curve = []
    for target in STEPS:
        feed(store, target - fed, fed)
        fed = target
        curve.append((fed, tracemalloc.get_traced_memory()[0]))
    tracemalloc.stop()
    return curve


def main() -> None:
    exact = measure("exact")
    sketch = measure("sketch")

    print(f"{'distinct IPs':>14} │ {'exact store':>14} │ {'sketch store':>14}")
    print("─" * 14 + "─┼─" + "─" * 14 + "─┼─" + "─" * 14)
    for (n, e_mem), (_n, s_mem) in zip(exact, sketch):
        print(f"{n:>14,} │ {human_bytes(e_mem):>14} │ {human_bytes(s_mem):>14}")

    e_growth = exact[-1][1] / exact[0][1]
    s_growth = sketch[-1][1] / sketch[0][1]
    print(f"\n  exact store grew {e_growth:.1f}x over the run; "
          f"sketch store grew {s_growth:.2f}x.")
    print("  (sketch trades exactness for a fixed footprint — see sketch.py)")


if __name__ == "__main__":
    main()
