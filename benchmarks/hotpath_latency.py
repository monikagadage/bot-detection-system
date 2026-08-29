"""
How long does one request spend in the detector?

Times `pipeline.check()` end to end over a realistic traffic mix, then times
each stage in isolation so you can see where the milliseconds go.

    python3 benchmarks/hotpath_latency.py
"""

from __future__ import annotations

import time

from _bench import flatten, percentiles

from botshield import rules
from botshield.bootstrap import build_system
from botshield.decision import decide
from botshield.features import event_target_key, extract
from botshield.synth import generate_sessions

WARMUP = 2000


def main() -> None:
    sysm = build_system(seed=0)
    events = flatten(generate_sessions(
        {"human": 400, "naive_scraper": 60, "crawler": 40, "mimic": 40,
         "stuffing_campaign": 8, "form_spammer": 40, "credential_stuffer": 30},
        seed=7, ts_start=2_000_000.0))

    # warm up (fill windows, prime caches)
    for ev in events[:WARMUP]:
        sysm.pipeline.check(ev)

    sample = events[WARMUP:]
    end_to_end = []
    for ev in sample:
        t0 = time.perf_counter()
        sysm.pipeline.check(ev)
        end_to_end.append(time.perf_counter() - t0)

    print(f"pipeline.check()  —  {len(sample)} requests, warm store\n")
    pct = percentiles(end_to_end)
    for k, v in pct.items():
        print(f"  {k:>6}  {v:8.1f} µs")
    print(f"  {'mean':>6}  {sum(end_to_end) / len(end_to_end) * 1e6:8.1f} µs")
    print(f"\n  sustained: ~{len(sample) / sum(end_to_end):,.0f} req/s single-threaded\n")

    # --- per-stage breakdown on a fresh warm system --------------------
    sysm2 = build_system(seed=0)
    for ev in events[:WARMUP]:
        sysm2.pipeline.check(ev)
    store = sysm2.store
    model = sysm2.model
    stage = {"record": [], "extract": [], "rules": [], "model+decide": []}
    for ev in sample[:5000]:
        t = time.perf_counter()
        store.record(ev.ip, ev.ts, ev.path)
        store.record_target(event_target_key(ev), ev.ts, ev.ip)
        stage["record"].append(time.perf_counter() - t)

        t = time.perf_counter()
        feats = extract(ev, store)
        stage["extract"].append(time.perf_counter() - t)

        t = time.perf_counter()
        hard, rscore, hits = rules.evaluate(ev, feats, store)
        stage["rules"].append(time.perf_counter() - t)

        t = time.perf_counter()
        decide(ev, feats, hard, rscore, hits, model, store)
        stage["model+decide"].append(time.perf_counter() - t)

    print("  per-stage median (µs):")
    for name, xs in stage.items():
        print(f"    {name:<14} {sorted(xs)[len(xs) // 2] * 1e6:7.1f}")


if __name__ == "__main__":
    main()
