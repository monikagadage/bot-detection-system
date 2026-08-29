"""
Model serving and training cost.

- `predict_proba`: the only model work on the hot path — how many ns?
- `fit`: retraining is off the hot path, but it still has to finish between
  label batches. How does it scale with the training-set size?

    python3 benchmarks/model_cost.py
"""

from __future__ import annotations

import time

import _bench  # noqa: F401  (puts the repo root on sys.path)

from botshield.bootstrap import build_system
from botshield.features import FEATURE_NAMES
from botshield.model import LogisticRegression
from botshield.synth import build_dataset, generate_sessions


def main() -> None:
    sysm = build_system(seed=0)

    # --- inference ---------------------------------------------------
    n = 200_000
    x = sysm.seed_X[0]
    t0 = time.perf_counter()
    for _ in range(n):
        sysm.model.predict_proba(x)
    dt = time.perf_counter() - t0
    print(f"predict_proba: {dt / n * 1e9:6.0f} ns/call   "
          f"({n / dt:,.0f} calls/s, {len(FEATURE_NAMES)} features)\n")

    # --- training vs dataset size ----------------------------------
    print(f"{'train rows':>12} │ {'fit() time':>12} │ {'per epoch':>10}")
    print("─" * 12 + "─┼─" + "─" * 12 + "─┼─" + "─" * 10)
    for scale in (1, 2, 4, 8):
        sessions = generate_sessions(
            {"human": 60 * scale, "naive_scraper": 20 * scale, "crawler": 15 * scale,
             "form_spammer": 12 * scale, "credential_stuffer": 10 * scale,
             "stuffing_campaign": 4 * scale}, seed=scale)
        X, y = build_dataset(sessions)
        m = LogisticRegression(len(FEATURE_NAMES))
        t0 = time.perf_counter()
        m.fit(X, y, epochs=400)
        dt = time.perf_counter() - t0
        print(f"{len(X):>12,} │ {dt:>10.2f} s │ {dt / 400 * 1e3:>8.1f} ms")


if __name__ == "__main__":
    main()
