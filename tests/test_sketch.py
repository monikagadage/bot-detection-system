"""Count-min sketch: never under-counts, tight on heavy hitters, flat memory."""

from __future__ import annotations

import random

from botshield.sketch import CountMinSketch


def test_never_undercounts():
    cms = CountMinSketch(width=512, depth=4)
    truth: dict[str, int] = {}
    rng = random.Random(0)
    for _ in range(10_000):
        k = f"ip-{rng.randint(0, 800)}"
        cms.add(k)
        truth[k] = truth.get(k, 0) + 1
    assert all(cms.estimate(k) >= v for k, v in truth.items())


def test_heavy_hitter_estimate_is_tight():
    cms = CountMinSketch(width=2048, depth=5)
    for _ in range(5_000):
        cms.add("whale")
    for i in range(5_000):
        cms.add(f"minnow-{i}")
    # Within epsilon * total of the truth; the whale stays accurate.
    assert abs(cms.estimate("whale") - 5_000) <= 0.02 * cms.total()


def test_memory_is_constant_in_key_count():
    small = CountMinSketch(width=1024, depth=4)
    big = CountMinSketch(width=1024, depth=4)
    for i in range(100):
        small.add(f"k{i}")
    for i in range(1_000_000):
        big.add(f"k{i}")
    assert small.memory_bytes() == big.memory_bytes()


def test_unseen_key_estimates_zero():
    cms = CountMinSketch(width=1024, depth=4)
    cms.add("seen")
    assert cms.estimate("never-added") == 0


def test_total_tracks_adds():
    cms = CountMinSketch()
    cms.add("a", 3)
    cms.add("b")
    assert cms.total() == 4
