"""
Count-Min Sketch — estimate "how many times have I seen this key" using a
fixed amount of memory, no matter how many distinct keys there are.

The problem it solves: under a botnet attack you might see millions of
distinct IPs in a minute. An exact `dict[ip] -> count` grows with the number
of attackers — exactly when you can least afford it. A sketch trades a
little accuracy for flat memory.

How it works:

  - `depth` independent hash functions, each mapping a key to one of `width`
    counters (a 2D array, `depth` rows x `width` columns).
  - `add(key)` increments one counter per row.
  - `estimate(key)` reads one counter per row and returns the **minimum**.
    Every counter is >= the true count (other keys only ever add to it), so
    the smallest is the tightest over-estimate. It never under-counts.

Error is bounded: with width = e/epsilon and depth = ln(1/delta), the
estimate is within `epsilon * total_count` of the truth with probability
`1 - delta`. Heavy hitters (the IPs flooding you) stay accurate; the noise
is in the long tail you don't care about.

This is the same structure behind Redis's `CMS.*` commands and a lot of
streaming-analytics systems.
"""

from __future__ import annotations

import hashlib


class CountMinSketch:
    def __init__(self, width: int = 2048, depth: int = 4) -> None:
        self.width = width
        self.depth = depth
        self._rows = [[0] * width for _ in range(depth)]
        self._total = 0

    def _cols(self, key: str) -> list[int]:
        # One digest, sliced into `depth` independent column indices.
        digest = hashlib.blake2b(key.encode(), digest_size=self.depth * 4).digest()
        return [
            int.from_bytes(digest[i * 4:(i + 1) * 4], "big") % self.width
            for i in range(self.depth)
        ]

    def add(self, key: str, count: int = 1) -> None:
        for row, col in zip(self._rows, self._cols(key)):
            row[col] += count
        self._total += count

    def estimate(self, key: str) -> int:
        return min(row[col] for row, col in zip(self._rows, self._cols(key)))

    def total(self) -> int:
        return self._total

    def memory_bytes(self) -> int:
        """Rough fixed footprint of the counter grid (ints ~ 28 bytes each in
        CPython, but the grid size is what matters — it never grows)."""
        return self.width * self.depth * 28
