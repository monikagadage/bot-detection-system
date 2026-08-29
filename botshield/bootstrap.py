"""
Bootstrap — build a ready-to-serve system: seed dataset, trained model,
fresh store, wired pipeline.

Returned so callers (the HTTP server, the simulator, the self-check) all
start from the same place.
"""

from __future__ import annotations

from dataclasses import dataclass

from .features import FEATURE_NAMES
from .model import LogisticRegression
from .pipeline import Pipeline
from .store import Store
from .synth import build_dataset, generate_sessions, seed_session_counts


@dataclass
class System:
    store: Store
    model: LogisticRegression
    pipeline: Pipeline
    seed_X: list[list[float]]
    seed_y: list[int]


def build_system(seed: int = 0) -> System:
    sessions = generate_sessions(seed_session_counts(), seed=seed)
    seed_X, seed_y = build_dataset(sessions)

    model = LogisticRegression(len(FEATURE_NAMES))
    model.fit(seed_X, seed_y)

    store = Store()
    pipeline = Pipeline(store, model)
    return System(store, model, pipeline, seed_X, seed_y)
