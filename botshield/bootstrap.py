"""
Bootstrap — build a ready-to-serve system: seed dataset, trained model,
store, wired pipeline.

Returned so callers (the HTTP server, the simulator, the self-check) all
start from the same place.

With `persist_path` / `model_path` set, state and the learned model survive
a restart: the model is loaded from JSON if the file exists (otherwise
trained from the seed set and saved), and if the store came back with
labels from disk the model is retrained to fold them in.
"""

from __future__ import annotations

import os
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


def build_system(
    seed: int = 0,
    persist_path: str | None = None,
    model_path: str | None = None,
    counter: str = "exact",
) -> System:
    sessions = generate_sessions(seed_session_counts(), seed=seed)
    seed_X, seed_y = build_dataset(sessions)

    if model_path and os.path.exists(model_path):
        with open(model_path) as fh:
            model = LogisticRegression.from_json(fh.read())
    else:
        model = LogisticRegression(len(FEATURE_NAMES))
        model.fit(seed_X, seed_y)

    store = Store(persist_path=persist_path, counter=counter)
    pipeline = Pipeline(store, model, model_path=model_path)

    # If durable labels came back from disk, catch the freshly built model up.
    if store.labels():
        pipeline.retrain(seed_X, seed_y)
    elif model_path and not os.path.exists(model_path):
        pipeline.save_model()

    return System(store, model, pipeline, seed_X, seed_y)
