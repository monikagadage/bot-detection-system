"""Shared fixtures.

The seed model is expensive to train (a few hundred epochs of hand-rolled
gradient descent), so it is built once per test session and handed out
read-only. Tests that mutate a model ask for `fresh_model` instead.
"""

from __future__ import annotations

import pytest

from botshield.bootstrap import build_system
from botshield.events import RequestEvent
from botshield.model import LogisticRegression
from botshield.pipeline import Pipeline
from botshield.store import Store


@pytest.fixture(scope="session")
def system():
    """A fully wired, trained system. Do not mutate — session-scoped."""
    return build_system(seed=0)


@pytest.fixture
def store():
    """An empty in-memory store."""
    return Store()


@pytest.fixture
def fresh_model(system):
    """A detached copy of the seed model, safe to retrain / nudge."""
    return LogisticRegression.from_json(system.model.to_json())


@pytest.fixture
def pipeline(system, fresh_model):
    """A pipeline with an empty store but the trained seed model."""
    return Pipeline(Store(), fresh_model)


@pytest.fixture
def make_event():
    """Factory for RequestEvents with browser-ish defaults, overridable per call."""

    def _make(**overrides):
        base = dict(
            ip="73.5.9.2",
            path="/article/3",
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15) Safari/605.1.15",
            accept="text/html",
            accept_language="en-US",
            cookie="session=42",
            ts=1_000_000.0,
        )
        base.update(overrides)
        return RequestEvent(**base)

    return _make
