"""Route normalization and per-target keying."""

from __future__ import annotations

import pytest

from botshield.routes import route_of, target_key


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/article/5", "/article/{id}"),
        ("/article/123456", "/article/{id}"),
        ("/user/deadbeefcafe0", "/user/{id}"),
        ("/user/550e8400-e29b-41d4-a716-446655440000", "/user/{id}"),
        ("/login", "/login"),
        ("/", "/"),
        ("/article/5/", "/article/{id}"),
        ("/search?q=hello", "/search"),
        ("/api/v2/orders/42", "/api/v2/orders/{id}"),
    ],
)
def test_route_of_collapses_ids(path, expected):
    assert route_of(path) == expected


def test_short_hex_is_not_an_id():
    # "abc123" is a plausible slug, not an id — only long hex collapses.
    assert route_of("/tag/abc123") == "/tag/abc123"


def test_target_key_splits_by_asn():
    assert target_key("/login", "datacenter") != target_key("/login", "residential")
    assert target_key("/login", "datacenter") == "/login|datacenter"
