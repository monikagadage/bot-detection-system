"""
Route normalization — collapse `/article/5`, `/article/6`, `/user/abc123`
into a single *route* (`/article/{id}`) so per-endpoint aggregation counts
"traffic to the article page" rather than treating every article as its own
endpoint.

Real gateways get the route pattern for free from the router
(`/article/:id`). Here we approximate it by replacing path segments that
look like ids (all digits, or long hex/uuid-ish tokens) with `{id}`.
"""

from __future__ import annotations

import re

_ID_SEGMENT = re.compile(r"^(\d+|[0-9a-fA-F]{12,}|[0-9a-fA-F-]{16,})$")

# Endpoints where automated abuse is both common and high-impact. Used to
# weight the "distributed attack" signal — a swarm on /login matters more
# than a swarm on /about.
SENSITIVE_ROUTES = {"/login", "/signup", "/checkout", "/password-reset", "/api/token"}


def route_of(path: str) -> str:
    path = path.split("?", 1)[0].rstrip("/") or "/"
    segments = path.split("/")
    normalized = ["{id}" if _ID_SEGMENT.match(seg) else seg for seg in segments]
    return "/".join(normalized) or "/"


def target_key(route: str, asn_type: str) -> str:
    """The key per-endpoint windows are bucketed under: a route plus the kind
    of network it's coming from. Splitting by ASN separates 'lots of
    residential users on /login' (normal) from 'lots of datacenter IPs on
    /login' (an attack)."""
    return f"{route}|{asn_type}"
