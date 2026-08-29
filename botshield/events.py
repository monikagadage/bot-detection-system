"""
The request event — the single unit of input the system reasons about.

In a real deployment this is assembled at the edge (a reverse proxy, API
gateway, or CDN worker) from the incoming HTTP request: client IP, the
headers, the path, the TLS fingerprint, and so on. Here it is a plain
dataclass so tests and the traffic simulator can build one directly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# Substrings that, if present in a User-Agent, are a dead giveaway that the
# client is a script or crawler rather than a browser a human is driving.
# Real systems keep a much larger, frequently-updated list (and also look at
# the *absence* of tokens a real browser always sends).
BOT_UA_TOKENS = (
    "bot", "spider", "crawler", "slurp",
    "curl", "wget", "python-requests", "python-urllib", "aiohttp",
    "scrapy", "httpclient", "okhttp", "go-http-client", "java/",
    "headless", "phantomjs", "puppeteer", "playwright",
)


@dataclass
class RequestEvent:
    """One HTTP request as the detector sees it."""

    ip: str
    method: str = "GET"
    path: str = "/"
    user_agent: str = ""
    accept: str = ""
    accept_language: str = ""
    cookie: str = ""
    # A hidden form field styled off-screen. Real users never see it, so they
    # never fill it. A dumb form bot fills every field it finds.
    honeypot_value: str = ""
    # Which kind of network the IP belongs to. "datacenter" = a hosting /
    # cloud provider (AWS, OVH, Hetzner...), where almost no real end users
    # browse from; "residential" = a home ISP or mobile carrier.
    asn_type: str = "residential"
    ts: float = field(default_factory=time.time)

    @classmethod
    def from_json(cls, d: dict) -> "RequestEvent":
        """Build an event from a decoded JSON body (the HTTP server's input)."""
        return cls(
            ip=str(d.get("ip", "0.0.0.0")),
            method=str(d.get("method", "GET")),
            path=str(d.get("path", "/")),
            user_agent=str(d.get("user_agent", "")),
            accept=str(d.get("accept", "")),
            accept_language=str(d.get("accept_language", "")),
            cookie=str(d.get("cookie", "")),
            honeypot_value=str(d.get("honeypot_value", "")),
            asn_type=str(d.get("asn_type", "residential")),
            ts=float(d.get("ts", time.time())),
        )
