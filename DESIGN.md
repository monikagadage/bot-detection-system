# Bot Detection — Design

A small, working model of how a site tells automated clients apart from
humans in the request path: feature extraction, a deterministic rules
engine, a learned scorer, an `ALLOW` / `CHALLENGE` / `BLOCK` policy, and a
feedback loop that retrains on what it got wrong. Not production
infrastructure — the same *ideas* stripped down to something readable end to
end. See [README.md](README.md) to run it.

## The problem

A public website gets traffic from two kinds of clients: **humans driving
browsers**, and **automated programs** — scrapers copying your content,
price bots, credential-stuffing tools trying leaked passwords against your
login, scalpers buying limited stock, spam bots filling forms, and crawlers
you never invited. Somewhere between 30% and 50% of raw traffic to a typical
site is automated.

You want to let the humans through untouched and stop (or slow, or
challenge) the bots — **in the request path, in a few milliseconds**, and
**without blocking real customers**, because a blocked customer is lost
revenue and a support ticket.

This is an **arms race**: every signal you use, a determined attacker will
eventually mimic. So the system is never "done" — it has to keep learning.

## Vocabulary

- **Signal / feature** — one measurable property of a request (its
  User-Agent, how fast this IP is going, whether it sent a cookie). The
  detector's whole job is combining many weak signals into one decision.
- **False positive (FP)** — a human classified as a bot. The expensive
  mistake: it blocks a paying customer. Bot systems are tuned to keep the FP
  rate tiny even if that means missing some bots.
- **False negative (FN)** — a bot classified as human. Costs you scraping
  bandwidth, fraud, or inventory, but not a customer.
- **Challenge** — instead of allow-or-block, make the client *prove* it's a
  browser: run some JavaScript, solve a proof-of-work, or (last resort) a
  CAPTCHA. Cheap for a real browser, expensive at bot-farm scale. This is the
  "I'm not sure" action.
- **Honeypot** — a trap that only a bot trips. Classic version: a form field
  hidden with CSS. A human never sees it, so a filled-in value ⇒ bot, with
  near-certainty.
- **Fingerprint** — a semi-stable identifier derived from *how* a client
  connects, not *who* it is: the exact TLS handshake (JA3/JA4), HTTP/2
  settings, header order, or a hash of browser/JS properties. Two requests
  with the same fingerprint are probably the same software.
- **Reputation** — a score attached to an IP, ASN, or fingerprint based on
  its past behaviour across the whole network. "This /24 sent 4M login
  attempts last week" is reputation.
- **ASN** — Autonomous System Number, i.e. which network operator owns an IP
  range. **Datacenter** ASNs (AWS, OVH, Hetzner) host almost no real end
  users; **residential** ASNs (Comcast, Vodafone) are where humans browse.
  Residential-proxy services exist precisely to launder bot traffic through
  residential ASNs.
- **Training/serving skew** — when features are computed one way during
  model training and a subtly different way in production, so the live model
  behaves worse than the offline metrics promised. Avoided here by having
  `features.py` be the *only* place features are computed.

## Where this sits in a real stack

```
        client
          │  HTTP request
          ▼
   ┌──────────────┐     the bot detector runs as a hook here — a CDN worker,
   │ edge / proxy │◀──  a reverse-proxy module (nginx/Envoy), an API gateway
   │  + detector  │     plugin, or a dedicated managed service (Cloudflare Bot
   └──────┬───────┘     Management, DataDome, HUMAN, Akamai Bot Manager).
          │  ALLOW (or CHALLENGE passed)
          ▼
     origin servers
```

The detector must answer **synchronously** and **fast** (single-digit ms),
because every request waits on it. Everything it *learns* from happens
**asynchronously**, off the request path.

## Architecture

```
                       ┌──────────────────────── async ────────────────────────┐
                       │                                                        │
  request              │   decision log ──▶ data lake ──▶ offline training ──┐  │
    │                  │        ▲                                            │  │
    ▼                  │        │                                     model  │  │
┌─────────┐  ┌──────────┴──┐ ┌──┴────────┐  ┌──────────┐  ┌───────────┐ registry│
│ ingest  │─▶│  feature    │▶│  rules    │─▶│  model   │─▶│ decision  │   │     │
│ (edge)  │  │ extraction  │ │  engine   │  │  scorer  │  │ + policy  │◀──┘     │
└─────────┘  └──────┬──────┘ └───────────┘  └──────────┘  └─────┬─────┘         │
                    │                                          │               │
             ┌──────▼───────┐                          ALLOW / CHALLENGE / BLOCK
             │ state store  │                                   │               │
             │ (Redis-like) │   labels ◀── challenge outcomes ──┤               │
             │ sliding      │         ◀── honeypot hits ────────┤               │
             │ windows,     │         ◀── operator /feedback ───┘               │
             │ reputation   │                                                   │
             └──────────────┘───────────────────────────────────────────────────┘
```

1. **Ingest** — the edge hook assembles a *request event*: IP, headers,
   path, method, TLS fingerprint, timestamp. (`botshield/events.py`)
2. **Feature extraction** — combine the event with per-IP window state from
   the store into a fixed numeric row. (`botshield/features.py`)
3. **State store** — sliding-window request counters, keyed two ways: **per
   IP** ("how fast is this client going?") and **per target** (route + ASN —
   "is this endpoint under coordinated load?"). Plus the reputation feed,
   labels, and challenge tokens. In production: Redis for counters (sharded
   by key), a separate feature store / warehouse for labels.
   (`botshield/store.py`)
4. **Rules engine** — deterministic checks in two tiers. **Hard rules**
   (honeypot filled, 50 req/10s, blocklisted IP) short-circuit to BLOCK.
   **Escalation rules** (`distributed_attack`) force *at least* a CHALLENGE
   even if the blended score would have allowed the request. Everything else
   is a **soft rule** contributing to a saturating weighted score. Fast to
   write, instantly deployable, explainable. (`botshield/rules.py`)
5. **Model scorer** — a classifier trained offline on labeled traffic,
   served as fixed weights. Catches combinations of weak signals that no
   single rule expresses. Here it's logistic regression from scratch; real
   systems use gradient-boosted trees or deep nets.
   (`botshield/model.py`)
6. **Decision + policy** — blend `rule_score` and `model_proba` into one
   number, then threshold it: `< 0.45` ALLOW, `0.45–0.80` CHALLENGE, `≥ 0.80`
   BLOCK. The thresholds are the operator's dial against the FP rate.
   (`botshield/decision.py`)
7. **Feedback loop** — challenge outcomes, honeypot hits, and operator
   labels flow back as training data. A retrain folds them in and swaps new
   weights into the live model. (`botshield/pipeline.py`)

## Signals, by layer (what real systems use)

| Layer | Examples | Why bots differ |
|---|---|---|
| Network | IP/ASN reputation, datacenter vs residential, requests/sec per IP | Bots run from cloud hosts and hit rates humans can't |
| Per-target | requests/min and distinct-IP count *per endpoint*, not per client | A distributed attack is invisible per-IP (each IP sends one request) but obvious per-endpoint (10k IPs on /login in a minute) |
| Transport | TLS fingerprint (JA3/JA4), HTTP/2 settings | A Python client's TLS stack ≠ Chrome's, even with a spoofed UA |
| HTTP | UA string, header presence + *order*, Accept/Accept-Language | Scripts omit headers browsers always send, or send them in the wrong order |
| Behavioral | inter-request timing regularity, navigation graph, mouse/scroll/keyboard events | Humans are irregular and bursty; loops are metronomes |
| Challenge | JS proof-of-work result, CAPTCHA, canvas/WebGL fingerprint | Simple bots can't run JS; headless browsers leave tells |
| Reputation | this fingerprint's history across the whole network | The first site sees a new attack; the 1000th site blocks it on arrival |

This mini version implements the **network**, **per-target**, **HTTP**, and
**behavioral** rows, plus a **honeypot** and a **challenge** action.

### Per-target aggregation (the credential-stuffing answer)

Per-IP features are blind to a distributed attack: run 10,000 login attempts
through a residential proxy pool and every individual IP looks like one
ordinary visitor — clean reputation, real browser headers, a single request.
The attack only exists in aggregate.

So the store keeps a second set of sliding windows keyed by
`route + ASN` (`/login|residential`) instead of by IP, and feature
extraction adds `target_req_60s` and `target_distinct_ips_60s`. When an
endpoint sees many requests from many distinct IPs at once, the
`distributed_attack` rule fires — and because it's an **escalation rule**,
every request to that endpoint is forced to at least a CHALLENGE. Real users
pass it transparently; the bot farm, which can't solve challenges at scale,
stalls. `simulate.py`'s `stuffing_campaign` archetype shows this: the same
requests that are `ALLOW`ed one at a time are `CHALLENGE`d / `BLOCK`ed once
they arrive as a swarm.

## Key design decisions

- **Three actions, not two.** Forcing every "not sure" into ALLOW or BLOCK
  is what generates blocked-customer tickets. CHALLENGE absorbs the
  uncertain middle: a real browser passes transparently, a bot farm eats the
  cost.
- **Rules *and* a model, not one or the other.** Rules give you speed
  (minutes to ship), explainability (support can read them), and hard
  guarantees. The model gives you coverage of signal *combinations* nobody
  wrote a rule for. They cover each other's weaknesses.
- **Evaluate sync, learn async.** The request path only reads fixed model
  weights and O(1) counters. Training, which is expensive and bursty,
  happens on logged data completely off the hot path.
- **One feature module.** `features.py` is imported by the live path *and*
  the trainer, so training/serving skew is impossible by construction.
- **Cheap labels first.** Honeypot hits and challenge outcomes are ground
  truth you get for free, at volume, continuously. Manual review is the
  expensive supplement, not the main source.

## Scaling & operations

- **Counters** are sharded by IP (consistent hashing) across a Redis fleet;
  each shard is O(1) per request. Approximate structures keep memory flat
  under attack: this repo's `counter="sketch"` mode
  (`BOTSHIELD_COUNTER=sketch`) estimates request *rates* from a fixed-size
  count-min sketch (`sketch.py`) instead of exact per-IP history, so a flood
  of millions of distinct IPs costs the same memory as an idle server. The
  trade-off: the estimate can over-count (never under-count), timing /
  distinct-path features fall back to a bounded recent tail, and the model —
  trained on exact features — sees slight training/serving skew.
- **Scorers** are stateless — the model is a few KB of weights shipped to
  every edge node. Scale them horizontally behind the load balancer.
- **Model rollout** is canaried: serve the new model to 1% of traffic, watch
  the FP rate and catch rate, then ramp. Keep the previous model hot for
  instant rollback.
- **Model decay** is constant because attackers adapt — retrain daily to
  weekly. A flat catch rate over time is the metric to alarm on.

## Failure modes

- **Store outage** → fail *open* to rules-only (a slow site loses less money
  than a down one). A login or checkout endpoint might instead fail
  *closed*. This is a per-route policy choice.
- **Model bug ships** → canary + one-click rollback to the last known-good
  weights in the registry.
- **Latency spike in the detector** → hard timeout, then fail-open; never
  let the detector become the outage.
- **Attacker probes your thresholds** → don't return *why* a request was
  blocked to the client; add noise to thresholds; rotate honeypots.
- **Feedback poisoning** → an attacker who can submit labels teaches your
  model wrong. Labels must be authenticated and rate-limited, and retrains
  gated on offline metrics before rollout.

## What this mini version simplifies

- **No TLS/HTTP-2 fingerprinting** — the highest-signal features in real
  systems. We only see application-level headers.
- **State is one in-memory `Store` object**, not a sharded Redis fleet;
  counters are exact lists, not sketches. It can optionally write labels /
  decisions / challenges through to a local SQLite file and reload them on
  restart (`persist.py`, `BOTSHIELD_DATA=./data`), with the learned model
  saved alongside as JSON — but that's a single-node stand-in for what would
  be a warehouse + feature store + model registry.
- **The model is logistic regression** trained in-process on a few thousand
  synthetic rows. Real models are GBMs/DNNs trained on billions of real,
  human-labeled requests.
- **Per-target aggregation is coarse.** It keys on `route + ASN` and counts
  requests + distinct IPs. A real system also tracks failure ratio (5xx /
  401 / 403), distinct usernames tried, and geographic spread, and keys on
  finer buckets. The distinct-IP count here is exact; at scale it'd be a
  HyperLogLog sketch.
- **No canary / model registry / rollback** — `retrain` swaps weights in
  immediately.
- **The challenge is a stub** — solving it just flips a boolean. A real one
  is a signed, single-use, short-TTL token bound to the client fingerprint.

## The building blocks underneath

This system is mostly a recombination of standard distributed-systems
pieces:

- **Hash maps** — the store is a hash map of IP → request history; the
  reputation feed is a hash lookup.
- **Sliding-window counters** — "requests in the last 60s" is exactly a
  sliding-window rate limiter, reused here as a model feature instead of as
  a limit.
- **LRU cache** — challenge tokens and per-fingerprint verdicts belong in a
  bounded LRU so memory stays flat.
- **Consistent hashing** — how the counter store is sharded across a Redis
  fleet without reshuffling every key when a node joins or leaves.
- **Load balancing** — the stateless scorers sit behind an LB and scale
  horizontally.
- **Bloom filters** — "have we seen this fingerprint before?" at memory
  scale is a Bloom-filter question.

## Run it

See [README.md](README.md). Short version:

```bash
python3 selfcheck.py     # [PASS]/[FAIL] for every component
python3 simulate.py      # end-to-end: traffic -> decisions -> feedback -> retrain
python3 server.py        # the actual service, then curl localhost:8500/check
```
