# botshield — a bot-detection service for the request path

[![CI](https://github.com/monikagadage/bot-detection-system/actions/workflows/ci.yml/badge.svg)](https://github.com/monikagadage/bot-detection-system/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)
![Dependencies](https://img.shields.io/badge/runtime%20deps-0-brightgreen)

Automated clients — scrapers, price bots, credential-stuffing tools, form
spammers, uninvited crawlers — are 30–50% of raw traffic to a typical public
site. `botshield` decides `ALLOW` / `CHALLENGE` / `BLOCK` for each request
**synchronously, in tens of microseconds**, and keeps a tight lid on false
positives (blocking a real customer is the expensive mistake). It combines a
deterministic **rules engine** with a **learned scorer**, and an
asynchronous **feedback loop** retrains the model on requests it got wrong.

The architecture follows the managed bot-management services (Cloudflare Bot
Management, DataDome, HUMAN, Akamai Bot Manager): a synchronous detector at
the edge, everything it learns from happening off the request path. The
implementation is standard-library-only and small enough to read in full —
[DESIGN.md](DESIGN.md) covers the signal layers, scaling, and failure modes.

## Design decisions

| Decision | Why |
|---|---|
| **Rules engine *and* a model, not just a model** | Rules give an explainable reason ("honeypot field was filled"), ship in minutes when a new attack starts at 3am, and give hard guarantees for near-certain signals. The model catches the fuzzy combinations rules miss. `decision.py` blends them (weighted) and lets a hard rule short-circuit. |
| **Three actions, not two** | `CHALLENGE` (JS proof-of-work / CAPTCHA) is the "not sure" action — cheap for a human, expensive at bot-farm scale. Splitting "unsure" out of allow-or-block is what keeps false positives from becoming lost revenue. |
| **One feature module, shared by training and serving** | `features.py` is the *only* place a request becomes numbers. Training/serving skew — offline metrics that don't survive contact with production — is designed out rather than monitored for. |
| **Per-endpoint windows, keyed by route + ASN** | A credential-stuffing campaign through a residential-proxy pool shows every IP as clean in isolation. Aggregating by `(/login, datacenter)` makes the *swarm* visible even when no single IP is. |
| **Count-min sketch counter mode** | Under a high-cardinality attack (millions of distinct IPs/minute) an exact `dict[ip]→count` grows exactly when you can least afford it. `BOTSHIELD_COUNTER=sketch` swaps in fixed-memory rate estimates — flat footprint regardless of attacker count. |
| **Online nudge + periodic full retrain** | `/feedback` applies a single online gradient step so the effect is immediate; a full retrain on the seed set + all collected labels catches up later. The live model object is swapped atomically so in-flight requests never see half-updated weights. |

## Run it

```bash
python3 selfcheck.py    # [PASS]/[FAIL] for every component
python3 simulate.py     # the whole thing end to end (~10s)
python3 server.py       # the HTTP service + live dashboard on 127.0.0.1:8500
```

Or in a container:

```bash
docker build -t botshield .
docker run -p 8500:8500 -v botshield-data:/data botshield
```

Open **http://localhost:8500** for a live dashboard — decision tiles, a
traffic timeline, what the model is currently leaning on, and a rolling feed
of decisions. **Replay demo traffic** classifies a mixed human/bot stream in
real time; **Retrain model** folds in collected labels.

## Develop

```bash
make install        # venv + pytest + ruff
make check          # ruff + pytest + selfcheck  (what CI runs)
make bench          # latency / throughput / memory / model-cost benchmarks
```

CI (`.github/workflows/ci.yml`) runs lint, the pytest suite, the component
self-check, and the end-to-end simulation on Python 3.10–3.13, and
smoke-runs the benchmarks.

## `simulate.py` — the feedback loop, made visible

Starts the server in a background thread, fires ~124 mixed human/bot
sessions at it over real HTTP, and prints a confusion matrix plus a
per-archetype `ALLOW` / `CHALLENGE` / `BLOCK` breakdown. Then an "operator"
labels the bots that slipped through, the model retrains, and a fresh batch
runs.

Watch the **`mimic`** archetype — a scraper with a real browser's headers, a
cookie, and a residential-proxy IP, but robotic request timing. The seed
model waves it through; after feedback + retrain it gets caught, with no new
false positives on humans.

```
=== PHASE 1 — seed model, no feedback yet ===
  bots caught (recall):      80.8%   (63/78)
  precision on 'bot' calls: 100.0%   (63/63 flagged were bots)
  humans wrongly flagged:    0/60
  mimic                    15          0      0      <- all allowed
  stuffing_campaign         0          0      6      <- caught per-endpoint

=== PHASE 3 — after feedback + retrain (fresh traffic) ===
  bots caught (recall):     100.0%   (78/78)
  precision on 'bot' calls: 100.0%   (78/78 flagged were bots)
  humans wrongly flagged:    0/60
  mimic                     0         15      0      <- now challenged
```

The **`stuffing_campaign`** archetype (credential stuffing through a
residential-proxy pool — every attempt a fresh, clean-looking IP) is caught
from the start, but only because of **per-endpoint** windows: each IP in
isolation is `ALLOW`ed; the swarm on `/login` is not. See
[DESIGN.md](DESIGN.md#per-target-aggregation-the-credential-stuffing-answer).

## Talk to it

```bash
python3 server.py
```

```bash
# a plausible human -> ALLOW
curl -s localhost:8500/check -d '{
  "ip":"73.5.9.2","path":"/article/3",
  "user_agent":"Mozilla/5.0 (Macintosh) Safari/605.1.15",
  "accept":"text/html","accept_language":"en-US","cookie":"session=42"}'

# a crude scraper from a datacenter IP -> CHALLENGE / BLOCK
curl -s localhost:8500/check -d '{
  "ip":"185.10.20.30","path":"/article/3",
  "user_agent":"python-requests/2.31.0"}'

# a form bot that fell for the honeypot -> BLOCK (hard rule)
curl -s localhost:8500/check -d '{
  "ip":"12.1.1.1","method":"POST","path":"/signup",
  "user_agent":"Mozilla/5.0","honeypot_value":"http://spam.example"}'
```

| Method + path | Body | Does |
|---|---|---|
| `POST /check` | a request event | `{request_id, action, score, model_proba, rule_score, reasons, challenge_token}` |
| `POST /challenge` | `{"token": "..."}` | mark a challenge passed |
| `POST /feedback` | `{"request_id": "...", "label": 0\|1}` | teach it: 0 = human, 1 = bot |
| `POST /retrain` | `{}` | refit the model on the seed set + all collected labels |
| `POST /replay` | `{}` | run a demo traffic mix through the live pipeline |
| `GET /stats` | — | counters + the model's current top features |
| `GET /recent?n=50` | — | the latest decisions (the dashboard's feed) |
| `GET /timeseries` | — | decisions bucketed by time |
| `GET /` | — | the live dashboard (HTML); `GET /api` for this list as JSON |

### Configuration (env vars)

| Var | Default | Effect |
|---|---|---|
| `BOTSHIELD_HOST` / `BOTSHIELD_PORT` | `127.0.0.1` / `8500` | bind address (the Docker image sets host `0.0.0.0`) |
| `BOTSHIELD_DATA` | *(unset — in-memory)* | mirror labels, decision log, and the trained model to SQLite + `model.json` under this dir; reload on restart |
| `BOTSHIELD_COUNTER` | `exact` | `sketch` = estimate request rates from a fixed-size count-min sketch (flat memory under a distinct-IP flood) |

## Benchmarks

```bash
make bench     # or run benchmarks/*.py individually
```

Representative numbers (one laptop, single-threaded, Python 3.10 — run them
yourself; CI hardware is not a stable baseline):

| | |
|---|---|
| `pipeline.check()` latency | p50 ≈ 16 µs, p99 ≈ 26 µs |
| per-stage median | record 1.5 µs · extract 7.3 µs · rules 2.4 µs · model+decide 3.6 µs |
| in-process throughput | ≈ 45k req/s (≈ 3.9k req/s over HTTP, threaded stdlib server) |
| memory, 500k distinct attacking IPs | exact ≈ 400 MB · sketch ≈ 4 MB (grows 1.01× over the run) |
| `predict_proba` | ≈ 1.8 µs/call |

## Layout

```
botshield/
  events.py      the request event — the unit of input
  routes.py      normalize /article/5 -> /article/{id}; per-endpoint keys
  store.py       sliding-window counters (per-IP AND per-endpoint) + reputation + labels
  persist.py     optional SQLite write-through for labels / decisions / challenges
  sketch.py      count-min sketch: fixed-memory rate estimates under attack
  features.py    event + window state  ->  a 15-value feature row
  rules.py       hard rules block, escalation rules force a challenge, soft rules score
  model.py       logistic regression from scratch (batch + online training)
  decision.py    blend rule score + model score  ->  an action + thresholds
  pipeline.py    wires the stages; owns the feedback loop (retrain)
  synth.py       labeled synthetic traffic: 6 bot archetypes + humans
  bootstrap.py   build a trained, wired system in one call
server.py        the HTTP service + JSON API
dashboard/       single-file live dashboard served at GET /
simulate.py      end-to-end demo
selfcheck.py     [PASS]/[FAIL] for every component
tests/           pytest suite (unit + integration)
benchmarks/      latency, throughput, memory-under-attack, model cost
DESIGN.md        architecture, signals, trade-offs, failure modes
```

## Tech notes

Python 3.10+, standard library only: `http.server` for the service,
hand-rolled logistic regression (no numpy), `hashlib.blake2b` for the
sketch's hash family. Production bot detection uses gradient-boosted trees
or deep nets trained on a cluster over billions of labeled requests — the
serving shape is the same: fixed weights in, a score out, well under a
millisecond.

## Knobs to experiment with

- `botshield/decision.py` — `CHALLENGE_AT`, `BLOCK_AT`, `RULE_WEIGHT` /
  `MODEL_WEIGHT`. Lower the thresholds and watch false positives climb.
- `botshield/rules.py` — add a rule, change a weight, tune `SATURATION`, or
  add a name to `ESCALATE_RULES` to make it force a challenge.
- `botshield/synth.py` — add a bot archetype, or make `mimic` sneakier, and
  see whether the model can still learn it from feedback.
- `botshield/model.py` — `epochs`, `lr`, `l2` in `fit`.
