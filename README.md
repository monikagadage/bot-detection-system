# Bot Detection System — a small, working model of how sites stop bots

A runnable model of the core of a bot-management service (Cloudflare Bot
Management, DataDome, HUMAN, Akamai): request events come in, feature
extraction and a deterministic rules engine and a learned scorer combine
into an `ALLOW` / `CHALLENGE` / `BLOCK` decision in the request path, and a
feedback loop retrains the model on what it got wrong. It's not production
infrastructure — it's the same *algorithms* stripped down to something you
can read end to end in one sitting. See [DESIGN.md](DESIGN.md) for the
architecture, the signal layers real systems use, scaling, and failure
modes.

**No dependencies** — Python 3.10+ standard library only. No `pip install`,
no framework.

## Run it

```bash
cd bot-detection-system
python3 selfcheck.py    # [PASS]/[FAIL] for every component
python3 simulate.py     # the whole thing end to end (~10s)
python3 server.py       # the HTTP service + live dashboard on 127.0.0.1:8500
```

Then open **http://localhost:8500** for a live dashboard — decision tiles, a
traffic timeline, what the model is currently leaning on, and a rolling feed
of decisions. Hit **Replay demo traffic** to watch it classify a mixed
human/bot stream in real time, or **Retrain model** to fold in collected
labels.

### `simulate.py` — the feedback loop, made visible

Starts the server in a background thread, fires ~124 mixed human/bot
sessions at it over real HTTP, and prints a confusion matrix plus a
per-archetype breakdown of `ALLOW` / `CHALLENGE` / `BLOCK`. Then an
"operator" labels the bots that slipped through, the model retrains, and a
fresh batch runs.

Watch the **`mimic`** archetype — a scraper with a real browser's headers, a
cookie, and a residential-proxy IP, but robotic request timing. The seed
model waves it through (Phase 1); after feedback + retrain it gets caught
(Phase 3), with no new false positives on humans.

```
=== PHASE 1 — seed model, no feedback yet ===
  bots caught (recall):     80.8%   (63/78)
  humans wrongly flagged:    0/60
  mimic                    15          0      0      <- all allowed
  stuffing_campaign         0          0      6      <- caught per-endpoint

=== PHASE 3 — after feedback + retrain (fresh traffic) ===
  bots caught (recall):     100.0%   (78/78)
  humans wrongly flagged:    0/60
  mimic                     0         15      0      <- now challenged
```

The **`stuffing_campaign`** archetype (credential stuffing through a
residential proxy pool — every attempt a fresh, clean-looking IP) is caught
from the start, but only because of **per-endpoint** windows: each IP in
isolation is `ALLOW`ed; the swarm on `/login` is not. See
[DESIGN.md](DESIGN.md#per-target-aggregation-the-credential-stuffing-answer).

### `server.py` — talk to it yourself

```bash
python3 server.py
```

```bash
# a plausible human -> ALLOW
curl -s localhost:8500/check -d '{
  "ip":"73.5.9.2","path":"/article/3",
  "user_agent":"Mozilla/5.0 (Macintosh) Safari/605.1.15",
  "accept":"text/html","accept_language":"en-US","cookie":"session=42"}'
```

```bash
# a crude scraper from a datacenter IP -> CHALLENGE / BLOCK
curl -s localhost:8500/check -d '{
  "ip":"185.10.20.30","path":"/article/3",
  "user_agent":"python-requests/2.31.0"}'
```

```bash
# a form bot that fell for the honeypot -> BLOCK (hard rule)
curl -s localhost:8500/check -d '{
  "ip":"12.1.1.1","method":"POST","path":"/signup",
  "user_agent":"Mozilla/5.0","honeypot_value":"http://spam.example"}'
```

```bash
curl -s localhost:8500/stats
```

| Method + path | Body | Does |
|---|---|---|
| `POST /check` | a request event | `{request_id, action, score, model_proba, rule_score, reasons, challenge_token}` |
| `POST /challenge` | `{"token": "..."}` | mark a challenge passed |
| `POST /feedback` | `{"request_id": "...", "label": 0\|1}` | teach it: 0 = human, 1 = bot |
| `POST /retrain` | `{}` | refit the model on the seed set + all collected labels |
| `POST /replay` | `{}` | run a demo traffic mix through the live pipeline (for the dashboard) |
| `GET /stats` | — | counters + the model's current top features |
| `GET /recent?n=50` | — | the latest decisions (the dashboard's feed) |
| `GET /timeseries` | — | decisions bucketed by time |
| `GET /` | — | the live dashboard (HTML); `GET /api` for this list as JSON |

### Persistence (optional)

```bash
BOTSHIELD_DATA=./data python3 server.py
```

Mirrors the label store, the decision log, and challenge tokens to
`data/botshield.db` (SQLite) and the trained model to `data/model.json`. On
restart the model and labels reload, and `/feedback` still resolves
`request_id`s issued before the restart. Without the env var everything
stays in memory (the default).

### Approximate counters (optional)

```bash
BOTSHIELD_COUNTER=sketch python3 server.py
```

Request-rate features (`req_10s`, `req_60s`, `target_req_60s`) are estimated
from a fixed-size **count-min sketch** (`botshield/sketch.py`) instead of
exact per-IP history. Memory stays flat no matter how many distinct IPs
attack you — the point of the structure. `benchmarks/memory_under_attack.py`
shows the difference.

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
benchmarks/      latency, throughput, memory-under-attack, model cost
DESIGN.md        architecture, signals, trade-offs, failure modes
```

## Benchmarks

```bash
python3 benchmarks/hotpath_latency.py      # per-request latency + per-stage breakdown
python3 benchmarks/throughput.py           # req/s in-process vs over HTTP
python3 benchmarks/memory_under_attack.py  # exact store vs count-min sketch
python3 benchmarks/model_cost.py           # predict_proba ns/call; fit() vs dataset size
```

Representative numbers (one laptop, single-threaded — run them yourself):

| | |
|---|---|
| `pipeline.check()` latency | p50 ≈ 15 µs, p99 ≈ 26 µs |
| in-process throughput | ≈ 50k req/s |
| memory, 500k distinct attacking IPs | exact ≈ 400 MB, sketch ≈ 4 MB |
| `predict_proba` | ≈ 1.9 µs/call |

## Tech stack

Python 3.10+, standard library only: `http.server` for the service,
`urllib` for the simulator's client, hand-rolled logistic regression (no
numpy). The point is to build the thing, not import a library that already
built it.

## Knobs to experiment with

- `botshield/decision.py` — `CHALLENGE_AT`, `BLOCK_AT`, `RULE_WEIGHT` /
  `MODEL_WEIGHT`. Lower the thresholds and watch false positives climb.
- `botshield/rules.py` — add a rule, change a weight, tune `SATURATION`, or
  add a name to `ESCALATE_RULES` to make it force a challenge.
- `botshield/synth.py` — add a bot archetype, or make `mimic` sneakier, and
  see whether the model can still learn it from feedback.
- `botshield/model.py` — `epochs`, `lr`, `l2` in `fit`.
