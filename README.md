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
python3 server.py       # the actual HTTP service on 127.0.0.1:8500
```

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
  bots caught (recall):     79.2%   (57/72)
  humans wrongly flagged:    0/60
  mimic                    15          0      0      <- all allowed

=== PHASE 3 — after feedback + retrain (fresh traffic) ===
  bots caught (recall):     100.0%   (72/72)
  humans wrongly flagged:    0/60
  mimic                     0         15      0      <- now challenged
```

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
| `GET /stats` | — | counters + the model's current top features |

## Layout

```
botshield/
  events.py      the request event — the unit of input
  store.py       sliding-window counters + reputation + labels  (the "Redis")
  features.py    event + window state  ->  a 13-value feature row
  rules.py       deterministic checks: hard rules block, soft rules score
  model.py       logistic regression from scratch (batch + online training)
  decision.py    blend rule score + model score  ->  an action + thresholds
  pipeline.py    wires the stages; owns the feedback loop (retrain)
  synth.py       labeled synthetic traffic: 5 bot archetypes + humans
  bootstrap.py   build a trained, wired system in one call
server.py        the HTTP service
simulate.py      end-to-end demo
selfcheck.py     [PASS]/[FAIL] for every component
DESIGN.md        architecture, signals, trade-offs, failure modes
```

## Tech stack

Python 3.10+, standard library only: `http.server` for the service,
`urllib` for the simulator's client, hand-rolled logistic regression (no
numpy). The point is to build the thing, not import a library that already
built it.

## Knobs to experiment with

- `botshield/decision.py` — `CHALLENGE_AT`, `BLOCK_AT`, `RULE_WEIGHT` /
  `MODEL_WEIGHT`. Lower the thresholds and watch false positives climb.
- `botshield/rules.py` — add a rule, or change a weight.
- `botshield/synth.py` — add a bot archetype, or make `mimic` sneakier, and
  see whether the model can still learn it from feedback.
- `botshield/model.py` — `epochs`, `lr`, `l2` in `fit`.
