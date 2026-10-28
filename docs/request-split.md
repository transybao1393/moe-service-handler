# Request split with worker management

Status: phase 1 (worker management) built: `split = "requests"`, tested on the mock and on
OLMoE (2 workers, each running the whole model, match the reference with 4 requests at once).
Phase 2 (the controller deciding and commanding) is designed below, not built. Chosen 2026-10-28 (D30 in the decision log) because it uses
streammoe exactly as it is: each worker runs the whole model in mode A, and nothing in
streammoe has to change.

## Roles

```
                client
                  │ prompts        ▲ tokens
                  ▼                │
        ┌──────────────────────────────────────┐
        │ m4 · controller (no model)           │
        │ HTTP API, tokenizer, queue,          │
        │ assigns each request to a worker,    │
        │ worker management: status, progress, │
        │ stuck and lost workers, drain        │
        └──────────────────────────────────────┘
          │ request A    ▲ tokens     │ request B    ▲ tokens
          ▼ (same time)  │            ▼ (same time)  │
     ┌───────────────────┐       ┌───────────────────┐
     │ m1 · worker        │       │ m4b · worker       │
     │ streammoe, whole   │       │ streammoe, whole   │
     │ model, own SSD     │       │ model, own SSD     │
     └───────────────────┘       └───────────────────┘
```

- **Controller:** runs no engine and holds no weights. It owns the HTTP API, the tokenizer and
  chat template, the queue, the choice of worker for each request, and the management of
  every worker.
- **Worker:** runs streammoe on the whole model inside its cap, streaming experts from its own
  SSD (each worker has a copy of the model). It runs whole requests: prefill, then decode
  until the controller's stop condition, sending each token as it is made.
- Workers never talk to each other. One answer runs at one Mac's speed; two workers serve
  twice the requests.

## The flow

1. A prompt comes in to the controller, which tokenizes it.
2. The controller looks it up in its answer cache (below). On a hit it streams the kept answer
   itself, and no worker is used.
3. Otherwise the controller picks a worker: a ready one with the fewest requests running (then the
   faster recent tokens per second) and sends `Start`: the request id, the prompt's token ids
   and the token limit.
4. The worker prefills and decodes, sending each `Token` as it is made.
5. The controller streams each token to the client. The worker ends the request itself at the
   token limit or an end-of-sequence token and sends `Done`; the controller sends `Drop` only
   to end one early (cancel, stall, client gone), and the slot stays taken until the worker's
   `Done` comes back. An answer that ended at the token limit or end of sequence goes into
   the cache.
6. Both workers do this at the same time, each for its own requests.

## Answer cache (on the controller)

Each part keeps its own cache, and none is shared: the workers' KV caches stay inside
streammoe; the controller has an answer cache in its own RAM, inside the service allowance.

- **Where:** in the request board, looked up after the prompt is tokenized and before it is
  queued, and filled when a worker's `Done` comes back with reason length or end of sequence
  (not a dropped or failed request).
- **Key:** the prompt's tokens (a keyed 128-bit hash and the length), `max_new`, and a hash of
  the model, its engine settings and the chat template. Sampling is greedy, so the same key
  gets the same answer.
- **Modes** (`[cache] mode`): `"exact"` (default) answers only the same key; `"similar"` then
  also looks for the nearest earlier prompt with the same model and `max_new`, and answers
  from it at cosine `similar_threshold` (0.98) or above; `"off"`.
- **Prompt vectors:** the controller runs no model, so a vector is made from the tokens:
  token 1-, 2- and 3-grams hashed with a sign into 256 coordinates, normalised. It measures
  similar wording, not similar meaning.
- **TurboQuant:** each vector is kept as a TurboQuant code, `bits` (default 4) per coordinate:
  a seeded random rotation, then a fixed Lloyd-Max codebook per coordinate. Cosines are
  estimated on the codes directly. Exact mode keeps no codes.
- **Limits:** `max_entries` (10 000) and `max_mib` (32); past either, the least recently used go.
- **Seen from outside:** `GET /v1/cache` (entries, bytes, hits and misses by kind, limits),
  `POST /v1/cache/clear`, `msh_cache_*` series; a hit shows in `/v1/requests` as phase `cached`
  on worker `controller`, with the kind, similarity and the worker that made the answer.

## Worker management

The controller keeps a table with one row per worker:

| Field | Meaning |
| --- | --- |
| `state` | `connecting`, `ready`, `draining`, `stuck`, `down` |
| `since` | when it entered that state |
| `ping_ms`, `last_pong` | the last health ping's round trip and when it came back |
| `requests` | what it is doing now: per request the id, `prefill` or `decode`, position, tokens sent, started at, last progress at |
| `done`, `failed` | requests finished and failed on it |
| `tokens_per_s` | recent decode speed |
| `footprint`, `cap` | from the worker's status report |
| `last_error` | the last error it sent, with time and request |

- **Status report:** every `Pong` carries the worker's status: its footprint and cap, and per
  request the position and tokens made. So the controller knows what each worker is doing
  without asking separately.
- **Stuck worker:** a ping not back within `health_timeout_ms`. A worker answers pings from its
  reader thread even during a long step, so "stuck" means a frozen process or link, not slow
  work. It gets no new requests; its
  requests are marked stuck with what they were on (e.g. "decode, token 212 of 500, no
  progress for 31 s").
- **Stuck request:** no progress for `stall_timeout_ms` (default 30 s) while the worker still
  answers pings. Progress is a new token or, during prefill, prompt positions advancing in the
  status report, so a long prompt is not failed as stalled. Only that request is failed and dropped on its worker.
- **Lost worker:** its connection breaks. It is marked `down`; its requests that have not sent
  a token yet go to another worker, the others fail (the client already has part of the
  answer). The controller reconnects in the background, and the worker is `ready` again once
  it says hello.
- **The cluster stays up while at least one worker is ready.** With none, new and queued
  requests fail (HTTP 503).
- `connecting` is the state before a worker's first `Hello`; after a loss it stays `down` until
  it says hello again. An `Error` from a worker ends only its request, with no `Done` after it.
- **Drain:** a worker can be set to `draining`. It finishes what it runs and takes nothing new
  (for a restart or an update).

What the operator sees, all on the controller:

| Where | What |
| --- | --- |
| `GET /v1/workers` | the table above as JSON |
| `GET /v1/workers/{name}` | one worker |
| `GET /v1/requests` | requests in flight and queued: worker, phase, position, tokens, waiting time |
| `POST /v1/workers/{name}/drain`, `/resume` | stop or restart new requests to a worker |
| `POST /v1/requests/{id}/cancel` | cancel one request |
| Prometheus | per-worker series with `worker="NAME"`: state, requests running, tokens per second, ping round trip, footprint, done and failed |
| Logs | one line per state change: worker, old and new state, reason |

## Control: the controller decides and commands (phase 2)

The controller is not only a monitor: it decides what to do, where and when, and sends the
commands. It works only through what streammoe already offers: its open settings, and the
measurements it reports (expert I/O and hot-tier hits, prefetch hits, footprint and peak
against the cap, per-command timings).

**Commands to a worker** (new `Command` frame, answered with the result):

| Command | What the worker does | Allowed when |
| --- | --- | --- |
| drain / resume | takes no new requests / takes them again (controller side) | always |
| cancel | drops one request | always |
| `reopen` | closes streammoe and opens it again with new open settings: cap, 4-bit or 3-bit experts, KV format, prefetch guess and count, attention threads, request slots | drained (no requests) |
| `restart` | exits; launchd starts it again | drained, or stuck |
| `reset_stats` | clears streammoe's counters, to start a measurement | always |

**Decisions** (a policy loop on the controller, every few seconds):

| Where | Decision | Based on |
| --- | --- | --- |
| Routing | which worker gets a request: the least busy, and a chat's next turn goes to the worker that holds that conversation (prefix reuse, no prefill again) | requests running, tokens/s, who holds which conversation |
| Load | how many requests each worker runs at once: lower the limit if one more request lowers its total tokens/s | measured tokens/s per worker by requests running |
| Fixing | stuck worker → drain → reopen; stuck again soon, or reopen fails → restart; errors k times in a window → take it out of service and report | ping, progress, errors in a time window |
| Memory | footprint near cap + allowance → fewer requests; well under the cap with a low hot-tier hit rate → reopen with a larger cap (up to a configured maximum) | footprint, peak, hot-tier hits |
| Tuning | try one open setting on one worker (e.g. prefetch guess `prev` vs `cross`, 4-bit vs 3-bit experts), compare over the same kind of traffic, keep the winner, roll back a loser | tokens/s, hot-tier and prefetch hits, before and after |

**When:**
- **Routing and load limits:** all the time.
- **Reopen, restart and tuning:** only on a drained worker, **one worker at a time**, and only
  while the other worker can carry the queue. Leave a minimum gap between actions on the same
  worker (e.g. 10 minutes), because a reopen reloads the model.

**Safety:**
- **`[control] mode`:**
  - `"advise"` (the default): the controller only records what it would do, at
    `GET /v1/recommendations`. You apply one with `POST /v1/recommendations/{id}/apply`.
  - `"act"`: it does them itself.
- **Every action is recorded:** one log line, plus a row in `GET /v1/actions` with what, why,
  the numbers before and after, and the outcome. A change that made things worse is rolled
  back.
- **Limits in the config:** the largest cap per worker, which settings may be changed, and the
  number of actions per hour.

## Communication

The traffic is small: a prompt goes out once (a few KB of token ids) and each token comes back
as a 28-byte frame. A token takes the worker tens of milliseconds to make; a wired link adds
well under 1 ms. So reliability and simplicity matter more than raw speed. Figures are typical
values, not measured on the Mac minis; `msh-probe serve` / `run` measures each link.

**Link:**

| Link | Bandwidth (typical) | Small round trip (typical) | On these Mac minis | Fit |
| --- | --- | --- | --- | --- |
| Thunderbolt bridge | ~10–20 Gb/s | ~50–150 µs | all have Thunderbolt; 2 cables from the controller, no switch | **best** |
| 10 Gb Ethernet | ~9.4 Gb/s | ~30–100 µs | only if ordered with it, plus a 10 GbE switch | very good |
| 1 Gb Ethernet (built in) | ~0.94 Gb/s | ~100–300 µs | all | **good enough**, no extra hardware |
| Wi-Fi | 0.2–1 Gb/s, varies | 1–10 ms, spikes | all | avoid: spikes look like stuck workers |
| RDMA over Thunderbolt 5 | ~80 Gb/s | a few µs | needs Thunderbolt 5 on both ends; the M1 and base M4 lack it | not possible here |

**On the link** (built): one persistent TCP connection per worker, used both ways. A message
over `[net] max_chunk_kib` (64) goes in chunks; one writer thread per link sends small frames
(pings, tokens, `Start`, `Drop`) between the chunks of a big one. TCP keep-alive is on. A lost
worker is reconnected in the background, waiting `reconnect_min_ms` (500) doubling up to
`reconnect_max_ms` (10 000), with some randomness; a worker that hears nothing for
`health_timeout_ms` drops the link and waits for the controller again.

**Protocol:**

| Protocol | Overhead | Fit |
| --- | --- | --- |
| Persistent TCP + our 24-byte binary frames (in use) | 24-byte header, no encoding, Nagle's batching off | **best**: built, tested, lowest of the TCP options |
| TLS over that TCP | encryption | only if the network is shared or untrusted |
| gRPC (HTTP/2 + protobuf) | headers + encoding per message | worse: needs an async runtime on workers, heavy dependencies |
| HTTP + JSON | text | only for the operator's management endpoints |
| ZeroMQ / nanomsg | small | a C library for nothing we lack |
| QUIC | encryption, multiplexing | helps on lossy links only |
| Custom UDP | smallest | must rebuild retransmission and ordering |

Copying the model to each worker once (about 19 GB at 4-bit) is where link speed matters:
about 3 minutes over 1 Gb Ethernet, well under a minute over Thunderbolt.

## Frames

Beside `Hello`, `Ping`/`Pong`, `Token`, `Drop`, `Error`:

| Type | Direction | Payload |
| --- | --- | --- |
| `Start` = 14 | controller → worker | request id in the header; `max_new u32, count u32`, then `token u32` × count |
| `Token` | worker → controller | request id in the header, position = the token's position; the token id |
| `Done` = 15 | worker → controller | request id, position = tokens made; one byte: `0` length, `1` end of sequence, `2` dropped |
| `Pong` | worker → controller | the status report: tagged sections (`tag u16, len u32, bytes`), unknown tags skipped. Tag 1 memory (footprint, cap); tag 2 requests (count, entry length, then per request id, decoding, position, tokens made). New fields go at the end of an entry or in a new tag |

A message in chunks is several frames with the same type, request id and position: all but
the last carry the `more` flag (`0x80`), and the first starts with the whole payload length
(u32). Chunks of different requests may interleave; a request's own never do.

Protocol version 7.

## What stays and what changes

- **Stays:** the controller-to-each-worker connections, frames and health ping, the HTTP API,
  metrics, logs and the memory allowance. The layer split stays as `split = "layers"`; the
  expert split stays mock-only and experimental.
- **Not in this mode yet:** speculative decoding and prefix reuse (config validation refuses
  them), the per-step metrics CSV, and routing a chat's next turn to the worker that holds it.
- **New:** `split = "requests"`, the worker table and its endpoints, per-request progress,
  stuck and lost handling per worker, reconnect, drain, the controller's answer cache, chunked
  links.
- **Next, with streammoe as it is:** decode several requests on a worker in one batch
  (`eval_requests`) and speculative decoding on the worker (`eval_tokens`). Both work on the
  whole model today.
