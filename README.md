# moe-service-handler

<p align="center">
  <img src="docs/assets/dataflow-requests-v2.svg" width="900" alt="Data flow, request split: two prompts come in to the m4 controller, which runs no model; it gives request A to the m1 worker and request B to the m4b worker at the same time; each worker runs streammoe on the whole model from its own SSD and sends every token back as it is made, and the controller streams it to the client. Every second a ping goes out and each worker's status comes back. The controller's worker table shows each worker's state and what it is doing. A third prompt that asks the first one again is answered from the controller's own cache, with no worker. A worker that hangs is marked stuck, with what it was on, and gets no new work. The workers never talk to each other.">
</p>

Turns streammoe into a service that runs a large MoE model, Qwen3.6-35B-A3B, across three
Mac minis: an M4 as the controller and an M1 and a second M4 as workers. Each worker runs
streammoe on the whole model, inside its memory cap; the controller runs no model and manages
everything: it takes the requests, gives each to a worker, streams the answers back and keeps
track of what every worker is doing. streammoe owns everything inside one machine (memory cap,
expert streaming, kernels); this repo owns everything that spans machines or requests, and
adapts to what streammoe offers. Full requirements:
[`moe-service-handler-prd.md`](moe-service-handler-prd.md).

## Architecture

The controller holds one persistent TCP connection to each worker; workers have no
connections to each other. The config's `split` picks how the work is divided.

**Request split** (`split = "requests"`; design: [`docs/request-split.md`](docs/request-split.md)).
The controller runs no model: it has the HTTP API, the tokenizer, the queue and the worker
management. Each worker runs streammoe on the whole model from its own SSD, as streammoe is
built to run. For each request:

1. The prompt comes in to the controller, which tokenizes it.
2. If the controller has answered it before, its own cache answers it, with no worker (below).
3. Otherwise the controller gives it to the ready worker with the fewest requests running (`Start`);
   both workers work at the same time, each on its own requests.
4. The worker prefills and decodes, sending each `Token` as it is made; the controller
   streams it to the client.
5. The worker ends the request at the token limit or the end of text (`Done`); the controller
   sends `Drop` only to end one early. A finished answer goes into the controller's cache.

One answer runs at one Mac's speed; two workers serve two requests at once. At a 9 GiB cap a
worker holds 3.1 GiB of weights, KV and buffers plus a 5.9 GiB hot tier of the 16.9 GiB of
experts (from `--plan`); the rest streams from its SSD.

**Worker management** (on the controller, request split only):

- **Worker table:** each worker's state (`connecting`, `ready`, `draining`, `stuck`, `down`)
  and since when, ping round trip, the requests it runs (phase, position, tokens sent, last
  progress), done and failed counts, tokens per second over 10 s, errors over a minute, the
  last error, and its footprint and cap from the status each `Pong` carries.
- **Stuck worker:** a ping not back within `health_timeout_ms` (5 s). It gets no new work and
  its requests show what they were on ("decode, token 212 of 500, no progress for 31 s").
- **Stalled request:** no progress for `stall_timeout_ms` (30 s): only that request fails.
- **Lost worker:** requests that sent nothing yet go to the other worker; the rest fail. The
  controller reconnects, and the worker serves again after its `Hello`.
- **The cluster stays up while one worker is ready**; with none, requests get 503.
- **Endpoints:** `GET /v1/workers`, `GET /v1/workers/{name}`, `GET /v1/requests`,
  `POST /v1/workers/{name}/drain` and `/resume`, `POST /v1/requests/{id}/cancel`; per-worker
  Prometheus series (`worker="NAME"`) and a log line per state change.
- **Answer cache** (`[cache]`, on the controller only): prompts it has answered, kept in its own
  RAM and never shared; the workers keep their own KV caches inside streammoe. The key is the
  prompt's tokens, `max_new`, and the model, engine settings and chat template; sampling is
  greedy, so the same key gets the same answer. `mode = "exact"` (default) answers only the
  same prompt; `"similar"` also answers a prompt worded almost the same, with a prompt vector
  (hashed token 1- to 3-grams, so similar wording, not similar meaning) at cosine
  `similar_threshold` (0.98) or above; `"off"` turns it off. The vectors are kept as
  TurboQuant codes (`bits`, default 4: a random rotation, then a fixed codebook per
  coordinate) and compared on the codes. Past `max_entries` (10 000) or `max_mib` (32), the
  least recently used go first. `GET /v1/cache` (entries, hits, misses), `POST /v1/cache/clear`,
  `msh_cache_*` series; a hit shows in `/v1/requests` as phase `cached` on worker `controller`.
- **Next (designed, not built):** the controller decides and commands: routing a chat's next
  turn to the worker that holds it, per-worker load limits, reopening or restarting a stuck
  worker, and tuning streammoe's open settings one worker at a time.

**Layer split** (`split = "layers"`, the default). Each node runs a contiguous layer range
(streammoe's `layer_range`). The controller runs its layers and sends the hidden state to m1,
which sends its result back; the controller sends that on to m4b, which runs the last layers
and the head and sends the token back. The workers take turns on every token, since m4b needs
m1's result. The 20 / 10 / 10 split of Qwen3.6's 40 layers is
[`configs/qwen36-3node.toml`](configs/qwen36-3node.toml); leave out `layers` and the planner
picks one from each node's speed and memory.

**Expert split** (`split = "experts"`, experimental, mock engine only). The controller runs
every layer's attention and router and sends each layer's picked experts to both workers at
once ([`docs/expert-parallel.md`](docs/expert-parallel.md)). It needs streammoe to run a layer
without its experts and to compute a chosen set of experts, which it does not offer, so it
stays on the mock.

The notes below cover all three; those about hidden states, chunked prefill, speculative
decoding and prefix reuse apply to the layer split.

- **Links:** one persistent TCP connection per worker, used both ways. A message bigger than
  `[net] max_chunk_kib` (64) goes in chunks, and one writer thread per link puts small frames
  (pings, tokens, `Start`, `Drop`) between the chunks of big ones, so a big message never holds
  up the rest. On loopback, a ping behind a 512 KiB hidden state came back in 20 µs (p50)
  instead of 59 µs. TCP keep-alive is on.
- **Frames:** 24-byte little-endian header (magic, version, type, request id, position,
  length) + payload. Types: `Hidden`, `Token`, `Drop`, `Rewind`, `Ping`/`Pong`, `Hello`, `Error`;
  for the request split also `Start`, `Done` and a status report in `Pong`; for the expert
  split `Experts`, `Partial`, `Own`, `Move`, `Release`. A message in chunks is several frames
  with the same type and request id: all but the last carry a `more` flag, and the first
  starts with the whole length (protocol version 7).
- **f32 on the wire:** the engine works in f32, so a split pipeline stays bit-identical to
  one process. Hidden states go on the wire straight from the engine's output buffer and are
  read straight into the next engine's input, with no encode or decode copy (NET-6). The
  controller passes a worker's result on to the next worker unchanged, as raw bytes.
- **`Drop`** goes through every worker; when it returns, every node has freed the request.
- **Errors** on a worker reach the coordinator as an `Error` frame tagged with the request;
  only that request fails and the cluster stays up.
- **Scheduler:** up to `max_in_flight` requests (default 3) step through the cluster at once, so
  each node works on a different request; extra requests queue up to `max_queue`, then are
  rejected. Dropping a request's event channel cancels it.
- **Prefill in chunks:** the prompt goes through the workers `prefill_chunk` positions at a time
  (default 64) in one `Hidden` frame, which streammoe runs on its batched path. Up to
  `prefill_depth` chunks (default 3) of a request are in the cluster at once, so the nodes work
  on consecutive chunks together.
- **Speculative decoding** (`max_draft` > 0, off by default): the coordinator drafts tokens by
  prompt lookup and sends them with the real one as one step; the head returns a token per
  position, matching drafts are kept, and `Rewind` drops the rest on every node. Output equals
  plain greedy decoding. Works with the mock engine; streammoe needs per-position logits and
  rewind on a layer range first. See [`docs/speed.md`](docs/speed.md).
- **Layer split:** give every node `layers = [a, b]`, or none and let every node plan the
  same split from the config (CL-3): the slowest node's time (its layers ÷ its `speed`) as
  small as possible, then as even as possible, within each node's memory cap under streammoe.
  `msh-node --config FILE --plan` prints the split and each node's memory budget;
  `msh-node --speeds-from run.csv` turns a previous run's metrics into `speed` values.
- **Prefix reuse** (`prefix_cache` > 0): finished requests keep their state on every node; the
  next chat turn takes it over and prefills only the new message. Off by default.
- **Health and reconnect:** a `Ping` goes to every worker each second. A lost worker is
  reconnected in the background, waiting `[net] reconnect_min_ms` (500) doubling up to
  `reconnect_max_ms` (10 000), with some randomness. A worker that hears nothing for
  `health_timeout_ms` drops the link and waits for the controller again; it never exits. In the
  request split a broken or hung worker is taken out of service on its own (above). In the
  layer and expert splits every node is needed for every token, so a lost worker fails the
  requests in flight and new ones get 503 until every worker is back; then the cluster serves
  again on its own (`msh_cluster_up`).
- **Metrics:** each worker appends its compute time to the frame it sends back, so the
  coordinator splits every step into per-node compute and network/queueing without clock
  sync. `--metrics FILE` or `[metrics] csv` writes one CSV row per step (a prompt chunk, or a
  decode step with its drafts) with that split, queue wait, requests in flight, ping round
  trip, reused and drafted/accepted positions, and run metadata (git hash, hardware, split).
- **Logs** (NF-5): `tracing` to stderr, text or JSON lines (`[log] json`), level from
  `[log] level` or `RUST_LOG`. Every line names the node; every line about a request carries
  `request` (the client's id, also in `chatcmpl-<id>`) and `state` (its id in frames, metrics
  and worker logs). `RUST_LOG=debug` adds one line per step on every node.
- **Prometheus** (CL-12): `prometheus = "ADDR"` in a node's `[[nodes]]` entry serves
  `GET /metrics` on that node: footprint and its limit, and compute time per step on every
  node; on the coordinator also requests by outcome, active and waiting requests, tokens,
  time to first token, decode-step round trip, drafts and prefix reuse, and the ping's round
  trip. Every series carries `node="NAME"`.
- **Service memory** (NF-2): the service's own memory (network buffers, requests, HTTP,
  metrics) has a fixed allowance beside streammoe's cap, `[service] mem_allowance_mib`
  (default 256). A node measures its footprint before the engine opens, adds the most its
  buffers can grow to, and refuses to start over the allowance; while serving it warns if the
  footprint passes cap + allowance. HTTP bodies over 1 MiB get 413, and requests beyond what
  the scheduler can hold get 429 before their body is read. Measured at start: a worker
  needs ~5 MiB; the coordinator with Qwen3.6's tokenizer ~140 MiB at worst (69 MiB at start,
  67 MiB of it HTTP bodies). OLMoE on one node with a 4 GiB cap peaked at 1.16 GiB.
- **KV format on the workers** (`[engine] kv_format`): `f32` (default), `f16`, `turbo4`,
  `turbo3`, passed to streammoe as it is. Not set in any config: on OLMoE the turbo formats
  change greedy output within a few tokens. `--plan` shows the KV size for each.
- **API:** the coordinator serves OpenAI-compatible `GET /v1/models` and
  `POST /v1/chat/completions` (plain or streamed), with greedy sampling. A full queue answers
  429, a cluster that is down 503. In the request split the controller also serves the worker
  management and cache endpoints above.

## Crates

| Crate | Kind | Purpose | State |
| --- | --- | --- | --- |
| `msh-net` | library + `msh-probe` | Frames, chunked two-way connections, latency probe | M0 done; chunking and keep-alive |
| `msh-node` | binary | Controller / worker, config, planner, star links, scheduler, HTTP API, metrics, engines | Request split with worker management (tested on OLMoE: 2 workers, whole model each), the controller's answer cache with TurboQuant prompt vectors, controller-to-worker links with reconnect, scheduler (CL-5), planner (CL-3), dead-node handling (CL-8), HTTP API (CL-2), chunked prefill (CL-6), speculative decoding and prefix reuse (mock engine), run metrics (CL-10, CL-11), Prometheus (CL-12), logs (NF-5), service memory allowance (NF-2); runs real streammoe (tested on OLMoE) |
| `msh-policy` | library | Per-expert 4-bit / 2-bit / skip rules | Not started |

The engine sits behind an `Engine` trait shaped like streammoe's `eval_request`: tokens or
hidden states in, for one position or a chunk; hidden states out, or on the head node the
sampled token for the last position or for every position (to verify drafts); plus `rewind`
and `free`. Two implementations:

- `MockEngine`: deterministic, no model needed; the default.
- `StreamMoeEngine`: streammoe on the node's layer range, with one request slot per request
  in flight plus one per kept conversation (`max_in_flight + prefix_cache`). It cannot
  verify drafts yet, so speculative decoding stays off with it. Built with a model feature
  (`--features olmoe` or `qwen36`) and selected with `[engine] kind = "streammoe"`.
  streammoe comes from [transybao1393/streamoe](https://github.com/transybao1393/streamoe),
  pinned to a commit in `crates/msh-node/Cargo.toml` (0.2.0-beta.1).

Each node opens streammoe for its own role in the cluster:

- **No checkpoints:** a stage never snapshots, and under a cap each checkpoint would take a
  full state's memory from the expert hot tier.
- **Prefill scratch sized to `prefill_chunk`**, the most positions one step brings, instead of
  streammoe's 512.
- **Frame buffers counted in the cap** (`host_buffers`).
- `[engine]` passes on `kv_format` (`f32` default, `f16`, `turbo4`, `turbo3`), `attn_threads`,
  `runtime_overhead_gib` and the expert streaming settings (`experts_dir`, `prefetch_guess`,
  `prefetch_experts`, `fast_experts_dir`; see [`docs/speed.md`](docs/speed.md)). Only `f32`
  keeps a split bit-identical to one node; `f16` gave the same tokens in our runs, TurboQuant
  did not.

streammoe's multi-request decode (`eval_requests`) and multi-token passes (`eval_tokens`, for
speculative decoding) run on the whole model only, so the cluster cannot use them yet; each
node runs one request's step at a time.

For development the real engine runs **OLMoE-1B-7B** (16 layers, 3.9 GB at 4-bit) from
`~/models/olmoe-1b-7b`: a 3-process 8/4/4 split reproduces streammoe's MLX reference answer
token for token.

## Quick start

```bash
cargo test --release

B=target/release/msh-node

# request split on one machine: a controller and two workers, each running the whole (mock) model
$B --config configs/local-requests.toml --node m1 &
$B --config configs/local-requests.toml --node m4b &
$B --config configs/local-requests.toml --node m4 --max-tokens 500 --requests 4

# the same with OLMoE over HTTP, and the worker table (build with --features msh-node/olmoe)
#   start m1 and m4b, then m4, with configs/olmoe-requests.toml, then:
curl localhost:8080/v1/workers
curl localhost:8080/v1/requests
curl localhost:8080/v1/cache          # the controller's answer cache: entries, hits, misses

# layer split, three nodes on one machine (the controller waits up to 60 s for the workers)
$B --config configs/local-3node.toml --node m4b &
$B --config configs/local-3node.toml --node m1 &
$B --config configs/local-3node.toml --node m4 --max-tokens 500 --requests 3 --metrics results/run.csv

# the real engine on OLMoE, served over HTTP, and its model-backed tests
cargo build --release --features msh-node/olmoe
$B --config configs/olmoe-3node.toml --plan    # split and memory budget per node
$B --config configs/olmoe-3node.toml --node m4b &
$B --config configs/olmoe-3node.toml --node m1 &
$B --config configs/olmoe-3node.toml --node m4 &
curl localhost:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is the capital of France?"}]}'
cargo test -p msh-node --features olmoe --release -- --ignored --test-threads=1

# after a run with --metrics: measured speed per node, for the planner
$B --speeds-from results/run.csv

# network round-trip probe: serve on one machine, run from another
target/release/msh-probe serve 0.0.0.0:7070
target/release/msh-probe run <host>:7070 ethernet results/m0-<name>.csv
```

## Baseline numbers

Loopback on a MacBook Pro M3 Pro ([`results/m0-m3pro-local.csv`](results/m0-m3pro-local.csv)),
the software floor for the runs between the Mac minis:

| Payload | Round trip p50 | One hop |
| --- | --- | --- |
| 4 KB | 13.7 µs | ~7 µs |
| 1 MB | 137 µs | ~68 µs |
| 16 MB | 3.2 ms | ~1.6 ms |

Sending a hidden-state frame over loopback and reading it back
(`cargo test -p msh-net --release --test loopback -- --ignored --nocapture`):

| Frame | Encoded payload (before) | Direct (NET-6) |
| --- | --- | --- |
| 1 position (8 KiB, a decode step) | 4.8 µs | 2.4 µs |
| 64 positions (512 KiB, a prompt chunk) | 133.5 µs | 33.5 µs |

Prefill of a 400-token prompt, OLMoE 8/4/4 split in 3 processes on the M3 Pro, by
`prefill_chunk` (time to first token):

| Chunk | 1 | 64 (default) | 128 | 256 |
| --- | --- | --- | --- | --- |
| Time to first token | 5.6 s | 1.4 s | 1.2 s | 0.95 s |

Bigger chunks are faster but hold the coordinator longer, delaying other requests' decode
steps (a 64-token chunk takes ~180 ms on m4's 8 layers).

Moving to streammoe 0.2.0-beta.1, same setup (3 GiB cap per node, 400-token prompt, 128
tokens):

| | Hot tier (m4) | Prefill on m4 | Decode |
| --- | --- | --- | --- |
| streammoe defaults | 0.31 GiB | 2.7 s | 42–44 tok/s |
| Opened for the cluster (no checkpoints, 64-token scratch) | 1.45 GiB | 1.5 s | 51 tok/s |

## License

[PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0) —
see [`LICENSE.md`](LICENSE.md). Free for noncommercial use (personal, research, education,
nonprofits); commercial use needs a separate license from the author.
