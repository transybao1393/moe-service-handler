# moe-service-handler

Turns streammoe into a service that runs Qwen3.6-35B-A3B across a three-node pipeline of Mac
minis (mode B). streammoe owns everything inside one machine (memory cap, expert streaming,
kernels); this repo owns everything that spans machines or requests. Full requirements: [`moe-service-handler-prd.md`](moe-service-handler-prd.md).

## Architecture

Each node runs a contiguous layer range. Nodes form a ring: hidden states flow forward,
the last node samples a token and sends it back to the coordinator.

```mermaid
flowchart LR
    C(["client"]) -- "request" --> M4
    subgraph R["ring · one persistent TCP connection per hop"]
        direction LR
        M4["<b>m4 · coordinator</b><br/>Mac mini M4<br/>embed + layers 0–19"]
        M1A["<b>m1a · worker</b><br/>Mac mini M1<br/>layers 20–29"]
        M1B["<b>m1b · worker</b><br/>Mac mini M1<br/>layers 30–39 + head"]
        M4 -- "Hidden · f32 · 8 KB/token" --> M1A
        M1A -- "Hidden" --> M1B
        M1B -- "Token · u32" --> M4
    end
```

- **Frames:** 24-byte little-endian header (magic, version, type, request id, position,
  length) + payload. Types: `Hidden`, `Token`, `Drop`, `Ping`/`Pong`, `Hello`, `Error`.
- **f32 on the wire:** the engine works in f32, so a split pipeline stays bit-identical to
  one process.
- **`Drop`** travels the whole ring; when it returns, every node has freed the request.
- **Errors** on a worker reach the coordinator as an `Error` frame tagged with the request;
  only that request fails and the ring stays up.
- **Scheduler:** up to `max_in_flight` requests (default 3) step through the ring at once, so
  each node works on a different request; extra requests queue up to `max_queue`, then are
  rejected. Dropping a request's event channel cancels it.
- **Health:** a `Ping` travels the ring every second. A broken connection, or a ping not back
  within 5 s (a hung node), marks the ring down: in-flight requests fail and new ones are
  rejected until the cluster is restarted.
- **Metrics:** each worker appends its compute time to the frame it forwards, so the
  coordinator splits every step into per-node compute and network/queueing without clock
  sync. `--metrics FILE` or `[metrics] csv` writes one CSV row per position with that split,
  queue wait, requests in flight, ring round trip and run metadata (git hash, hardware, split).
- **API:** the coordinator serves OpenAI-compatible `GET /v1/models` and
  `POST /v1/chat/completions` (plain or streamed), with greedy sampling. A full queue answers
  429, a ring that is down 503.

## Crates

| Crate | Kind | Purpose | State |
| --- | --- | --- | --- |
| `msh-net` | library + `msh-probe` | Frames, connections, latency probe | M0 done |
| `msh-node` | binary | Coordinator / worker, config, ring, scheduler, HTTP API, metrics, engines | Ring, scheduler (CL-5), dead-node handling (CL-8), HTTP API (CL-2), run metrics (CL-10, CL-11); runs real streammoe (tested on OLMoE) |
| `msh-policy` | library | Per-expert 4-bit / 2-bit / skip rules | Not started |

The engine sits behind an `Engine` trait shaped like streammoe's `eval_request` (token or
hidden state in; hidden state or sampled token out). Two implementations:

- `MockEngine`: deterministic, no model needed; the default.
- `StreamMoeEngine`: streammoe on the node's layer range, one streammoe request slot per request in
  flight. Built with a model feature (`--features olmoe` or `qwen36`) and selected with
  `[engine] kind = "streammoe"`. streammoe comes from
  [transybao1393/streamoe](https://github.com/transybao1393/streamoe), pinned to a commit in
  `crates/msh-node/Cargo.toml`.

For development the real engine runs **OLMoE-1B-7B** (16 layers, 3.9 GB at 4-bit) from
`~/models/olmoe-1b-7b`: a 3-process 8/4/4 split reproduces streammoe's MLX reference answer
token for token.

## Quick start

```bash
cargo test --release

# three nodes on one machine over loopback
B=target/release/msh-node
$B --config configs/local-3node.toml --node m1b &
$B --config configs/local-3node.toml --node m1a &
$B --config configs/local-3node.toml --node m4 --max-tokens 500 --requests 3 --metrics results/run.csv

# the real engine on OLMoE, served over HTTP, and its model-backed tests
cargo build --release --features msh-node/olmoe
$B --config configs/olmoe-3node.toml --node m1b &
$B --config configs/olmoe-3node.toml --node m1a &
$B --config configs/olmoe-3node.toml --node m4 &
curl localhost:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is the capital of France?"}]}'
cargo test -p msh-node --features olmoe --release -- --ignored --test-threads=1

# network round-trip probe: serve on one machine, run from another
target/release/msh-probe serve 0.0.0.0:7070
target/release/msh-probe run <host>:7070 ethernet results/m0-<name>.csv
```

## Baseline numbers

Loopback on a MacBook Pro M3 Pro ([`results/m0-m3pro-local.csv`](results/m0-m3pro-local.csv)),
the software floor for the M4↔M1 runs:

| Payload | Round trip p50 | One hop |
| --- | --- | --- |
| 4 KB | 13.7 µs | ~7 µs |
| 1 MB | 137 µs | ~68 µs |
| 16 MB | 3.2 ms | ~1.6 ms |

## License

[PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0) —
see [`LICENSE.md`](LICENSE.md). Free for noncommercial use (personal, research, education,
nonprofits); commercial use needs a separate license from the author.
