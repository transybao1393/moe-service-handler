# PRD: moe-service-handler

**Author:** Tran Sy Bao (Johnny)
**Date:** 2026-10-25
**Status:** Draft
**Related:** streammoe architecture and goals (fork README), *Streaming Mixture-of-Experts from an External SSD on a Mac mini* (engineering post, Part 1)

---

## 1. Summary

`moe-service-handler` is a Rust workspace that turns streammoe into a service running Qwen3.6-35B-A3B on one or three Mac minis. It has three crates:

| Crate | Kind | One-line purpose | Mode |
| --- | --- | --- | --- |
| `msh-node` | binary | Coordinator or worker: planner, scheduler, health checks, HTTP endpoint, run-level metrics | A, B |
| `msh-net` | library | Node-to-node transport for hidden states and tokens | B |
| `msh-policy` | library | Rules for each selected expert (4-bit, 2-bit or skip), applied through streammoe's router hook | A |

**Rule for the boundary:** anything that touches one machine's memory or GPU lives in streammoe. Anything that spans several machines or requests, or is a policy choice, lives in moe-service-handler.

### 1.1 Where each part lives

| Part | Lives in | Note |
| --- | --- | --- |
| Hard memory cap (budget) | streammoe | Reserved at `open`; covers every engine allocation |
| Hot tier, slot ring | streammoe | Written straight into the buffers the GPU reads |
| SSD reads (`pread` + `F_NOCACHE`) | streammoe | |
| Expert caching and prefetch | streammoe | Prefetch needs the next layer's gate |
| Layer-range execution (embed, range, head) | streammoe | |
| Per-node speed benchmark | streammoe | Used by the planner |
| KV and DeltaNet state per request | streammoe | |
| Router hook | streammoe | Exposed for `msh-policy` |
| 2-bit expert path | streammoe | |
| In-place measurement (per token, layer, tier) | streammoe | Exposed through a callback |
| Planner (layer split per machine) | moe-service-handler | `msh-node` |
| Scheduler (ordering and cancelling requests) | moe-service-handler | `msh-node` |
| Coordinator / worker roles | moe-service-handler | `msh-node` |
| Health checks, dead-node detection | moe-service-handler | `msh-node` |
| OpenAI-compatible HTTP API | moe-service-handler | `msh-node` |
| Hidden-state and token transport | moe-service-handler | `msh-net` |
| Policy rules (skip / 2-bit / 4-bit) | moe-service-handler | `msh-policy`; calls streammoe's router hook |
| Joining measurements with network wait | moe-service-handler | `msh-node` |
| CSV and Prometheus export | moe-service-handler | `msh-node` |
| The service's own memory (network buffers, HTTP) | moe-service-handler | Counted separately, outside streammoe's cap |

## 2. Problem

streammoe runs a range of layers inside a hard memory cap on one machine. It does not:

- move hidden states between machines;
- decide which machine runs which layers;
- accept requests from applications or schedule several of them through a pipeline;
- choose, per expert, whether to skip it or run a cheaper copy.

Without these, mode B (three-node pipeline) cannot run, and mode A cannot use the decision layer.

## 3. Goals and non-goals

### Goals

1. Run Qwen3.6-35B-A3B in mode B across a Mac mini M4 and two Mac mini M1 (16 GB), with each node running streammoe on its own layer range.
2. Serve applications on the LAN through an OpenAI-compatible HTTP endpoint, in both modes.
3. Keep several requests in flight through the mode B pipeline, so all three nodes stay busy.
4. Let a policy skip low-weight experts or run them from a 2-bit copy in mode A.
5. Combine streammoe's per-node measurements with network timing into one run-level record per token.

### Non-goals

- Anything inside one node's engine: memory cap, hot tier, streaming, caching, prefetch, kernels.
- Models other than Qwen3.6-35B-A3B in the first release.
- Expert parallelism (experts of one layer split across nodes): it needs up to one network round trip per layer, against 2–3 per token for a split by layer.
- Redundancy or failover beyond detecting a dead node and failing its requests cleanly.
- Non-Apple hardware.

## 4. Users and use cases

| User | Use case | Mode |
| --- | --- | --- |
| Johnny, as developer | Benchmarks and ablations for Part 2 of the engineering post | A, B |
| Local tools (coding agent, content generation) | Call the HTTP endpoint on the LAN | A, B |
| Small team on the LAN | Several concurrent chat sessions | B |

## 5. System overview

```
             ┌──────────────────────── Mac mini M4 ─────────────────────────┐
 client ───▶ │ msh-node --coordinator (HTTP, scheduler, planner)            │
             │   └─ streammoe: embedding + layers 1–20                        │
             └───────────────┬──────────────────────────────────────────────┘
                             │ msh-net: hidden state, ~4 KB per token
             ┌───────────────▼──────────────┐      ┌──────────────────────────────┐
             │ Mac mini M1 (a)              │      │ Mac mini M1 (b)              │
             │ msh-node --worker            │ ───▶ │ msh-node --worker            │
             │   └─ streammoe: layers 21–30   │      │   └─ streammoe: layers 31–40   │
             │                              │      │      + head                  │
             └──────────────────────────────┘      └───────────────┬──────────────┘
                             ▲                                     │ token id
                             └─────────────────────────────────────┘ back to M4
```

Every node runs the same binary. The M4 runs as coordinator; the M1s run as workers. Mode A is a one-node cluster: the coordinator runs streammoe on all 40 layers and no transport is used.

The 20 / 10 / 10 split is the starting plan: each layer is about 0.44 GB at 4-bit (calculated), so 20 layers fit in the M4's 10 GB cap and the M1s keep spare memory for KV cache. The planner may change it based on measured speed.

## 6. Dependencies on streammoe

moe-service-handler needs these from streammoe. Each one is a streammoe goal or a small addition to it.

| ID | streammoe must provide | Used by |
| --- | --- | --- |
| DEP-1 | Open a context on a contiguous layer range, with flags for embedding (first node) and head (last node) | Planner, coordinator, workers |
| DEP-2 | Entry points that take and return a hidden state: embed tokens, run the range, apply the head | Coordinator, workers |
| DEP-3 | Separate KV and DeltaNet state per request id, with a call to free one request's state | Scheduler |
| DEP-4 | `open` fails cleanly when the layer range does not fit the configured cap | Planner |
| DEP-5 | A router hook called after routing each layer, receiving the selected experts and gate weights and returning a decision per expert and, optionally, adjusted gate weights (streammoe keeps the default 4-bit path when no hook is set) | `msh-policy` |
| DEP-6 | A 2-bit expert path, used only when the hook asks for it | `msh-policy` |
| DEP-7 | Per-token, per-layer, per-tier timing and hit rates, delivered through a callback or ring buffer; no file or HTTP output inside streammoe | Run-level metrics |
| DEP-8 | A short built-in benchmark (decode time per layer) the planner can call at start-up | Planner |

---

## 7. Component requirements

Requirement IDs are referenced by tests and milestones. **Must** = required for the milestone that introduces the crate; **Should** = wanted, may slip.

### 7.1 `msh-node` (binary)

**Purpose:** run a node as coordinator or worker, plan the split, serve and schedule requests, and record run-level metrics.

| ID | Requirement | Priority |
| --- | --- | --- |
| CL-1 | One binary, two roles: `--coordinator` and `--worker`, configured by a TOML file listing nodes, addresses and memory caps | Must |
| CL-2 | Coordinator exposes an OpenAI-compatible `POST /v1/chat/completions` with streaming (server-sent events) | Must |
| CL-3 | Planner runs at start-up: reads each node's cap, runs streammoe's benchmark (DEP-8), assigns contiguous layer ranges, refuses to start if the model does not fit | Must |
| CL-4 | A manual layer split in the config overrides the planner | Must |
| CL-5 | Scheduler keeps up to N requests in flight (N configurable, default 3) and interleaves their tokens so each node works on a different request | Must |
| CL-6 | Prefill runs in chunks so nodes overlap on long prompts | Should |
| CL-7 | Mode A runs as a one-node cluster with the same binary and config format | Must |
| CL-8 | Health check per node; a missing worker fails in-flight requests with a clear error and the coordinator stops accepting work | Must |
| CL-9 | A finished, cancelled or failed request frees its state on every node (DEP-3) | Must |
| CL-10 | Run-level metrics: per token, each node's streammoe timing (DEP-7) plus network wait and queueing time, joined by request id and position | Must |
| CL-11 | CSV export per run, one row per token, with run metadata (git hash, config, hardware) | Must |
| CL-12 | Optional Prometheus endpoint on each node | Should |

### 7.2 `msh-net` (library)

**Purpose:** move hidden states forward and token ids back with the lowest possible latency.

| ID | Requirement | Priority |
| --- | --- | --- |
| NET-1 | Frame format: magic, version, message type, request id (u64), position (u32), payload length, payload; little-endian | Must |
| NET-2 | Message types: `Hidden` (BF16, `seq_len × 2048` values), `Token` (u32), `Drop` (free a request), `Ping` / `Pong` | Must |
| NET-3 | TCP with `TCP_NODELAY`, one persistent connection per neighbouring node | Must |
| NET-4 | Works over Ethernet or a Thunderbolt bridge interface with only the address changed | Must |
| NET-5 | Latency probe: round-trip time for a 4 KB frame, reported in run-level metrics | Must |
| NET-6 | Send directly from streammoe's output buffer, without an extra copy, where the OS allows | Should |
| NET-7 | Version mismatch between nodes is refused at connect time | Must |

### 7.3 `msh-policy` (library)

**Purpose:** decide, through streammoe's router hook (DEP-5), how each selected expert runs.

| ID | Requirement | Priority |
| --- | --- | --- |
| POL-1 | Implements the router hook: given a layer and its (expert, gate) picks, returns `Full`, `LowBit` or `Skip` per expert | Must |
| POL-2 | `AlwaysFull` as the default and as the reference for equivalence tests | Must |
| POL-3 | Threshold policy: skip below gate threshold T1; 2-bit for experts outside the hot tier with gate below T2; both from config | Must |
| POL-4 | After a skip, returns renormalised gate weights for the remaining experts through the hook | Must |
| POL-5 | Decisions are logged per token and layer for the ablation tables | Must |

---

## 8. Non-functional requirements

| ID | Requirement |
| --- | --- |
| NF-1 | **Correctness:** with `AlwaysFull`, the pipeline's logits match streammoe running all 40 layers in one process on the same prompt, within the tolerance streammoe's differential tests already use. |
| NF-2 | **Isolation:** moe-service-handler keeps its own memory (network buffers, HTTP, metrics) inside a fixed allowance set in config, counted separately from streammoe's cap; each node's total footprint stays within cap + allowance. |
| NF-3 | **Platform:** macOS on Apple Silicon (M1 and M4). Rust stable. No Python at runtime. |
| NF-4 | **Configuration:** one TOML file per node; every threshold, cap and split changes without rebuilding. |
| NF-5 | **Logging:** structured logs (`tracing`), request id on every line. |
| NF-6 | **License:** PolyForm Noncommercial 1.0.0. |

## 9. Success metrics

The tok/s figures are calculated ceilings from published bandwidths, not measurements. The release is judged on measuring each one and explaining the gap.

| Metric | Mode | Ceiling (calculated) | Release bar |
| --- | --- | --- | --- |
| Decode, one request | B | ~58 tok/s, before network latency | Measured and explained |
| Decode, one request | A | ~17–23 tok/s at a 70–80% expert hit rate | Measured and explained |
| Decode, three requests together | B | Up to ~3× the one-request rate | Higher than the one-request rate |
| Output mismatch with `AlwaysFull` | A, B | — | None beyond test tolerance |
| Node footprint above cap + allowance | A, B | — | Never |

Ceiling inputs: M4 memory bandwidth 120 GB/s; M1 about 68.25 GB/s; Thunderbolt 4 SSD read about 3.2 GB/s on a Mac mini M4; about 1.5 GB of active weights and 0.5 GB of routed experts per token at 4-bit.

## 10. Milestones

| Milestone | Delivers | Depends on streammoe | Exit criterion |
| --- | --- | --- | --- |
| M0 Baseline | Network round-trip probe between M4 and M1 | — | Latency recorded |
| M1 Two nodes | `msh-net`; coordinator and one worker with a manual split | DEP-1, DEP-2 | NF-1 holds; decode measured |
| M2 Three nodes, many requests | Planner, scheduler, HTTP endpoint | DEP-3, DEP-4, DEP-8 | Three-request throughput above one-request rate |
| M3 Policy in mode A | `msh-policy` | DEP-5, DEP-6 | Mode A decode measured with and without each policy |
| M4 Metrics complete | Run-level metrics and CSV export | DEP-7 | Part 2 tables reproducible from one command |

## 11. Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Network round trip between Mac minis is high | Each token pays it 2–3 times in mode B | Measure in M0; try a Thunderbolt bridge; keep several requests in flight to hide it |
| streammoe runs slowly on the M1 GPU | Mode B ceiling drops | Planner gives the M4 more layers |
| streammoe APIs (DEP-1 to DEP-8) change while both are in development | Rework | Pin a streammoe version per milestone; write the dependencies down as tests |
| 2-bit experts hurt output quality | Policy unusable | Flash-MoE found all-2-bit experts broke JSON and tool calling; use 2-bit only for low-gate experts and judge on task accuracy |
| One node fails | Whole pipeline stops | Health checks and clean failure (CL-8); no redundancy in this release |

## 12. Open questions

- [ ] Round-trip latency of a 4 KB frame between M4 and M1, over Ethernet and over a Thunderbolt bridge.
- [ ] Default number of requests in flight (CL-5) once real latencies are known.
- [ ] Size of the per-node allowance for the service itself (NF-2).
- [ ] Whether the router hook (DEP-5) and the 2-bit path (DEP-6) are planned in streammoe, or need to be proposed there.
