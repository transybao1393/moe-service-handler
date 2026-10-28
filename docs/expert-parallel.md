# Expert-parallel design

Status: the cluster side is built and tested on the mock engine (`split = "experts"`, beside
the default layer split). The real engine needs streammoe APIs that do not exist yet (below),
so until then streammoe runs the layer split.

## Why

With the layer split, a token goes through m4's layers, then m1's, then m4b's: m4b needs m1's
output, so the workers take turns and one token is never faster than on one machine. Splitting
the **experts** instead lets the controller send to both workers at the same time.

## The flow, for each token

```
                 client
                   │ prompt            ▲ token
                   ▼                   │
        ┌─────────────────────────────────────┐
        │ m4 · controller                     │
        │ embed, every layer's attention,     │
        │ router, shared expert, KV, head     │
        └─────────────────────────────────────┘
          │ hidden + experts     ▲ partial results
          │ (both at once)       │ (both back)
     ┌────┴──────┐          ┌────┴──────┐
     │ m1        │          │ m4b       │
     │ its       │          │ its       │
     │ experts   │          │ experts   │
     └───────────┘          └───────────┘
```

1. The prompt comes in to the controller.
2. For layer L, the controller runs attention, the shared expert and the router. The router
   picks 8 of the layer's 256 experts for each position.
3. The controller sends the hidden state and the chosen experts (ids and gate weights) to
   **both workers at the same time**. Each worker gets only the experts it owns.
4. Each worker computes its experts and sends the outputs back.
5. The controller adds them up and goes on to layer L+1. Steps 2–5 repeat for all 40 layers.
6. After the last layer the controller runs the head, gets the token and streams it to the
   client. The next token starts again at step 2.

The router for layer L needs layer L−1's output, so the controller cannot know a token's
experts for every layer up front; it decides one layer at a time. That is about 40 small round
trips per token instead of 2.

## Expert ownership

- Every expert has **one home worker**, so no expert is held twice. The controller keeps the
  table (256 × 40 entries) and sends each worker the list it owns at start.
- At start the table splits each layer's experts evenly. The controller counts how often each
  expert is picked and, every `rebalance_every` positions, moves up to `max_moves` experts so
  both workers get about the same work per token (each about 4 of the 8 experts per layer).
- A move sends the expert's weights to the new owner (`Move`, about 1.7 MB at 4-bit). The
  table changes only when the new owner confirms; then the old owner frees its copy
  (`Release`). Steps already sent to the old owner still find the expert, since each link
  keeps its order.
- The weights for a move come from the **controller**, which keeps a copy of every expert on
  its SSD (about 17 GiB at 4-bit), since workers never talk to each other. It reads them only
  when an expert moves, so they need no RAM.
- Not yet: a minimum gain per move. With none, experts can move back and forth; add one
  before real runs.

## Memory and loading

The experts of Qwen3.6-35B-A3B are 16.9 GiB at 4-bit (about 12.6 GiB at 3-bit), so half is
8.44 GiB (6.3 GiB). Each worker has at least 10 GB of RAM to spare, so **a worker keeps every
expert it owns in memory**: loaded once at start, no SSD reads while serving.

| Node | Holds | Memory |
| --- | --- | --- |
| m4 controller | embedding, every layer's attention, router and shared expert, the head, all KV; every expert on SSD, for moves | ~1.3 GiB weights + KV (KV size to be planned from streammoe's budget) + service ~140 MiB |
| m1 worker | half the experts | 8.44 GiB at 4-bit, 6.3 GiB at 3-bit + service ~5 MiB |
| m4b worker | the other half | same |

Weights come from `model_weights.json`; expert sizes from streammoe's 4-bit expert size.
10 GB is 9.3 GiB, so 4-bit leaves under 1 GiB for the runtime; 3-bit leaves about 3 GiB.

Workers keep **no state per request** (the KV cache stays on the controller), so `Drop` and
`Rewind` never go to them.

**Where a worker loads its experts from:** its own internal SSD (copied once at setup) or, if
it has none, from the controller over the network at start (8.5 GB, about 8 s on 10 Gb
Ethernet). Either way, only hidden states cross the network while serving.

**If a worker's memory is smaller than what it owns**, it keeps a hot set: it loads what is
picked, keeps what is used, and frees what has gone cold to make room. The worker does this on
its own, since it sees every request for its experts; the controller is not involved.

## Frames

New message types, beside `Hello`, `Ping`/`Pong`, `Error` (protocol version 5; every node
needs the same build). Little-endian, after the usual 24-byte header:

| Type | Direction | Payload |
| --- | --- | --- |
| `Own` | controller → worker | `count u32`, then `(layer u16, expert u16)` × count; the worker answers with an empty `Own` once it holds them all |
| `Experts` | controller → worker | `layer u16, picks u32`, then `(row u16, expert u16, gate f32)` × picks, then every row of the step as f32 |
| `Partial` | worker → controller | `layer u16, compute_us u32`, then one f32 row per pick, gate × expert output, in the order asked |
| `Move` | controller → worker | `layer u16, expert u16`, then the weights; the reply is the same frame without weights |
| `Release` | controller → worker | `layer u16, expert u16`: free a moved expert |

`Experts` and `Partial` carry the request id and the step's position in the header; `Own`,
`Move` and `Release` use request id 0, and an `Error` with id 0 answers one of them (a failed
`Move` leaves the old owner in place). Rows go on the wire in place, with no encode copy.

The worker applies the gate; `Partial` returns each pick's output separately (not summed), so
the controller adds them in a fixed order (router rank within each row) and the result stays
bit-identical to one machine. A decode token sends 8 KiB to each worker and gets back about
4 × 8 KiB per layer.

## Speed

- **Network:** 40 round trips per token, a few KiB each. On a real link a small round trip is
  roughly 50–200 µs, so 2–8 ms per token. To measure on the Mac minis.
- **Compute:** the experts, most of the per-token work, run on both workers at once.
- **Several requests:** while the workers compute layer L for one request, the controller runs
  attention for another, so `max_in_flight` keeps every node busy.
- **One step per request at a time** (`prefill_depth` is 1 here): a later prompt chunk could
  otherwise reach layer L+1 before the earlier chunk had updated that layer's attention state.

## What streammoe needs to add

1. On the controller: run one layer **without its routed experts** (attention, shared expert,
   router), returning the hidden state and the picked experts; then take the summed expert
   outputs and finish the layer.
2. On a worker: hold a **chosen set of experts** in memory and compute them for given rows.

Both belong to the streammoe-dependent work on hold.

## What changes in this repo

- **Stays:** the controller-to-each-worker connections, health checks, HTTP API, metrics,
  logs, service memory allowance, speculative decoding and prefix reuse (both live on the
  controller).
- **Changes:** frames (above), the scheduler step (40 fan-outs per token instead of one pass),
  the worker loop (expert store instead of a layer range), the planner (expert ownership
  instead of layer ranges), the memory budget per node, the README and the diagram.
- **Built on the mock engine:** frames, the ownership table and rebalancing (`owners.rs`), the
  controller's per-layer fan-out (`scheduler.rs`), the worker's expert store (`node.rs`), the
  memory estimate, metrics (`workers_us`: the slowest worker per layer, summed) and tests
  (`tests/experts.rs`: the 3-node cluster equals one process, with rebalancing, a hot set,
  speculative decoding, worker errors and lost or hung workers).
- **Not yet:** the hot set has no config key (under streammoe it would follow the memory cap),
  and the controller copies each `Partial` once after reading it.
