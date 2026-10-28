# Speed in the cluster

How the three-node cluster (Mac mini M4 controller, Mac mini M1, Mac mini M4) gets tokens out
faster, which parts are built, and which wait on streammoe.

## Where a token's time goes

```
                      ┌──hop──▶ worker m1 (M1)  [layers a..b]
controller (M4)       │  ◀─hop──┘
[layers 0..a] ────────┤
                      │ ──hop──▶ worker m4b (M4) [layers b..n + head]
                      └─◀─hop──┘ token
```

The controller talks to each worker; workers never talk to each other. One request's token is
serial: every node's layers run in turn, plus four network hops (to m1, back, to m4b, back).
While one node computes, the other two wait for it. There are two ways to fill that idle
time:

- **More work per trip:** prompt chunks, draft tokens, several requests in one pass. A trip
  through the workers carrying k positions costs about the same four hops as one position.
- **More trips in the cluster at once:** several requests, so each node works on a different
  one.

Then each node's own step gets faster: expert streaming, prefetch and a balanced split.

## Techniques

| Technique | What it buys | Where | State |
| --- | --- | --- | --- |
| Several requests in flight | Every node busy; total tok/s | scheduler | Done |
| Chunked, pipelined prefill | Time to first token (5.6 → 1.4 s on 400 tokens, OLMoE) | scheduler, engine | Done |
| Speculative decoding, prompt lookup | Several tokens per trip through the workers when text repeats | scheduler, `draft.rs`, protocol | Cluster side done; streammoe API needed |
| MTP drafting | Drafts for any text, not just repeats | head node + `Drafter` | Needs MTP weights and streammoe support |
| Prefix reuse across chat turns | No re-prefill of the conversation so far | scheduler | Done (`prefix_cache`) |
| Batched decode per node | Several requests' tokens in one pass on each node | engine | streammoe API needed |
| Cross-node expert prefetch | Next node starts its SSD reads before the hidden state arrives | protocol, engine | streammoe API needed |
| Streaming settings per node | Fewer SSD stalls: prefetch guesses, packed 3-bit experts, fast disk tier | `[engine]` | Passed through; streammoe defaults kept until measured |
| Planner (CL-3) | Layers split by each node's memory and speed, so no node holds the others back | `planner.rs`, every node | Done; speeds from config or `--speeds-from` |

## Planner

Every node runs the same planner on the same config, so they agree on the split without
talking. It picks the split whose slowest node (layers ÷ `speed`) is fastest, then the most
even one, and gives ties to later nodes. Under streammoe a node may only take a range whose
fixed memory (weights, KV, scratch, buffers) fits its `mem_cap_gib`; what is left of the cap
is its expert hot tier, which `--plan` prints. A range that only just fits leaves a tiny hot
tier and streams nearly every expert from SSD, so check `--plan` before a run.

`speed` defaults to 1. After a run with `--metrics`, `--speeds-from` gives each node's
measured layers per millisecond, scaled so the slowest is 1. Those speeds were measured with
that run's split and hot tiers, so re-plan and re-measure once or twice.

## Speculative decoding in the cluster

The controller guesses the next tokens and sends them with the real one as a single step.

1. **Draft:** the controller looks up the latest n-gram (at least `draft_min_ngram` tokens)
   earlier in the prompt and output, and proposes up to `max_draft` tokens that followed it
   (`PromptLookup`). Nothing is drafted when there is no match, so plain text costs nothing
   extra.
2. **Verify:** the step carries `[next token, d1..dk]`. The `Hidden` frame sets
   `FLAG_EVERY_TOKEN`, and the head returns a token after every position in one `Token`
   frame.
3. **Accept:** the controller keeps drafts while they equal the head's tokens, and emits
   those plus the head's next token: between 1 and k + 1 tokens per trip. The output is the
   same as plain greedy decoding.
4. **Rewind:** if a draft was rejected, the controller rewinds its own engine and sends
   `Rewind(keep)` through the workers. The next step follows it on the same FIFO links, so a
   rejection costs no extra round trip.

`max_draft = 0` (the default) turns it off, and so does an engine that cannot verify and
rewind (`Engine::can_speculate`). The metrics CSV records `drafted` and `accepted` for every
step.

streammoe measured prompt-lookup speculation on one node at 0.79–1.02× plain decode. Its
verify pass is disk-bound: the drafts route to experts that are not in RAM. The cluster may
come out differently, because the hops are paid once per trip instead of once per token,
and three nodes' RAM holds more experts. That has to be measured on the Mac minis before it
is turned on by default.

## Expert streaming settings

Each node streams its own layers' experts from its SSD, keeping the most used in a RAM hot
tier. `[engine]` passes streammoe's settings on:

- `prefetch_guess`: `prev` (streammoe's default) or `cross`. streamoe measured `cross` with 6
  experts 14% faster than its default on Qwen3.6-35B-A3B at a 4 GiB cap, and no faster on
  OLMoE. In a cluster each node's first layer cannot use `cross` (it has no previous layer to
  guess from), so the gain may be smaller.
- `prefetch_experts`: 4 by default under a cap; 6 was best in that measurement.
- `experts_dir`: `root` (4-bit) or `root-3bit` (packed 3-bit: 13 vs 17 GB on a3b, 9–15%
  faster in streamoe's runs, slightly less accurate).
- `fast_experts_dir`: copies of the hottest layers' experts on a faster disk.

These stay at streammoe's defaults until they are measured on the Mac minis.
`StreamMoeEngine::expert_io_stats` gives a node's disk reads, prefetch reads and hits, and
I/O wait.

## Prefix reuse

With `prefix_cache = N`, the last N finished requests keep their state on every node. A new
prompt that starts with everything such a request evaluated, plus at least one more token,
takes over its request id and prefills only the rest. In a chat, the next turn's prompt is
the conversation so far, so only the new message is prefilled. The oldest parked request is
dropped on every node when more than N are kept.

Each parked request holds a request slot of KV memory on every node, so engines open with
`max_in_flight + prefix_cache` slots. A request that ends in the middle of a verify step is
not parked, because its engines still hold rejected drafts. The metrics' `cached` column
shows the positions each request reused.

## MTP

Qwen3.6-35B-A3B's config has one MTP layer (`mtp_num_hidden_layers: 1`), but the MLX 4-bit
conversion in `~/models/qwen3-6-35b-a3b/mlx-4bit` has no MTP tensors, and streammoe has no
MTP code. Once both exist, the node holding the head runs the MTP layer after the head and
returns its guesses with the token. The controller then drafts from those instead of prompt
lookup, through the same `Drafter` trait and verify path.

## What the cluster needs from streammoe

Every node runs a layer range, but these streammoe features currently run on the whole model
only:

1. **Logits for every position on the head's layer range** (`eval_request` returns the last
   position only). Needed to verify drafts.
2. **Rewinding a request on a layer range** (`rewind_tokens` is whole-model and
   single-sequence). Needed to drop rejected drafts; for the linear-attention layers it means
   restoring the recurrence.
3. **`eval_requests` with hidden input and output on a layer range**: one token from each of
   several requests in one pass, on every node.
4. **An expert prefetch hint for a layer from a hidden state**, so node i can tell node i+1
   which experts its first layer will likely need while node i finishes its own last layer.
5. **MTP**, plus the MTP weights from the original checkpoint.

`StreamMoeEngine` reports `can_speculate() == false` until 1 and 2 exist; the cluster side is
tested on the mock engine.

## Testing

Each part is tested on its own on the dev Mac: mock-engine clusters for the scheduler and
protocol, and small OLMoE stage tests for the engine. Results go to `results/`. Whole-cluster
and big-model runs happen on the Mac minis.
