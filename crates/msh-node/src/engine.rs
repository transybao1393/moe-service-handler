//! What a node needs from streammoe (DEP-1, DEP-2, DEP-3), as a trait shaped like streammoe's
//! `eval_request`: a stage takes a token (first node) or a hidden state, and produces a hidden
//! state for the next node or, on the last node, the sampled token.
//!
//! `MockEngine` keeps per-request state that depends on every earlier position, like a
//! KV/DeltaNet cache, so ordering and freeing bugs show up as wrong tokens or errors.
//!
//! The expert split (`split = "experts"`, docs/expert-parallel.md) needs more: on the
//! controller, a layer run up to its routed experts and then finished with their outputs; on a
//! worker, a set of held experts run on given rows. Those methods fail by default; the mock
//! has them with [`MockEngine::with_experts`]. Its layers are then attention (the per-request
//! state), a shared expert, a router picking `top_k` experts per position, and the routed
//! experts, whose gated outputs are added in pick order.

use std::collections::{HashMap, HashSet};
use std::ops::Range;

pub use msh_net::Pick;

use crate::Error;

/// The model's layers, experts per layer and experts picked per position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeShape {
    pub layers: u32,
    pub experts: u32,
    pub top_k: u32,
}

fn no_experts<T>() -> Result<T, Error> {
    Err(Error::Engine("this engine cannot split by experts".into()))
}

/// Input to a run of consecutive positions on a node's stage: one position while decoding, a
/// chunk of the prompt during prefill (CL-6).
pub enum StageIn<'a> {
    /// Token ids; only the first node (it holds the embedding).
    Tokens(&'a [u32]),
    /// The previous node's output, `hidden_dim` f32s per position, row-major.
    Hidden(&'a [f32]),
}

impl StageIn<'_> {
    /// Positions carried, given the model's hidden size.
    pub fn len(&self, hidden_dim: usize) -> usize {
        match self {
            StageIn::Tokens(t) => t.len(),
            StageIn::Hidden(h) => h.len() / hidden_dim,
        }
    }
}

/// Which tokens the node holding the head samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sample {
    /// After the last position only: a prompt chunk or a plain decode step.
    Last,
    /// After every position: verifying draft tokens (speculative decoding).
    Every,
}

pub trait Engine: Send {
    fn hidden_dim(&self) -> usize;
    /// End-of-sequence token: generation stops there, and it is not reported as output.
    fn eos(&self) -> Option<u32> {
        None
    }
    /// Whether this node holds the head, and so samples tokens instead of passing on hidden
    /// states.
    fn has_head(&self) -> bool;
    /// Whether `Sample::Every` and `rewind` work, which speculative decoding needs on every
    /// node.
    fn can_speculate(&self) -> bool {
        false
    }
    /// Runs this node's layer range for positions `start_pos..` of request `req`. On the node
    /// holding the head, sets `tokens` to the greedy tokens `sample` asks for; elsewhere
    /// resizes `hidden_out` to the next node's input, one row per position. A request's
    /// positions must arrive in order from 0.
    fn step(
        &mut self,
        req: u64,
        start_pos: u32,
        input: StageIn<'_>,
        sample: Sample,
        hidden_out: &mut Vec<f32>,
        tokens: &mut Vec<u32>,
    ) -> Result<(), Error>;
    /// Keeps the first `len` positions of `req`, dropping rejected draft tokens. Only
    /// positions of the request's last step can be dropped.
    fn rewind(&mut self, req: u64, len: u32) -> Result<(), Error> {
        let _ = len;
        Err(Error::Engine(format!("request {req}: this engine cannot rewind")))
    }
    /// Frees all state held for `req`. Freeing an unknown request is a no-op.
    fn free(&mut self, req: u64);

    // The expert split. The controller calls `embed`, then `route` and `combine` for every
    // layer in order, then `head`; the workers' `run_experts` comes in between. Only the
    // controller keeps state per request.

    /// The routed experts' shape, if this engine can split by experts.
    fn moe_shape(&self) -> Option<MoeShape> {
        None
    }
    /// Controller: starts a step of positions `start_pos..` of `req`, setting `hidden` to the
    /// embedding of `tokens`, one row per position. `sample` is what `head` will take.
    fn embed(&mut self, req: u64, start_pos: u32, tokens: &[u32], sample: Sample, hidden: &mut Vec<f32>) -> Result<(), Error> {
        let _ = (req, start_pos, tokens, sample, hidden);
        no_experts()
    }
    /// Controller: runs `layer` of the step up to its routed experts. Turns `hidden` into the
    /// layer's output without them (attention, shared expert, residual), sets `x` to the rows
    /// the experts take and `picks` to each row's experts in the order they are added.
    fn route(&mut self, req: u64, layer: u32, hidden: &mut [f32], x: &mut Vec<f32>, picks: &mut Vec<Pick>) -> Result<(), Error> {
        let _ = (req, layer, hidden, x, picks);
        no_experts()
    }
    /// Controller: finishes `layer` by adding the experts' outputs, one row per pick in
    /// `picks` order, to `hidden`. The order is fixed, so the sum is the same on every split.
    fn combine(&self, layer: u32, picks: &[Pick], outputs: &[f32], hidden: &mut [f32]) {
        let _ = layer;
        let dim = self.hidden_dim();
        for (p, out) in picks.iter().zip(outputs.chunks_exact(dim)) {
            let row = &mut hidden[p.row as usize * dim..][..dim];
            for (h, o) in row.iter_mut().zip(out) {
                *h += o;
            }
        }
    }
    /// Controller: the head after the last layer; sets `tokens` as `step` does.
    fn head(&mut self, req: u64, hidden: &[f32], sample: Sample, tokens: &mut Vec<u32>) -> Result<(), Error> {
        let _ = (req, hidden, sample, tokens);
        no_experts()
    }
    /// Controller: an expert's weights, as a `Move` carries them to its new owner.
    fn expert_weights(&self, layer: u32, expert: u32, out: &mut Vec<u8>) -> Result<(), Error> {
        let _ = (layer, expert, out);
        no_experts()
    }
    /// Worker: holds these `(layer, expert)`s, replacing any held before.
    fn own_experts(&mut self, experts: &[(u32, u32)]) -> Result<(), Error> {
        let _ = experts;
        no_experts()
    }
    /// Worker: takes over an expert from its weights.
    fn load_expert(&mut self, layer: u32, expert: u32, weights: &[u8]) -> Result<(), Error> {
        let _ = (layer, expert, weights);
        no_experts()
    }
    /// Worker: no longer holds an expert.
    fn release_expert(&mut self, layer: u32, expert: u32) {
        let _ = (layer, expert);
    }
    /// Worker: each pick's expert of `layer` on its row of `rows`, scaled by its gate; one
    /// output row per pick in `out`, in `picks` order. `req` is for errors and logs only.
    fn run_experts(&mut self, req: u64, layer: u32, rows: &[f32], picks: &[Pick], out: &mut Vec<f32>) -> Result<(), Error> {
        let _ = (req, layer, rows, picks, out);
        no_experts()
    }
}

pub struct MockEngine {
    layers: Range<u32>,
    head: bool,
    dim: usize,
    vocab: u32,
    state: HashMap<u64, MockState>,
    fail_on: Option<u64>,
    /// Set: layers have routed experts.
    moe: Option<MoeShape>,
    /// A worker's experts.
    store: ExpertStore,
}

struct MockState {
    next_pos: u32,
    /// One state vector per layer.
    layers: Vec<f32>,
    /// The state before each position of the last step, for `rewind`.
    before: Vec<(u32, Vec<f32>)>,
    /// Positions of the step in progress (expert split), and whether it fills `before`.
    step: Range<u32>,
    every: bool,
}

/// The experts a worker owns. Up to `capacity` are in memory; the rest are loaded when picked,
/// in place of the least recently used (the hot set). The mock's disk is its weight formula.
#[derive(Default)]
struct ExpertStore {
    owned: HashSet<(u32, u32)>,
    /// Weights in memory, with when each was last used.
    resident: HashMap<(u32, u32), (Vec<f32>, u64)>,
    capacity: Option<usize>,
    clock: u64,
    /// Experts loaded while serving (picked but not in memory), and taken over by `Move`.
    loads: usize,
    moved_in: usize,
}

impl ExpertStore {
    /// Frees the least recently used expert if memory is full.
    fn make_room(&mut self) {
        if self.capacity.is_some_and(|c| self.resident.len() >= c) {
            let coldest = *self.resident.iter().min_by_key(|(_, (_, t))| *t).unwrap().0;
            self.resident.remove(&coldest);
        }
    }
}

/// A deterministic value in `[0, 1)` for a weight.
fn unit(kind: u64, layer: u32, expert: u32, i: usize) -> f32 {
    let mut z = (kind << 60) ^ ((layer as u64) << 44) ^ ((expert as u64) << 24) ^ i as u64;
    // splitmix64
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 40) as f32 / (1u64 << 24) as f32
}

/// Expert `expert` of `layer`.
fn expert_weights(layer: u32, expert: u32, dim: usize) -> Vec<f32> {
    (0..dim).map(|i| 0.5 + unit(1, layer, expert, i)).collect()
}

/// One pick's gated expert output for its row.
fn run_expert(w: &[f32], gate: f32, x: &[f32], out: &mut Vec<f32>) {
    out.extend(x.iter().zip(w).map(|(x, w)| gate * (x * w).tanh()));
}

impl MockEngine {
    /// A stage running `layers` of a `num_layers` model; the last stage holds the head.
    pub fn new(layers: Range<u32>, num_layers: u32, dim: usize) -> Self {
        let head = layers.end == num_layers;
        MockEngine {
            layers,
            head,
            dim,
            vocab: 151_936,
            state: HashMap::new(),
            fail_on: None,
            moe: None,
            store: ExpertStore::default(),
        }
    }

    /// Gives every layer `experts` routed experts, `top_k` picked per position. The same
    /// engine serves as the controller (it holds every layer and the head), a worker (it holds
    /// the experts it is told to own) or a single process.
    pub fn with_experts(mut self, experts: u32, top_k: u32) -> Self {
        assert!(top_k >= 1 && top_k <= experts && experts <= 1 << 16, "{top_k} of {experts} experts");
        self.moe = Some(MoeShape { layers: self.layers.end, experts, top_k });
        self
    }

    /// On a worker: keeps at most `n` experts in memory, loading the others when picked.
    pub fn with_expert_capacity(mut self, n: usize) -> Self {
        self.store.capacity = Some(n.max(1));
        self
    }

    /// Experts this worker owns.
    pub fn held_experts(&self) -> usize {
        self.store.owned.len()
    }

    /// Experts loaded while serving because they were not in memory, and experts taken over by
    /// `Move`.
    pub fn expert_loads(&self) -> (usize, usize) {
        (self.store.loads, self.store.moved_in)
    }

    /// Checks a step's input and starts it on `req`'s state: `n` positions from `start_pos`.
    fn begin(&mut self, req: u64, start_pos: u32, n: usize, sample: Sample) -> Result<(), Error> {
        if self.fail_on == Some(req) {
            return Err(Error::Engine(format!("mock failure on request {req}")));
        }
        if n == 0 {
            return Err(Error::Engine("no positions".into()));
        }
        let state_len = self.layers.len() * self.dim;
        let st = self.state.entry(req).or_insert_with(|| MockState {
            next_pos: 0,
            layers: vec![0.0; state_len],
            before: Vec::new(),
            step: 0..0,
            every: false,
        });
        if start_pos != st.next_pos {
            return Err(Error::Engine(format!(
                "request {req}: position {start_pos} out of order, expected {}",
                st.next_pos
            )));
        }
        st.before.clear();
        st.every = sample == Sample::Every;
        if st.every {
            // Every layer's slice is set before its position runs; see `route`.
            st.before.extend((0..n as u32).map(|p| (start_pos + p, st.layers.clone())));
        }
        st.step = start_pos..start_pos + n as u32;
        st.next_pos += n as u32;
        Ok(())
    }

    /// The embedding of `ids`, one row each, appended to `out`.
    fn embed_rows(&self, ids: &[u32], out: &mut Vec<f32>) {
        let dim = self.dim;
        for &token in ids {
            out.extend((0..dim).map(|i| ((token as f32 + 1.0) * (i as f32 + 1.0) * 1e-3).sin()));
        }
    }

    /// Greedy tokens from the last layer's output, as `sample` asks.
    fn sample(&self, out: &[f32], sample: Sample, tokens: &mut Vec<u32>) {
        let dim = self.dim;
        let n = out.len() / dim;
        let rows = match sample {
            Sample::Last => &out[(n - 1) * dim..],
            Sample::Every => out,
        };
        // Bit-sensitive on purpose: any difference in the hidden state changes the token.
        for h in rows.chunks_exact(dim) {
            let s: f32 = h.iter().enumerate().map(|(i, v)| v * (i % 7 + 1) as f32).sum();
            tokens.push(((s.to_bits() as u64).wrapping_mul(2_654_435_761) % self.vocab as u64) as u32);
        }
    }

    /// `route` for the mock: attention updates the request's state for `layer`, a shared
    /// expert adds to it, and the router picks `top_k` experts per row, with softmax gates.
    fn route_rows(&mut self, req: u64, layer: u32, hidden: &mut [f32], x: &mut Vec<f32>, picks: &mut Vec<Pick>) -> Result<(), Error> {
        let (dim, shape) = (self.dim, self.moe.ok_or_else(|| Error::Engine("the mock has no experts".into()))?);
        if !self.layers.contains(&layer) {
            return Err(Error::Engine(format!("layer {layer} is not in {:?}", self.layers)));
        }
        let k = (layer - self.layers.start) as usize;
        let st = self.state.get_mut(&req).ok_or_else(|| Error::Engine(format!("request {req}: no step started")))?;
        let n = st.step.len();
        if n == 0 || hidden.len() != n * dim {
            return Err(Error::Engine(format!("request {req}: {} values for a step of {n} positions", hidden.len())));
        }
        x.clear();
        picks.clear();
        let mut logits = Vec::with_capacity(shape.experts as usize);
        let scale = 1.0 / (dim as f32).sqrt();
        for (p, h) in hidden.chunks_exact_mut(dim).enumerate() {
            let s = &mut st.layers[k * dim..(k + 1) * dim];
            if st.every {
                st.before[p].1[k * dim..(k + 1) * dim].copy_from_slice(s);
            }
            for i in 0..dim {
                let a = (h[i] * Self::weight(layer, i) + s[i]).tanh();
                s[i] = 0.9 * s[i] + 0.1 * a;
                x.push(a);
                h[i] = a + 0.5 * (a * (0.5 + unit(3, layer, 0, i))).tanh();
            }
            let a = &x[p * dim..];
            logits.clear();
            for e in 0..shape.experts {
                let z: f32 = a.iter().take(dim).enumerate().map(|(i, v)| v * (unit(2, layer, e, i) - 0.5)).sum();
                // Low ids are picked a little more often, so use is uneven, as in a real model.
                logits.push((z * scale - 0.3 * e as f32 / shape.experts as f32, e));
            }
            logits.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            let top = &logits[..shape.top_k as usize];
            let total: f32 = top.iter().map(|(z, _)| (z - top[0].0).exp()).sum();
            picks.extend(top.iter().map(|&(z, e)| Pick { row: p as u16, expert: e as u16, gate: (z - top[0].0).exp() / total }));
        }
        Ok(())
    }

    /// Makes sure a held expert is in memory, evicting the least recently used if full.
    fn resident(&mut self, layer: u32, expert: u32) -> Result<&[f32], Error> {
        let key = (layer, expert);
        let st = &mut self.store;
        if !st.owned.contains(&key) {
            return Err(Error::Engine(format!("expert {expert} of layer {layer} is not on this node")));
        }
        st.clock += 1;
        if !st.resident.contains_key(&key) {
            st.make_room();
            st.loads += 1;
            st.resident.insert(key, (expert_weights(layer, expert, self.dim), 0));
        }
        let entry = st.resident.get_mut(&key).unwrap();
        entry.1 = st.clock;
        Ok(&entry.0)
    }

    /// Samples from `vocab` token ids instead of the default 151,936.
    pub fn with_vocab(mut self, vocab: u32) -> Self {
        self.vocab = vocab;
        self
    }

    /// Makes `step` fail for request `req`, to test error paths.
    pub fn failing_on(mut self, req: u64) -> Self {
        self.fail_on = Some(req);
        self
    }

    pub fn live_requests(&self) -> usize {
        self.state.len()
    }

    fn weight(layer: u32, i: usize) -> f32 {
        0.5 + ((layer as usize * 2048 + i) as f32 * 0.618_034).fract()
    }
}

impl Engine for MockEngine {
    fn hidden_dim(&self) -> usize {
        self.dim
    }

    fn has_head(&self) -> bool {
        self.head
    }

    fn can_speculate(&self) -> bool {
        true
    }

    fn step(
        &mut self,
        req: u64,
        start_pos: u32,
        input: StageIn<'_>,
        sample: Sample,
        out: &mut Vec<f32>,
        tokens: &mut Vec<u32>,
    ) -> Result<(), Error> {
        if self.fail_on == Some(req) {
            return Err(Error::Engine(format!("mock failure on request {req}")));
        }
        let dim = self.dim;
        let n = input.len(dim);
        out.clear();
        tokens.clear();
        match input {
            StageIn::Tokens(_) if self.layers.start != 0 => {
                return Err(Error::Engine("token input on a stage without the embedding".into()))
            }
            StageIn::Tokens(ids) => self.embed_rows(ids, out),
            StageIn::Hidden(src) if n == 0 || src.len() != n * dim => {
                return Err(Error::Engine(format!("hidden input of {} values, not whole rows", src.len())))
            }
            StageIn::Hidden(src) => out.extend_from_slice(src),
        }
        self.begin(req, start_pos, n, sample)?;
        if self.moe.is_some() {
            // Layer by layer, the way the controller and its workers run it, with every expert
            // here.
            let (mut x, mut picks, mut outputs) = (Vec::new(), Vec::new(), Vec::new());
            for layer in self.layers.clone() {
                self.route_rows(req, layer, out, &mut x, &mut picks)?;
                outputs.clear();
                for p in &picks {
                    let w = expert_weights(layer, p.expert as u32, dim);
                    run_expert(&w, p.gate, &x[p.row as usize * dim..][..dim], &mut outputs);
                }
                self.combine(layer, &picks, &outputs, out);
            }
        } else {
            let st = self.state.get_mut(&req).unwrap();
            // Position by position, so a chunk gives exactly what single positions would.
            for (p, h) in out.chunks_exact_mut(dim).enumerate() {
                if st.every {
                    st.before[p].1.copy_from_slice(&st.layers);
                }
                for (k, layer) in self.layers.clone().enumerate() {
                    let s = &mut st.layers[k * dim..(k + 1) * dim];
                    for i in 0..dim {
                        let x = (h[i] * Self::weight(layer, i) + s[i]).tanh();
                        s[i] = 0.9 * s[i] + 0.1 * x;
                        h[i] = x;
                    }
                }
            }
        }
        if self.head {
            self.sample(out, sample, tokens);
        }
        Ok(())
    }

    fn rewind(&mut self, req: u64, len: u32) -> Result<(), Error> {
        let Some(st) = self.state.get_mut(&req) else { return Ok(()) };
        if len == st.next_pos {
            return Ok(());
        }
        let i = st.before.iter().position(|(p, _)| *p == len).ok_or_else(|| {
            Error::Engine(format!("request {req}: cannot rewind to {len} from {}", st.next_pos))
        })?;
        st.layers = st.before[i].1.clone();
        st.before.truncate(i);
        st.next_pos = len;
        Ok(())
    }

    fn free(&mut self, req: u64) {
        self.state.remove(&req);
    }

    fn moe_shape(&self) -> Option<MoeShape> {
        self.moe
    }

    fn embed(&mut self, req: u64, start_pos: u32, tokens: &[u32], sample: Sample, hidden: &mut Vec<f32>) -> Result<(), Error> {
        if self.moe.is_none() || self.layers.start != 0 {
            return Err(Error::Engine("embed needs the mock with experts and the embedding".into()));
        }
        hidden.clear();
        self.begin(req, start_pos, tokens.len(), sample)?;
        self.embed_rows(tokens, hidden);
        Ok(())
    }

    fn route(&mut self, req: u64, layer: u32, hidden: &mut [f32], x: &mut Vec<f32>, picks: &mut Vec<Pick>) -> Result<(), Error> {
        self.route_rows(req, layer, hidden, x, picks)
    }

    fn head(&mut self, req: u64, hidden: &[f32], sample: Sample, tokens: &mut Vec<u32>) -> Result<(), Error> {
        tokens.clear();
        if !self.head {
            return Err(Error::Engine("no head on this node".into()));
        }
        let n = self.state.get(&req).map_or(0, |s| s.step.len());
        if n == 0 || hidden.len() != n * self.dim {
            return Err(Error::Engine(format!("request {req}: {} values for a step of {n} positions", hidden.len())));
        }
        self.sample(hidden, sample, tokens);
        Ok(())
    }

    fn expert_weights(&self, layer: u32, expert: u32, out: &mut Vec<u8>) -> Result<(), Error> {
        out.clear();
        for w in expert_weights(layer, expert, self.dim) {
            out.extend_from_slice(&w.to_le_bytes());
        }
        Ok(())
    }

    fn own_experts(&mut self, experts: &[(u32, u32)]) -> Result<(), Error> {
        if self.moe.is_none() {
            return no_experts();
        }
        let st = &mut self.store;
        st.owned = experts.iter().copied().collect();
        st.resident.clear();
        // Loaded once at start, as many as fit.
        for &(layer, expert) in experts.iter().take(st.capacity.unwrap_or(usize::MAX)) {
            st.resident.insert((layer, expert), (expert_weights(layer, expert, self.dim), 0));
        }
        Ok(())
    }

    fn load_expert(&mut self, layer: u32, expert: u32, weights: &[u8]) -> Result<(), Error> {
        if self.moe.is_none() {
            return no_experts();
        }
        if weights.len() != self.dim * 4 {
            return Err(Error::Engine(format!("expert {expert} of layer {layer}: {} bytes of weights", weights.len())));
        }
        let w = weights.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let st = &mut self.store;
        st.owned.insert((layer, expert));
        st.resident.remove(&(layer, expert));
        st.make_room();
        st.resident.insert((layer, expert), (w, st.clock));
        st.moved_in += 1;
        Ok(())
    }

    fn release_expert(&mut self, layer: u32, expert: u32) {
        self.store.owned.remove(&(layer, expert));
        self.store.resident.remove(&(layer, expert));
    }

    fn run_experts(&mut self, req: u64, layer: u32, rows: &[f32], picks: &[Pick], out: &mut Vec<f32>) -> Result<(), Error> {
        if self.fail_on == Some(req) {
            return Err(Error::Engine(format!("mock failure on request {req}")));
        }
        let dim = self.dim;
        out.clear();
        for p in picks {
            let x = rows.get(p.row as usize * dim..(p.row as usize + 1) * dim);
            let x = x.ok_or_else(|| Error::Engine(format!("pick for row {} of {}", p.row, rows.len() / dim)))?;
            let w = self.resident(layer, p.expert as u32)?;
            run_expert(w, p.gate, x, out);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs one step and returns the head's tokens (empty on other stages).
    fn run(e: &mut MockEngine, start: u32, input: StageIn, sample: Sample, h: &mut Vec<f32>) -> Vec<u32> {
        let mut t = Vec::new();
        e.step(1, start, input, sample, h, &mut t).unwrap();
        t
    }

    #[test]
    fn split_ranges_compose_to_the_full_range() {
        let mut full = MockEngine::new(0..40, 40, 64);
        let mut parts = [MockEngine::new(0..20, 40, 64), MockEngine::new(20..30, 40, 64), MockEngine::new(30..40, 40, 64)];
        let (mut a, mut b, mut c) = (Vec::new(), Vec::new(), Vec::new());
        for pos in 0..5 {
            let t = [pos + 3];
            let want = run(&mut full, pos, StageIn::Tokens(&t), Sample::Last, &mut a);
            assert!(run(&mut parts[0], pos, StageIn::Tokens(&t), Sample::Last, &mut b).is_empty());
            assert!(run(&mut parts[1], pos, StageIn::Hidden(&b), Sample::Last, &mut c).is_empty());
            assert_eq!(run(&mut parts[2], pos, StageIn::Hidden(&c), Sample::Last, &mut b), want);
            assert_eq!(want.len(), 1);
        }
    }

    #[test]
    fn a_chunk_matches_its_positions_one_by_one() {
        let prompt = [5, 9, 2, 7, 7, 1, 3];
        let mut one = MockEngine::new(0..4, 4, 16);
        let mut h = Vec::new();
        let mut each = Vec::new();
        for (pos, t) in prompt.iter().enumerate() {
            each.extend(run(&mut one, pos as u32, StageIn::Tokens(std::slice::from_ref(t)), Sample::Last, &mut h));
        }
        let mut chunked = MockEngine::new(0..4, 4, 16);
        run(&mut chunked, 0, StageIn::Tokens(&prompt[..3]), Sample::Last, &mut h);
        assert_eq!(h.len(), 3 * 16);
        assert_eq!(run(&mut chunked, 3, StageIn::Tokens(&prompt[3..]), Sample::Last, &mut h), each[6..]);
        // Sampling every position gives each position's token.
        let mut every = MockEngine::new(0..4, 4, 16);
        assert_eq!(run(&mut every, 0, StageIn::Tokens(&prompt), Sample::Every, &mut h), each);
        // Decoding carries on from the same state either way.
        let next = |e: &mut MockEngine, h: &mut Vec<f32>| run(e, 7, StageIn::Tokens(&[4]), Sample::Last, h);
        assert_eq!(next(&mut one, &mut h), next(&mut chunked, &mut h));
    }

    #[test]
    fn rewind_drops_positions_of_the_last_step() {
        let mut h = Vec::new();
        let mut want = MockEngine::new(0..2, 2, 8);
        run(&mut want, 0, StageIn::Tokens(&[1, 2, 3]), Sample::Last, &mut h);
        let after = run(&mut want, 3, StageIn::Tokens(&[9]), Sample::Last, &mut h);

        let mut e = MockEngine::new(0..2, 2, 8);
        run(&mut e, 0, StageIn::Tokens(&[1]), Sample::Last, &mut h);
        // Two of four drafts accepted: keep positions 0..3.
        run(&mut e, 1, StageIn::Tokens(&[2, 3, 5, 6]), Sample::Every, &mut h);
        e.rewind(1, 3).unwrap();
        assert_eq!(run(&mut e, 3, StageIn::Tokens(&[9]), Sample::Last, &mut h), after);
        // Only the last step's positions can go.
        assert!(e.rewind(1, 1).is_err());
    }

    #[test]
    fn rejects_out_of_order_positions_until_freed() {
        let mut e = MockEngine::new(0..2, 2, 8);
        let (mut h, mut t) = (Vec::new(), Vec::new());
        e.step(7, 0, StageIn::Tokens(&[1, 2]), Sample::Last, &mut h, &mut t).unwrap();
        assert!(e.step(7, 1, StageIn::Tokens(&[1]), Sample::Last, &mut h, &mut t).is_err());
        e.free(7);
        e.step(7, 0, StageIn::Tokens(&[1]), Sample::Last, &mut h, &mut t).unwrap();
    }

    const E: u32 = 16;

    fn moe(dim: usize) -> MockEngine {
        MockEngine::new(0..4, 4, dim).with_experts(E, 4)
    }

    /// One step the way the controller and two workers run it: expert `e` lives on worker
    /// `e % 2`.
    fn split_step(c: &mut MockEngine, w: &mut [MockEngine; 2], start: u32, ids: &[u32], sample: Sample) -> Vec<u32> {
        let (mut h, mut x, mut picks, mut outputs, mut out, mut t) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        c.embed(1, start, ids, sample, &mut h).unwrap();
        for layer in 0..4 {
            c.route(1, layer, &mut h, &mut x, &mut picks).unwrap();
            assert_eq!(picks.len(), 4 * ids.len());
            outputs.clear();
            for p in &picks {
                let worker = &mut w[p.expert as usize % 2];
                worker.run_experts(1, layer, &x, std::slice::from_ref(p), &mut out).unwrap();
                outputs.extend_from_slice(&out);
            }
            c.combine(layer, &picks, &outputs, &mut h);
        }
        c.head(1, &h, sample, &mut t).unwrap();
        t
    }

    fn workers(dim: usize) -> [MockEngine; 2] {
        let mut w = [moe(dim), moe(dim)];
        for (i, w) in w.iter_mut().enumerate() {
            let own: Vec<_> = (0..4).flat_map(|l| (0..E).filter(move |e| *e as usize % 2 == i).map(move |e| (l, e))).collect();
            w.own_experts(&own).unwrap();
        }
        w
    }

    #[test]
    fn expert_split_matches_one_engine_exactly() {
        let prompt = [5, 9, 2, 7, 7, 1, 3];
        let mut one = moe(16);
        let mut h = Vec::new();
        let mut want = Vec::new();
        for (pos, t) in prompt.iter().enumerate() {
            want.extend(run(&mut one, pos as u32, StageIn::Tokens(std::slice::from_ref(t)), Sample::Last, &mut h));
        }
        let (mut c, mut w) = (moe(16), workers(16));
        // A chunk, then single positions.
        let mut got = split_step(&mut c, &mut w, 0, &prompt[..3], Sample::Every);
        for pos in 3..7 {
            got.extend(split_step(&mut c, &mut w, pos, &prompt[pos as usize..][..1], Sample::Last));
        }
        assert_eq!(got, want);
        // Workers hold no request state.
        assert_eq!((w[0].live_requests(), w[1].live_requests()), (0, 0));
        // Without experts the layers differ.
        let mut plain = MockEngine::new(0..4, 4, 16);
        assert_ne!(run(&mut plain, 0, StageIn::Tokens(&prompt), Sample::Every, &mut h), want);
    }

    #[test]
    fn hot_set_and_moved_experts_give_the_same_outputs() {
        let ids = [4, 8, 15, 16, 23, 42];
        let (mut c, mut w) = (moe(8), workers(8));
        let want = split_step(&mut c, &mut w, 0, &ids, Sample::Every);
        // Two experts in memory per worker: most picks load one, evicting the coldest.
        let (mut c, mut small) = (moe(8), workers(8));
        for w in &mut small {
            *w = std::mem::replace(w, moe(8)).with_expert_capacity(2);
            let own: Vec<_> = w.store.owned.iter().copied().collect();
            w.own_experts(&own).unwrap();
        }
        assert_eq!(split_step(&mut c, &mut small, 0, &ids, Sample::Every), want);
        assert!(small[0].expert_loads().0 > 0 && small[0].store.resident.len() <= 2);
        // Expert 3 of layer 0 moves from worker 1 to worker 0 with its weights.
        let (c0, mut bytes) = (moe(8), Vec::new());
        c0.expert_weights(0, 3, &mut bytes).unwrap();
        w[0].load_expert(0, 3, &bytes).unwrap();
        w[1].release_expert(0, 3);
        assert_eq!((w[0].held_experts(), w[1].held_experts()), (33, 31));
        let rows = [0.5; 8];
        let pick = [Pick { row: 0, expert: 3, gate: 0.5 }];
        let (mut a, mut b) = (Vec::new(), Vec::new());
        assert!(w[1].run_experts(1, 0, &rows, &pick, &mut a).is_err(), "released");
        w[0].run_experts(1, 0, &rows, &pick, &mut a).unwrap();
        let mut fresh = workers(8);
        fresh[1].run_experts(1, 0, &rows, &pick, &mut b).unwrap();
        assert_eq!(a, b);
        assert!(w[0].load_expert(0, 3, &bytes[1..]).is_err());
    }

    #[test]
    fn expert_split_rewinds_on_the_controller() {
        let mut h = Vec::new();
        let mut want = moe(8);
        run(&mut want, 0, StageIn::Tokens(&[1, 2, 3]), Sample::Last, &mut h);
        let after = run(&mut want, 3, StageIn::Tokens(&[9]), Sample::Last, &mut h);
        let (mut c, mut w) = (moe(8), workers(8));
        split_step(&mut c, &mut w, 0, &[1], Sample::Last);
        split_step(&mut c, &mut w, 1, &[2, 3, 5, 6], Sample::Every);
        c.rewind(1, 3).unwrap();
        assert_eq!(split_step(&mut c, &mut w, 3, &[9], Sample::Last), after);
    }

    #[test]
    fn engines_without_experts_refuse_the_expert_split() {
        let mut e = MockEngine::new(0..2, 2, 8);
        assert_eq!(e.moe_shape(), None);
        assert!(e.embed(1, 0, &[1], Sample::Last, &mut Vec::new()).is_err());
        assert!(e.own_experts(&[(0, 0)]).is_err());
    }

    #[test]
    fn rejects_tokens_on_a_later_stage_and_ragged_hidden() {
        let mut e = MockEngine::new(2..4, 4, 8);
        let (mut h, mut t) = (Vec::new(), Vec::new());
        assert!(e.step(1, 0, StageIn::Tokens(&[1]), Sample::Last, &mut h, &mut t).is_err());
        assert!(e.step(1, 0, StageIn::Hidden(&[0.0; 12]), Sample::Last, &mut h, &mut t).is_err());
    }
}
