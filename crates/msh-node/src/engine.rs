//! What a node needs from streammoe (DEP-1, DEP-2, DEP-3), as a trait shaped like streammoe's
//! `eval_request`: a stage takes a token (first node) or a hidden state, and produces a hidden
//! state for the next node or, on the last node, the sampled token.
//!
//! `MockEngine` keeps per-request state that depends on every earlier position, like a
//! KV/DeltaNet cache, so ordering and freeing bugs show up as wrong tokens or errors.

use std::collections::HashMap;
use std::ops::Range;

use crate::Error;

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
}

pub struct MockEngine {
    layers: Range<u32>,
    head: bool,
    dim: usize,
    vocab: u32,
    state: HashMap<u64, MockState>,
    fail_on: Option<u64>,
}

struct MockState {
    next_pos: u32,
    /// One state vector per layer.
    layers: Vec<f32>,
    /// The state before each position of the last step, for `rewind`.
    before: Vec<(u32, Vec<f32>)>,
}

impl MockEngine {
    /// A stage running `layers` of a `num_layers` model; the last stage holds the head.
    pub fn new(layers: Range<u32>, num_layers: u32, dim: usize) -> Self {
        let head = layers.end == num_layers;
        MockEngine { layers, head, dim, vocab: 151_936, state: HashMap::new(), fail_on: None }
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
            StageIn::Tokens(ids) => {
                for &token in ids {
                    out.extend((0..dim).map(|i| ((token as f32 + 1.0) * (i as f32 + 1.0) * 1e-3).sin()));
                }
            }
            StageIn::Hidden(src) if n == 0 || src.len() != n * dim => {
                return Err(Error::Engine(format!("hidden input of {} values, not whole rows", src.len())))
            }
            StageIn::Hidden(src) => out.extend_from_slice(src),
        }
        if n == 0 {
            return Err(Error::Engine("no positions".into()));
        }
        let state_len = self.layers.len() * dim;
        let st = self.state.entry(req).or_insert_with(|| MockState {
            next_pos: 0,
            layers: vec![0.0; state_len],
            before: Vec::new(),
        });
        if start_pos != st.next_pos {
            return Err(Error::Engine(format!(
                "request {req}: position {start_pos} out of order, expected {}",
                st.next_pos
            )));
        }
        st.before.clear();
        // Position by position, so a chunk gives exactly what single positions would.
        for (p, h) in out.chunks_exact_mut(dim).enumerate() {
            if sample == Sample::Every {
                st.before.push((start_pos + p as u32, st.layers.clone()));
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
        st.next_pos += n as u32;
        if !self.head {
            return Ok(());
        }
        let rows = match sample {
            Sample::Last => &out[(n - 1) * dim..],
            Sample::Every => &out[..],
        };
        // Bit-sensitive on purpose: any difference in the hidden state changes the token.
        for h in rows.chunks_exact(dim) {
            let s: f32 = h.iter().enumerate().map(|(i, v)| v * (i % 7 + 1) as f32).sum();
            tokens.push(((s.to_bits() as u64).wrapping_mul(2_654_435_761) % self.vocab as u64) as u32);
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

    #[test]
    fn rejects_tokens_on_a_later_stage_and_ragged_hidden() {
        let mut e = MockEngine::new(2..4, 4, 8);
        let (mut h, mut t) = (Vec::new(), Vec::new());
        assert!(e.step(1, 0, StageIn::Tokens(&[1]), Sample::Last, &mut h, &mut t).is_err());
        assert!(e.step(1, 0, StageIn::Hidden(&[0.0; 12]), Sample::Last, &mut h, &mut t).is_err());
    }
}
