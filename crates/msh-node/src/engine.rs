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

pub trait Engine: Send {
    fn hidden_dim(&self) -> usize;
    /// End-of-sequence token: generation stops there, and it is not reported as output.
    fn eos(&self) -> Option<u32> {
        None
    }
    /// Runs this node's layer range for positions `start_pos..` of request `req`. On the node
    /// holding the head, returns the greedy token after the last position; elsewhere resizes
    /// `hidden_out` to the next node's input, one row per position, and returns `None`. A
    /// request's positions must arrive in order from 0.
    fn step(
        &mut self,
        req: u64,
        start_pos: u32,
        input: StageIn<'_>,
        hidden_out: &mut Vec<f32>,
    ) -> Result<Option<u32>, Error>;
    /// Frees all state held for `req`. Freeing an unknown request is a no-op.
    fn free(&mut self, req: u64);
}

pub struct MockEngine {
    layers: Range<u32>,
    head: bool,
    dim: usize,
    vocab: u32,
    /// Per request: next expected position, and one state vector per layer.
    state: HashMap<u64, (u32, Vec<f32>)>,
    fail_on: Option<u64>,
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

    fn step(
        &mut self,
        req: u64,
        start_pos: u32,
        input: StageIn<'_>,
        out: &mut Vec<f32>,
    ) -> Result<Option<u32>, Error> {
        if self.fail_on == Some(req) {
            return Err(Error::Engine(format!("mock failure on request {req}")));
        }
        let dim = self.dim;
        let n = input.len(dim);
        out.clear();
        match input {
            StageIn::Tokens(_) if self.layers.start != 0 => {
                return Err(Error::Engine("token input on a stage without the embedding".into()))
            }
            StageIn::Tokens(tokens) => {
                for &token in tokens {
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
        let (next_pos, st) = self.state.entry(req).or_insert_with(|| (0, vec![0.0; state_len]));
        if start_pos != *next_pos {
            return Err(Error::Engine(format!(
                "request {req}: position {start_pos} out of order, expected {next_pos}"
            )));
        }
        *next_pos += n as u32;
        // Position by position, so a chunk gives exactly what single positions would.
        for h in out.chunks_exact_mut(dim) {
            for (k, layer) in self.layers.clone().enumerate() {
                let s = &mut st[k * dim..(k + 1) * dim];
                for i in 0..dim {
                    let x = (h[i] * Self::weight(layer, i) + s[i]).tanh();
                    s[i] = 0.9 * s[i] + 0.1 * x;
                    h[i] = x;
                }
            }
        }
        if !self.head {
            return Ok(None);
        }
        // Bit-sensitive on purpose: any difference in the hidden state changes the token.
        let h = &out[(n - 1) * dim..];
        let s: f32 = h.iter().enumerate().map(|(i, v)| v * (i % 7 + 1) as f32).sum();
        Ok(Some(((s.to_bits() as u64).wrapping_mul(2_654_435_761) % self.vocab as u64) as u32))
    }

    fn free(&mut self, req: u64) {
        self.state.remove(&req);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_ranges_compose_to_the_full_range() {
        let mut full = MockEngine::new(0..40, 40, 64);
        let mut parts = [MockEngine::new(0..20, 40, 64), MockEngine::new(20..30, 40, 64), MockEngine::new(30..40, 40, 64)];
        let (mut a, mut b, mut c) = (Vec::new(), Vec::new(), Vec::new());
        for pos in 0..5 {
            let t = [pos + 3];
            let want = full.step(1, pos, StageIn::Tokens(&t), &mut a).unwrap();
            assert_eq!(parts[0].step(1, pos, StageIn::Tokens(&t), &mut b).unwrap(), None);
            assert_eq!(parts[1].step(1, pos, StageIn::Hidden(&b), &mut c).unwrap(), None);
            assert_eq!(parts[2].step(1, pos, StageIn::Hidden(&c), &mut b).unwrap(), want);
            assert!(want.is_some());
        }
    }

    #[test]
    fn a_chunk_matches_its_positions_one_by_one() {
        let prompt = [5, 9, 2, 7, 7, 1, 3];
        let mut one = MockEngine::new(0..4, 4, 16);
        let mut h = Vec::new();
        let mut want = None;
        for (pos, t) in prompt.iter().enumerate() {
            want = one.step(1, pos as u32, StageIn::Tokens(std::slice::from_ref(t)), &mut h).unwrap();
        }
        let mut chunked = MockEngine::new(0..4, 4, 16);
        chunked.step(1, 0, StageIn::Tokens(&prompt[..3]), &mut h).unwrap();
        assert_eq!(h.len(), 3 * 16);
        assert_eq!(chunked.step(1, 3, StageIn::Tokens(&prompt[3..]), &mut h).unwrap(), want);
        // Decoding carries on from the same state either way.
        let next = |e: &mut MockEngine, h: &mut Vec<f32>| e.step(1, 7, StageIn::Tokens(&[4]), h).unwrap();
        assert_eq!(next(&mut one, &mut h), next(&mut chunked, &mut h));
    }

    #[test]
    fn rejects_out_of_order_positions_until_freed() {
        let mut e = MockEngine::new(0..2, 2, 8);
        let mut h = Vec::new();
        e.step(7, 0, StageIn::Tokens(&[1, 2]), &mut h).unwrap();
        assert!(e.step(7, 1, StageIn::Tokens(&[1]), &mut h).is_err());
        e.free(7);
        e.step(7, 0, StageIn::Tokens(&[1]), &mut h).unwrap();
    }

    #[test]
    fn rejects_tokens_on_a_later_stage_and_ragged_hidden() {
        let mut e = MockEngine::new(2..4, 4, 8);
        assert!(e.step(1, 0, StageIn::Tokens(&[1]), &mut Vec::new()).is_err());
        assert!(e.step(1, 0, StageIn::Hidden(&[0.0; 12]), &mut Vec::new()).is_err());
    }
}
