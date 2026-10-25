//! What a node needs from moeflux (DEP-1, DEP-2, DEP-3), as a trait shaped like moeflux's
//! `eval_request`: a stage takes a token (first node) or a hidden state, and produces a hidden
//! state for the next node or, on the last node, the sampled token.
//!
//! `MockEngine` keeps per-request state that depends on every earlier position, like a
//! KV/DeltaNet cache, so ordering and freeing bugs show up as wrong tokens or errors.

use std::collections::HashMap;
use std::ops::Range;

use crate::Error;

/// Input to one position of a node's stage.
pub enum StageIn<'a> {
    /// Token id; only the first node (it holds the embedding).
    Token(u32),
    /// The previous node's output, `hidden_dim` f32s.
    Hidden(&'a [f32]),
}

pub trait Engine: Send {
    fn hidden_dim(&self) -> usize;
    /// End-of-sequence token: generation stops there, and it is not reported as output.
    fn eos(&self) -> Option<u32> {
        None
    }
    /// Runs this node's layer range for one position of request `req`. On the node holding the
    /// head, returns the greedy token; elsewhere writes the next node's input to `hidden_out`
    /// and returns `None`. A request's positions must arrive in order from 0.
    fn step(
        &mut self,
        req: u64,
        pos: u32,
        input: StageIn<'_>,
        hidden_out: &mut [f32],
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
        pos: u32,
        input: StageIn<'_>,
        h: &mut [f32],
    ) -> Result<Option<u32>, Error> {
        if self.fail_on == Some(req) {
            return Err(Error::Engine(format!("mock failure on request {req}")));
        }
        match input {
            StageIn::Token(_) if self.layers.start != 0 => {
                return Err(Error::Engine("token input on a stage without the embedding".into()))
            }
            StageIn::Token(token) => {
                for (i, v) in h.iter_mut().enumerate() {
                    *v = ((token as f32 + 1.0) * (i as f32 + 1.0) * 1e-3).sin();
                }
            }
            StageIn::Hidden(src) => h.copy_from_slice(src),
        }
        let n = self.layers.len() * self.dim;
        let (next_pos, st) = self.state.entry(req).or_insert_with(|| (0, vec![0.0; n]));
        if pos != *next_pos {
            return Err(Error::Engine(format!(
                "request {req}: position {pos} out of order, expected {next_pos}"
            )));
        }
        *next_pos += 1;
        for (k, layer) in self.layers.clone().enumerate() {
            let s = &mut st[k * self.dim..(k + 1) * self.dim];
            for i in 0..self.dim {
                let x = (h[i] * Self::weight(layer, i) + s[i]).tanh();
                s[i] = 0.9 * s[i] + 0.1 * x;
                h[i] = x;
            }
        }
        if !self.head {
            return Ok(None);
        }
        // Bit-sensitive on purpose: any difference in the hidden state changes the token.
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
        let (mut a, mut b, mut c) = (vec![0.0; 64], vec![0.0; 64], vec![0.0; 64]);
        for pos in 0..5 {
            let want = full.step(1, pos, StageIn::Token(pos + 3), &mut a).unwrap();
            assert_eq!(parts[0].step(1, pos, StageIn::Token(pos + 3), &mut b).unwrap(), None);
            assert_eq!(parts[1].step(1, pos, StageIn::Hidden(&b), &mut c).unwrap(), None);
            assert_eq!(parts[2].step(1, pos, StageIn::Hidden(&c), &mut b).unwrap(), want);
            assert!(want.is_some());
        }
    }

    #[test]
    fn rejects_out_of_order_positions_until_freed() {
        let mut e = MockEngine::new(0..2, 2, 8);
        let mut h = vec![0.0; 8];
        e.step(7, 0, StageIn::Token(1), &mut h).unwrap();
        assert!(e.step(7, 2, StageIn::Token(1), &mut h).is_err());
        e.free(7);
        e.step(7, 0, StageIn::Token(1), &mut h).unwrap();
    }

    #[test]
    fn rejects_tokens_on_a_later_stage() {
        let mut e = MockEngine::new(2..4, 4, 8);
        assert!(e.step(1, 0, StageIn::Token(1), &mut vec![0.0; 8]).is_err());
    }
}
