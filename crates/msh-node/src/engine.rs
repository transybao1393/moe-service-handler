//! What a node needs from moeflux (DEP-1, DEP-2, DEP-3), as a trait.
//!
//! `MockEngine` stands in until moeflux's layer-range execution lands. It keeps per-request
//! state that depends on every earlier position, like a KV/DeltaNet cache, so ordering and
//! freeing bugs show up as wrong tokens or errors.

use std::collections::HashMap;
use std::ops::Range;

use crate::Error;

pub trait Engine: Send {
    fn hidden_dim(&self) -> usize;
    /// Embedding lookup. Called only on the first node.
    fn embed(&mut self, token: u32, out: &mut [f32]) -> Result<(), Error>;
    /// Runs this node's layer range in place for one position of one request.
    fn forward(&mut self, req: u64, pos: u32, hidden: &mut [f32]) -> Result<(), Error>;
    /// Final norm, LM head and greedy sampling. Called only on the last node.
    fn head(&mut self, hidden: &[f32]) -> Result<u32, Error>;
    /// Frees all state held for `req`. Freeing an unknown request is a no-op.
    fn free(&mut self, req: u64);
}

pub struct MockEngine {
    layers: Range<u32>,
    dim: usize,
    vocab: u32,
    /// Per request: next expected position, and one state vector per layer.
    state: HashMap<u64, (u32, Vec<f32>)>,
}

impl MockEngine {
    pub fn new(layers: Range<u32>, dim: usize) -> Self {
        MockEngine { layers, dim, vocab: 151_936, state: HashMap::new() }
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

    fn embed(&mut self, token: u32, out: &mut [f32]) -> Result<(), Error> {
        for (i, v) in out.iter_mut().enumerate() {
            *v = ((token as f32 + 1.0) * (i as f32 + 1.0) * 1e-3).sin();
        }
        Ok(())
    }

    fn forward(&mut self, req: u64, pos: u32, h: &mut [f32]) -> Result<(), Error> {
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
        Ok(())
    }

    fn head(&mut self, h: &[f32]) -> Result<u32, Error> {
        // Bit-sensitive on purpose: any difference in the hidden state changes the token.
        let s: f32 = h.iter().enumerate().map(|(i, v)| v * (i % 7 + 1) as f32).sum();
        Ok(((s.to_bits() as u64).wrapping_mul(2_654_435_761) % self.vocab as u64) as u32)
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
        let mut full = MockEngine::new(0..40, 64);
        let mut parts = [MockEngine::new(0..20, 64), MockEngine::new(20..30, 64), MockEngine::new(30..40, 64)];
        for pos in 0..5 {
            let mut a = vec![0.0; 64];
            full.embed(pos + 3, &mut a).unwrap();
            let mut b = a.clone();
            full.forward(1, pos, &mut a).unwrap();
            for p in &mut parts {
                p.forward(1, pos, &mut b).unwrap();
            }
            assert_eq!(full.head(&a).unwrap(), parts[2].head(&b).unwrap());
        }
    }

    #[test]
    fn rejects_out_of_order_positions_until_freed() {
        let mut e = MockEngine::new(0..2, 8);
        let mut h = vec![0.0; 8];
        e.forward(7, 0, &mut h).unwrap();
        assert!(e.forward(7, 2, &mut h).is_err());
        e.free(7);
        e.forward(7, 0, &mut h).unwrap();
    }
}
