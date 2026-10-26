//! The real engine: a streammoe context opened on this node's layer range (DEP-1 to DEP-4).
//!
//! Each request gets a slot in streammoe's request pool, admitted at its first position and
//! released on `free`. The pool is sized from the scheduler's `max_in_flight`; the coordinator
//! admits a new request only after the previous one's `Drop` has gone round the ring, so no
//! node's pool overflows. Sampling is greedy on the node holding the head.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use streammoe::{Ctx, OpenConfig, ReqId, StageInput, StageOutput};

use crate::engine::{Engine, StageIn};
use crate::Error;

/// How to open one node's context.
pub struct OpenSpec<'a> {
    /// Holds `artifacts/{model_weights.bin,model_weights.json,vocab.bin}` and `root/`
    /// (packed experts), the layout streammoe's converters write.
    pub model_dir: &'a Path,
    pub layers: std::ops::Range<u32>,
    pub n_ctx: usize,
    pub mem_cap: Option<u64>,
    pub max_requests: usize,
}

pub struct StreamMoeEngine {
    ctx: Ctx,
    admitted: HashSet<u64>,
    logits: Vec<f32>,
}

impl StreamMoeEngine {
    pub fn open(spec: &OpenSpec) -> Result<Self, Error> {
        let art = spec.model_dir.join("artifacts");
        let cfg = OpenConfig {
            n_ctx: spec.n_ctx,
            mem_cap: spec.mem_cap,
            layer_range: Some(spec.layers.start as usize..spec.layers.end as usize),
            // Slot 0 is streammoe's always-admitted default request, which this engine never
            // uses, so N requests in flight need N + 1 slots (one idle slot of KV memory).
            max_requests: spec.max_requests + 1,
            ..OpenConfig::default()
        };
        let ctx = Ctx::open_with(
            &art.join("model_weights.bin"),
            &art.join("model_weights.json"),
            &art.join("vocab.bin"),
            &spec.model_dir.join("root"),
            /* use_2bit = */ false,
            &cfg,
        )
        .map_err(|e| Error::Engine(format!("open {}: {e}", spec.model_dir.display())))?;
        let logits = vec![0.0; ctx.n_vocab()];
        Ok(StreamMoeEngine { ctx, admitted: HashSet::new(), logits })
    }
}

impl Engine for StreamMoeEngine {
    fn hidden_dim(&self) -> usize {
        streammoe::riir::VARIANT.hidden_dim
    }

    fn eos(&self) -> Option<u32> {
        Some(self.ctx.eos() as u32)
    }

    fn step(
        &mut self,
        req: u64,
        pos: u32,
        input: StageIn<'_>,
        hidden_out: &mut [f32],
    ) -> Result<Option<u32>, Error> {
        let engine_err = |e: streammoe::Error| Error::Engine(format!("request {req} position {pos}: {e}"));
        if !self.admitted.contains(&req) {
            if pos != 0 {
                return Err(Error::Engine(format!("request {req}: first position {pos}, expected 0")));
            }
            self.ctx.admit_request(ReqId(req)).map_err(engine_err)?;
            self.admitted.insert(req);
        }
        let token;
        let input = match input {
            StageIn::Token(t) => {
                token = [t as i32];
                StageInput::Tokens(&token)
            }
            StageIn::Hidden(h) => StageInput::Hidden(h),
        };
        if !self.ctx.has_head() {
            let out = StageOutput::Hidden(hidden_out);
            self.ctx.eval_request(ReqId(req), input, pos as usize, out).map_err(engine_err)?;
            return Ok(None);
        }
        let out = StageOutput::Logits(&mut self.logits);
        self.ctx.eval_request(ReqId(req), input, pos as usize, out).map_err(engine_err)?;
        Ok(Some(argmax(&self.logits)))
    }

    fn free(&mut self, req: u64) {
        if self.admitted.remove(&req) {
            self.ctx.drop_request(ReqId(req));
        }
    }
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best as u32
}

/// Expands a leading `~/` to `$HOME`.
pub fn expand_home(p: &str) -> PathBuf {
    match (p.strip_prefix("~/"), std::env::var_os("HOME")) {
        (Some(rest), Some(home)) => PathBuf::from(home).join(rest),
        _ => PathBuf::from(p),
    }
}
