//! The real engine: a streammoe context opened on this node's layer range (DEP-1 to DEP-4).
//!
//! Each request gets a slot in streammoe's request pool, admitted at its first position and
//! released on `free`. The pool is sized from the scheduler's `max_in_flight`; the coordinator
//! admits a new request only after the previous one's `Drop` has gone round the ring, so no
//! node's pool overflows. Sampling is greedy on the node holding the head.
//!
//! A prompt chunk goes to streammoe as one call, which takes its batched prefill path.

use std::collections::HashSet;
use std::path::Path;

use streammoe::{Ctx, OpenConfig, ReqId, StageInput, StageOutput};

use crate::config::KvFormat;
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
    /// The cluster's prefill chunk: the most positions one step brings, which sizes
    /// streammoe's prefill scratch and this node's frame buffers.
    pub prefill_chunk: usize,
    pub kv_format: KvFormat,
    /// `None` keeps streammoe's default.
    pub attn_threads: Option<usize>,
    /// `None` keeps streammoe's default.
    pub runtime_overhead: Option<u64>,
}

impl<'a> OpenSpec<'a> {
    /// The whole of `layers` with every setting at its default.
    pub fn new(model_dir: &'a Path, layers: std::ops::Range<u32>) -> Self {
        OpenSpec {
            model_dir,
            layers,
            n_ctx: 2048,
            mem_cap: None,
            max_requests: 1,
            prefill_chunk: 64,
            kv_format: KvFormat::F32,
            attn_threads: None,
            runtime_overhead: None,
        }
    }
}

pub struct StreamMoeEngine {
    ctx: Ctx,
    admitted: HashSet<u64>,
    logits: Vec<f32>,
    /// Token ids as streammoe takes them.
    tokens: Vec<i32>,
}

impl StreamMoeEngine {
    pub fn open(spec: &OpenSpec) -> Result<Self, Error> {
        let art = spec.model_dir.join("artifacts");
        let defaults = OpenConfig::default();
        let chunk = spec.prefill_chunk.max(1);
        let cfg = OpenConfig {
            n_ctx: spec.n_ctx,
            mem_cap: spec.mem_cap,
            layer_range: Some(spec.layers.start as usize..spec.layers.end as usize),
            // Slot 0 is streammoe's always-admitted default request, which this engine never
            // uses, so N requests in flight need N + 1 slots (one idle slot of KV memory).
            max_requests: spec.max_requests + 1,
            // A pipeline stage never snapshots; under a cap each checkpoint would take a full
            // state's memory from the hot tier.
            max_checkpoints: 0,
            prefill_chunk: Some(chunk),
            // A worker's receive, input, output and send buffers each hold one chunk of f32
            // rows; the coordinator holds fewer.
            host_buffers: (4 * chunk * streammoe::riir::VARIANT.hidden_dim * 4) as u64,
            kv_format: match spec.kv_format {
                KvFormat::F32 => streammoe::KvFormat::F32,
                KvFormat::F16 => streammoe::KvFormat::F16,
                KvFormat::Turbo4 => streammoe::KvFormat::Turbo4,
                KvFormat::Turbo3 => streammoe::KvFormat::Turbo3,
            },
            attn_threads: spec.attn_threads.unwrap_or(defaults.attn_threads),
            runtime_overhead: spec.runtime_overhead.unwrap_or(defaults.runtime_overhead),
            ..defaults
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
        Ok(StreamMoeEngine { ctx, admitted: HashSet::new(), logits, tokens: Vec::new() })
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
        start_pos: u32,
        input: StageIn<'_>,
        hidden_out: &mut Vec<f32>,
    ) -> Result<Option<u32>, Error> {
        let engine_err = |e: streammoe::Error| Error::Engine(format!("request {req} position {start_pos}: {e}"));
        let dim = self.hidden_dim();
        let n = input.len(dim);
        if n == 0 {
            return Err(Error::Engine(format!("request {req}: no positions")));
        }
        if !self.admitted.contains(&req) {
            if start_pos != 0 {
                return Err(Error::Engine(format!("request {req}: first position {start_pos}, expected 0")));
            }
            self.ctx.admit_request(ReqId(req)).map_err(engine_err)?;
            self.admitted.insert(req);
        }
        let input = match input {
            StageIn::Tokens(t) => {
                self.tokens.clear();
                self.tokens.extend(t.iter().map(|&t| t as i32));
                StageInput::Tokens(&self.tokens)
            }
            StageIn::Hidden(h) => StageInput::Hidden(h),
        };
        let out = if self.ctx.has_head() {
            StageOutput::Logits(&mut self.logits)
        } else {
            hidden_out.resize(n * dim, 0.0);
            StageOutput::Hidden(hidden_out)
        };
        self.ctx.eval_request(ReqId(req), input, start_pos as usize, out).map_err(engine_err)?;
        Ok(self.ctx.has_head().then(|| argmax(&self.logits)))
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
