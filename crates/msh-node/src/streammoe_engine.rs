//! The real engine: a streammoe context opened on this node's layer range (DEP-1 to DEP-4).
//!
//! Each request gets a slot in streammoe's request pool, admitted at its first position and
//! released on `free`. The pool is sized from the scheduler's `max_in_flight`; the coordinator
//! admits a new request only after the previous one's `Drop` has gone through the workers, so no
//! node's pool overflows. Sampling is greedy on the node holding the head.
//!
//! A prompt chunk goes to streammoe as one call, which takes its batched prefill path.

use std::collections::HashSet;
use std::path::Path;

use streammoe::{Ctx, OpenConfig, ReqId, StageInput, StageOutput};

use crate::config::{KvFormat, PrefetchGuess};
use crate::engine::{Engine, Sample, StageIn};
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
    /// Expert files, relative to `model_dir`.
    pub experts_dir: &'a str,
    pub prefetch_guess: Option<PrefetchGuess>,
    pub prefetch_experts: Option<usize>,
    pub fast_experts_dir: Option<&'a Path>,
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
            experts_dir: "root",
            prefetch_guess: None,
            prefetch_experts: None,
            fast_experts_dir: None,
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

/// streammoe's settings for a node's context.
fn open_config(spec: &OpenSpec) -> OpenConfig {
    let defaults = OpenConfig::default();
    let chunk = spec.prefill_chunk.max(1);
    OpenConfig {
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
        prefetch_guess: match spec.prefetch_guess {
            None => defaults.prefetch_guess,
            Some(PrefetchGuess::Prev) => streammoe::PrefetchGuess::PrevToken,
            Some(PrefetchGuess::Cross) => streammoe::PrefetchGuess::CrossLayer,
        },
        prefetch_experts: spec.prefetch_experts.or(defaults.prefetch_experts),
        fast_experts_dir: spec.fast_experts_dir.map(Path::to_path_buf),
        ..defaults
    }
}

/// A node's memory for its layer range under a cap, from streammoe's own budget plan, without
/// opening the model.
#[derive(Clone, Copy, Debug)]
pub struct RangeBudget {
    /// Everything but the expert hot tier: weights, KV, expert slots, prefill buffers,
    /// runtime overhead, frame buffers.
    pub fixed: u64,
    /// Expert hot tier the rest of the cap holds, at most every expert of the range.
    pub hot_tier: u64,
    /// Every expert of the range (4-bit size).
    pub range_experts: u64,
}

/// Plans layer ranges against a cap as streammoe's open does, for the planner. Experts count
/// at their 4-bit size, so packed 3-bit experts make this an overestimate. The open itself is
/// the exact check (DEP-4).
pub struct Budgeter {
    cfg: OpenConfig,
    /// Non-expert weights of each single-layer range; the first includes the embedding and the
    /// last the head, so a range's weights are the sum over its layers.
    layer_weights: Vec<u64>,
}

impl Budgeter {
    /// Reads the weight manifest once. `spec.layers` and `spec.mem_cap` are not used.
    pub fn new(spec: &OpenSpec) -> Result<Self, Error> {
        use streammoe::riir::io::weight_file::filtered_bytes;
        use streammoe::riir::range::keeps_tensor;
        let art = spec.model_dir.join("artifacts");
        let (bin, manifest) = (art.join("model_weights.bin"), art.join("model_weights.json"));
        let layer_weights = (0..streammoe::riir::VARIANT.num_layers)
            .map(|l| {
                let keep = |name: &str| keeps_tensor(&(l..l + 1), name);
                filtered_bytes(&bin, &manifest, Some(&keep))
                    .map_err(|e| Error::Engine(format!("weights of layer {l}: {e}")))
            })
            .collect::<Result<_, _>>()?;
        Ok(Budgeter { cfg: open_config(spec), layer_weights })
    }

    pub fn budget(&self, layers: std::ops::Range<u32>, cap: u64) -> RangeBudget {
        use streammoe::riir::budget::{kv_bytes_per_token_fmt, kv_scratch_bytes, Account, BudgetPlan};
        let v = streammoe::riir::VARIANT;
        let cfg = &self.cfg;
        let range = layers.start as usize..layers.end as usize;
        let weights = self.layer_weights[range.clone()].iter().sum();
        let mut plan = BudgetPlan::for_range(
            cfg.n_ctx,
            streammoe::ExpertIoMode::Pread,
            weights,
            cfg.runtime_overhead,
            &range,
            cfg.max_requests,
            cfg.resolved_prefill_chunk(),
        );
        plan.set(Account::HostBuffers, cfg.host_buffers);
        if cfg.kv_format != streammoe::KvFormat::F32 {
            let kv = kv_bytes_per_token_fmt(v, &range, cfg.kv_format);
            plan.set(Account::Kv, cfg.max_requests as u64 * kv * cfg.n_ctx as u64);
            plan.set(Account::KvScratch, kv_scratch_bytes(v, &range, cfg.n_ctx, cfg.kv_format));
        }
        let range_experts = range.len() as u64 * v.num_experts as u64 * v.expert_size_4bit() as u64;
        let fixed = plan.total();
        RangeBudget { fixed, hot_tier: cap.saturating_sub(fixed).min(range_experts), range_experts }
    }
}

impl StreamMoeEngine {
    pub fn open(spec: &OpenSpec) -> Result<Self, Error> {
        let art = spec.model_dir.join("artifacts");
        let cfg = open_config(spec);
        let ctx = Ctx::open_with(
            &art.join("model_weights.bin"),
            &art.join("model_weights.json"),
            &art.join("vocab.bin"),
            &spec.model_dir.join(spec.experts_dir),
            /* use_2bit = */ false,
            &cfg,
        )
        .map_err(|e| Error::Engine(format!("open {}: {e}", spec.model_dir.display())))?;
        let logits = vec![0.0; ctx.n_vocab()];
        Ok(StreamMoeEngine { ctx, admitted: HashSet::new(), logits, tokens: Vec::new() })
    }

    /// streammoe's expert read counters for this node: disk reads by kind, prefetch hits,
    /// time waiting on I/O.
    pub fn expert_io_stats(&self) -> streammoe::ExpertIoStats {
        self.ctx.expert_io_stats()
    }
}

impl Engine for StreamMoeEngine {
    fn hidden_dim(&self) -> usize {
        streammoe::riir::VARIANT.hidden_dim
    }

    fn eos(&self) -> Option<u32> {
        Some(self.ctx.eos() as u32)
    }

    fn has_head(&self) -> bool {
        self.ctx.has_head()
    }

    fn step(
        &mut self,
        req: u64,
        start_pos: u32,
        input: StageIn<'_>,
        sample: Sample,
        hidden_out: &mut Vec<f32>,
        tokens: &mut Vec<u32>,
    ) -> Result<(), Error> {
        let engine_err = |e: streammoe::Error| Error::Engine(format!("request {req} position {start_pos}: {e}"));
        let dim = self.hidden_dim();
        let n = input.len(dim);
        if n == 0 {
            return Err(Error::Engine(format!("request {req}: no positions")));
        }
        let head = self.ctx.has_head();
        if head && sample == Sample::Every && n > 1 {
            // On a layer range streammoe returns the last position's logits only.
            return Err(Error::Engine(format!("request {req}: streammoe cannot sample every position")));
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
        let out = if head {
            StageOutput::Logits(&mut self.logits)
        } else {
            hidden_out.resize(n * dim, 0.0);
            StageOutput::Hidden(hidden_out)
        };
        self.ctx.eval_request(ReqId(req), input, start_pos as usize, out).map_err(engine_err)?;
        tokens.clear();
        if head {
            tokens.push(argmax(&self.logits));
        }
        Ok(())
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
