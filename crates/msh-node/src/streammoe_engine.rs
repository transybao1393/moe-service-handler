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

use crate::config::{expand_home, Config, KvFormat, PrefetchGuess};
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

/// Calls `f` with the open settings for `layers` on this config. Every split opens and plans
/// through here, so the `[engine]` table reaches streammoe the same way on each.
pub fn with_spec<T>(
    cfg: &Config,
    layers: std::ops::Range<u32>,
    mem_cap: Option<u64>,
    f: impl FnOnce(&OpenSpec) -> T,
) -> T {
    let model_dir = expand_home(&cfg.engine.model_dir);
    let fast_experts_dir = cfg.engine.fast_experts_dir.as_deref().map(expand_home);
    f(&OpenSpec {
        model_dir: &model_dir,
        layers,
        n_ctx: cfg.engine.n_ctx,
        mem_cap,
        max_requests: cfg.scheduler.engine_slots(),
        prefill_chunk: cfg.scheduler.prefill_chunk,
        kv_format: cfg.engine.kv_format,
        attn_threads: cfg.engine.attn_threads,
        runtime_overhead: cfg.engine.runtime_overhead(),
        experts_dir: &cfg.engine.experts_dir,
        prefetch_guess: cfg.engine.prefetch_guess,
        prefetch_experts: cfg.engine.prefetch_experts,
        fast_experts_dir: fast_experts_dir.as_deref(),
    })
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
    /// Part of `fixed`: the KV cache of every request slot at `n_ctx`.
    pub kv: u64,
    /// Part of `fixed`: the f32 prefill scratch a quantized KV cache needs (0 for f32).
    pub kv_scratch: u64,
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
        let kv = cfg.max_requests as u64 * kv_bytes_per_token_fmt(v, &range, cfg.kv_format) * cfg.n_ctx as u64;
        let mut kv_scratch = 0;
        if cfg.kv_format != streammoe::KvFormat::F32 {
            kv_scratch = kv_scratch_bytes(v, &range, cfg.n_ctx, cfg.kv_format);
            plan.set(Account::Kv, kv);
            plan.set(Account::KvScratch, kv_scratch);
        }
        let range_experts = range.len() as u64 * v.num_experts as u64 * v.expert_size_4bit() as u64;
        let fixed = plan.total();
        let hot_tier = cap.saturating_sub(fixed).min(range_experts);
        RangeBudget { fixed, hot_tier, range_experts, kv, kv_scratch }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Split;

    const NODES: &str = r#"
        [[nodes]]
        name = "a"
        addr = "127.0.0.1:1"
        LAYERS_A
        [[nodes]]
        name = "b"
        addr = "127.0.0.1:2"
        LAYERS_B
    "#;

    fn config(split: &str, kv: &str, layers: [&str; 2]) -> Config {
        let text = format!(
            "hidden_dim = 2048\nnum_layers = 16\nsplit = \"{split}\"\n[engine]\nkind = \"streammoe\"\nmodel_dir = \"/m\"\nkv_format = \"{kv}\"\n{}",
            NODES.replace("LAYERS_A", layers[0]).replace("LAYERS_B", layers[1])
        );
        Config::parse(&text).unwrap()
    }

    #[test]
    fn kv_format_reaches_streammoe_on_every_split_that_opens_it() {
        let formats = [
            ("f32", streammoe::KvFormat::F32),
            ("f16", streammoe::KvFormat::F16),
            ("turbo4", streammoe::KvFormat::Turbo4),
            ("turbo3", streammoe::KvFormat::Turbo3),
        ];
        for (name, want) in formats {
            let layers = config("layers", name, ["layers = [0, 8]", "layers = [8, 16]"]);
            let requests = config("requests", name, ["", ""]);
            assert_eq!(requests.split, Split::Requests);
            for cfg in [&layers, &requests] {
                // Every node that opens an engine: both on the layer split, the workers on the
                // request split (its controller opens none).
                let opened: Vec<_> = (0..cfg.nodes.len()).filter_map(|i| cfg.layers_of(i)).collect();
                assert!(!opened.is_empty());
                for r in opened {
                    let got = with_spec(cfg, r, Some(3 << 30), |s| (s.kv_format, open_config(s).kv_format));
                    assert!(got.0 == cfg.engine.kv_format && got.1 == want, "{name} on {:?}", cfg.split);
                }
            }
        }
    }
}
