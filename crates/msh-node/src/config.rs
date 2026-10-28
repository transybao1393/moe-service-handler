//! Cluster config (CL-1, CL-4, CL-7). One TOML file, shared by every node:
//!
//! ```toml
//! hidden_dim = 2048
//! num_layers = 40
//!
//! [[nodes]]                # first node = controller (coordinator), runs the embedding
//! name = "m4"
//! addr = "192.168.1.10:7100"
//! layers = [0, 20]         # half-open layer range
//! prometheus = "0.0.0.0:9100"  # optional: GET /metrics on this node (CL-12)
//!
//! [[nodes]]                # a worker; the last one runs the head and returns the tokens
//! name = "m1a"
//! addr = "192.168.1.11:7100"
//! layers = [20, 40]
//! ```
//!
//! The controller connects to every worker's `addr`; workers never connect to each other. A
//! step goes to the first worker, back to the controller, on to the next worker, and so on.
//!
//! `layers` (a half-open range) goes on every node or on none. Without it, every node plans
//! the same split (CL-3) from each node's optional `speed` (default 1) and, under streammoe,
//! its memory cap. A single node runs the whole model. An optional `[scheduler]` table sets
//! `max_in_flight` (default 3), `max_queue` (64), `health_interval_ms` (1000),
//! `health_timeout_ms` (5000), `stall_timeout_ms` (30000; request split only),
//! `prefill_chunk` (64), `prefill_depth` (3), `max_draft` (0: no speculative decoding),
//! `draft_min_ngram` (3) and `prefix_cache` (0: no prefix reuse).
//!
//! `split` picks how the model is divided. `"layers"` (the default) is the layer split above.
//! `"experts"` is the expert split (docs/expert-parallel.md): the controller runs every layer's
//! attention, shared expert and router, the embedding and the head, and sends each layer's
//! hidden rows to every worker at once; each worker runs the routed experts it owns. No node
//! takes `layers`. For now only the mock engine can split by experts. An optional `[experts]`
//! table sets `count` (routed experts per layer, default 256), `top_k` (experts picked per
//! position, 8), `rebalance_every` (positions routed between rebalances, 1024; 0 never moves
//! experts), `max_moves` (experts moved per rebalance, 8) and `extra_experts` (how many more
//! than an even share a worker may come to own, 64):
//!
//! ```toml
//! split = "experts"
//! [experts]
//! count = 256
//! top_k = 8
//! ```
//!
//! `"requests"` is the request split (docs/request-split.md): the controller runs no engine and
//! holds no weights; it takes the HTTP requests, queues them and gives each whole request to
//! one worker, which runs every layer (streammoe's whole model inside its cap, mode A) and sends
//! the tokens back as it makes them. No node takes `layers`, and at least one worker is needed.
//! `max_in_flight` is then per worker, `max_queue` is the controller's queue, a worker that
//! does not answer a health ping within `health_timeout_ms` is stuck and gets no new requests,
//! and a request that makes no progress for `stall_timeout_ms` (default 30000) fails alone.
//! Speculative decoding (`max_draft`) and prefix reuse (`prefix_cache`) are not available here:
//!
//! ```toml
//! split = "requests"
//! [scheduler]
//! max_in_flight = 2        # per worker
//! stall_timeout_ms = 30000
//! ```
//!
//! An optional `[engine]` table picks the engine (default: the mock):
//!
//! ```toml
//! [engine]
//! kind = "streammoe"                     # needs a build with a model feature, e.g. --features olmoe
//! model_dir = "~/models/olmoe-1b-7b"   # artifacts/ and root/ as streammoe's converters write them
//! n_ctx = 2048
//! mem_cap_gib = 3.0                    # per node; a node can override it with its own mem_cap_gib
//! kv_format = "f16"                    # optional: f32 (default), f16, turbo4, turbo3
//! attn_threads = 4                     # optional: CPU decode attention threads
//! runtime_overhead_gib = 0.75          # optional: unitemized room inside the cap
//! experts_dir = "root-3bit"            # optional: packed 3-bit experts (default "root")
//! prefetch_guess = "cross"             # optional: prev (streammoe default) or cross
//! prefetch_experts = 6                 # optional: experts prefetched per guess
//! fast_experts_dir = "/Volumes/fast/x" # optional: faster disk with copies of some layers
//! ```
//!
//! An optional `[http]` table makes the coordinator serve the OpenAI-compatible API:
//!
//! ```toml
//! [http]
//! listen = "0.0.0.0:8080"
//! tokenizer = "~/models/olmoe-1b-7b/hf/tokenizer.json"
//! chat_template = "olmoe"              # or "chatml" (Qwen)
//! model = "olmoe-1b-7b"                # name reported to clients
//! ```
//!
//! An optional `[metrics]` table makes the coordinator write one CSV row per step
//! (CL-10, CL-11); `msh-node --metrics FILE` does the same for one run:
//!
//! ```toml
//! [metrics]
//! csv = "results/olmoe-3node.csv"
//! ```
//!
//! An optional `[service]` table sets `mem_allowance_mib` (default 256): memory for the
//! service itself (network buffers, requests, HTTP, metrics), beside the engine's cap (NF-2).
//!
//! An optional `[log]` table sets the log (NF-5). `RUST_LOG` overrides `level`.
//!
//! ```toml
//! [log]
//! level = "info"                       # or a filter, e.g. "msh_node=debug"
//! json = true                          # one JSON object per line instead of text
//! ```

use std::ops::Range;
use std::path::PathBuf;

use serde::Deserialize;

use crate::chat::ChatTemplate;
use crate::scheduler::SchedulerConfig;
use crate::Error;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub hidden_dim: usize,
    pub num_layers: u32,
    #[serde(default)]
    pub split: Split,
    #[serde(default)]
    pub experts: ExpertsConfig,
    #[serde(default)]
    pub scheduler: SchedulerConfig,
    #[serde(default)]
    pub engine: EngineConfig,
    #[serde(default)]
    pub http: Option<HttpConfig>,
    #[serde(default)]
    pub metrics: Option<MetricsConfig>,
    #[serde(default)]
    pub log: LogConfig,
    #[serde(default)]
    pub service: ServiceConfig,
    pub nodes: Vec<NodeConfig>,
}

/// How the model is divided across the nodes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Split {
    /// Each node runs a contiguous layer range; a step visits the workers in turn.
    #[default]
    Layers,
    /// The controller runs everything but the routed experts; each layer goes to every worker
    /// at once, and each worker runs the experts it owns.
    Experts,
    /// The controller runs no engine; each worker runs the whole model and serves whole
    /// requests.
    Requests,
}

/// The expert split's shape and rebalancing.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ExpertsConfig {
    /// Routed experts per layer.
    pub count: u32,
    /// Experts the router picks per position.
    pub top_k: u32,
    /// Positions routed between rebalances; 0 keeps the starting split.
    pub rebalance_every: u64,
    /// Most experts moved per rebalance.
    pub max_moves: usize,
    /// How many experts beyond an even share a worker may come to own.
    pub extra_experts: usize,
}

impl Default for ExpertsConfig {
    fn default() -> Self {
        ExpertsConfig { count: 256, top_k: 8, rebalance_every: 1024, max_moves: 8, extra_experts: 64 }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServiceConfig {
    /// Memory for the service itself, beside streammoe's cap (NF-2).
    pub mem_allowance_mib: f64,
}

impl Default for ServiceConfig {
    fn default() -> Self {
        ServiceConfig { mem_allowance_mib: 256.0 }
    }
}

impl ServiceConfig {
    pub fn mem_allowance(&self) -> u64 {
        (self.mem_allowance_mib * (1u64 << 20) as f64) as u64
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LogConfig {
    /// A `tracing` filter: a level, or per-module levels.
    pub level: String,
    pub json: bool,
}

impl Default for LogConfig {
    fn default() -> Self {
        LogConfig { level: "info".into(), json: false }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricsConfig {
    /// CSV file to append rows to.
    pub csv: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HttpConfig {
    pub listen: String,
    /// Path to a Hugging Face `tokenizer.json`.
    pub tokenizer: String,
    pub chat_template: ChatTemplate,
    pub model: String,
}

/// Expands a leading `~/` to `$HOME`.
pub fn expand_home(p: &str) -> PathBuf {
    match (p.strip_prefix("~/"), std::env::var_os("HOME")) {
        (Some(rest), Some(home)) => PathBuf::from(home).join(rest),
        _ => PathBuf::from(p),
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EngineKind {
    #[default]
    Mock,
    StreamMoe,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EngineConfig {
    pub kind: EngineKind,
    pub model_dir: String,
    pub n_ctx: usize,
    /// Hard memory cap for streammoe on each node, in GiB. `None` = uncapped.
    pub mem_cap_gib: Option<f64>,
    /// Storage of each node's KV cache. Smaller formats leave more of the cap to the expert
    /// hot tier.
    pub kv_format: KvFormat,
    /// Threads for streammoe's CPU decode attention; `None` keeps its default.
    pub attn_threads: Option<usize>,
    /// Room inside the cap for memory streammoe does not itemize, in GiB; `None` keeps its
    /// default (0.5). Long decodes can need 0.75.
    pub runtime_overhead_gib: Option<f64>,
    /// Expert files under `model_dir`: `root` (4-bit, default) or `root-3bit` (packed 3-bit,
    /// smaller and faster, slightly less accurate). streammoe detects the layout.
    pub experts_dir: String,
    /// How decode guesses the experts to prefetch; `None` keeps streammoe's default (`prev`).
    pub prefetch_guess: Option<PrefetchGuess>,
    /// Experts per layer each guess prefetches; `None` keeps streammoe's default (4 under a
    /// cap). `0` turns speculative prefetch off.
    pub prefetch_experts: Option<usize>,
    /// A faster disk holding copies of some layers' expert files (same layout); `None` reads
    /// everything from `experts_dir`.
    pub fast_experts_dir: Option<String>,
}

/// Where decode's expert prefetch guesses come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PrefetchGuess {
    /// The previous token's experts for the same layer.
    Prev,
    /// The current hidden state through the next layer's router. Not possible for the first
    /// layer of a node's range, which falls back to `prev`.
    Cross,
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig {
            kind: EngineKind::Mock,
            model_dir: String::new(),
            n_ctx: 2048,
            mem_cap_gib: None,
            kv_format: KvFormat::F32,
            attn_threads: None,
            runtime_overhead_gib: None,
            experts_dir: "root".into(),
            prefetch_guess: None,
            prefetch_experts: None,
            fast_experts_dir: None,
        }
    }
}

impl EngineConfig {
    pub fn runtime_overhead(&self) -> Option<u64> {
        self.runtime_overhead_gib.map(gib)
    }
}

/// KV cache storage: f32, f16, or TurboQuant 4 / 3-bit (smallest, slightly less accurate).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KvFormat {
    #[default]
    F32,
    F16,
    Turbo4,
    Turbo3,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NodeConfig {
    pub name: String,
    /// A worker's address, where the controller connects. Unused on the controller.
    pub addr: String,
    /// Address to bind; defaults to `addr`. Use `0.0.0.0:PORT` to listen on every interface.
    pub listen: Option<String>,
    /// Half-open layer range. Set on every node for a manual split (CL-4), or on none to let
    /// the planner choose (CL-3).
    pub layers: Option<[u32; 2]>,
    /// Relative speed for the planner (layers per unit of time; default 1). See
    /// `msh-node --speeds-from`.
    pub speed: Option<f64>,
    /// Overrides `engine.mem_cap_gib` on this node.
    pub mem_cap_gib: Option<f64>,
    /// Address for this node's Prometheus endpoint, `GET /metrics` (CL-12). Off when unset.
    pub prometheus: Option<String>,
}

impl NodeConfig {
    /// The node's layers. Panics before the split is planned ([`Config::plan_layers`]).
    pub fn layer_range(&self) -> Range<u32> {
        let [a, b] = self.layers.expect("layer split not planned");
        a..b
    }

    pub fn listen_addr(&self) -> &str {
        self.listen.as_deref().unwrap_or(&self.addr)
    }

    /// This node's streammoe memory cap in bytes.
    pub fn mem_cap(&self, engine: &EngineConfig) -> Option<u64> {
        self.mem_cap_gib.or(engine.mem_cap_gib).map(gib)
    }
}

fn gib(g: f64) -> u64 {
    (g * (1u64 << 30) as f64) as u64
}

impl Config {
    pub fn parse(text: &str) -> Result<Config, Error> {
        let cfg: Config = toml::from_str(text).map_err(|e| Error::Config(e.to_string()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// A manual split must cover `0..num_layers` contiguously, in node order.
    pub fn validate(&self) -> Result<(), Error> {
        let bad = |m: String| Err(Error::Config(m));
        if !(self.service.mem_allowance_mib > 0.0 && self.service.mem_allowance_mib.is_finite()) {
            return bad("service.mem_allowance_mib must be positive".into());
        }
        if self.nodes.is_empty() {
            return bad("no nodes".into());
        }
        for (i, n) in self.nodes.iter().enumerate() {
            if self.nodes[..i].iter().any(|m| m.name == n.name) {
                return bad(format!("duplicate node name {}", n.name));
            }
            if n.speed.is_some_and(|s| !(s > 0.0 && s.is_finite())) {
                return bad(format!("node {}: speed must be positive", n.name));
            }
        }
        if self.split == Split::Experts {
            return self.validate_experts();
        }
        if self.split == Split::Requests {
            return self.validate_requests();
        }
        match self.nodes.iter().filter(|n| n.layers.is_some()).count() {
            0 if self.nodes.len() > self.num_layers as usize => {
                return bad(format!("{} nodes for {} layers", self.nodes.len(), self.num_layers));
            }
            0 => {}
            k if k == self.nodes.len() => {
                let mut next = 0;
                for n in &self.nodes {
                    let [lo, hi] = n.layers.unwrap();
                    if lo != next || hi <= lo {
                        let m = format!("node {}: layers [{lo}, {hi}) must start at {next} and be non-empty", n.name);
                        return bad(m);
                    }
                    next = hi;
                }
                if next != self.num_layers {
                    return bad(format!("layer ranges end at {next}, model has {}", self.num_layers));
                }
            }
            _ => return bad("set layers on every node (manual split) or on none (planned)".into()),
        }
        if self.engine.kind == EngineKind::StreamMoe && self.engine.model_dir.is_empty() {
            return bad("engine.model_dir is required for kind = \"streammoe\"".into());
        }
        Ok(())
    }

    fn validate_experts(&self) -> Result<(), Error> {
        let bad = |m: String| Err(Error::Config(m));
        if let Some(n) = self.nodes.iter().find(|n| n.layers.is_some()) {
            return bad(format!("node {}: split = \"experts\" takes no layers", n.name));
        }
        if self.engine.kind != EngineKind::Mock {
            return bad("split = \"experts\" needs the mock engine until streammoe can run a layer without its experts".into());
        }
        let e = &self.experts;
        // Layer and expert ids are u16 on the wire; the owner table keeps a byte per expert.
        if e.count == 0 || e.count > 1 << 16 || self.num_layers > 1 << 16 {
            return bad(format!("{} experts per layer, {} layers: both must be 1..=65536", e.count, self.num_layers));
        }
        if e.top_k == 0 || e.top_k > e.count {
            return bad(format!("experts.top_k = {} must be 1..={}", e.top_k, e.count));
        }
        if self.nodes.len() > 256 {
            return bad(format!("{} nodes; the expert split takes at most 255 workers", self.nodes.len()));
        }
        Ok(())
    }

    fn validate_requests(&self) -> Result<(), Error> {
        let bad = |m: String| Err(Error::Config(m));
        if let Some(n) = self.nodes.iter().find(|n| n.layers.is_some()) {
            return bad(format!("node {}: split = \"requests\" takes no layers (every worker runs all of them)", n.name));
        }
        if self.nodes.len() < 2 {
            return bad("split = \"requests\" needs at least one worker".into());
        }
        let s = &self.scheduler;
        if s.max_draft > 0 || s.prefix_cache > 0 {
            return bad("split = \"requests\" has no speculative decoding or prefix reuse: set max_draft and prefix_cache to 0".into());
        }
        if s.stall_timeout_ms == 0 || s.health_timeout_ms == 0 {
            return bad("scheduler.stall_timeout_ms and health_timeout_ms must be positive".into());
        }
        if self.engine.kind == EngineKind::StreamMoe && self.engine.model_dir.is_empty() {
            return bad("engine.model_dir is required for kind = \"streammoe\"".into());
        }
        Ok(())
    }

    /// The layers node `i` runs: its range on the layer split, every layer on a request-split
    /// worker, `None` on the expert split and on the request split's controller.
    pub fn layers_of(&self, i: usize) -> Option<Range<u32>> {
        match self.split {
            Split::Layers => Some(self.nodes[i].layer_range()),
            Split::Experts => None,
            Split::Requests => (i > 0).then_some(0..self.num_layers),
        }
    }

    /// Whether the config leaves the layer split to the planner.
    pub fn is_planned(&self) -> bool {
        self.split == Split::Layers && self.nodes.iter().all(|n| n.layers.is_none())
    }

    /// Fills in every node's layers with the planner's split (CL-3) unless the config gives
    /// them (CL-4) or splits by experts. `fits(i, range)` says whether node `i` can hold `range`.
    pub fn plan_layers(&mut self, fits: impl Fn(usize, Range<u32>) -> bool) -> Result<(), Error> {
        if !self.is_planned() {
            return Ok(());
        }
        let speeds: Vec<f64> = self.nodes.iter().map(|n| n.speed.unwrap_or(1.0)).collect();
        let ranges = crate::planner::plan(self.num_layers, &speeds, fits).map_err(Error::Config)?;
        for (n, r) in self.nodes.iter_mut().zip(ranges) {
            n.layers = Some([r.start, r.end]);
        }
        Ok(())
    }

    pub fn index_of(&self, name: &str) -> Result<usize, Error> {
        self.nodes
            .iter()
            .position(|n| n.name == name)
            .ok_or_else(|| Error::Config(format!("no node named {name}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const THREE: &str = r#"
        hidden_dim = 2048
        num_layers = 40
        [[nodes]]
        name = "m4"
        addr = "10.0.0.1:7100"
        listen = "0.0.0.0:7100"
        layers = [0, 20]
        [[nodes]]
        name = "m1a"
        addr = "10.0.0.2:7100"
        layers = [20, 30]
        [[nodes]]
        name = "m1b"
        addr = "10.0.0.3:7100"
        layers = [30, 40]
    "#;

    #[test]
    fn planner_fills_in_a_missing_split() {
        let text = THREE
            .replace("layers = [0, 20]", "speed = 2.0")
            .replace("layers = [20, 30]", "")
            .replace("layers = [30, 40]", "speed = 2.0");
        let mut c = Config::parse(&text).unwrap();
        assert!(c.is_planned());
        c.plan_layers(|_, _| true).unwrap();
        let split: Vec<_> = c.nodes.iter().map(|n| n.layer_range()).collect();
        assert_eq!(split, [0..16, 16..24, 24..40]);
        // A manual split is kept as it is.
        let mut manual = Config::parse(THREE).unwrap();
        manual.plan_layers(|_, _| false).unwrap();
        assert_eq!(manual.nodes[1].layer_range(), 20..30);
        // Nothing fits: refuse to start.
        let mut c = Config::parse(&text).unwrap();
        assert!(c.plan_layers(|_, r| r.len() < 10).is_err());
    }

    #[test]
    fn split_is_manual_on_every_node_or_none() {
        assert!(Config::parse(&THREE.replace("layers = [20, 30]", "")).is_err());
        assert!(Config::parse(&THREE.replace("layers = [0, 20]", "layers = [0, 20]\nspeed = 0.0")).is_err());
    }

    #[test]
    fn parses_three_node_split() {
        let c = Config::parse(THREE).unwrap();
        assert_eq!(c.nodes[1].layer_range(), 20..30);
        assert_eq!(c.nodes[0].listen_addr(), "0.0.0.0:7100");
        assert_eq!(c.nodes[2].listen_addr(), "10.0.0.3:7100");
        assert_eq!(c.index_of("m1b").unwrap(), 2);
        assert_eq!(c.scheduler.max_in_flight, 3);
    }

    #[test]
    fn scheduler_section_overrides_defaults() {
        let text = THREE.replace("num_layers = 40", "num_layers = 40\n[scheduler]\nmax_in_flight = 5");
        let c = Config::parse(&text).unwrap();
        assert_eq!((c.scheduler.max_in_flight, c.scheduler.max_queue), (5, 64));
    }

    #[test]
    fn engine_section_and_per_node_cap() {
        let text = THREE
            .replace("num_layers = 40", "num_layers = 40\n[engine]\nkind = \"streammoe\"\nmodel_dir = \"/m\"\nmem_cap_gib = 2.0")
            .replace("layers = [20, 30]", "layers = [20, 30]\nmem_cap_gib = 0.5");
        let c = Config::parse(&text).unwrap();
        assert_eq!(c.engine.kind, EngineKind::StreamMoe);
        assert_eq!(c.nodes[0].mem_cap(&c.engine), Some(2 << 30));
        assert_eq!(c.nodes[1].mem_cap(&c.engine), Some(1 << 29));
        assert_eq!(Config::parse(THREE).unwrap().engine.kind, EngineKind::Mock);
        assert!(Config::parse(&text.replace("model_dir = \"/m\"", "")).is_err());
    }

    #[test]
    fn engine_tuning() {
        let text = THREE.replace(
            "num_layers = 40",
            "num_layers = 40\n[engine]\nkind = \"streammoe\"\nmodel_dir = \"/m\"\nkv_format = \"turbo4\"\nattn_threads = 8\nruntime_overhead_gib = 0.75",
        );
        let e = Config::parse(&text).unwrap().engine;
        assert_eq!((e.kv_format, e.attn_threads, e.runtime_overhead()), (KvFormat::Turbo4, Some(8), Some(3 << 28)));
        let streaming = text.replace(
            "kv_format = \"turbo4\"",
            "experts_dir = \"root-3bit\"\nprefetch_guess = \"cross\"\nprefetch_experts = 6\nfast_experts_dir = \"/f\"",
        );
        let e = Config::parse(&streaming).unwrap().engine;
        assert_eq!(
            (e.experts_dir.as_str(), e.prefetch_guess, e.prefetch_experts, e.fast_experts_dir.as_deref()),
            ("root-3bit", Some(PrefetchGuess::Cross), Some(6), Some("/f"))
        );
        let e = Config::parse(THREE).unwrap().engine;
        assert_eq!((e.kv_format, e.attn_threads, e.runtime_overhead()), (KvFormat::F32, None, None));
        assert!(Config::parse(&text.replace("turbo4", "q8")).is_err());
    }

    #[test]
    fn http_section() {
        let text = THREE.replace(
            "num_layers = 40",
            "num_layers = 40\n[http]\nlisten = \"0.0.0.0:8080\"\ntokenizer = \"/t.json\"\nchat_template = \"chatml\"\nmodel = \"q\"",
        );
        let http = Config::parse(&text).unwrap().http.unwrap();
        assert_eq!((http.chat_template, http.model.as_str()), (ChatTemplate::Chatml, "q"));
        assert!(Config::parse(THREE).unwrap().http.is_none());
    }

    #[test]
    fn expert_split_takes_no_layers() {
        let layerless = THREE.replace("layers = [0, 20]", "").replace("layers = [20, 30]", "").replace("layers = [30, 40]", "");
        let text = layerless.replace("num_layers = 40", "num_layers = 40\nsplit = \"experts\"\n[experts]\ncount = 16\ntop_k = 4");
        let mut c = Config::parse(&text).unwrap();
        assert_eq!((c.split, c.experts.count, c.experts.top_k, c.experts.max_moves), (Split::Experts, 16, 4, 8));
        assert!(!c.is_planned());
        c.plan_layers(|_, _| true).unwrap();
        assert!(c.nodes.iter().all(|n| n.layers.is_none()), "no layer split is planned");
        assert_eq!(Config::parse(&layerless).unwrap().split, Split::Layers);
        assert!(Config::parse(&THREE.replace("num_layers = 40", "num_layers = 40\nsplit = \"experts\"")).is_err());
        assert!(Config::parse(&text.replace("top_k = 4", "top_k = 17")).is_err());
        assert!(Config::parse(&text.replace("split = \"experts\"", "split = \"rows\"")).is_err());
        let streammoe = text.replace("[experts]", "[engine]\nkind = \"streammoe\"\nmodel_dir = \"/m\"\n[experts]");
        assert!(Config::parse(&streammoe).is_err());
    }

    #[test]
    fn request_split_takes_no_layers_and_needs_a_worker() {
        let layerless = THREE.replace("layers = [0, 20]", "").replace("layers = [20, 30]", "").replace("layers = [30, 40]", "");
        let text = layerless.replace("num_layers = 40", "num_layers = 40\nsplit = \"requests\"\n[scheduler]\nmax_in_flight = 2");
        let c = Config::parse(&text).unwrap();
        assert_eq!((c.split, c.scheduler.max_in_flight, c.scheduler.stall_timeout_ms), (Split::Requests, 2, 30_000));
        assert_eq!((c.layers_of(0), c.layers_of(2)), (None, Some(0..40)));
        assert!(!c.is_planned());
        let stall = text.replace("max_in_flight = 2", "stall_timeout_ms = 500");
        assert_eq!(Config::parse(&stall).unwrap().scheduler.stall_timeout_ms, 500);
        assert!(Config::parse(&THREE.replace("num_layers = 40", "num_layers = 40\nsplit = \"requests\"")).is_err());
        assert!(Config::parse(&text.replace("max_in_flight = 2", "max_draft = 4")).is_err());
        assert!(Config::parse(&text.replace("max_in_flight = 2", "prefix_cache = 1")).is_err());
        assert!(Config::parse(&text.replace("max_in_flight = 2", "stall_timeout_ms = 0")).is_err());
        let alone = text.split("[[nodes]]").take(2).collect::<Vec<_>>().join("[[nodes]]");
        assert!(Config::parse(&alone).is_err(), "no worker");
        let streammoe = text.replace("[scheduler]", "[engine]\nkind = \"streammoe\"\nmodel_dir = \"/m\"\n[scheduler]");
        assert!(Config::parse(&streammoe).is_ok());
    }

    #[test]
    fn rejects_gaps_overlaps_and_short_coverage() {
        for (from, to) in [("[20, 30]", "[21, 30]"), ("[20, 30]", "[19, 30]"), ("[30, 40]", "[30, 39]")] {
            assert!(Config::parse(&THREE.replace(from, to)).is_err(), "{to} accepted");
        }
    }
}
