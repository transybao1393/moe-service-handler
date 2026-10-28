//! msh-node --config FILE --node NAME [--prompt 1,2,3] [--max-tokens N] [--requests K]
//!          [--metrics FILE]
//! msh-node --config FILE --plan
//! msh-node --speeds-from METRICS.csv
//!
//! Without `layers` on the nodes, every node plans the same split from the config (CL-3):
//! node speeds, and under `[engine] kind = "streammoe"` each node's memory cap. `--plan` prints
//! the split (and each node's memory budget) and exits. `--speeds-from` prints each node's
//! measured speed from a previous run's metrics, to copy into the config's `speed` entries.
//!
//! The first node in the config is the coordinator. With an `[http]` table and no `--prompt`,
//! it serves the OpenAI-compatible API. With `--prompt` (token ids) it submits K concurrent
//! greedy generations, prints throughput and exits. `--metrics FILE` (or `[metrics] csv`)
//! appends one row per position to a CSV.
//! Every other node is a worker and serves until the coordinator disconnects.
//! `[engine] kind` picks the mock (default) or streammoe, which needs a model feature at build
//! time, e.g. `cargo build --release --features olmoe`.

use std::net::TcpListener;
#[cfg(feature = "streammoe")]
use std::ops::Range;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use std::sync::Arc;

use msh_node::chat::Codec;
use msh_node::config::{expand_home, Config, EngineKind, HttpConfig, LogConfig, Split};
use tracing::info;
use msh_node::http::{self, HttpState};
use msh_node::{memory, stats};
use msh_node::metrics::{Metrics, RunInfo};
use msh_node::node::Star;
use msh_node::engine::{Engine, MockEngine};
use msh_node::node::{accept_controller, connect_workers, Worker};
use msh_node::planner;
use msh_node::scheduler::{self, collect};

/// Logs to stderr (NF-5): `RUST_LOG` if set, else `[log] level`.
fn init_logs(cfg: &LogConfig) -> Result<(), Box<dyn std::error::Error>> {
    use tracing_subscriber::EnvFilter;
    let filter = match std::env::var("RUST_LOG") {
        Ok(f) => EnvFilter::try_new(f)?,
        Err(_) => EnvFilter::try_new(&cfg.level).map_err(|e| format!("[log] level {:?}: {e}", cfg.level))?,
    };
    use std::io::IsTerminal;
    let logs = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_ansi(std::io::stderr().is_terminal());
    if cfg.json {
        logs.json().init();
    } else {
        logs.init();
    }
    Ok(())
}

const CONNECT_TIMEOUT: Duration = Duration::from_secs(60);

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).map(String::as_str);
    let usage = "usage: msh-node --config FILE --node NAME [--prompt 1,2,3] [--max-tokens N] [--requests K] \
                 [--metrics FILE]\n       msh-node --config FILE --plan\n       msh-node --speeds-from METRICS.csv";
    if let Some(csv) = flag("--speeds-from") {
        let speeds = planner::speeds_from_metrics(&std::fs::read_to_string(csv).map_err(|e| format!("{csv}: {e}"))?)?;
        let slowest = speeds.iter().map(|(_, s)| *s).fold(f64::INFINITY, f64::min);
        for (name, speed) in speeds {
            println!("{name}: speed = {:.2}  ({speed:.3} layers/ms)", speed / slowest);
        }
        return Ok(());
    }
    let path = flag("--config").ok_or(usage)?;
    let mut cfg = Config::parse(&std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?)?;
    init_logs(&cfg.log)?;
    let print_plan = args.iter().any(|a| a == "--plan");
    plan_split(&mut cfg, print_plan)?;
    if print_plan {
        return Ok(());
    }
    let name = flag("--node").ok_or(usage)?;
    let idx = cfg.index_of(name)?;
    // Every log line of this process, on every thread, names the node.
    let _span = tracing::info_span!("node", node = name).entered();
    let node = &cfg.nodes[idx];
    // The tokenizer is the service's; load it before measuring the service's memory (NF-2).
    let http = cfg.http.as_ref().filter(|_| idx == 0 && flag("--prompt").is_none());
    let codec = http.map(|h| Codec::load(&expand_home(&h.tokenizer), h.chat_template)).transpose()?;
    memory::check_start(&cfg, &memory::growth(&cfg, idx, http.is_some()))?;
    // Open the engine before joining the cluster, so a range that does not fit fails here (DEP-4).
    match (cfg.engine.kind, cfg.split) {
        (EngineKind::Mock, Split::Layers) => {
            serve(&cfg, idx, MockEngine::new(node.layer_range(), cfg.num_layers, cfg.hidden_dim), codec, &flag)
        }
        // Every node builds the whole mock; a worker holds only the experts it is sent.
        (EngineKind::Mock, Split::Experts) => {
            let e = MockEngine::new(0..cfg.num_layers, cfg.num_layers, cfg.hidden_dim);
            serve(&cfg, idx, e.with_experts(cfg.experts.count, cfg.experts.top_k), codec, &flag)
        }
        (EngineKind::StreamMoe, _) => serve(&cfg, idx, open_streammoe(&cfg, idx)?, codec, &flag),
    }
}

/// What node `i` runs, for logs: its layers, or its part of the expert split.
fn role(cfg: &Config, i: usize) -> String {
    match (cfg.split, i) {
        (Split::Layers, _) => format!("{:?}", cfg.nodes[i].layer_range()),
        (Split::Experts, 0) => "all layers but the routed experts".into(),
        (Split::Experts, _) => "routed experts".into(),
    }
}

/// Starts the scheduler for the config's split.
fn start_scheduler<E: Engine + 'static>(
    cfg: &Config,
    engine: E,
    star: Option<Star>,
    metrics: Option<Metrics>,
) -> Result<(scheduler::Client, std::thread::JoinHandle<E>), Box<dyn std::error::Error>> {
    Ok(match cfg.split {
        Split::Layers => scheduler::spawn_with_metrics(engine, star, cfg.scheduler, metrics),
        Split::Experts => scheduler::spawn_experts(engine, star, cfg.scheduler, cfg.experts, metrics)?,
    })
}

/// Plans the split unless the config gives it, logs it, and with `print` shows it on stdout.
#[cfg(feature = "streammoe")]
fn plan_split(cfg: &mut Config, print: bool) -> Result<(), Box<dyn std::error::Error>> {
    use msh_node::streammoe_engine::Budgeter;
    let caps: Vec<Option<u64>> = cfg.nodes.iter().map(|n| n.mem_cap(&cfg.engine)).collect();
    let budgeter = match cfg.engine.kind {
        EngineKind::StreamMoe if caps.iter().any(Option::is_some) => {
            Some(with_spec(cfg, 0..cfg.num_layers, None, Budgeter::new)?)
        }
        _ => None,
    };
    let budget = |i: usize, r: Range<u32>| Some(budgeter.as_ref()?.budget(r, caps[i]?));
    let source = if cfg.is_planned() { "planner" } else { "config" };
    cfg.plan_layers(|i, r| budget(i, r).is_none_or(|b| b.fixed <= caps[i].unwrap_or(u64::MAX)))?;
    show_split(cfg, source, print, |i| {
        budget(i, cfg.nodes[i].layer_range()).map(|b| {
            format!(
                "fixed {:.2} GiB, hot tier {:.2} of {:.2} GiB experts",
                b.fixed as f64 / GIB,
                b.hot_tier as f64 / GIB,
                b.range_experts as f64 / GIB
            )
        })
    });
    Ok(())
}

#[cfg(not(feature = "streammoe"))]
fn plan_split(cfg: &mut Config, print: bool) -> Result<(), Box<dyn std::error::Error>> {
    let source = if cfg.is_planned() { "planner" } else { "config" };
    cfg.plan_layers(|_, _| true)?;
    show_split(cfg, source, print, |_| None);
    Ok(())
}

const GIB: f64 = (1u64 << 30) as f64;
const MIB: f64 = (1u64 << 20) as f64;

fn show_split(cfg: &Config, source: &str, print: bool, budget: impl Fn(usize) -> Option<String>) {
    if cfg.split == Split::Experts {
        let workers = cfg.nodes.len() - 1;
        let e = &cfg.experts;
        info!(workers, experts = e.count, top_k = e.top_k, "expert split");
        if print {
            for (i, n) in cfg.nodes.iter().enumerate() {
                let what = match i {
                    0 => role(cfg, 0),
                    // Even shares to start, as the controller assigns them.
                    _ => {
                        let share = msh_node::owners::Owners::even(1, e.count, workers).owned_by(i - 1).len();
                        format!("{share} experts per layer to start")
                    }
                };
                println!("{:<8} {what}", n.name);
            }
        }
        return;
    }
    let split: Vec<String> = cfg.nodes.iter().map(|n| format!("{}:{:?}", n.name, n.layer_range())).collect();
    info!(source, split = split.join(" "), "layer split");
    if !print {
        return;
    }
    for (i, n) in cfg.nodes.iter().enumerate() {
        let r = n.layer_range();
        let mut line = format!("{:<8} layers {:>3}..{:<3} ({:>2})  speed {:.2}", n.name, r.start, r.end, r.len(), n.speed.unwrap_or(1.0));
        if let Some(cap) = n.mem_cap(&cfg.engine) {
            line += &format!("  cap {:.2} GiB", cap as f64 / GIB);
        }
        if let Some(b) = budget(i) {
            line += &format!("  {b}");
        }
        println!("{line}");
    }
}

/// Calls `f` with the open settings for `layers` on this config.
#[cfg(feature = "streammoe")]
fn with_spec<T>(
    cfg: &Config,
    layers: Range<u32>,
    mem_cap: Option<u64>,
    f: impl FnOnce(&msh_node::streammoe_engine::OpenSpec) -> T,
) -> T {
    let model_dir = expand_home(&cfg.engine.model_dir);
    let fast_experts_dir = cfg.engine.fast_experts_dir.as_deref().map(expand_home);
    f(&msh_node::streammoe_engine::OpenSpec {
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

#[cfg(feature = "streammoe")]
fn open_streammoe(cfg: &Config, idx: usize) -> Result<impl Engine + 'static, Box<dyn std::error::Error>> {
    use msh_node::streammoe_engine::StreamMoeEngine;
    let node = &cfg.nodes[idx];
    let t0 = Instant::now();
    let engine = with_spec(cfg, node.layer_range(), node.mem_cap(&cfg.engine), StreamMoeEngine::open)?;
    info!(layers = ?node.layer_range(), secs = format_args!("{:.1}", t0.elapsed().as_secs_f64()), "streammoe opened");
    Ok(engine)
}

#[cfg(not(feature = "streammoe"))]
fn open_streammoe(_: &Config, _: usize) -> Result<MockEngine, Box<dyn std::error::Error>> {
    Err("engine kind \"streammoe\" needs a build with a model feature, e.g. --features olmoe".into())
}

fn serve<'a, E: Engine + 'static>(
    cfg: &Config,
    idx: usize,
    engine: E,
    codec: Option<Codec>,
    flag: &dyn Fn(&str) -> Option<&'a str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let node = &cfg.nodes[idx];
    let name = node.name.as_str();
    let limit = node.mem_cap(&cfg.engine).map(|cap| cap + cfg.service.mem_allowance());
    stats::STATS.footprint_limit.store(limit.unwrap_or(0), std::sync::atomic::Ordering::Relaxed);
    memory::watch(limit, Duration::from_secs(5));
    if let Some(addr) = &node.prometheus {
        let listener = TcpListener::bind(addr).map_err(|e| format!("bind prometheus {addr}: {e}"))?;
        stats::serve(listener, name.to_string(), idx == 0);
    }
    // Workers listen for the controller; the controller connects to each of them.
    if idx > 0 {
        let listener = TcpListener::bind(node.listen_addr()).map_err(|e| format!("bind {}: {e}", node.listen_addr()))?;
        info!(role = "worker", runs = role(cfg, idx), listen = node.listen_addr(), "waiting for the controller");
        let link = accept_controller(cfg, idx, listener)?;
        info!(controller = link.peer_name(), "connected");
        Worker::new(engine, link, name).run()?;
        info!("controller disconnected, exiting");
        return Ok(());
    }
    info!(role = "controller", runs = role(cfg, 0), workers = cfg.nodes.len() - 1, "connecting to workers");
    let star = connect_workers(cfg, CONNECT_TIMEOUT)?;
    info!(nodes = cfg.nodes.len(), "cluster ready");

    let metrics = open_metrics(cfg, flag("--metrics"))?;
    if let (Some(codec), Some(http)) = (codec, &cfg.http) {
        return serve_http(cfg, http, codec, engine, star, metrics);
    }
    let prompt: Vec<u32> = flag("--prompt")
        .unwrap_or("1,2,3,4,5,6,7,8")
        .split(',')
        .map(|t| t.trim().parse())
        .collect::<Result<_, _>>()
        .map_err(|e| format!("--prompt: {e}"))?;
    let max_new: usize = flag("--max-tokens").unwrap_or("32").parse().map_err(|e| format!("--max-tokens: {e}"))?;

    let requests: u32 = flag("--requests").unwrap_or("1").parse().map_err(|e| format!("--requests: {e}"))?;

    let (client, sched) = start_scheduler(cfg, engine, star, metrics)?;
    let t0 = Instant::now();
    let rxs: Vec<_> = (0..requests).map(|_| client.submit(prompt.clone(), max_new)).collect();
    let mut total = 0;
    let mut decode = Vec::new();
    for (i, rx) in rxs.into_iter().enumerate() {
        let g = collect(rx)?;
        total += g.tokens.len();
        // The first latency includes the prompt's last position; the rest are decode steps.
        decode.extend(g.latencies.iter().skip(1).map(|d| d.as_secs_f64() * 1e6));
        if i == 0 {
            println!("request 1 tokens: {:?}", &g.tokens[..g.tokens.len().min(32)]);
        }
    }
    let wall = t0.elapsed().as_secs_f64();
    client.shutdown();
    sched.join().map_err(|_| "scheduler panicked")?;
    decode.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = decode.get(decode.len() / 2).copied().unwrap_or(f64::NAN);
    println!(
        "{requests} requests (max {} in flight), {total} tokens in {wall:.3} s: {:.0} tok/s total, \
         per-token latency p50 {p50:.0} µs",
        cfg.scheduler.max_in_flight,
        total as f64 / wall
    );
    if let Some(now) = memory::footprint() {
        let limit = limit.map_or("no cap".into(), |l| format!("cap + allowance {:.0} MiB", l as f64 / MIB));
        println!("footprint {:.0} MiB ({limit})", now as f64 / MIB);
    }
    Ok(())
}

/// Runs the scheduler behind the HTTP API until the process is stopped.
fn serve_http<E: Engine + 'static>(
    cfg: &Config,
    http: &HttpConfig,
    codec: Codec,
    engine: E,
    star: Option<Star>,
    metrics: Option<Metrics>,
) -> Result<(), Box<dyn std::error::Error>> {
    let (client, _sched) = start_scheduler(cfg, engine, star, metrics)?;
    let state = Arc::new(HttpState::new(client, codec, http.model.clone(), cfg.engine.n_ctx, memory::http_max_open(cfg)));
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let listener = tokio::net::TcpListener::bind(&http.listen).await.map_err(|e| format!("bind {}: {e}", http.listen))?;
        info!(model = http.model, url = format!("http://{}/v1", http.listen), "serving");
        axum::serve(listener, http::router(state)).await?;
        Ok(())
    })
}

/// Opens the metrics CSV from `--metrics` or `[metrics] csv`, if either is set.
fn open_metrics(cfg: &Config, flag: Option<&str>) -> Result<Option<Metrics>, Box<dyn std::error::Error>> {
    let Some(path) = flag.map(String::from).or_else(|| cfg.metrics.as_ref().map(|m| m.csv.clone())) else {
        return Ok(None);
    };
    let split: Vec<String> = match cfg.split {
        Split::Layers => {
            cfg.nodes.iter().map(|n| format!("{}:{}-{}", n.name, n.layer_range().start, n.layer_range().end)).collect()
        }
        Split::Experts => {
            let mut s = vec![format!("experts {}/{}:", cfg.experts.top_k, cfg.experts.count)];
            s.extend(cfg.nodes.iter().map(|n| n.name.clone()));
            s
        }
    };
    let engine = match cfg.engine.kind {
        EngineKind::Mock => "mock".to_string(),
        EngineKind::StreamMoe => format!("streammoe {}", cfg.engine.model_dir),
    };
    let run = RunInfo {
        run_id: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs(),
        git_hash: msh_net::host::GIT_HASH.into(),
        host: msh_net::host::hostname(),
        hardware: msh_net::host::hardware(),
        split: split.join(" "),
        engine,
    };
    let path = expand_home(&path);
    let m = Metrics::create(&path, &run).map_err(|e| format!("metrics {}: {e}", path.display()))?;
    info!(path = %path.display(), "writing metrics");
    Ok(Some(m))
}
