//! msh-node --config FILE --node NAME [--prompt 1,2,3] [--max-tokens N] [--requests K]
//!          [--metrics FILE]
//!
//! The first node in the config is the coordinator. With an `[http]` table and no `--prompt`,
//! it serves the OpenAI-compatible API. With `--prompt` (token ids) it submits K concurrent
//! greedy generations, prints throughput and exits. `--metrics FILE` (or `[metrics] csv`)
//! appends one row per position to a CSV.
//! Every other node is a worker and serves until the coordinator disconnects.
//! `[engine] kind` picks the mock (default) or streammoe, which needs a model feature at build
//! time, e.g. `cargo build --release --features olmoe`.

use std::net::TcpListener;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use msh_node::chat::Codec;
use msh_node::config::{expand_home, Config, EngineKind, HttpConfig};
use msh_node::http::{self, HttpState};
use msh_node::metrics::{Metrics, RunInfo};
use msh_node::node::Ring;
use msh_node::engine::{Engine, MockEngine};
use msh_node::node::{join_ring, Worker};
use msh_node::scheduler::{self, collect};

const RING_TIMEOUT: Duration = Duration::from_secs(60);

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
                 [--metrics FILE]";
    let path = flag("--config").ok_or(usage)?;
    let name = flag("--node").ok_or(usage)?;

    let cfg = Config::parse(&std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?)?;
    let idx = cfg.index_of(name)?;
    let node = &cfg.nodes[idx];
    // Open the engine before joining the ring, so a range that does not fit fails here (DEP-4).
    match cfg.engine.kind {
        EngineKind::Mock => serve(&cfg, idx, MockEngine::new(node.layer_range(), cfg.num_layers, cfg.hidden_dim), &flag),
        EngineKind::StreamMoe => serve(&cfg, idx, open_streammoe(&cfg, idx)?, &flag),
    }
}

#[cfg(feature = "streammoe")]
fn open_streammoe(cfg: &Config, idx: usize) -> Result<impl Engine + 'static, Box<dyn std::error::Error>> {
    use msh_node::streammoe_engine::{OpenSpec, StreamMoeEngine};
    let node = &cfg.nodes[idx];
    let model_dir = expand_home(&cfg.engine.model_dir);
    let t0 = Instant::now();
    let engine = StreamMoeEngine::open(&OpenSpec {
        model_dir: &model_dir,
        layers: node.layer_range(),
        n_ctx: cfg.engine.n_ctx,
        mem_cap: node.mem_cap(&cfg.engine),
        max_requests: cfg.scheduler.engine_slots(),
        prefill_chunk: cfg.scheduler.prefill_chunk,
        kv_format: cfg.engine.kv_format,
        attn_threads: cfg.engine.attn_threads,
        runtime_overhead: cfg.engine.runtime_overhead(),
    })?;
    eprintln!("{}: streammoe opened layers {:?} in {:.1} s", node.name, node.layer_range(), t0.elapsed().as_secs_f64());
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
    flag: &dyn Fn(&str) -> Option<&'a str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let node = &cfg.nodes[idx];
    let name = node.name.as_str();
    let listener = TcpListener::bind(node.listen_addr()).map_err(|e| format!("bind {}: {e}", node.listen_addr()))?;
    let role = if idx == 0 { "coordinator" } else { "worker" };
    eprintln!("{name}: {role}, layers {:?}, listening on {}", node.layer_range(), node.listen_addr());
    let ring = join_ring(&cfg, idx, listener, RING_TIMEOUT)?;
    eprintln!("{name}: ring ready ({} nodes)", cfg.nodes.len());

    if idx > 0 {
        Worker::new(engine, ring.expect("multi-node ring"), name).run()?;
        eprintln!("{name}: coordinator disconnected, exiting");
        return Ok(());
    }

    let metrics = open_metrics(cfg, flag("--metrics"))?;
    if let (None, Some(http)) = (flag("--prompt"), &cfg.http) {
        return serve_http(cfg, http, engine, ring, metrics);
    }
    let prompt: Vec<u32> = flag("--prompt")
        .unwrap_or("1,2,3,4,5,6,7,8")
        .split(',')
        .map(|t| t.trim().parse())
        .collect::<Result<_, _>>()
        .map_err(|e| format!("--prompt: {e}"))?;
    let max_new: usize = flag("--max-tokens").unwrap_or("32").parse().map_err(|e| format!("--max-tokens: {e}"))?;

    let requests: u32 = flag("--requests").unwrap_or("1").parse().map_err(|e| format!("--requests: {e}"))?;

    let (client, sched) = scheduler::spawn_with_metrics(engine, ring, cfg.scheduler, metrics);
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
    Ok(())
}

/// Runs the scheduler behind the HTTP API until the process is stopped.
fn serve_http<E: Engine + 'static>(
    cfg: &Config,
    http: &HttpConfig,
    engine: E,
    ring: Option<Ring>,
    metrics: Option<Metrics>,
) -> Result<(), Box<dyn std::error::Error>> {
    let codec = Codec::load(&expand_home(&http.tokenizer), http.chat_template)?;
    let (client, _sched) = scheduler::spawn_with_metrics(engine, ring, cfg.scheduler, metrics);
    let state = Arc::new(HttpState {
        client,
        codec,
        model: http.model.clone(),
        n_ctx: cfg.engine.n_ctx,
        next_id: AtomicU64::new(1),
    });
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let listener = tokio::net::TcpListener::bind(&http.listen).await.map_err(|e| format!("bind {}: {e}", http.listen))?;
        eprintln!("{}: serving {} on http://{}/v1", cfg.nodes[0].name, http.model, http.listen);
        axum::serve(listener, http::router(state)).await?;
        Ok(())
    })
}

/// Opens the metrics CSV from `--metrics` or `[metrics] csv`, if either is set.
fn open_metrics(cfg: &Config, flag: Option<&str>) -> Result<Option<Metrics>, Box<dyn std::error::Error>> {
    let Some(path) = flag.map(String::from).or_else(|| cfg.metrics.as_ref().map(|m| m.csv.clone())) else {
        return Ok(None);
    };
    let split: Vec<String> =
        cfg.nodes.iter().map(|n| format!("{}:{}-{}", n.name, n.layers[0], n.layers[1])).collect();
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
    eprintln!("{}: writing metrics to {}", cfg.nodes[0].name, path.display());
    Ok(Some(m))
}
