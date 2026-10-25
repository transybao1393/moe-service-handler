//! msh-node --config FILE --node NAME [--prompt 1,2,3] [--max-tokens N] [--requests K]
//!
//! The first node in the config is the coordinator; it submits K concurrent greedy generations
//! over the prompt (token ids, until the HTTP endpoint lands) and prints throughput.
//! Every other node is a worker and serves until the coordinator disconnects.
//! `[engine] kind` picks the mock (default) or moeflux, which needs a model feature at build
//! time, e.g. `cargo build --release --features olmoe`.

use std::net::TcpListener;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use msh_node::config::{Config, EngineKind};
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
    let usage = "usage: msh-node --config FILE --node NAME [--prompt 1,2,3] [--max-tokens N] [--requests K]";
    let path = flag("--config").ok_or(usage)?;
    let name = flag("--node").ok_or(usage)?;

    let cfg = Config::parse(&std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?)?;
    let idx = cfg.index_of(name)?;
    let node = &cfg.nodes[idx];
    // Open the engine before joining the ring, so a range that does not fit fails here (DEP-4).
    match cfg.engine.kind {
        EngineKind::Mock => serve(&cfg, idx, MockEngine::new(node.layer_range(), cfg.num_layers, cfg.hidden_dim), &flag),
        EngineKind::Moeflux => serve(&cfg, idx, open_moeflux(&cfg, idx)?, &flag),
    }
}

#[cfg(feature = "moeflux")]
fn open_moeflux(cfg: &Config, idx: usize) -> Result<impl Engine + 'static, Box<dyn std::error::Error>> {
    use msh_node::moeflux_engine::{expand_home, MoefluxEngine, OpenSpec};
    let node = &cfg.nodes[idx];
    let model_dir = expand_home(&cfg.engine.model_dir);
    let t0 = Instant::now();
    let engine = MoefluxEngine::open(&OpenSpec {
        model_dir: &model_dir,
        layers: node.layer_range(),
        n_ctx: cfg.engine.n_ctx,
        mem_cap: node.mem_cap(&cfg.engine),
        max_requests: cfg.scheduler.max_in_flight,
    })?;
    eprintln!("{}: moeflux opened layers {:?} in {:.1} s", node.name, node.layer_range(), t0.elapsed().as_secs_f64());
    Ok(engine)
}

#[cfg(not(feature = "moeflux"))]
fn open_moeflux(_: &Config, _: usize) -> Result<MockEngine, Box<dyn std::error::Error>> {
    Err("engine kind \"moeflux\" needs a build with a model feature, e.g. --features olmoe".into())
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

    let prompt: Vec<u32> = flag("--prompt")
        .unwrap_or("1,2,3,4,5,6,7,8")
        .split(',')
        .map(|t| t.trim().parse())
        .collect::<Result<_, _>>()
        .map_err(|e| format!("--prompt: {e}"))?;
    let max_new: usize = flag("--max-tokens").unwrap_or("32").parse().map_err(|e| format!("--max-tokens: {e}"))?;

    let requests: u32 = flag("--requests").unwrap_or("1").parse().map_err(|e| format!("--requests: {e}"))?;

    let (client, sched) = scheduler::spawn(engine, ring, cfg.scheduler);
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
