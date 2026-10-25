//! msh-node --config FILE --node NAME [--prompt 1,2,3] [--max-tokens N]
//!
//! The first node in the config is the coordinator; it runs one greedy generation over the
//! prompt (token ids, until the HTTP endpoint lands in M2) and prints per-token timing.
//! Every other node is a worker and serves until the coordinator disconnects.
//! The engine is `MockEngine` until moeflux's layer-range execution (DEP-1, DEP-2) lands.

use std::net::TcpListener;
use std::process::ExitCode;
use std::time::Duration;

use msh_node::config::Config;
use msh_node::engine::MockEngine;
use msh_node::node::{join_ring, Coordinator, Worker};

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
    let usage = "usage: msh-node --config FILE --node NAME [--prompt 1,2,3] [--max-tokens N]";
    let path = flag("--config").ok_or(usage)?;
    let name = flag("--node").ok_or(usage)?;

    let cfg = Config::parse(&std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?)?;
    let idx = cfg.index_of(name)?;
    let node = &cfg.nodes[idx];
    let engine = MockEngine::new(node.layer_range(), cfg.hidden_dim);

    let listener = TcpListener::bind(node.listen_addr()).map_err(|e| format!("bind {}: {e}", node.listen_addr()))?;
    let role = if idx == 0 { "coordinator" } else { "worker" };
    eprintln!("{name}: {role}, layers {:?}, listening on {}", node.layer_range(), node.listen_addr());
    let ring = join_ring(&cfg, idx, listener, RING_TIMEOUT)?;
    eprintln!("{name}: ring ready ({} nodes)", cfg.nodes.len());

    if idx > 0 {
        let is_last = idx == cfg.nodes.len() - 1;
        Worker::new(engine, ring.expect("multi-node ring"), name, is_last).run()?;
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

    let mut coord = Coordinator::new(engine, ring);
    let g = coord.generate(1, &prompt, max_new)?;
    let us: Vec<f64> = g.step_times.iter().map(|d| d.as_secs_f64() * 1e6).collect();
    let mut sorted = us.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let decode = &us[1.min(us.len())..]; // first entry includes the prompt's last position
    let mean = decode.iter().sum::<f64>() / decode.len().max(1) as f64;
    println!("tokens: {:?}", g.tokens);
    println!(
        "{} tokens, step p50 {:.0} µs, decode mean {:.0} µs ({:.0} tok/s)",
        g.tokens.len(),
        sorted[sorted.len() / 2],
        mean,
        1e6 / mean
    );
    Ok(())
}
