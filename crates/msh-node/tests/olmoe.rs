//! The real engine on OLMoE-1B-7B: one node against a three-node ring over loopback, checked
//! against moeflux's MLX reference continuation.
//!
//! Needs the converted model at `$HOME/models/olmoe-1b-7b` (or `MSH_OLMOE_DIR`), so it is
//! `#[ignore]`d. Run with:
//!
//! ```text
//! cargo test -p msh-node --features olmoe --release --test olmoe -- --ignored --test-threads=1
//! ```

#![cfg(feature = "olmoe")]

use std::net::TcpListener;
use std::path::PathBuf;
use std::thread;
use std::time::Duration;

use msh_node::config::{Config, NodeConfig};
use msh_node::moeflux_engine::{expand_home, MoefluxEngine, OpenSpec};
use msh_node::node::{join_ring, Worker};
use msh_node::scheduler::{self, collect, Client, SchedulerConfig};

/// "What is the capital of France? Answer in one sentence." in OLMoE's chat template.
const PROMPT: &[u32] = &[
    50279, 29, 93, 4537, 49651, 187, 1276, 310, 253, 5347, 273, 6181, 32, 37741, 275, 581, 6197,
    15, 187, 29, 93, 515, 5567, 49651, 187,
];
/// MLX greedy continuation, "The capital of France is Paris.", without the final EOS.
const REF: &[u32] = &[510, 5347, 273, 6181, 310, 7785, 15];

fn model_dir() -> PathBuf {
    std::env::var("MSH_OLMOE_DIR").map(PathBuf::from).unwrap_or_else(|_| expand_home("~/models/olmoe-1b-7b"))
}

/// moeflux's cap is checked against the whole process's footprint, so it only holds with one
/// context per process, as in a real deployment. A ring of threads runs uncapped.
fn open(layers: std::ops::Range<u32>, max_requests: usize, mem_cap: Option<u64>) -> MoefluxEngine {
    let dir = model_dir();
    MoefluxEngine::open(&OpenSpec { model_dir: &dir, layers, n_ctx: 512, mem_cap, max_requests })
        .expect("open OLMoE (is the model converted at ~/models/olmoe-1b-7b?)")
}

/// Starts `splits` on loopback threads and returns the coordinator's client.
fn start(splits: &[[u32; 2]]) -> Client {
    let cap = (splits.len() == 1).then_some(3 << 30);
    let sched = SchedulerConfig::default();
    let listeners: Vec<TcpListener> = splits.iter().map(|_| TcpListener::bind("127.0.0.1:0").unwrap()).collect();
    let cfg = Config {
        hidden_dim: 2048,
        num_layers: 16,
        scheduler: sched,
        engine: Default::default(),
        nodes: splits
            .iter()
            .zip(&listeners)
            .enumerate()
            .map(|(i, (layers, l))| NodeConfig {
                name: format!("n{i}"),
                addr: l.local_addr().unwrap().to_string(),
                listen: None,
                layers: *layers,
                mem_cap_gib: None,
            })
            .collect(),
    };
    let mut listeners = listeners.into_iter();
    let coord_listener = listeners.next().unwrap();
    for (k, l) in listeners.enumerate() {
        let (i, cfg) = (k + 1, cfg.clone());
        let engine = open(cfg.nodes[i].layer_range(), sched.max_in_flight, cap);
        thread::spawn(move || {
            let ring = join_ring(&cfg, i, l, Duration::from_secs(30)).unwrap().unwrap();
            Worker::new(engine, ring, &cfg.nodes[i].name).run().unwrap();
        });
    }
    let engine = open(cfg.nodes[0].layer_range(), sched.max_in_flight, cap);
    let ring = join_ring(&cfg, 0, coord_listener, Duration::from_secs(30)).unwrap();
    scheduler::spawn(engine, ring, sched).0
}

#[test]
#[ignore = "needs the OLMoE model on disk"]
fn one_node_matches_mlx_reference() {
    let client = start(&[[0, 16]]);
    assert_eq!(client.generate(PROMPT.to_vec(), 32).unwrap().tokens, REF);
}

#[test]
#[ignore = "needs the OLMoE model on disk"]
fn three_node_ring_matches_mlx_reference_with_requests_in_flight() {
    let client = start(&[[0, 8], [8, 12], [12, 16]]);
    // Three at once: each request's state must stay separate on every node.
    let rxs: Vec<_> = (0..3).map(|_| client.submit(PROMPT.to_vec(), 32)).collect();
    for rx in rxs {
        assert_eq!(collect(rx).unwrap().tokens, REF);
    }
    // Slots were released on every node: the pool (3) admits three more.
    let rxs: Vec<_> = (0..3).map(|_| client.submit(PROMPT.to_vec(), 32)).collect();
    for rx in rxs {
        assert_eq!(collect(rx).unwrap().tokens, REF);
    }
}
