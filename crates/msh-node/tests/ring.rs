use std::net::TcpListener;
use std::thread;
use std::time::Duration;

use msh_node::config::{Config, NodeConfig};
use msh_node::engine::MockEngine;
use msh_node::node::{join_ring, Coordinator, Worker};

const DIM: usize = 2048;
const PROMPT: &[u32] = &[11, 22, 33, 44, 55];

/// Starts every node of `splits` on loopback threads and returns the coordinator plus the
/// worker threads, which return their engines when the coordinator is dropped.
fn start(splits: &[[u32; 2]]) -> (Coordinator<MockEngine>, Vec<thread::JoinHandle<MockEngine>>) {
    let listeners: Vec<TcpListener> = splits.iter().map(|_| TcpListener::bind("127.0.0.1:0").unwrap()).collect();
    let cfg = Config {
        hidden_dim: DIM,
        num_layers: 40,
        nodes: splits
            .iter()
            .zip(&listeners)
            .enumerate()
            .map(|(i, (layers, l))| NodeConfig {
                name: format!("n{i}"),
                addr: l.local_addr().unwrap().to_string(),
                listen: None,
                layers: *layers,
            })
            .collect(),
    };
    cfg.validate().unwrap();

    let mut workers = Vec::new();
    let mut coord_listener = None;
    for (i, l) in listeners.into_iter().enumerate() {
        if i == 0 {
            coord_listener = Some(l);
            continue;
        }
        let cfg = cfg.clone();
        workers.push(thread::spawn(move || {
            let ring = join_ring(&cfg, i, l, Duration::from_secs(5)).unwrap().unwrap();
            let engine = MockEngine::new(cfg.nodes[i].layer_range(), DIM);
            Worker::new(engine, ring, &cfg.nodes[i].name, i == cfg.nodes.len() - 1).run().unwrap()
        }));
    }
    let ring = join_ring(&cfg, 0, coord_listener.unwrap(), Duration::from_secs(5)).unwrap();
    (Coordinator::new(MockEngine::new(cfg.nodes[0].layer_range(), DIM), ring), workers)
}

#[test]
fn three_node_split_matches_single_node_exactly() {
    // NF-1 with the mock engine: f32 on the wire must leave the result bit-identical.
    let (mut single, _) = start(&[[0, 40]]);
    let want = single.generate(1, PROMPT, 16).unwrap().tokens;

    let (mut ring, workers) = start(&[[0, 20], [20, 30], [30, 40]]);
    let got = ring.generate(1, PROMPT, 16).unwrap();
    assert_eq!(got.tokens, want);
    assert_eq!(got.step_times.len(), 16);

    // Same request id again: only works if Drop freed position state on every node (CL-9).
    assert_eq!(ring.generate(1, PROMPT, 16).unwrap().tokens, want);
    assert_eq!(ring.engine().live_requests(), 0);

    drop(ring);
    for w in workers {
        assert_eq!(w.join().unwrap().live_requests(), 0);
    }
}

#[test]
fn two_node_split_matches_too() {
    let (mut single, _) = start(&[[0, 40]]);
    let (mut ring, _) = start(&[[0, 25], [25, 40]]);
    assert_eq!(ring.generate(9, PROMPT, 8).unwrap().tokens, single.generate(9, PROMPT, 8).unwrap().tokens);
}
