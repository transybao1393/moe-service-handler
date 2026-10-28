//! The real engine on OLMoE-1B-7B: one node against a controller and two workers over loopback, checked
//! against streammoe's MLX reference continuation.
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

use msh_node::config::{expand_home, Config, NodeConfig, PrefetchGuess};
use msh_node::streammoe_engine::{OpenSpec, StreamMoeEngine};
use msh_node::node::{accept_controller, connect_workers, Worker};
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

/// streammoe's cap is checked against the whole process's footprint, so it only holds with one
/// context per process, as in a real deployment. A cluster of threads runs uncapped.
fn open(layers: std::ops::Range<u32>, max_requests: usize, mem_cap: Option<u64>) -> StreamMoeEngine {
    let dir = model_dir();
    StreamMoeEngine::open(&OpenSpec { n_ctx: 512, mem_cap, max_requests, ..OpenSpec::new(&dir, layers) })
        .expect("open OLMoE (is the model converted at ~/models/olmoe-1b-7b?)")
}

/// A cluster on loopback threads. Dropping it shuts it down and waits until every node has
/// released its context, so the next test starts with the process's memory back.
struct Cluster {
    client: Option<Client>,
    threads: Vec<thread::JoinHandle<()>>,
}

impl Drop for Cluster {
    fn drop(&mut self) {
        drop(self.client.take());
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

impl std::ops::Deref for Cluster {
    type Target = Client;
    fn deref(&self) -> &Client {
        self.client.as_ref().unwrap()
    }
}

fn start(splits: &[[u32; 2]]) -> Cluster {
    start_with(splits, SchedulerConfig::default())
}

fn start_with(splits: &[[u32; 2]], sched: SchedulerConfig) -> Cluster {
    let cap = (splits.len() == 1).then_some(3 << 30);
    let listeners: Vec<TcpListener> = splits.iter().map(|_| TcpListener::bind("127.0.0.1:0").unwrap()).collect();
    let cfg = Config {
        hidden_dim: 2048,
        num_layers: 16,
        split: Default::default(),
        experts: Default::default(),
        scheduler: sched,
        engine: Default::default(),
        http: None,
        metrics: None,
        log: Default::default(),
        service: Default::default(),
        nodes: splits
            .iter()
            .zip(&listeners)
            .enumerate()
            .map(|(i, (layers, l))| NodeConfig {
                name: format!("n{i}"),
                addr: l.local_addr().unwrap().to_string(),
                listen: None,
                layers: Some(*layers),
                speed: None,
                mem_cap_gib: None,
                prometheus: None,
            })
            .collect(),
    };
    let mut listeners = listeners.into_iter();
    let coord_listener = listeners.next().unwrap();
    let mut threads = Vec::new();
    for (k, l) in listeners.enumerate() {
        let (i, cfg) = (k + 1, cfg.clone());
        let engine = open(cfg.nodes[i].layer_range(), sched.engine_slots(), cap);
        threads.push(thread::spawn(move || {
            let link = accept_controller(&cfg, i, l).unwrap();
            Worker::new(engine, link, &cfg.nodes[i].name).run().unwrap();
        }));
    }
    let engine = open(cfg.nodes[0].layer_range(), sched.engine_slots(), cap);
    drop(coord_listener);
    let star = connect_workers(&cfg, Duration::from_secs(30)).unwrap();
    let (client, coordinator) = scheduler::spawn(engine, star, sched);
    threads.push(thread::spawn(move || drop(coordinator.join())));
    Cluster { client: Some(client), threads }
}

#[test]
#[ignore = "needs the OLMoE model on disk"]
fn one_node_matches_mlx_reference() {
    // Position by position, and the whole prompt as one batched chunk.
    for prefill_chunk in [1, 64] {
        let client = start_with(&[[0, 16]], SchedulerConfig { prefill_chunk, ..Default::default() });
        assert_eq!(client.generate(PROMPT.to_vec(), 32).unwrap().tokens, REF, "chunk {prefill_chunk}");
    }
}

#[test]
#[ignore = "needs the OLMoE model on disk"]
fn three_node_cluster_matches_mlx_reference_with_requests_in_flight() {
    // The 25-token prompt in chunks of 8, 8, 8 and 1, pipelined through the workers.
    let sched = SchedulerConfig { prefill_chunk: 8, ..Default::default() };
    let client = start_with(&[[0, 8], [8, 12], [12, 16]], sched);
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

#[test]
#[ignore = "needs the OLMoE model on disk"]
fn request_split_two_workers_match_the_reference() {
    // Each worker opens the whole model (mode A); the controller opens nothing.
    use msh_node::config::Split;
    use msh_node::node::RequestWorker;
    use msh_node::requests;

    let sched = SchedulerConfig { max_in_flight: 2, prefill_chunk: 8, ..Default::default() };
    let listeners: Vec<TcpListener> = (0..2).map(|_| TcpListener::bind("127.0.0.1:0").unwrap()).collect();
    let node = |i: usize, addr: String| NodeConfig {
        name: format!("n{i}"),
        addr,
        listen: None,
        layers: None,
        speed: None,
        mem_cap_gib: None,
        prometheus: None,
    };
    let mut nodes = vec![node(0, "127.0.0.1:1".into())];
    nodes.extend(listeners.iter().enumerate().map(|(k, l)| node(k + 1, l.local_addr().unwrap().to_string())));
    let cfg = Config {
        hidden_dim: 2048,
        num_layers: 16,
        split: Split::Requests,
        experts: Default::default(),
        scheduler: sched,
        engine: Default::default(),
        http: None,
        metrics: None,
        log: Default::default(),
        service: Default::default(),
        nodes,
    };
    cfg.validate().unwrap();
    let workers: Vec<_> = listeners
        .into_iter()
        .enumerate()
        .map(|(k, l)| {
            let (i, cfg) = (k + 1, cfg.clone());
            let engine = open(0..16, sched.engine_slots(), None);
            thread::spawn(move || {
                let mut w = RequestWorker::new(engine, &cfg.nodes[i].name, sched.prefill_chunk, None);
                w.serve(accept_controller(&cfg, i, l).unwrap()).unwrap();
            })
        })
        .collect();
    let (client, coordinator) = requests::spawn(&cfg).unwrap();
    let board = client.board().unwrap().clone();
    assert_eq!(board.wait_ready(2, Duration::from_secs(30)), 2);
    // Four at once: two on each worker, interleaved there.
    let rxs: Vec<_> = (0..4).map(|_| client.submit(PROMPT.to_vec(), 32)).collect();
    for rx in rxs {
        assert_eq!(collect(rx).unwrap().tokens, REF);
    }
    assert!(board.workers().iter().all(|w| w.done == 2), "{:?}", board.workers());
    drop(client);
    coordinator.join().unwrap();
    for w in workers {
        w.join().unwrap();
    }
}

#[test]
#[ignore = "needs the OLMoE model on disk"]
fn planner_memory_estimate_matches_the_open() {
    // The planner's estimate for a node must agree with what streammoe's open accepts.
    use msh_node::streammoe_engine::Budgeter;
    let dir = model_dir();
    let spec = |mem_cap| OpenSpec { n_ctx: 512, mem_cap, max_requests: 3, ..OpenSpec::new(&dir, 0..8) };
    let est = Budgeter::new(&spec(None)).unwrap().budget(0..8, 8 << 30);
    assert_eq!(est.hot_tier, est.range_experts, "an 8 GiB cap holds every expert of 8 layers");
    let below = est.fixed - (64 << 20);
    assert!(StreamMoeEngine::open(&spec(Some(below))).is_err(), "opened under the estimate");
    let above = est.fixed + (256 << 20);
    let engine = StreamMoeEngine::open(&spec(Some(above)));
    assert!(engine.is_ok(), "estimate {} B, open at {above} B: {:?}", est.fixed, engine.err());
}

#[test]
#[ignore = "needs the OLMoE model on disk"]
fn prefetch_settings_keep_the_output() {
    // Prefetch only changes what is read ahead, never the math.
    let dir = model_dir();
    let mut prefetch_reads = Vec::new();
    for prefetch_experts in [0, 6] {
        let spec = OpenSpec {
            n_ctx: 512,
            mem_cap: Some(3 << 30),
            prefetch_guess: Some(PrefetchGuess::Cross),
            prefetch_experts: Some(prefetch_experts),
            ..OpenSpec::new(&dir, 0..16)
        };
        let engine = StreamMoeEngine::open(&spec).expect("open OLMoE");
        let (client, coordinator) = scheduler::spawn(engine, None, SchedulerConfig::default());
        assert_eq!(client.generate(PROMPT.to_vec(), 32).unwrap().tokens, REF, "prefetch {prefetch_experts}");
        drop(client);
        prefetch_reads.push(coordinator.join().unwrap().expert_io_stats().prefetch.reads);
    }
    assert_eq!(prefetch_reads[0], 0, "prefetch off reads nothing ahead");
    assert!(prefetch_reads[1] > 0, "prefetch 6 reads ahead: {prefetch_reads:?}");
}

#[test]
#[ignore = "needs the OLMoE model on disk"]
fn next_turn_continues_the_parked_request_exactly() {
    // Turn 1 is the reference question; turn 2 extends the conversation with more tokens.
    let p2: Vec<u32> = [PROMPT, REF, &[50279, 187], &PROMPT[1..]].concat();
    let fresh = start(&[[0, 16]]).generate(p2.clone(), 16).unwrap().tokens;
    let client = start_with(&[[0, 16]], SchedulerConfig { prefix_cache: 1, ..Default::default() });
    assert_eq!(client.generate(PROMPT.to_vec(), 32).unwrap().tokens, REF);
    assert_eq!(client.generate(p2, 16).unwrap().tokens, fresh);
}

#[test]
#[ignore = "needs the OLMoE model on disk"]
fn http_chat_completion_answers_with_the_reference() {
    use std::sync::Arc;

    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use msh_node::chat::{ChatTemplate, Codec};
    use msh_node::http::{router, HttpState};
    use tower::ServiceExt;

    let codec = Codec::load(&model_dir().join("hf/tokenizer.json"), ChatTemplate::Olmoe).unwrap();
    let cluster = start(&[[0, 8], [8, 12], [12, 16]]);
    let state = HttpState::new((*cluster).clone(), codec, "olmoe".into(), 512, 64);
    let body = serde_json::json!({
        "messages": [{ "role": "user", "content": "What is the capital of France? Answer in one sentence." }],
    });
    let req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let rt = tokio::runtime::Runtime::new().unwrap();
    let resp: serde_json::Value = rt.block_on(async {
        let resp = router(Arc::new(state)).oneshot(req).await.unwrap();
        serde_json::from_slice(&resp.into_body().collect().await.unwrap().to_bytes()).unwrap()
    });
    assert_eq!(resp["choices"][0]["message"]["content"], "The capital of France is Paris.");
    assert_eq!(resp["usage"]["prompt_tokens"], PROMPT.len());
}
