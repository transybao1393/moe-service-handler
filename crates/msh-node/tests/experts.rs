//! The expert split on the mock engine: a controller and two workers on loopback threads,
//! checked token for token against one process running every expert itself.

use std::net::TcpListener;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use msh_net::{Connection, MsgType};
use msh_node::config::{Config, ExpertsConfig, NodeConfig, Split};
use msh_node::engine::MockEngine;
use msh_node::metrics::{Metrics, RunInfo};
use msh_node::node::{accept_controller, connect_workers, Worker};
use msh_node::scheduler::{self, collect, Client, SchedulerConfig};
use msh_node::Error;

const DIM: usize = 32;
const LAYERS: u32 = 4;
const EXPERTS: u32 = 16;
const TOP_K: u32 = 4;
const VOCAB: u32 = 151_936;

fn engine(vocab: u32) -> MockEngine {
    MockEngine::new(0..LAYERS, LAYERS, DIM).with_experts(EXPERTS, TOP_K).with_vocab(vocab)
}

fn experts() -> ExpertsConfig {
    ExpertsConfig { count: EXPERTS, top_k: TOP_K, rebalance_every: 0, ..Default::default() }
}

fn prompt(i: u32) -> Vec<u32> {
    (0..5 + i).map(|k| 11 * (k + 1) + 100 * i).collect()
}

/// Reference output: one process with every expert, one request at a time, one position per
/// step.
fn reference(prompts: &[Vec<u32>], max_new: usize, vocab: u32) -> Vec<Vec<u32>> {
    let sched = SchedulerConfig { prefill_chunk: 1, ..Default::default() };
    let (client, _) = scheduler::spawn(engine(vocab), None, sched);
    prompts.iter().map(|p| client.generate(p.clone(), max_new).unwrap().tokens).collect()
}

fn config(listeners: &[TcpListener], sched: SchedulerConfig, experts: ExpertsConfig) -> Config {
    let cfg = Config {
        hidden_dim: DIM,
        num_layers: LAYERS,
        split: Split::Experts,
        experts,
        scheduler: sched,
        engine: Default::default(),
        http: None,
        metrics: None,
        log: Default::default(),
        service: Default::default(),
        nodes: listeners
            .iter()
            .enumerate()
            .map(|(i, l)| NodeConfig {
                name: format!("n{i}"),
                addr: l.local_addr().unwrap().to_string(),
                listen: None,
                layers: None,
                speed: None,
                mem_cap_gib: None,
                prometheus: None,
            })
            .collect(),
    };
    cfg.validate().unwrap();
    cfg
}

struct Cluster {
    client: Client,
    coordinator: JoinHandle<MockEngine>,
    workers: Vec<JoinHandle<MockEngine>>,
}

impl Cluster {
    /// A controller and two workers. `worker(i, engine)` adjusts worker `i`'s engine.
    fn start(
        sched: SchedulerConfig,
        experts: ExpertsConfig,
        vocab: u32,
        metrics: Option<Metrics>,
        worker: impl Fn(usize, MockEngine) -> MockEngine,
    ) -> Cluster {
        let listeners: Vec<TcpListener> = (0..3).map(|_| TcpListener::bind("127.0.0.1:0").unwrap()).collect();
        let cfg = config(&listeners, sched, experts);
        let mut listeners = listeners.into_iter();
        drop(listeners.next());
        let workers = listeners
            .enumerate()
            .map(|(k, l)| {
                let (i, cfg, e) = (k + 1, cfg.clone(), worker(k + 1, engine(vocab)));
                thread::spawn(move || {
                    let link = accept_controller(&cfg, i, l).unwrap();
                    Worker::new(e, link, &cfg.nodes[i].name).run().unwrap()
                })
            })
            .collect();
        let star = connect_workers(&cfg, Duration::from_secs(5)).unwrap();
        let (client, coordinator) =
            scheduler::spawn_experts(engine(vocab), star, cfg.scheduler, cfg.experts, metrics).unwrap();
        Cluster { client, coordinator, workers }
    }

    fn run(&self, prompts: &[Vec<u32>], max_new: usize) -> Vec<Vec<u32>> {
        let rxs: Vec<_> = prompts.iter().map(|p| self.client.submit(p.clone(), max_new)).collect();
        rxs.into_iter().map(|rx| collect(rx).unwrap().tokens).collect()
    }

    /// Shuts down, checks that only the controller held request state and freed it, and
    /// returns the workers' engines.
    fn stop(self) -> Vec<MockEngine> {
        self.client.shutdown();
        assert_eq!(self.coordinator.join().unwrap().live_requests(), 0, "coordinator");
        let workers: Vec<MockEngine> = self.workers.into_iter().map(|w| w.join().unwrap()).collect();
        for (i, w) in workers.iter().enumerate() {
            assert_eq!(w.live_requests(), 0, "worker {}", i + 1);
        }
        workers
    }
}

#[test]
fn three_nodes_match_one_process_with_requests_in_flight() {
    let prompts: Vec<_> = (0..6).map(prompt).collect();
    let want = reference(&prompts, 12, VOCAB);
    // Prompt chunks of 3 and three requests in flight at once.
    let sched = SchedulerConfig { prefill_chunk: 3, max_in_flight: 3, ..Default::default() };
    let c = Cluster::start(sched, experts(), VOCAB, None, |_, e| e);
    assert_eq!(c.run(&prompts, 12), want);
    let workers = c.stop();
    // An even split, untouched: half of every layer's experts each.
    let held: Vec<usize> = workers.iter().map(MockEngine::held_experts).collect();
    assert_eq!(held, [32, 32]);
    assert!(workers.iter().all(|w| w.expert_loads() == (0, 0)));
}

#[test]
fn rebalancing_mid_run_keeps_the_output() {
    let prompts: Vec<_> = (0..4).map(prompt).collect();
    let want = reference(&prompts, 30, VOCAB);
    let moving = ExpertsConfig { rebalance_every: 3, max_moves: 2, ..experts() };
    let c = Cluster::start(SchedulerConfig::default(), moving, VOCAB, None, |_, e| e);
    assert_eq!(c.run(&prompts, 30), want);
    let workers = c.stop();
    let moved: usize = workers.iter().map(|w| w.expert_loads().1).sum();
    assert!(moved > 0, "no expert moved");
    // Every expert still has exactly one home.
    assert_eq!(workers.iter().map(MockEngine::held_experts).sum::<usize>(), (LAYERS * EXPERTS) as usize);
    // Both workers took experts: with a rebalance every 3 positions the counts are small and
    // the busier worker changes from one rebalance to the next.
    assert!(workers.iter().all(|w| w.expert_loads().1 > 0), "moves both ways");
}

#[test]
fn a_worker_with_a_small_hot_set_gives_the_same_tokens() {
    let prompts: Vec<_> = (0..3).map(prompt).collect();
    let want = reference(&prompts, 10, VOCAB);
    // Worker 2 keeps 5 of its 32 experts in memory.
    let c = Cluster::start(SchedulerConfig::default(), experts(), VOCAB, None, |i, e| {
        if i == 2 { e.with_expert_capacity(5) } else { e }
    });
    assert_eq!(c.run(&prompts, 10), want);
    let workers = c.stop();
    assert!(workers[1].expert_loads().0 > 0, "loads while serving");
    assert_eq!(workers[0].expert_loads().0, 0);
}

#[test]
fn speculative_decoding_matches_plain_decoding() {
    // Six token ids make the text repeat, so prompt lookup drafts often.
    let vocab = 6;
    let prompts: Vec<Vec<u32>> = (0..3).map(|i| (0..20).map(|k| (k * (i + 1)) % 5).collect()).collect();
    let want = reference(&prompts, 40, vocab);
    let sched = SchedulerConfig { max_draft: 4, draft_min_ngram: 2, prefix_cache: 2, ..Default::default() };
    let c = Cluster::start(sched, experts(), vocab, None, |_, e| e);
    assert_eq!(c.run(&prompts, 40), want);
    c.stop();
}

#[test]
fn worker_error_fails_only_its_request() {
    // Request ids count from 1 in submission order: request 2 is the second submit.
    let c = Cluster::start(SchedulerConfig::default(), experts(), VOCAB, None, |i, e| {
        if i == 1 { e.failing_on(2) } else { e }
    });
    let rxs: Vec<_> = (0..3).map(|i| c.client.submit(prompt(i), 8)).collect();
    let results: Vec<_> = rxs.into_iter().map(collect).collect();
    let err = results[1].as_ref().unwrap_err().to_string();
    assert!(err.contains("n1") && err.contains("request 2"), "{err}");
    assert!(results[0].is_ok() && results[2].is_ok());
    // The cluster keeps serving.
    assert_eq!(c.client.generate(prompt(4), 4).unwrap().tokens, reference(&[prompt(4)], 4, VOCAB)[0]);
    c.stop();
}

#[test]
fn metrics_time_every_node_and_the_workers_path() {
    let dir = std::env::temp_dir().join(format!("msh-experts-metrics-{}", std::process::id()));
    let path = dir.join("run.csv");
    let run = RunInfo {
        run_id: 1,
        git_hash: "test".into(),
        host: "h".into(),
        hardware: "hw".into(),
        split: "experts 4/16: n0 n1 n2".into(),
        engine: "mock".into(),
    };
    let sched = SchedulerConfig { prefill_chunk: 2, ..Default::default() };
    let c = Cluster::start(sched, experts(), VOCAB, Some(Metrics::create(&path, &run).unwrap()), |_, e| e);
    c.run(&[prompt(0), prompt(1)], 6);
    c.stop();
    let text = std::fs::read_to_string(&path).unwrap();
    let rows: Vec<Vec<&str>> = text.lines().skip(1).map(|l| l.split(',').collect()).collect();
    // Prompts of 5 and 6 positions in chunks of 2 (3 steps each), then 5 decode steps each.
    assert_eq!(rows.len(), 3 + 3 + 2 * 5);
    for r in &rows {
        let (step, coord, workers, network) = (r[11], r[12], r[13], r[14]);
        let stages: Vec<u64> = r[15].split(';').map(|t| t.parse().unwrap()).collect();
        assert_eq!(stages.len(), 3, "every node reports a time: {r:?}");
        let n = |s: &str| s.parse::<u64>().unwrap();
        // The workers ran at once: their share is at most the slower one's total.
        assert!(n(workers) <= stages[1].max(stages[2]) * 2 && n(workers) <= stages[1] + stages[2], "{r:?}");
        assert!(n(coord) + n(workers) + n(network) >= n(step), "{r:?}");
    }
    std::fs::remove_dir_all(dir).unwrap();
}

/// A controller and one worker that runs `fake` once it has answered `Own`.
fn with_fake_worker(sched: SchedulerConfig, fake: impl FnOnce(Connection) + Send + 'static) -> Client {
    let listeners = [TcpListener::bind("127.0.0.1:0").unwrap(), TcpListener::bind("127.0.0.1:0").unwrap()];
    let cfg = config(&listeners, sched, experts());
    let [l0, l1] = listeners;
    let worker_cfg = cfg.clone();
    thread::spawn(move || {
        let mut link = accept_controller(&worker_cfg, 1, l1).unwrap();
        let mut buf = Vec::new();
        assert_eq!(link.recv_frame(&mut buf).unwrap().msg_type, MsgType::Own);
        link.send(MsgType::Own, 0, 0, &[]).unwrap();
        fake(link)
    });
    drop(l0);
    let star = connect_workers(&cfg, Duration::from_secs(5)).unwrap();
    scheduler::spawn_experts(engine(VOCAB), star, sched, cfg.experts, None).unwrap().0
}

/// Reads frames until the first layer's experts.
fn wait_for_experts(link: &mut Connection) {
    let mut buf = Vec::new();
    while link.recv_frame(&mut buf).unwrap().msg_type != MsgType::Experts {}
}

#[test]
fn lost_worker_marks_the_cluster_down() {
    let client = with_fake_worker(SchedulerConfig::default(), |mut link| {
        wait_for_experts(&mut link);
        // Returning drops the connection, as a crashed process would.
    });
    let err = client.generate(prompt(0), 8).unwrap_err().to_string();
    assert!(err.contains("cluster down"), "{err}");
    let err = client.generate(prompt(1), 8).unwrap_err();
    assert!(matches!(&err, Error::Rejected(m) if m.contains("cluster down")), "{err}");
}

#[test]
fn hung_worker_is_detected_by_health_ping() {
    let sched = SchedulerConfig { health_interval_ms: 20, health_timeout_ms: 200, ..Default::default() };
    let client = with_fake_worker(sched, |mut link| {
        wait_for_experts(&mut link);
        thread::sleep(Duration::from_secs(3));
        drop(link);
    });
    let t0 = Instant::now();
    let err = client.generate(prompt(0), 8).unwrap_err().to_string();
    assert!(err.contains("health ping"), "{err}");
    assert!(t0.elapsed() < Duration::from_secs(2), "took {:?}", t0.elapsed());
}

#[test]
fn engine_must_match_the_expert_config() {
    let wrong = ExpertsConfig { top_k: 2, ..experts() };
    let err = scheduler::spawn_experts(engine(VOCAB), None, SchedulerConfig::default(), wrong, None).err().unwrap();
    assert!(matches!(err, Error::Config(_)), "{err}");
    let plain = MockEngine::new(0..LAYERS, LAYERS, DIM);
    assert!(scheduler::spawn_experts(plain, None, SchedulerConfig::default(), experts(), None).is_err());
    // Without workers, the controller runs every expert itself.
    let (client, _) = scheduler::spawn_experts(engine(VOCAB), None, SchedulerConfig::default(), experts(), None).unwrap();
    assert_eq!(client.generate(prompt(2), 6).unwrap().tokens, reference(&[prompt(2)], 6, VOCAB)[0]);
}
