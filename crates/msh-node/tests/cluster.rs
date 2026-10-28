use std::io;
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use msh_net::MsgType;
use msh_node::config::{Config, NetConfig, NodeConfig};
use msh_node::engine::MockEngine;
use msh_node::metrics::{Metrics, RunInfo};
use msh_net::{Connection, FrameError};
use msh_node::node::{accept_controller, connect_workers, Worker};
use msh_node::scheduler::{self, collect, Client, Event, SchedulerConfig};
use msh_node::Error;

const DIM: usize = 2048;
const THREE: &[[u32; 2]] = &[[0, 20], [20, 30], [30, 40]];
/// The mock's default vocabulary: generated text practically never repeats.
const VOCAB: u32 = 151_936;

fn prompt(i: u32) -> Vec<u32> {
    (0..5).map(|k| 11 * (k + 1) + 100 * i).collect()
}

struct Cluster {
    client: Client,
    coordinator: JoinHandle<MockEngine>,
    workers: Vec<JoinHandle<MockEngine>>,
}

impl Cluster {
    fn start(splits: &[[u32; 2]]) -> Cluster {
        Cluster::start_with(splits, SchedulerConfig::default(), None)
    }

    /// Starts every node on loopback threads. With `fail = Some((node, req))`, that node's
    /// engine fails every step of request `req`.
    fn start_with(splits: &[[u32; 2]], sched: SchedulerConfig, fail: Option<(usize, u64)>) -> Cluster {
        Cluster::start_full(splits, sched, fail, None, VOCAB)
    }

    fn start_full(
        splits: &[[u32; 2]],
        sched: SchedulerConfig,
        fail: Option<(usize, u64)>,
        metrics: Option<Metrics>,
        vocab: u32,
    ) -> Cluster {
        Cluster::start_net(splits, sched, fail, metrics, vocab, NetConfig::default())
    }

    fn start_net(
        splits: &[[u32; 2]],
        sched: SchedulerConfig,
        fail: Option<(usize, u64)>,
        metrics: Option<Metrics>,
        vocab: u32,
        net: NetConfig,
    ) -> Cluster {
        let listeners: Vec<TcpListener> =
            splits.iter().map(|_| TcpListener::bind("127.0.0.1:0").unwrap()).collect();
        let cfg = Config {
            hidden_dim: DIM,
            num_layers: 40,
            split: Default::default(),
            experts: Default::default(),
            scheduler: sched,
            engine: Default::default(),
            http: None,
            metrics: None,
            log: Default::default(),
            service: Default::default(),
            net,
            cache: Default::default(),
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
        cfg.validate().unwrap();

        let engine = |i: usize| {
            let e = MockEngine::new(cfg.nodes[i].layer_range(), 40, DIM).with_vocab(vocab);
            match fail {
                Some((node, req)) if node == i => e.failing_on(req),
                _ => e,
            }
        };
        let mut listeners = listeners.into_iter();
        let coord_listener = listeners.next().unwrap();
        let workers = listeners
            .enumerate()
            .map(|(k, l)| {
                let i = k + 1;
                let (cfg, engine) = (cfg.clone(), engine(i));
                thread::spawn(move || {
                    let link = accept_controller(&cfg, i, l).unwrap();
                    Worker::new(engine, link, &cfg.nodes[i].name).run().unwrap()
                })
            })
            .collect();
        // The controller connects out; it listens on nothing.
        drop(coord_listener);
        let star = connect_workers(&cfg, Duration::from_secs(5)).unwrap();
        let (client, coordinator) = scheduler::spawn_with_metrics(engine(0), star, cfg.scheduler, metrics);
        Cluster { client, coordinator, workers }
    }

    /// Shuts down and checks that no node still holds request state (CL-9).
    fn stop_and_check_freed(self) {
        self.client.shutdown();
        assert_eq!(self.coordinator.join().unwrap().live_requests(), 0, "coordinator");
        for (i, w) in self.workers.into_iter().enumerate() {
            assert_eq!(w.join().unwrap().live_requests(), 0, "worker {}", i + 1);
        }
    }
}

/// Reference output: one node, all 40 layers, one request at a time, one position per step.
fn reference(prompts: &[Vec<u32>], max_new: usize) -> Vec<Vec<u32>> {
    reference_with_vocab(prompts, max_new, VOCAB)
}

fn reference_with_vocab(prompts: &[Vec<u32>], max_new: usize, vocab: u32) -> Vec<Vec<u32>> {
    let sched = SchedulerConfig { prefill_chunk: 1, ..Default::default() };
    let c = Cluster::start_full(&[[0, 40]], sched, None, None, vocab);
    prompts.iter().map(|p| c.client.generate(p.clone(), max_new).unwrap().tokens).collect()
}

#[test]
fn three_node_split_matches_single_node_exactly() {
    // NF-1 with the mock engine: f32 on the wire leaves the result bit-identical.
    let want = reference(&[prompt(0)], 16);
    let c = Cluster::start(THREE);
    let got = c.client.generate(prompt(0), 16).unwrap();
    assert_eq!(got.tokens, want[0]);
    assert_eq!(got.latencies.len(), 16);
    c.stop_and_check_freed();
}

#[test]
fn concurrent_requests_match_their_sequential_results() {
    for splits in [THREE, &[[0, 40]][..]] {
        let prompts: Vec<_> = (0..6).map(prompt).collect();
        let want = reference(&prompts, 12);
        let c = Cluster::start(splits);
        let rxs: Vec<_> = prompts.iter().map(|p| c.client.submit(p.clone(), 12)).collect();
        let got: Vec<_> = rxs.into_iter().map(|rx| collect(rx).unwrap().tokens).collect();
        assert_eq!(got, want, "splits {splits:?}");
        c.stop_and_check_freed();
    }
}

#[test]
fn pipelined_prefill_chunks_match_position_by_position() {
    // Prompts of 1 to 37 positions in chunks of 4, three chunks of a request in the cluster at
    // once, alongside other requests' chunks and decode steps.
    let prompts: Vec<Vec<u32>> =
        [37, 1, 4, 9, 23].iter().enumerate().map(|(i, &n)| (0..n).map(|k| 7 * k + 1000 * i as u32).collect()).collect();
    let want = reference(&prompts, 10);
    let sched = SchedulerConfig { prefill_chunk: 4, prefill_depth: 3, ..Default::default() };
    let c = Cluster::start_with(THREE, sched, None);
    let rxs: Vec<_> = prompts.iter().map(|p| c.client.submit(p.clone(), 10)).collect();
    let got: Vec<_> = rxs.into_iter().map(|rx| collect(rx).unwrap().tokens).collect();
    assert_eq!(got, want);
    c.stop_and_check_freed();
}

#[test]
fn worker_error_fails_only_its_request() {
    // Request ids are assigned in submission order from 1, so request 2 is the second submit.
    let c = Cluster::start_with(THREE, SchedulerConfig::default(), Some((1, 2)));
    let rxs: Vec<_> = (0..3).map(|i| c.client.submit(prompt(i), 8)).collect();
    let results: Vec<_> = rxs.into_iter().map(collect).collect();
    let err = results[1].as_ref().unwrap_err().to_string();
    assert!(err.contains("n1") && err.contains("request 2"), "{err}");
    assert!(results[0].is_ok() && results[2].is_ok());
    c.stop_and_check_freed();
}

#[test]
fn full_queue_rejects_and_admitted_requests_still_finish() {
    let sched = SchedulerConfig { max_in_flight: 1, max_queue: 1, ..Default::default() };
    let c = Cluster::start_with(THREE, sched, None);
    let rxs: Vec<_> = (0..4).map(|i| c.client.submit(prompt(i), 8)).collect();
    let results: Vec<_> = rxs.into_iter().map(collect).collect();
    let rejected = results.iter().filter(|r| matches!(r, Err(Error::Rejected(_)))).count();
    // One request in flight plus one queued at most; the rest are refused.
    assert!(rejected >= 2, "only {rejected} rejected");
    assert!(results.iter().all(|r| r.is_ok() || matches!(r, Err(Error::Rejected(_)))));
    c.stop_and_check_freed();
}

#[test]
fn dropping_the_receiver_cancels_and_frees_the_request() {
    let c = Cluster::start(THREE);
    let rx = c.client.submit(prompt(0), 10_000);
    for _ in 0..3 {
        assert!(matches!(rx.recv().unwrap(), Event::Token { .. }));
    }
    drop(rx);
    // The cluster keeps serving, and the cancelled request is freed everywhere.
    assert_eq!(c.client.generate(prompt(1), 4).unwrap().tokens, reference(&[prompt(1)], 4)[0]);
    c.stop_and_check_freed();
}

#[test]
fn rejects_empty_prompts() {
    let c = Cluster::start(THREE);
    assert!(matches!(c.client.generate(vec![], 4), Err(Error::Rejected(_))));
    c.stop_and_check_freed();
}

/// A controller and one worker that runs `fake` instead of a real worker loop.
fn with_fake_worker(sched: SchedulerConfig, fake: impl FnOnce(Connection) + Send + 'static) -> Client {
    let (l0, l1) = (TcpListener::bind("127.0.0.1:0").unwrap(), TcpListener::bind("127.0.0.1:0").unwrap());
    let node = |name: &str, l: &TcpListener, layers| NodeConfig {
        name: name.into(),
        addr: l.local_addr().unwrap().to_string(),
        listen: None,
        layers: Some(layers),
        speed: None,
        mem_cap_gib: None,
        prometheus: None,
    };
    let cfg = Config {
        hidden_dim: DIM,
        num_layers: 40,
        split: Default::default(),
        experts: Default::default(),
        scheduler: sched,
        engine: Default::default(),
        http: None,
        metrics: None,
        log: Default::default(),
        service: Default::default(),
        net: Default::default(),
        cache: Default::default(),
        nodes: vec![node("n0", &l0, [0, 20]), node("n1", &l1, [20, 40])],
    };
    let worker_cfg = cfg.clone();
    thread::spawn(move || fake(accept_controller(&worker_cfg, 1, l1).unwrap()));
    drop(l0);
    let star = connect_workers(&cfg, Duration::from_secs(5)).unwrap();
    scheduler::spawn(MockEngine::new(0..20, 40, DIM), star, sched).0
}

/// Reads frames until the first hidden state.
fn wait_for_hidden(link: &mut Connection) {
    let mut buf = Vec::new();
    while link.recv_frame(&mut buf).unwrap().msg_type != MsgType::Hidden {}
}

#[test]
fn crashed_worker_fails_requests_and_stops_accepting_work() {
    let client = with_fake_worker(SchedulerConfig::default(), |mut link| {
        wait_for_hidden(&mut link);
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
        wait_for_hidden(&mut link);
        // Hang with the connection still open, as a frozen machine would.
        thread::sleep(Duration::from_secs(3));
        drop(link);
    });
    let t0 = Instant::now();
    let err = client.generate(prompt(0), 8).unwrap_err().to_string();
    assert!(err.contains("health ping"), "{err}");
    assert!(t0.elapsed() < Duration::from_secs(2), "took {:?}", t0.elapsed());
}

#[test]
fn frequent_health_pings_do_not_disturb_generation() {
    let want = reference(&[prompt(0), prompt(1)], 200);
    let sched = SchedulerConfig { health_interval_ms: 1, ..Default::default() };
    let c = Cluster::start_with(THREE, sched, None);
    let rxs: Vec<_> = [prompt(0), prompt(1)].into_iter().map(|p| c.client.submit(p, 200)).collect();
    let got: Vec<_> = rxs.into_iter().map(|rx| collect(rx).unwrap().tokens).collect();
    assert_eq!(got, want);
    c.stop_and_check_freed();
}

#[test]
fn metrics_has_a_row_per_step_with_every_node_timed() {
    let dir = std::env::temp_dir().join(format!("msh-cluster-metrics-{}", std::process::id()));
    let path = dir.join("run.csv");
    let run = RunInfo {
        run_id: 1,
        git_hash: "test".into(),
        host: "h".into(),
        hardware: "hw".into(),
        split: "n0:0-20 n1:20-30 n2:30-40".into(),
        engine: "mock".into(),
    };
    let metrics = Metrics::create(&path, &run).unwrap();
    let sched = SchedulerConfig { prefill_chunk: 2, ..Default::default() };
    let c = Cluster::start_full(THREE, sched, None, Some(metrics), VOCAB);
    let rxs: Vec<_> = (0..2).map(|i| c.client.submit(prompt(i), 6)).collect();
    for rx in rxs {
        collect(rx).unwrap();
    }
    c.stop_and_check_freed();

    let text = std::fs::read_to_string(&path).unwrap();
    let rows: Vec<Vec<&str>> = text.lines().skip(1).map(|l| l.split(',').collect()).collect();
    // Two requests, each 5 prompt positions in chunks of 2, 2, 1 (the last one gives the first
    // token) and 5 more decode steps (the 6th token ends it).
    assert_eq!(rows.len(), 2 * 8);
    for r in &rows {
        let (kind, step, coord, workers, network, stages) = (r[9], r[11], r[12], r[13], r[14], r[15]);
        assert_eq!((r[19], r[20], r[21]), ("0", "0", "0"), "no reuse or drafting by default");
        assert_eq!(stages.split(';').count(), 3, "every node reports a time: {r:?}");
        let step: u64 = step.parse().unwrap();
        let total = coord.parse::<u64>().unwrap() + workers.parse::<u64>().unwrap() + network.parse::<u64>().unwrap();
        assert!(total >= step, "compute + network covers the step: {r:?}");
        assert!(["prefill", "first_token", "decode"].contains(&kind));
    }
    let count = |k| rows.iter().filter(|r| r[9] == k).count();
    assert_eq!((count("prefill"), count("first_token"), count("decode")), (4, 2, 10));
    let positions = |k| rows.iter().filter(|r| r[9] == k).map(|r| r[8].parse::<u32>().unwrap()).sum::<u32>();
    assert_eq!((positions("prefill"), positions("first_token")), (8, 2));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn speculative_decoding_matches_plain_decoding() {
    // Six token ids make the text repeat, so prompt lookup drafts often and the head both
    // confirms and rejects drafts.
    let vocab = 6;
    let prompts: Vec<Vec<u32>> = (0..4).map(|i| (0..20).map(|k| (k * (i + 1)) % 5).collect()).collect();
    let want = reference_with_vocab(&prompts, 60, vocab);
    for splits in [THREE, &[[0, 40]][..]] {
        let dir = std::env::temp_dir().join(format!("msh-spec-{}-{}", std::process::id(), splits.len()));
        let path = dir.join("run.csv");
        let metrics = Metrics::create(&path, &RunInfo {
            run_id: 1,
            git_hash: "test".into(),
            host: "h".into(),
            hardware: "hw".into(),
            split: format!("{} nodes", splits.len()),
            engine: "mock".into(),
        })
        .unwrap();
        let sched = SchedulerConfig { max_draft: 4, draft_min_ngram: 2, ..Default::default() };
        let c = Cluster::start_full(splits, sched, None, Some(metrics), vocab);
        let rxs: Vec<_> = prompts.iter().map(|p| c.client.submit(p.clone(), 60)).collect();
        let got: Vec<_> = rxs.into_iter().map(|rx| collect(rx).unwrap().tokens).collect();
        assert_eq!(got, want, "splits {splits:?}");
        c.stop_and_check_freed();

        let text = std::fs::read_to_string(&path).unwrap();
        let rows: Vec<Vec<&str>> = text.lines().skip(1).map(|l| l.split(',').collect()).collect();
        let sum = |col: usize| rows.iter().map(|r| r[col].parse::<u32>().unwrap()).sum::<u32>();
        let (drafted, accepted) = (sum(20), sum(21));
        let decode_steps = rows.iter().filter(|r| r[9] == "decode").count();
        assert!(accepted > 0 && accepted < drafted, "drafted {drafted}, accepted {accepted}");
        // Every accepted draft is a token that needed no step of its own.
        assert_eq!(decode_steps as u32 + accepted, 4 * 59, "splits {splits:?}");
        std::fs::remove_dir_all(dir).unwrap();
    }
}

#[test]
fn drafting_stops_at_max_tokens() {
    // A prompt that repeats exactly: drafts are long and right, and must not run past
    // max_new.
    let vocab = 6;
    let p: Vec<u32> = (0..30).map(|k| k % 3).collect();
    let want = reference_with_vocab(&[p.clone()], 5, vocab);
    let sched = SchedulerConfig { max_draft: 4, draft_min_ngram: 2, ..Default::default() };
    let c = Cluster::start_full(THREE, sched, None, None, vocab);
    assert_eq!(c.client.generate(p, 5).unwrap().tokens, want[0]);
    c.stop_and_check_freed();
}

/// Runs `turns` one after another on a three-node cluster and returns each one's tokens and the
/// positions it reused (the `cached` column of its first row).
fn chat_turns(sched: SchedulerConfig, vocab: u32, turns: &[(Vec<u32>, usize)]) -> Vec<(Vec<u32>, u32)> {
    static RUN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = RUN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("msh-prefix-{}-{n}", std::process::id()));
    let path = dir.join("run.csv");
    let _ = std::fs::remove_file(&path);
    let run = RunInfo {
        run_id: 1,
        git_hash: "test".into(),
        host: "h".into(),
        hardware: "hw".into(),
        split: "3 nodes".into(),
        engine: "mock".into(),
    };
    let c = Cluster::start_full(THREE, sched, None, Some(Metrics::create(&path, &run).unwrap()), vocab);
    let tokens: Vec<Vec<u32>> = turns.iter().map(|(p, n)| c.client.generate(p.clone(), *n).unwrap().tokens).collect();
    c.stop_and_check_freed();
    let text = std::fs::read_to_string(&path).unwrap();
    let rows: Vec<Vec<&str>> = text.lines().skip(1).map(|l| l.split(',').collect()).collect();
    // Requests run one at a time, and a reused one keeps the request id, so a turn starts at a
    // prompt step that follows a decode step.
    let mut cached = Vec::new();
    let mut decoding = true;
    for r in &rows {
        if r[9] != "decode" && decoding {
            cached.push(r[19].parse().unwrap());
        }
        decoding = r[9] == "decode";
    }
    std::fs::remove_dir_all(dir).unwrap();
    tokens.into_iter().zip(cached).collect()
}

#[test]
fn next_chat_turn_reuses_the_previous_one() {
    for (max_draft, vocab) in [(0, VOCAB), (4, 6)] {
        let p1: Vec<u32> = (0..12).map(|k| k % 5).collect();
        let g1 = reference_with_vocab(&[p1.clone()], 10, vocab).remove(0);
        // The next turn: the conversation so far plus a new message.
        let p2: Vec<u32> = [&p1[..], &g1[..], &[4, 3, 2, 1, 0]].concat();
        let other: Vec<u32> = (0..9).map(|k| (k + 2) % 5).collect();
        let want = reference_with_vocab(&[p2.clone(), other.clone()], 10, vocab);

        let sched = SchedulerConfig { prefix_cache: 2, max_draft, draft_min_ngram: 2, ..Default::default() };
        let got = chat_turns(sched, vocab, &[(p1.clone(), 10), (p2, 10), (other, 10)]);
        assert_eq!(got[0].0, g1);
        assert_eq!((&got[1].0, &got[2].0), (&want[0], &want[1]), "max_draft {max_draft}");
        // Turn 2 skips turn 1's prompt and all but its last token, which was never fed.
        assert_eq!(got[1].1, (p1.len() + g1.len() - 1) as u32, "max_draft {max_draft}");
        assert_eq!((got[0].1, got[2].1), (0, 0), "nothing to reuse");
    }
}

#[test]
fn oldest_parked_request_is_dropped_everywhere() {
    // One slot kept: the second conversation evicts the first, so its next turn starts over.
    let a: Vec<u32> = (0..8).collect();
    let b: Vec<u32> = (20..28).collect();
    let ga = reference(&[a.clone()], 4).remove(0);
    let a2: Vec<u32> = [&a[..], &ga[..], &[99]].concat();
    let want = reference(&[a2.clone()], 4).remove(0);
    let sched = SchedulerConfig { prefix_cache: 1, ..Default::default() };
    let got = chat_turns(sched, VOCAB, &[(a, 4), (b, 4), (a2, 4)]);
    assert_eq!(got[2], (want, 0));
}

#[test]
fn small_chunks_give_the_same_tokens() {
    // 4 KiB chunks: a prompt chunk of 16 positions (128 KiB) goes in 33 frames, between other
    // requests' frames, three chunks of a request in the cluster at once.
    let prompts: Vec<Vec<u32>> =
        [40, 3, 17, 33].iter().enumerate().map(|(i, &n)| (0..n).map(|k| 13 * k + 500 * i as u32).collect()).collect();
    let want = reference(&prompts, 8);
    let sched = SchedulerConfig { prefill_chunk: 16, prefill_depth: 3, ..Default::default() };
    let net = NetConfig { max_chunk_kib: 4, ..Default::default() };
    let c = Cluster::start_net(THREE, sched, None, None, VOCAB, net);
    let rxs: Vec<_> = prompts.iter().map(|p| c.client.submit(p.clone(), 8)).collect();
    let got: Vec<_> = rxs.into_iter().map(|rx| collect(rx).unwrap().tokens).collect();
    assert_eq!(got, want);
    c.stop_and_check_freed();
}

/// Forwards connections to `to` and can cut them all, as a pulled cable would.
struct Proxy {
    addr: String,
    conns: Arc<Mutex<Vec<TcpStream>>>,
}

impl Proxy {
    fn start(to: SocketAddr) -> Proxy {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap().to_string();
        let conns: Arc<Mutex<Vec<TcpStream>>> = Arc::default();
        let held = conns.clone();
        thread::spawn(move || {
            for down in l.incoming() {
                let down = down.unwrap();
                let up = TcpStream::connect(to).unwrap();
                held.lock().unwrap().extend([down.try_clone().unwrap(), up.try_clone().unwrap()]);
                for (mut from, mut to) in [(down.try_clone().unwrap(), up.try_clone().unwrap()), (up, down)] {
                    thread::spawn(move || {
                        let _ = io::copy(&mut from, &mut to);
                        let _ = to.shutdown(Shutdown::Write);
                    });
                }
            }
        });
        Proxy { addr, conns }
    }

    fn cut(&self) {
        for c in self.conns.lock().unwrap().drain(..) {
            let _ = c.shutdown(Shutdown::Both);
        }
    }
}

/// A layer-split config: node `i` at `addrs[i]` runs `splits[i]`.
fn config(addrs: &[String], splits: &[[u32; 2]], sched: SchedulerConfig, net: NetConfig) -> Config {
    let cfg = Config {
        hidden_dim: DIM,
        num_layers: 40,
        split: Default::default(),
        experts: Default::default(),
        scheduler: sched,
        engine: Default::default(),
        http: None,
        metrics: None,
        log: Default::default(),
        service: Default::default(),
        net,
        cache: Default::default(),
        nodes: addrs
            .iter()
            .zip(splits)
            .enumerate()
            .map(|(i, (addr, layers))| NodeConfig {
                name: format!("n{i}"),
                addr: addr.clone(),
                listen: None,
                layers: Some(*layers),
                speed: None,
                mem_cap_gib: None,
                prometheus: None,
            })
            .collect(),
    };
    cfg.validate().unwrap();
    cfg
}

fn bind() -> (TcpListener, String) {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap().to_string();
    (l, addr)
}

/// Worker `i` on `l`, serving one controller after another, as `msh-node` does.
fn worker_forever(cfg: &Config, i: usize, l: TcpListener) {
    let cfg = cfg.clone();
    thread::spawn(move || {
        let e = MockEngine::new(cfg.nodes[i].layer_range(), 40, DIM);
        Worker::idle(e, &cfg.nodes[i].name).serve_forever(&cfg, i, &l).unwrap();
    });
}

/// Generates once the cluster takes requests again.
fn once_up(client: &Client, p: Vec<u32>, n: usize) -> Vec<u32> {
    let t0 = Instant::now();
    loop {
        match client.generate(p.clone(), n) {
            Ok(g) => return g.tokens,
            Err(Error::Rejected(m)) if m.contains("cluster down") => {
                assert!(t0.elapsed() < Duration::from_secs(10), "still down: {m}");
                thread::sleep(Duration::from_millis(10));
            }
            Err(e) => panic!("{e}"),
        }
    }
}

fn quick() -> (SchedulerConfig, NetConfig) {
    let sched = SchedulerConfig { health_interval_ms: 20, health_timeout_ms: 1000, ..Default::default() };
    (sched, NetConfig { reconnect_min_ms: 20, reconnect_max_ms: 100, ..Default::default() })
}

#[test]
fn a_cut_link_takes_the_cluster_down_until_the_worker_is_back() {
    let (sched, net) = quick();
    let want = reference(&[prompt(0), prompt(1)], 6);
    let (nodes, mut addrs): (Vec<_>, Vec<_>) = (0..3).map(|_| bind()).unzip();
    let proxy = Proxy::start(addrs[2].parse().unwrap());
    addrs[2] = proxy.addr.clone();
    let cfg = config(&addrs, THREE, sched, net);
    for (i, l) in nodes.into_iter().enumerate().skip(1) {
        worker_forever(&cfg, i, l);
    }
    let star = connect_workers(&cfg, Duration::from_secs(5)).unwrap();
    let (client, _sched) = scheduler::spawn(MockEngine::new(0..20, 40, DIM), star, sched);
    assert_eq!(client.generate(prompt(0), 6).unwrap().tokens, want[0]);

    // A long generation in flight when the cable to n2 is pulled: it fails, its state is gone.
    let rx = client.submit(prompt(2), 100_000);
    assert!(matches!(rx.recv().unwrap(), Event::Token { .. }));
    proxy.cut();
    let err = collect(rx).unwrap_err().to_string();
    assert!(err.contains("cluster down"), "{err}");
    // Every worker is reconnected, from a clean state, and the cluster serves again by itself.
    assert_eq!(once_up(&client, prompt(1), 6), want[1]);
    assert_eq!(client.generate(prompt(0), 6).unwrap().tokens, want[0]);
}

#[test]
fn a_crashed_worker_restarted_on_its_address_rejoins() {
    let (sched, net) = quick();
    let ((l0, a0), (l1, a1)) = (bind(), bind());
    let cfg = config(&[a0, a1.clone()], &[[0, 20], [20, 40]], sched, net);
    drop(l0);
    let worker_cfg = cfg.clone();
    // n1 crashes at its first hidden state; its listener is gone with it.
    thread::spawn(move || wait_for_hidden(&mut accept_controller(&worker_cfg, 1, l1).unwrap()));
    let star = connect_workers(&cfg, Duration::from_secs(5)).unwrap();
    let (client, _sched) = scheduler::spawn(MockEngine::new(0..20, 40, DIM), star, sched);
    let err = client.generate(prompt(0), 8).unwrap_err().to_string();
    assert!(err.contains("cluster down"), "{err}");
    let err = client.generate(prompt(0), 8).unwrap_err();
    assert!(matches!(&err, Error::Rejected(m) if m.contains("cluster down")), "{err}");
    // Restarted on the same address: the controller finds it.
    thread::sleep(Duration::from_millis(100));
    worker_forever(&cfg, 1, TcpListener::bind(&a1).unwrap());
    assert_eq!(once_up(&client, prompt(1), 8), reference(&[prompt(1)], 8)[0]);
}

#[test]
fn a_worker_drops_a_silent_controller_and_takes_the_next() {
    let sched = SchedulerConfig { health_interval_ms: 20, health_timeout_ms: 200, ..Default::default() };
    let ((_l0, a0), (l1, a1)) = (bind(), bind());
    let cfg = config(&[a0, a1.clone()], &[[0, 20], [20, 40]], sched, NetConfig::default());
    worker_forever(&cfg, 1, l1);
    // A controller that says `Hello`, then nothing, as a frozen machine would.
    let mut frozen = Connection::connect(a1.as_str(), "n0", Duration::from_secs(1)).unwrap();
    let t0 = Instant::now();
    match frozen.recv_frame(&mut Vec::new()) {
        Err(FrameError::Io(e)) if e.kind() == io::ErrorKind::UnexpectedEof => {}
        other => panic!("expected the worker to drop the link, got {other:?}"),
    }
    assert!(t0.elapsed() >= Duration::from_millis(150) && t0.elapsed() < Duration::from_secs(3), "{:?}", t0.elapsed());
    // The next controller, pinging, is served.
    let star = connect_workers(&cfg, Duration::from_secs(5)).unwrap();
    let (client, _sched) = scheduler::spawn(MockEngine::new(0..20, 40, DIM), star, sched);
    assert_eq!(client.generate(prompt(3), 8).unwrap().tokens, reference(&[prompt(3)], 8)[0]);
    // It keeps the link while pings come.
    thread::sleep(Duration::from_millis(500));
    assert_eq!(client.generate(prompt(4), 4).unwrap().tokens, reference(&[prompt(4)], 4)[0]);
}
