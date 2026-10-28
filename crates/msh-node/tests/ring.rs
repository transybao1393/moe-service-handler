use std::net::TcpListener;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use msh_net::MsgType;
use msh_node::config::{Config, NodeConfig};
use msh_node::engine::MockEngine;
use msh_node::metrics::{Metrics, RunInfo};
use msh_node::node::{join_ring, Ring, Worker};
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
        let listeners: Vec<TcpListener> =
            splits.iter().map(|_| TcpListener::bind("127.0.0.1:0").unwrap()).collect();
        let cfg = Config {
            hidden_dim: DIM,
            num_layers: 40,
            scheduler: sched,
            engine: Default::default(),
            http: None,
            metrics: None,
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
                    let ring = join_ring(&cfg, i, l, Duration::from_secs(5)).unwrap().unwrap();
                    Worker::new(engine, ring, &cfg.nodes[i].name).run().unwrap()
                })
            })
            .collect();
        let ring = join_ring(&cfg, 0, coord_listener, Duration::from_secs(5)).unwrap();
        let (client, coordinator) = scheduler::spawn_with_metrics(engine(0), ring, cfg.scheduler, metrics);
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
    // Prompts of 1 to 37 positions in chunks of 4, three chunks of a request in the ring at
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
    // The ring keeps serving, and the cancelled request is freed everywhere.
    assert_eq!(c.client.generate(prompt(1), 4).unwrap().tokens, reference(&[prompt(1)], 4)[0]);
    c.stop_and_check_freed();
}

#[test]
fn rejects_empty_prompts() {
    let c = Cluster::start(THREE);
    assert!(matches!(c.client.generate(vec![], 4), Err(Error::Rejected(_))));
    c.stop_and_check_freed();
}

/// A two-node ring whose second node runs `fake` instead of a real worker.
fn with_fake_worker(sched: SchedulerConfig, fake: impl FnOnce(Ring) + Send + 'static) -> Client {
    let (l0, l1) = (TcpListener::bind("127.0.0.1:0").unwrap(), TcpListener::bind("127.0.0.1:0").unwrap());
    let node = |name: &str, l: &TcpListener, layers| NodeConfig {
        name: name.into(),
        addr: l.local_addr().unwrap().to_string(),
        listen: None,
        layers,
        mem_cap_gib: None,
    };
    let cfg = Config {
        hidden_dim: DIM,
        num_layers: 40,
        scheduler: sched,
        engine: Default::default(),
        http: None,
        metrics: None,
        nodes: vec![node("n0", &l0, [0, 20]), node("n1", &l1, [20, 40])],
    };
    let worker_cfg = cfg.clone();
    thread::spawn(move || fake(join_ring(&worker_cfg, 1, l1, Duration::from_secs(5)).unwrap().unwrap()));
    let ring = join_ring(&cfg, 0, l0, Duration::from_secs(5)).unwrap();
    scheduler::spawn(MockEngine::new(0..20, 40, DIM), ring, sched).0
}

/// Reads frames until the first hidden state.
fn wait_for_hidden(ring: &mut Ring) {
    let mut buf = Vec::new();
    while ring.prev.recv_frame(&mut buf).unwrap().msg_type != MsgType::Hidden {}
}

#[test]
fn crashed_worker_fails_requests_and_stops_accepting_work() {
    let client = with_fake_worker(SchedulerConfig::default(), |mut ring| {
        wait_for_hidden(&mut ring);
        // Returning drops both connections, as a crashed process would.
    });
    let err = client.generate(prompt(0), 8).unwrap_err().to_string();
    assert!(err.contains("ring down"), "{err}");
    let err = client.generate(prompt(1), 8).unwrap_err();
    assert!(matches!(&err, Error::Rejected(m) if m.contains("ring down")), "{err}");
}

#[test]
fn hung_worker_is_detected_by_health_ping() {
    let sched = SchedulerConfig { health_interval_ms: 20, health_timeout_ms: 200, ..Default::default() };
    let client = with_fake_worker(sched, |mut ring| {
        wait_for_hidden(&mut ring);
        // Hang with the connections still open, as a frozen machine would.
        thread::sleep(Duration::from_secs(3));
        drop(ring);
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
    let dir = std::env::temp_dir().join(format!("msh-ring-metrics-{}", std::process::id()));
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

/// Runs `turns` one after another on a three-node ring and returns each one's tokens and the
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
