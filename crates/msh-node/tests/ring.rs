use std::net::TcpListener;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use msh_net::MsgType;
use msh_node::config::{Config, NodeConfig};
use msh_node::engine::MockEngine;
use msh_node::node::{join_ring, Ring, Worker};
use msh_node::scheduler::{self, collect, Client, Event, SchedulerConfig};
use msh_node::Error;

const DIM: usize = 2048;
const THREE: &[[u32; 2]] = &[[0, 20], [20, 30], [30, 40]];

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
        let listeners: Vec<TcpListener> =
            splits.iter().map(|_| TcpListener::bind("127.0.0.1:0").unwrap()).collect();
        let cfg = Config {
            hidden_dim: DIM,
            num_layers: 40,
            scheduler: sched,
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

        let engine = |i: usize| {
            let e = MockEngine::new(cfg.nodes[i].layer_range(), 40, DIM);
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
        let (client, coordinator) = scheduler::spawn(engine(0), ring, cfg.scheduler);
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

/// Reference output: one node, all 40 layers, one request at a time.
fn reference(prompts: &[Vec<u32>], max_new: usize) -> Vec<Vec<u32>> {
    let c = Cluster::start(&[[0, 40]]);
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
    };
    let cfg = Config {
        hidden_dim: DIM,
        num_layers: 40,
        scheduler: sched,
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
