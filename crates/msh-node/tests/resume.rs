//! Resume on loss (`[scheduler] resume_on_loss`) on the mock engine: requests whose worker is
//! lost mid-answer go on elsewhere and give, token for token, what one process gives.

use std::io;
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use msh_net::{decode_start, encode_done, encode_status, encode_token, DoneReason, MsgType, Status};
use msh_node::config::{Config, NodeConfig, Split};
use msh_node::engine::{Engine, MockEngine, Sample, StageIn};
use msh_node::node::{accept_controller, RequestWorker};
use msh_node::requests::{self, Board, WorkerState};
use msh_node::scheduler::{self, collect, Client, Event, SchedulerConfig};
use msh_node::stats::STATS;
use msh_node::Error;

const DIM: usize = 64;
const LAYERS: u32 = 8;

fn mock() -> MockEngine {
    MockEngine::new(0..LAYERS, LAYERS, DIM).with_vocab(1000)
}

fn prompt(i: u32) -> Vec<u32> {
    (0..5 + i % 4).map(|k| 11 * (k + 1) + 100 * i).collect()
}

/// Reference output: one process, one request at a time, one position per step.
fn reference(prompts: &[Vec<u32>], max_new: usize) -> Vec<Vec<u32>> {
    let sched = SchedulerConfig { prefill_chunk: 1, ..Default::default() };
    let (client, _) = scheduler::spawn(mock(), None, sched);
    prompts.iter().map(|p| client.generate(p.clone(), max_new).unwrap().tokens).collect()
}

/// The mock, a little slower per step, so requests are still running when a test acts; with
/// an end-of-sequence token if given.
struct Slow(MockEngine, Duration, Option<u32>);

impl Engine for Slow {
    fn hidden_dim(&self) -> usize {
        self.0.hidden_dim()
    }
    fn eos(&self) -> Option<u32> {
        self.2
    }
    fn has_head(&self) -> bool {
        self.0.has_head()
    }
    fn step(&mut self, req: u64, pos: u32, input: StageIn<'_>, s: Sample, h: &mut Vec<f32>, t: &mut Vec<u32>) -> Result<(), Error> {
        thread::sleep(self.1);
        self.0.step(req, pos, input, s, h, t)
    }
    fn free(&mut self, req: u64) {
        self.0.free(req)
    }
}

fn slow(ms: u64) -> Slow {
    Slow(mock(), Duration::from_millis(ms), None)
}

/// A request-split config: controller `n0`, then a worker `n1`, `n2`, ... at each address.
fn config(addrs: &[String], sched: SchedulerConfig) -> Config {
    let node = |i: usize, addr: &str| NodeConfig {
        name: format!("n{i}"),
        addr: addr.into(),
        listen: None,
        layers: None,
        speed: None,
        mem_cap_gib: None,
        prometheus: None,
    };
    let mut nodes = vec![node(0, "127.0.0.1:1")];
    nodes.extend(addrs.iter().enumerate().map(|(k, a)| node(k + 1, a)));
    let cfg = Config {
        hidden_dim: DIM,
        num_layers: LAYERS,
        split: Split::Requests,
        experts: Default::default(),
        scheduler: sched,
        engine: Default::default(),
        http: None,
        metrics: None,
        log: Default::default(),
        service: Default::default(),
        net: Default::default(),
        cache: Default::default(),
        nodes,
    };
    cfg.validate().unwrap();
    cfg
}

fn listeners(n: usize) -> (Vec<TcpListener>, Vec<String>) {
    let ls: Vec<TcpListener> = (0..n).map(|_| TcpListener::bind("127.0.0.1:0").unwrap()).collect();
    let addrs = ls.iter().map(|l| l.local_addr().unwrap().to_string()).collect();
    (ls, addrs)
}

/// Worker `i` on `l`: serves `links` controller connections in turn, then hands back its engine.
fn worker<E: Engine + 'static>(cfg: &Config, i: usize, l: TcpListener, engine: E, links: usize) -> JoinHandle<E> {
    let cfg = cfg.clone();
    thread::spawn(move || {
        let mut w = RequestWorker::new(engine, &cfg.nodes[i].name, cfg.scheduler.prefill_chunk, None);
        for _ in 0..links {
            let link = accept_controller(&cfg, i, l.try_clone().unwrap()).unwrap();
            let _ = w.serve(link);
        }
        w.into_engine()
    })
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

/// `n` workers, each behind a proxy and serving `links` links, and a controller over them.
fn cluster(n: usize, links: usize, sched: SchedulerConfig) -> (Client, JoinHandle<()>, Vec<Proxy>, Vec<JoinHandle<Slow>>) {
    let (ls, real) = listeners(n);
    let proxies: Vec<Proxy> = real.iter().map(|a| Proxy::start(a.parse().unwrap())).collect();
    let addrs: Vec<String> = proxies.iter().map(|p| p.addr.clone()).collect();
    let cfg = config(&addrs, sched);
    let workers = ls.into_iter().enumerate().map(|(k, l)| worker(&cfg, k + 1, l, slow(2), links)).collect();
    let (client, sched) = requests::spawn(&cfg).unwrap();
    assert_eq!(board(&client).wait_ready(n, Duration::from_secs(5)), n);
    (client, sched, proxies, workers)
}

fn until(what: &str, f: impl Fn() -> bool) {
    let t0 = Instant::now();
    while !f() {
        assert!(t0.elapsed() < Duration::from_secs(10), "timed out waiting for {what}");
        thread::sleep(Duration::from_millis(2));
    }
}

fn board(c: &Client) -> &Board {
    c.board().expect("request split")
}

fn next_token(rx: &mpsc::Receiver<Event>) -> u32 {
    match rx.recv_timeout(Duration::from_secs(10)).unwrap() {
        Event::Token { token, .. } => token,
        other => panic!("expected a token, got {other:?}"),
    }
}

/// Reads `n` more tokens into `got`.
fn take(rx: &mpsc::Receiver<Event>, n: usize, got: &mut Vec<u32>) {
    for _ in 0..n {
        got.push(next_token(rx));
    }
}

/// Moves new work from worker `from` to worker `to`, then cuts `from`'s link.
fn lose(b: &Board, proxies: &[Proxy], from: usize, to: usize) {
    b.drain(&format!("n{from}")).unwrap();
    b.resume(&format!("n{to}")).unwrap();
    proxies[from - 1].cut();
}

fn on(b: &Board, id: u64) -> (Option<String>, u32) {
    b.request(id).map_or((None, u32::MAX), |r| (r.worker, r.resumed))
}

#[test]
fn several_requests_resume_on_the_other_worker_with_the_reference_tokens() {
    let prompts: Vec<_> = (0..3).map(prompt).collect();
    let want = reference(&prompts, 150);
    let sched = SchedulerConfig { max_in_flight: 4, health_interval_ms: 20, ..Default::default() };
    let (client, sched, proxies, _workers) = cluster(2, 2, sched);
    let b = board(&client);
    b.drain("n2").unwrap();
    let reqs: Vec<_> = prompts.iter().map(|p| client.submit_with_id(p.clone(), 150)).collect();
    let mut got = vec![Vec::new(); 3];
    for (k, (id, rx)) in reqs.iter().enumerate() {
        take(rx, 3 + k, &mut got[k]);
        assert_eq!(on(b, *id), (Some("n1".into()), 0));
    }
    let resumed_before = STATS.requests_resumed.get();
    lose(b, &proxies, 1, 2);

    // Each goes on on n2, moved once, keeping at least what the client already has.
    for (k, (id, _)) in reqs.iter().enumerate() {
        until("moved to n2", || on(b, *id) == (Some("n2".into()), 1));
        assert!(b.request(*id).unwrap().tokens_sent as usize >= got[k].len());
    }
    for (k, (_, rx)) in reqs.into_iter().enumerate() {
        got[k].extend(collect(rx).unwrap().tokens);
    }
    assert_eq!(got, want, "no token repeated or missed");
    // Other tests in this binary may resume too.
    assert!(STATS.requests_resumed.get() - resumed_before >= 3);
    let (n1, n2) = (b.worker("n1").unwrap(), b.worker("n2").unwrap());
    assert_eq!((n1.done, n1.failed, n2.done), (0, 0, 3));
    client.shutdown();
    sched.join().unwrap();
}

#[test]
fn moved_twice_finishes_and_a_third_loss_fails() {
    let want = reference(&[prompt(1)], 400)[0].clone();
    let sched = SchedulerConfig { max_in_flight: 2, health_interval_ms: 20, ..Default::default() };
    let (client, sched, proxies, _workers) = cluster(3, 3, sched);
    let b = board(&client);
    let up = |n: &str| until(&format!("{n} back"), || b.worker(n).unwrap().state != WorkerState::Down);

    // n1, then n2, then n3: two moves, and it ends as if nothing happened.
    b.drain("n2").unwrap();
    b.drain("n3").unwrap();
    let (a, a_rx) = client.submit_with_id(prompt(1), 400);
    let mut got = Vec::new();
    take(&a_rx, 5, &mut got);
    assert_eq!(on(b, a), (Some("n1".into()), 0));
    lose(b, &proxies, 1, 2);
    take(&a_rx, 5, &mut got);
    assert_eq!(on(b, a), (Some("n2".into()), 1));
    lose(b, &proxies, 2, 3);
    take(&a_rx, 5, &mut got);
    assert_eq!(on(b, a), (Some("n3".into()), 2));
    got.extend(collect(a_rx).unwrap().tokens);
    assert_eq!(got, want);

    // n3, then n1, then n2, then lost a third time: it fails.
    up("n1");
    up("n2");
    let (c, c_rx) = client.submit_with_id(prompt(2), 400);
    let mut got = Vec::new();
    take(&c_rx, 5, &mut got);
    assert_eq!(on(b, c), (Some("n3".into()), 0));
    lose(b, &proxies, 3, 1);
    take(&c_rx, 5, &mut got);
    assert_eq!(on(b, c), (Some("n1".into()), 1));
    lose(b, &proxies, 1, 2);
    take(&c_rx, 5, &mut got);
    assert_eq!(on(b, c), (Some("n2".into()), 2));
    up("n3");
    lose(b, &proxies, 2, 3);
    let err = collect(c_rx).unwrap_err().to_string();
    assert!(err.contains("worker n2 lost after"), "{err}");
    assert_eq!(got, reference(&[prompt(2)], 400)[0][..15]);
    assert!(b.request(c).is_none());
    client.shutdown();
    sched.join().unwrap();
}

#[test]
fn without_resume_on_loss_started_requests_fail() {
    let sched = SchedulerConfig { max_in_flight: 4, health_interval_ms: 20, resume_on_loss: false, ..Default::default() };
    let (client, sched, proxies, _workers) = cluster(2, 2, sched);
    let b = board(&client);
    b.drain("n2").unwrap();
    let reqs: Vec<_> = (0..2).map(|i| client.submit_with_id(prompt(i), 300)).collect();
    for (_, rx) in &reqs {
        next_token(rx);
    }
    lose(b, &proxies, 1, 2);
    for (id, rx) in reqs {
        let err = collect(rx).unwrap_err().to_string();
        assert!(err.contains("worker n1 lost after"), "{err}");
        assert!(b.request(id).is_none());
    }
    let n1 = b.worker("n1").unwrap();
    assert_eq!(n1.failed, 2);
    assert_eq!(b.worker("n2").unwrap().done, 0);
    client.shutdown();
    sched.join().unwrap();
}

#[test]
fn a_resumed_answer_is_cached_under_the_original_prompt() {
    let want = reference(&[prompt(5)], 120)[0].clone();
    let sched = SchedulerConfig { max_in_flight: 2, health_interval_ms: 20, ..Default::default() };
    let (client, sched, proxies, _workers) = cluster(2, 2, sched);
    let b = board(&client);
    b.drain("n2").unwrap();
    let (a, a_rx) = client.submit_with_id(prompt(5), 120);
    let mut got = Vec::new();
    take(&a_rx, 4, &mut got);
    lose(b, &proxies, 1, 2);
    until("moved to n2", || on(b, a).1 == 1);
    got.extend(collect(a_rx).unwrap().tokens);
    assert_eq!(got, want);
    // The scheduler fills the cache just after the client hears `Done`.
    until("the answer cached", || b.cache().entries == 1);

    let again = client.generate(prompt(5), 120).unwrap().tokens;
    assert_eq!(again, want);
    let s = b.cache();
    assert_eq!((s.entries, s.hits.exact, s.misses.exact), (1, 1, 1));
    assert_eq!(b.worker("n2").unwrap().done, 1, "the second answer came from the cache");
    client.shutdown();
    sched.join().unwrap();
}

#[test]
fn the_new_start_carries_the_tokens_so_far_and_the_limit_left() {
    let max_new = 60;
    let want = reference(&[prompt(3)], max_new)[0].clone();
    let sched = SchedulerConfig { max_in_flight: 2, health_interval_ms: 20, ..Default::default() };
    let (ls, real) = listeners(2);
    let proxy = Proxy::start(real[0].parse().unwrap());
    let cfg = config(&[proxy.addr.clone(), real[1].clone()], sched);
    let mut ls = ls.into_iter();
    worker(&cfg, 1, ls.next().unwrap(), slow(2), 2);
    // n2 records the `Start` it gets and answers it with tokens 900, 901, ... up to its limit.
    let (start_tx, starts) = mpsc::channel();
    let fake_cfg = cfg.clone();
    let l2 = ls.next().unwrap();
    thread::spawn(move || {
        let mut link = accept_controller(&fake_cfg, 2, l2).unwrap();
        let mut buf = Vec::new();
        let mut out = Vec::new();
        while let Ok(h) = link.recv_frame(&mut buf) {
            match h.msg_type {
                MsgType::Ping => {
                    encode_status(&Status { footprint: 1 << 30, cap: 2 << 30, requests: vec![] }, &mut out);
                    link.send(MsgType::Pong, h.request_id, 0, &out).unwrap();
                }
                MsgType::Start => {
                    let mut p = Vec::new();
                    let left = decode_start(&buf, &mut p).unwrap();
                    for k in 0..left {
                        encode_token(&[900 + k], &[1], &mut out);
                        link.send(MsgType::Token, h.request_id, p.len() as u32 + k, &out).unwrap();
                    }
                    encode_done(DoneReason::Length, &mut out);
                    link.send(MsgType::Done, h.request_id, left, &out).unwrap();
                    start_tx.send((left, p)).unwrap();
                }
                _ => {}
            }
        }
    });
    let (client, sched) = requests::spawn(&cfg).unwrap();
    let b = board(&client);
    assert_eq!(b.wait_ready(2, Duration::from_secs(5)), 2);
    b.drain("n2").unwrap();
    let (_, rx) = client.submit_with_id(prompt(3), max_new);
    let mut got = Vec::new();
    take(&rx, 7, &mut got);
    b.drain("n1").unwrap();
    b.resume("n2").unwrap();
    proxy.cut();
    got.extend(collect(rx).unwrap().tokens);

    let (left, p) = starts.recv_timeout(Duration::from_secs(5)).unwrap();
    let kept = p.len() - prompt(3).len();
    assert!(kept >= 7, "kept {kept}");
    assert_eq!(p, [prompt(3).as_slice(), &want[..kept]].concat(), "the prompt and the tokens so far");
    assert_eq!(left as usize, max_new - kept);
    assert_eq!(got.len(), max_new, "the original limit in all");
    assert_eq!(got[..kept], want[..kept]);
    assert_eq!(got[kept..], (900..900 + left).collect::<Vec<_>>()[..]);
    client.shutdown();
    sched.join().unwrap();
}

#[test]
fn a_resumed_request_still_stops_at_the_end_of_sequence() {
    // The first time token 40 shows up is the end of sequence.
    let plain = reference(&[prompt(6)], 200)[0].clone();
    let eos = plain[40];
    assert!(!plain[..40].contains(&eos), "pick another position");
    let sched = SchedulerConfig { max_in_flight: 2, health_interval_ms: 20, ..Default::default() };
    let (ls, real) = listeners(2);
    let proxy = Proxy::start(real[0].parse().unwrap());
    let cfg = config(&[proxy.addr.clone(), real[1].clone()], sched);
    let engine = || Slow(mock(), Duration::from_millis(2), Some(eos));
    let mut ls = ls.into_iter();
    worker(&cfg, 1, ls.next().unwrap(), engine(), 2);
    worker(&cfg, 2, ls.next().unwrap(), engine(), 1);
    let (client, sched) = requests::spawn(&cfg).unwrap();
    let b = board(&client);
    assert_eq!(b.wait_ready(2, Duration::from_secs(5)), 2);
    b.set_eos(Some(eos));
    b.drain("n2").unwrap();
    let (_, rx) = client.submit_with_id(prompt(6), 200);
    let mut got = Vec::new();
    take(&rx, 10, &mut got);
    b.drain("n1").unwrap();
    b.resume("n2").unwrap();
    proxy.cut();
    got.extend(collect(rx).unwrap().tokens);
    assert_eq!(got, plain[..40]);
    assert_eq!(b.worker("n2").unwrap().done, 1);
    client.shutdown();
    sched.join().unwrap();
}

#[test]
fn a_link_lost_after_the_end_of_sequence_ends_the_request_as_done() {
    // `RequestWorker` never sends the end-of-sequence token as a `Token` (it sends `Done`
    // instead), so a fake worker plays one that does and loses its link before its `Done`.
    let sched = SchedulerConfig { max_in_flight: 2, health_interval_ms: 20, ..Default::default() };
    let (ls, addrs) = listeners(2);
    let cfg = config(&addrs, sched);
    let mut ls = ls.into_iter();
    let (go_tx, go) = mpsc::channel::<()>();
    let fake_cfg = cfg.clone();
    let l1 = ls.next().unwrap();
    thread::spawn(move || {
        let mut link = accept_controller(&fake_cfg, 1, l1).unwrap();
        let mut buf = Vec::new();
        let mut out = Vec::new();
        while let Ok(h) = link.recv_frame(&mut buf) {
            match h.msg_type {
                MsgType::Ping => {
                    encode_status(&Status { footprint: 1 << 30, cap: 2 << 30, requests: vec![] }, &mut out);
                    link.send(MsgType::Pong, h.request_id, 0, &out).unwrap();
                }
                MsgType::Start => {
                    let mut p = Vec::new();
                    decode_start(&buf, &mut p).unwrap();
                    for k in 0..3 {
                        encode_token(&[40 + k], &[1], &mut out);
                        link.send(MsgType::Token, h.request_id, p.len() as u32 + k, &out).unwrap();
                    }
                    let _ = go.recv();
                    return;
                }
                _ => {}
            }
        }
    });
    worker(&cfg, 2, ls.next().unwrap(), slow(1), 1);
    let (client, sched) = requests::spawn(&cfg).unwrap();
    let b = board(&client);
    assert_eq!(b.wait_ready(2, Duration::from_secs(5)), 2);
    b.set_eos(Some(42));
    b.drain("n2").unwrap();
    let (a, rx) = client.submit_with_id(prompt(7), 50);
    let mut got = Vec::new();
    take(&rx, 3, &mut got);
    b.resume("n2").unwrap();
    go_tx.send(()).unwrap();
    got.extend(collect(rx).unwrap().tokens);
    assert_eq!(got, [40, 41, 42]);
    assert!(b.request(a).is_none());
    assert_eq!(b.worker("n2").unwrap().done, 0, "not resumed past it");
    client.shutdown();
    sched.join().unwrap();
}
