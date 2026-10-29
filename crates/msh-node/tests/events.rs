//! Worker events and controller commands on the request split, end to end over loopback links
//! with the mock engine: `PrefillDone` moves a request to decode, `MemoryHigh`/`MemoryOk` keep
//! new work off a worker, and `Board::command` gets each command's `Result`.

use std::net::TcpListener;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use msh_net::{
    decode_command, decode_start, encode_event, encode_status, CommandKind, Connection, EventKind, MsgType,
    RunStatus, Status,
};
use msh_node::config::{Config, NodeConfig, Split};
use msh_node::engine::{Engine, MockEngine, Sample, StageIn};
use msh_node::node::{accept_controller, RequestWorker};
use msh_node::requests::{self, Board, CommandOutcome, WorkerState};
use msh_node::scheduler::{collect, Client, SchedulerConfig};
use msh_node::Error;

const DIM: usize = 64;
const LAYERS: u32 = 8;

fn mock() -> MockEngine {
    MockEngine::new(0..LAYERS, LAYERS, DIM)
}

/// Distinct prompts, so the answer cache never answers one.
fn prompt(i: u32) -> Vec<u32> {
    (0..5 + i % 4).map(|k| 11 * (k + 1) + 100 * i + 7).collect()
}

/// The mock, a little slower per step, so requests are still running when a test acts.
struct Slow(MockEngine, Duration);

impl Engine for Slow {
    fn hidden_dim(&self) -> usize {
        self.0.hidden_dim()
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
    Slow(mock(), Duration::from_millis(ms))
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

/// Worker `i` on `l` with memory cap `cap`: serves one controller connection, then hands back
/// its engine.
fn worker<E: Engine + 'static>(cfg: &Config, i: usize, l: TcpListener, engine: E, cap: Option<u64>) -> JoinHandle<E> {
    let cfg = cfg.clone();
    thread::spawn(move || {
        let mut w = RequestWorker::new(engine, &cfg.nodes[i].name, cfg.scheduler.prefill_chunk, cap);
        let link = accept_controller(&cfg, i, l).unwrap();
        w.serve(link).unwrap();
        w.into_engine()
    })
}

/// A fake worker: answers `Hello`, then hands every frame to `f` until it returns false.
fn fake(cfg: &Config, i: usize, l: TcpListener, mut f: impl FnMut(&mut Connection, msh_net::Header, &[u8]) -> bool + Send + 'static) {
    let cfg = cfg.clone();
    thread::spawn(move || {
        let mut link = accept_controller(&cfg, i, l).unwrap();
        let mut buf = Vec::new();
        while let Ok(h) = link.recv_frame(&mut buf) {
            if !f(&mut link, h, &buf) {
                break;
            }
        }
    });
}

fn pong(link: &mut Connection, seq: u64, footprint: u64, requests: Vec<RunStatus>) {
    let mut out = Vec::new();
    encode_status(&Status { footprint, cap: 2 << 30, requests }, &mut out);
    link.send(MsgType::Pong, seq, 0, &out).unwrap();
}

fn event(link: &mut Connection, kind: EventKind, request: u64, position: u32, value: u64, text: &str) {
    let mut out = Vec::new();
    encode_event(kind, value, text, &mut out);
    link.send(MsgType::Event, request, position, &out).unwrap();
}

fn until(what: &str, f: impl Fn() -> bool) {
    let t0 = Instant::now();
    while !f() {
        assert!(t0.elapsed() < Duration::from_secs(5), "timed out waiting for {what}");
        thread::sleep(Duration::from_millis(5));
    }
}

fn board(c: &Client) -> &Board {
    c.board().expect("request split")
}

fn state(c: &Client, name: &str) -> WorkerState {
    board(c).worker(name).unwrap().state
}

fn outcome(rx: mpsc::Receiver<CommandOutcome>) -> CommandOutcome {
    rx.recv_timeout(Duration::from_secs(5)).expect("a command outcome")
}

fn events(b: &Board, name: &str, kind: &str) -> u64 {
    let series = b.series();
    let w = series.iter().find(|w| w.name == name).unwrap();
    w.events.iter().find(|(k, _)| *k == kind).map_or(0, |(_, n)| *n)
}

#[test]
fn prefill_done_moves_a_request_to_decode_before_any_token() {
    // n1 is a fake: it answers a start with only `PrefillDone`, so nothing else can set decode.
    let sched = SchedulerConfig { health_interval_ms: 10_000, health_timeout_ms: 20_000, ..Default::default() };
    let (ls, addrs) = listeners(1);
    let cfg = config(&addrs, sched);
    fake(&cfg, 1, ls.into_iter().next().unwrap(), |link, h, payload| {
        match h.msg_type {
            MsgType::Ping => pong(link, h.request_id, 1 << 30, vec![]),
            MsgType::Start => {
                let mut prompt = Vec::new();
                decode_start(payload, &mut prompt).unwrap();
                event(link, EventKind::PrefillDone, h.request_id, prompt.len() as u32, 1234, "");
            }
            _ => {}
        }
        true
    });
    let (client, _sched) = requests::spawn(&cfg).unwrap();
    let b = board(&client);
    assert_eq!(b.wait_ready(1, Duration::from_secs(5)), 1);
    let (id, _rx) = client.submit_with_id(prompt(0), 10);
    until("decode", || b.request(id).is_some_and(|r| r.phase == "decode"));
    let r = b.request(id).unwrap();
    assert_eq!((r.tokens_sent, r.position, r.worker.as_deref()), (0, prompt(0).len() as u32, Some("n1")));
    let w = b.worker("n1").unwrap();
    assert_eq!(w.last_event.as_ref().map(|e| e.kind), Some("prefill_done"), "{:?}", w.last_event);
    assert_eq!(events(b, "n1", "prefill_done"), 1);
    client.shutdown();
}

#[test]
fn a_real_worker_sends_prefill_done_and_it_is_counted() {
    let sched = SchedulerConfig { prefill_chunk: 64, ..Default::default() };
    let (ls, addrs) = listeners(1);
    let cfg = config(&addrs, sched);
    let n1 = worker(&cfg, 1, ls.into_iter().next().unwrap(), slow(30), None);
    let (client, sched) = requests::spawn(&cfg).unwrap();
    let b = board(&client);
    assert_eq!(b.wait_ready(1, Duration::from_secs(5)), 1);
    let (id, rx) = client.submit_with_id(prompt(1), 8);
    // The whole prompt is one step: decode right after it, with at most its first token out.
    until("decode", || b.request(id).is_some_and(|r| r.phase == "decode"));
    assert!(b.request(id).map_or(true, |r| r.tokens_sent <= 2), "{:?}", b.request(id));
    assert_eq!(collect(rx).unwrap().tokens.len(), 8);
    assert_eq!(events(b, "n1", "prefill_done"), 1);
    let text = msh_node::stats::render_workers("n0", &b.series());
    assert!(text.contains("msh_worker_events_total{node=\"n0\",worker=\"n1\",kind=\"prefill_done\"} 1"), "{text}");
    client.shutdown();
    sched.join().unwrap();
    assert_eq!(n1.join().unwrap().0.live_requests(), 0);
}

#[test]
fn memory_high_keeps_new_requests_off_a_worker() {
    // n1's cap is one byte, far under its footprint: it sends `MemoryHigh` on its first check.
    let (ls, addrs) = listeners(2);
    let cfg = config(&addrs, SchedulerConfig { max_in_flight: 4, ..Default::default() });
    let mut ls = ls.into_iter();
    let n1 = worker(&cfg, 1, ls.next().unwrap(), mock(), Some(1));
    let n2 = worker(&cfg, 2, ls.next().unwrap(), mock(), None);
    let (client, sched) = requests::spawn(&cfg).unwrap();
    let b = board(&client);
    assert_eq!(b.wait_ready(2, Duration::from_secs(5)), 2);
    until("n1 memory high", || b.worker("n1").unwrap().memory_high);
    let w = b.worker("n1").unwrap();
    let e = w.last_event.expect("last event");
    assert_eq!((e.kind, e.text.as_str()), ("memory_high", "1"));
    assert!(e.at > 0);
    assert!(!b.worker("n2").unwrap().memory_high);
    let rxs: Vec<_> = (2..8).map(|i| client.submit(prompt(i), 4)).collect();
    for rx in rxs {
        assert_eq!(collect(rx).unwrap().tokens.len(), 4);
    }
    assert_eq!((b.worker("n1").unwrap().done, b.worker("n2").unwrap().done), (0, 6));
    assert_eq!(events(b, "n1", "memory_high"), 1);
    client.shutdown();
    sched.join().unwrap();
    n1.join().unwrap();
    n2.join().unwrap();
}

#[test]
fn memory_ok_lets_a_worker_take_requests_again() {
    // A real worker's cap is fixed and its footprint does not shrink, so a fake sends the
    // events: `MemoryHigh` on its first ping, `MemoryOk` on the first ping after `ok` is set.
    let sched = SchedulerConfig { health_interval_ms: 20, ..Default::default() };
    let (ls, addrs) = listeners(1);
    let cfg = config(&addrs, sched);
    let ok = Arc::new(AtomicBool::new(false));
    let (started, starts) = mpsc::channel();
    let flag = ok.clone();
    let (mut high_sent, mut ok_sent) = (false, false);
    fake(&cfg, 1, ls.into_iter().next().unwrap(), move |link, h, _| {
        match h.msg_type {
            MsgType::Ping => {
                // The footprint in the report stays high, so only the event lifts the block.
                pong(link, h.request_id, 2 << 30, vec![]);
                if !high_sent {
                    event(link, EventKind::MemoryHigh, 0, 0, 2 << 30, &(2u64 << 30).to_string());
                    high_sent = true;
                } else if !ok_sent && flag.load(Ordering::SeqCst) {
                    event(link, EventKind::MemoryOk, 0, 0, 1 << 20, "");
                    ok_sent = true;
                }
            }
            MsgType::Start => {
                let _ = started.send(h.request_id);
            }
            _ => {}
        }
        true
    });
    let (client, _sched) = requests::spawn(&cfg).unwrap();
    let b = board(&client);
    assert_eq!(b.wait_ready(1, Duration::from_secs(5)), 1);
    until("n1 memory high", || b.worker("n1").unwrap().memory_high);
    let (id, _rx) = client.submit_with_id(prompt(8), 4);
    assert!(starts.recv_timeout(Duration::from_millis(200)).is_err(), "a start reached a worker with memory high");
    assert_eq!(b.request(id).unwrap().phase, "queued");
    ok.store(true, Ordering::SeqCst);
    assert_eq!(starts.recv_timeout(Duration::from_secs(5)).unwrap(), id);
    let w = b.worker("n1").unwrap();
    assert!(!w.memory_high);
    assert_eq!(w.last_event.map(|e| e.kind), Some("memory_ok"));
    assert_eq!((events(b, "n1", "memory_high"), events(b, "n1", "memory_ok")), (1, 1));
    client.shutdown();
}

#[test]
fn status_and_reset_stats_answer_ok() {
    let (ls, addrs) = listeners(1);
    let cfg = config(&addrs, SchedulerConfig::default());
    let n1 = worker(&cfg, 1, ls.into_iter().next().unwrap(), mock(), None);
    let (client, sched) = requests::spawn(&cfg).unwrap();
    let b = board(&client);
    assert_eq!(b.wait_ready(1, Duration::from_secs(5)), 1);
    assert_eq!(collect(client.submit(prompt(9), 3)).unwrap().tokens.len(), 3);

    let s = outcome(b.command("n1", CommandKind::Status, Duration::from_secs(5)).unwrap());
    assert!(s.ok, "{s:?}");
    assert!(s.message.contains("requests 0") && s.message.contains("footprint"), "{}", s.message);
    assert!(s.message.contains("served 1") && s.message.contains("made 3"), "{}", s.message);
    let r = outcome(b.command("n1", CommandKind::ResetStats, Duration::from_secs(5)).unwrap());
    assert!(r.ok, "{r:?}");
    let s = outcome(b.command("n1", CommandKind::Status, Duration::from_secs(5)).unwrap());
    assert!(s.message.contains("served 0") && s.message.contains("made 0"), "{}", s.message);
    // The worker stays up and serving.
    assert_eq!(state(&client, "n1"), WorkerState::Ready);
    let err = b.command("nobody", CommandKind::Status, Duration::from_secs(1)).unwrap_err();
    assert!(err.contains("nobody"), "{err}");
    client.shutdown();
    sched.join().unwrap();
    n1.join().unwrap();
}

#[test]
fn restart_is_refused_while_busy_and_ends_the_worker_when_idle() {
    let (ls, addrs) = listeners(1);
    let cfg = config(&addrs, SchedulerConfig::default());
    let l = ls.into_iter().next().unwrap();
    let w_cfg = cfg.clone();
    let n1 = thread::spawn(move || {
        let mut w = RequestWorker::new(slow(20), "n1", w_cfg.scheduler.prefill_chunk, None);
        w.serve_forever(&w_cfg, 1, &l).unwrap();
        // The listener goes with the thread, so the controller cannot reconnect.
        w.restart_requested()
    });
    let (client, _sched) = requests::spawn(&cfg).unwrap();
    let b = board(&client);
    assert_eq!(b.wait_ready(1, Duration::from_secs(5)), 1);

    let rx = client.submit(prompt(10), 30);
    until("running", || b.worker("n1").unwrap().requests.iter().any(|r| r.tokens_sent > 0));
    let busy = outcome(b.command("n1", CommandKind::Restart, Duration::from_secs(5)).unwrap());
    assert!(!busy.ok && busy.message.contains("busy"), "{busy:?}");
    assert_eq!(collect(rx).unwrap().tokens.len(), 30);

    until("idle", || b.worker("n1").unwrap().running == 0);
    let done = outcome(b.command("n1", CommandKind::Restart, Duration::from_secs(5)).unwrap());
    assert!(done.ok, "{done:?}");
    let t0 = Instant::now();
    while !n1.is_finished() {
        assert!(t0.elapsed() < Duration::from_secs(5), "serve_forever did not return");
        thread::sleep(Duration::from_millis(5));
    }
    assert!(n1.join().unwrap(), "restart_requested");
    until("n1 down", || state(&client, "n1") == WorkerState::Down);
    let err = b.command("n1", CommandKind::Status, Duration::from_secs(1)).unwrap_err();
    assert!(err.contains("not connected"), "{err}");
    client.shutdown();
}

#[test]
fn a_command_in_flight_gets_worker_down_when_the_link_breaks() {
    // n1 never answers a command; it hangs up on the first one.
    let (ls, addrs) = listeners(1);
    let cfg = config(&addrs, SchedulerConfig::default());
    fake(&cfg, 1, ls.into_iter().next().unwrap(), |link, h, payload| match h.msg_type {
        MsgType::Ping => {
            pong(link, h.request_id, 1 << 30, vec![]);
            true
        }
        MsgType::Command => {
            assert_eq!(decode_command(payload).unwrap(), CommandKind::Status);
            false
        }
        _ => true,
    });
    let (client, _sched) = requests::spawn(&cfg).unwrap();
    let b = board(&client);
    assert_eq!(b.wait_ready(1, Duration::from_secs(5)), 1);
    let o = outcome(b.command("n1", CommandKind::Status, Duration::from_secs(5)).unwrap());
    assert_eq!((o.ok, o.message.as_str()), (false, "worker down"));
    until("n1 down", || state(&client, "n1") != WorkerState::Ready);
    client.shutdown();
}

#[test]
fn a_command_not_answered_times_out() {
    let (ls, addrs) = listeners(1);
    let cfg = config(&addrs, SchedulerConfig::default());
    fake(&cfg, 1, ls.into_iter().next().unwrap(), |link, h, _| {
        if h.msg_type == MsgType::Ping {
            pong(link, h.request_id, 1 << 30, vec![]);
        }
        true
    });
    let (client, _sched) = requests::spawn(&cfg).unwrap();
    let b = board(&client);
    assert_eq!(b.wait_ready(1, Duration::from_secs(5)), 1);
    let o = outcome(b.command("n1", CommandKind::Status, Duration::from_millis(200)).unwrap());
    assert_eq!((o.ok, o.message.as_str()), (false, "timeout"));
    assert_eq!(state(&client, "n1"), WorkerState::Ready);
    client.shutdown();
}
