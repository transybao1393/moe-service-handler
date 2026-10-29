//! The request split on the mock engine: a controller with no engine and workers running the
//! whole model on loopback threads, checked token for token against one process, and its
//! worker management: lost, hung and stalled workers, drain, cancel and the endpoints.

use std::io;
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use msh_net::{
    decode_start, encode_done, encode_status, encode_token, Connection, DoneReason, FrameError, MsgType, RunStatus, Status,
};
use msh_node::config::{Config, NetConfig, NodeConfig, Split};
use msh_node::engine::{Engine, MockEngine, Sample, StageIn};
use msh_node::node::{accept_controller, RequestWorker};
use msh_node::requests::{self, Board, WorkerState};
use msh_node::scheduler::{self, collect, Client, Event, SchedulerConfig};
use msh_node::Error;

const DIM: usize = 64;
const LAYERS: u32 = 8;

fn mock() -> MockEngine {
    MockEngine::new(0..LAYERS, LAYERS, DIM)
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

/// Worker `i` on `l`: serves `links` controller connections in turn, then hands back its engine.
fn worker<E: Engine + 'static>(cfg: &Config, i: usize, l: TcpListener, engine: E, links: usize) -> JoinHandle<E> {
    let cfg = cfg.clone();
    thread::spawn(move || {
        let mut w = RequestWorker::new(engine, &cfg.nodes[i].name, cfg.scheduler.prefill_chunk, None);
        for _ in 0..links {
            let link = accept_controller(&cfg, i, l.try_clone().unwrap()).unwrap();
            w.serve(link).unwrap();
        }
        w.into_engine()
    })
}

fn until(what: &str, f: impl Fn() -> bool) {
    let t0 = Instant::now();
    while !f() {
        assert!(t0.elapsed() < Duration::from_secs(10), "timed out waiting for {what}");
        thread::sleep(Duration::from_millis(5));
    }
}

fn board(c: &Client) -> &Board {
    c.board().expect("request split")
}

fn state(c: &Client, name: &str) -> WorkerState {
    board(c).worker(name).unwrap().state
}

fn next_token(rx: &mpsc::Receiver<Event>) -> u32 {
    match rx.recv_timeout(Duration::from_secs(10)).unwrap() {
        Event::Token { token, .. } => token,
        other => panic!("expected a token, got {other:?}"),
    }
}

#[test]
fn requests_spread_over_two_workers_match_one_process() {
    let prompts: Vec<_> = (0..8).map(prompt).collect();
    let want = reference(&prompts, 12);
    let sched = SchedulerConfig { max_in_flight: 2, prefill_chunk: 3, ..Default::default() };
    let (ls, addrs) = listeners(2);
    let cfg = config(&addrs, sched);
    let workers: Vec<_> = ls.into_iter().enumerate().map(|(k, l)| worker(&cfg, k + 1, l, mock(), 1)).collect();
    let (client, sched) = requests::spawn(&cfg).unwrap();
    assert_eq!(board(&client).wait_ready(2, Duration::from_secs(5)), 2);
    let rxs: Vec<_> = prompts.iter().map(|p| client.submit(p.clone(), 12)).collect();
    let got: Vec<_> = rxs.into_iter().map(|rx| collect(rx).unwrap().tokens).collect();
    assert_eq!(got, want);
    // Both workers served, and every request finished on one of them.
    let done: Vec<u64> = board(&client).workers().iter().map(|w| w.done).collect();
    assert!(done.iter().all(|&d| d > 0), "{done:?}");
    assert_eq!(done.iter().sum::<u64>(), 8);
    client.shutdown();
    sched.join().unwrap();
    for w in workers {
        assert_eq!(w.join().unwrap().live_requests(), 0);
    }
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

#[test]
fn lost_worker_retries_unstarted_requests_and_comes_back() {
    // Without resume_on_loss, a request that sent tokens fails with its worker.
    let sched = SchedulerConfig {
        max_in_flight: 2,
        prefill_chunk: 1,
        health_interval_ms: 20,
        resume_on_loss: false,
        ..Default::default()
    };
    let (ls, real) = listeners(2);
    let proxy = Proxy::start(real[0].parse().unwrap());
    let cfg = config(&[proxy.addr.clone(), real[1].clone()], sched);
    let mut ls = ls.into_iter();
    // n1 serves the link that is cut, then the one after it.
    let n1 = worker(&cfg, 1, ls.next().unwrap(), slow(2), 2);
    let n2 = worker(&cfg, 2, ls.next().unwrap(), slow(2), 1);
    let (client, sched) = requests::spawn(&cfg).unwrap();
    let b = board(&client);
    assert_eq!(b.wait_ready(2, Duration::from_secs(5)), 2);

    // Both on n1: a long generation that has sent tokens, and a long prompt that has not.
    b.drain("n2").unwrap();
    let (a, a_rx) = client.submit_with_id(prompt(0), 100_000);
    next_token(&a_rx);
    let long: Vec<u32> = (0..300).map(|k| k * 7 % 1000).collect();
    let (c, c_rx) = client.submit_with_id(long.clone(), 8);
    let on = |id| b.request(id).unwrap();
    until("the long prompt started", || b.request(c).is_some_and(|r| r.phase == "prefill"));
    assert_eq!((on(a).worker.as_deref(), on(c).worker.as_deref()), (Some("n1"), Some("n1")));
    assert_eq!(on(c).tokens_sent, 0);
    b.resume("n2").unwrap();
    proxy.cut();

    // The one with tokens fails; the other starts over on n2 and gives the same tokens.
    let err = collect(a_rx).unwrap_err().to_string();
    assert!(err.contains("worker n1 lost after"), "{err}");
    assert_eq!(collect(c_rx).unwrap().tokens, reference(&[long], 8)[0]);
    let n1_view = b.worker("n1").unwrap();
    assert!(n1_view.last_error.unwrap().message.contains("link lost"));
    assert!(n1_view.failed >= 1 && n1_view.errors_last_min >= 1);
    assert_eq!(b.worker("n2").unwrap().done, 1);

    // n1 reconnects and serves again.
    until("n1 ready again", || state(&client, "n1") == WorkerState::Ready);
    b.drain("n2").unwrap();
    assert_eq!(client.generate(prompt(3), 6).unwrap().tokens, reference(&[prompt(3)], 6)[0]);
    assert_eq!(b.worker("n1").unwrap().done, 1);
    client.shutdown();
    sched.join().unwrap();
    assert_eq!(n1.join().unwrap().0.live_requests(), 0);
    assert_eq!(n2.join().unwrap().0.live_requests(), 0);
}

#[test]
fn lost_worker_resumes_started_requests_on_another() {
    let sched = SchedulerConfig { max_in_flight: 2, health_interval_ms: 20, ..Default::default() };
    let (ls, real) = listeners(2);
    let proxy = Proxy::start(real[0].parse().unwrap());
    let cfg = config(&[proxy.addr.clone(), real[1].clone()], sched);
    let mut ls = ls.into_iter();
    let n1 = worker(&cfg, 1, ls.next().unwrap(), slow(2), 2);
    let n2 = worker(&cfg, 2, ls.next().unwrap(), slow(2), 1);
    let (client, sched) = requests::spawn(&cfg).unwrap();
    let b = board(&client);
    assert_eq!(b.wait_ready(2, Duration::from_secs(5)), 2);

    b.drain("n2").unwrap();
    let (a, a_rx) = client.submit_with_id(prompt(0), 200);
    let first = next_token(&a_rx);
    assert_eq!(b.request(a).unwrap().worker.as_deref(), Some("n1"));
    b.resume("n2").unwrap();
    proxy.cut();

    // The request carries on from the tokens already sent and ends as if nothing happened.
    let mut tokens = vec![first];
    tokens.extend(collect(a_rx).unwrap().tokens);
    assert_eq!(tokens, reference(&[prompt(0)], 200)[0]);
    assert_eq!(b.worker("n2").unwrap().done, 1);
    client.shutdown();
    sched.join().unwrap();
    assert_eq!(n2.join().unwrap().0.live_requests(), 0);
    drop(n1);
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

fn pong(link: &mut Connection, seq: u64, requests: Vec<RunStatus>) {
    let mut out = Vec::new();
    encode_status(&Status { footprint: 1 << 30, cap: 2 << 30, requests }, &mut out);
    link.send(MsgType::Pong, seq, 0, &out).unwrap();
}

#[test]
fn hung_worker_becomes_stuck_and_gets_no_new_work() {
    let sched = SchedulerConfig {
        health_interval_ms: 20,
        health_timeout_ms: 300,
        stall_timeout_ms: 60_000,
        resume_on_loss: false,
        ..Default::default()
    };
    let (ls, addrs) = listeners(2);
    let cfg = config(&addrs, sched);
    let mut ls = ls.into_iter();
    let (release, hang) = mpsc::channel::<()>();
    // n1 answers pings until it gets a request, sends three tokens, then freezes with the
    // connection open until released.
    fake(&cfg, 1, ls.next().unwrap(), move |link, h, payload| {
        match h.msg_type {
            MsgType::Ping => pong(link, h.request_id, vec![]),
            MsgType::Start => {
                let mut prompt = Vec::new();
                decode_start(payload, &mut prompt).unwrap();
                let mut out = Vec::new();
                for k in 0..3 {
                    encode_token(&[40 + k], &[1], &mut out);
                    link.send(MsgType::Token, h.request_id, prompt.len() as u32 + k, &out).unwrap();
                }
                let _ = hang.recv();
                return false;
            }
            _ => {}
        }
        true
    });
    let n2 = worker(&cfg, 2, ls.next().unwrap(), mock(), 1);
    let (client, _sched) = requests::spawn(&cfg).unwrap();
    let b = board(&client);
    assert_eq!(b.wait_ready(2, Duration::from_secs(5)), 2);
    b.drain("n2").unwrap();
    let (a, a_rx) = client.submit_with_id(prompt(0), 500);
    assert_eq!([next_token(&a_rx), next_token(&a_rx), next_token(&a_rx)], [40, 41, 42]);
    b.resume("n2").unwrap();

    until("n1 stuck", || state(&client, "n1") == WorkerState::Stuck);
    let w = b.worker("n1").unwrap();
    assert!(w.reason.contains("health ping") && w.reason.contains("decode, token 3 of 500"), "{}", w.reason);
    let r = &w.requests[0];
    assert_eq!((r.id, r.phase, r.tokens_sent), (a, "decode", 3));
    assert!(r.stuck.as_deref().unwrap().starts_with("decode, token 3 of 500, no progress for"), "{r:?}");
    // New work goes to n2 only.
    let prompts: Vec<_> = (1..4).map(prompt).collect();
    let rxs: Vec<_> = prompts.iter().map(|p| client.submit(p.clone(), 6)).collect();
    let got: Vec<_> = rxs.into_iter().map(|rx| collect(rx).unwrap().tokens).collect();
    assert_eq!(got, reference(&prompts, 6));
    assert_eq!((b.worker("n1").unwrap().running, b.worker("n2").unwrap().done), (1, 3));

    // Its process dies: down, and the request it held fails.
    release.send(()).unwrap();
    until("n1 down", || state(&client, "n1") == WorkerState::Down);
    let err = collect(a_rx).unwrap_err().to_string();
    assert!(err.contains("worker n1 lost after 3 tokens"), "{err}");
    client.shutdown();
    assert_eq!(n2.join().unwrap().live_requests(), 0);
}

#[test]
fn stalled_request_fails_alone_and_is_dropped_on_its_worker() {
    let sched = SchedulerConfig { health_interval_ms: 20, stall_timeout_ms: 300, ..Default::default() };
    let (ls, addrs) = listeners(2);
    let cfg = config(&addrs, sched);
    let mut ls = ls.into_iter();
    let (dropped_tx, dropped) = mpsc::channel();
    // n1 takes a request and never makes progress on it, but answers every ping.
    let mut held: Option<u64> = None;
    fake(&cfg, 1, ls.next().unwrap(), move |link, h, _| {
        match h.msg_type {
            MsgType::Ping => {
                let runs = held.iter().map(|&request| RunStatus { request, decoding: false, position: 2, made: 0 }).collect();
                pong(link, h.request_id, runs);
            }
            MsgType::Start => held = Some(h.request_id),
            MsgType::Drop => {
                held = None;
                let mut out = Vec::new();
                encode_done(DoneReason::Dropped, &mut out);
                link.send(MsgType::Done, h.request_id, 0, &out).unwrap();
                dropped_tx.send(h.request_id).unwrap();
            }
            _ => {}
        }
        true
    });
    let n2 = worker(&cfg, 2, ls.next().unwrap(), slow(1), 1);
    let (client, sched) = requests::spawn(&cfg).unwrap();
    let b = board(&client);
    assert_eq!(b.wait_ready(2, Duration::from_secs(5)), 2);
    b.drain("n2").unwrap();
    let (s, s_rx) = client.submit_with_id(prompt(0), 50);
    until("the stalled request on n1", || b.request(s).is_some_and(|r| r.worker.as_deref() == Some("n1")));
    b.resume("n2").unwrap();
    let ok = client.submit(prompt(1), 80);

    let err = collect(s_rx).unwrap_err().to_string();
    assert!(err.contains("stalled") && err.contains("n1") && err.contains("prefill, position 2 of 5"), "{err}");
    assert_eq!(dropped.recv_timeout(Duration::from_secs(5)).unwrap(), s);
    assert_eq!(collect(ok).unwrap().tokens, reference(&[prompt(1)], 80)[0]);
    // The worker answers pings, so it stays ready; only the request failed.
    until("the slot freed", || b.worker("n1").unwrap().running == 0);
    let w = b.worker("n1").unwrap();
    assert_eq!((w.state, w.failed), (WorkerState::Ready, 1));
    assert!(w.last_error.unwrap().message.contains("stalled"));
    assert_eq!((w.footprint, w.cap), (1 << 30, 2 << 30), "from the status report");
    client.shutdown();
    sched.join().unwrap();
    assert_eq!(n2.join().unwrap().0.live_requests(), 0);
}

#[test]
fn drain_and_resume() {
    let sched = SchedulerConfig { max_in_flight: 2, ..Default::default() };
    let (ls, addrs) = listeners(2);
    let cfg = config(&addrs, sched);
    let workers: Vec<_> = ls.into_iter().enumerate().map(|(k, l)| worker(&cfg, k + 1, l, mock(), 1)).collect();
    let (client, sched) = requests::spawn(&cfg).unwrap();
    let b = board(&client);
    assert_eq!(b.wait_ready(2, Duration::from_secs(5)), 2);
    assert!(b.drain("nope").is_err());
    let v = b.drain("n1").unwrap();
    assert_eq!((v.state, v.reason.as_str()), (WorkerState::Draining, "drain asked"));
    let prompts: Vec<_> = (0..4).map(prompt).collect();
    let rxs: Vec<_> = prompts.iter().map(|p| client.submit(p.clone(), 6)).collect();
    let got: Vec<_> = rxs.into_iter().map(|rx| collect(rx).unwrap().tokens).collect();
    assert_eq!(got, reference(&prompts, 6));
    assert_eq!((b.worker("n1").unwrap().done, b.worker("n2").unwrap().done), (0, 4));
    // Both drained: nothing is ready, so new requests are refused.
    b.drain("n2").unwrap();
    let err = client.generate(prompt(0), 4).unwrap_err();
    assert!(matches!(&err, Error::Rejected(m) if m.contains("cluster down: no worker ready")), "{err}");
    assert_eq!(b.resume("n1").unwrap().state, WorkerState::Ready);
    // A new prompt: a repeat would be answered from the controller's cache.
    assert_eq!(client.generate(prompt(4), 6).unwrap().tokens, reference(&[prompt(4)], 6)[0]);
    assert_eq!(b.worker("n1").unwrap().done, 1);
    b.resume("n2").unwrap();
    client.shutdown();
    sched.join().unwrap();
    for w in workers {
        assert_eq!(w.join().unwrap().live_requests(), 0);
    }
}

#[test]
fn cancel_ends_a_running_or_queued_request() {
    let sched = SchedulerConfig { max_in_flight: 1, ..Default::default() };
    let (ls, addrs) = listeners(1);
    let cfg = config(&addrs, sched);
    let n1 = worker(&cfg, 1, ls.into_iter().next().unwrap(), slow(1), 1);
    let (client, sched) = requests::spawn(&cfg).unwrap();
    let b = board(&client);
    assert_eq!(b.wait_ready(1, Duration::from_secs(5)), 1);
    let (running, running_rx) = client.submit_with_id(prompt(0), 100_000);
    next_token(&running_rx);
    let (queued, queued_rx) = client.submit_with_id(prompt(1), 10);
    until("the second request queued", || b.request(queued).is_some_and(|r| r.phase == "queued"));
    assert_eq!(b.cancel(queued).unwrap().phase, "queued");
    assert_eq!(b.cancel(running).unwrap().phase, "decode");
    for rx in [queued_rx, running_rx] {
        let err = collect(rx).unwrap_err().to_string();
        assert!(err.contains("cancelled by the operator"), "{err}");
    }
    // The worker freed it and is free for the next one.
    until("the slot freed", || b.worker("n1").unwrap().running == 0 && b.requests().is_empty());
    assert!(b.cancel(running).is_err(), "already over");
    assert_eq!(client.generate(prompt(2), 5).unwrap().tokens, reference(&[prompt(2)], 5)[0]);
    client.shutdown();
    sched.join().unwrap();
    assert_eq!(n1.join().unwrap().0.live_requests(), 0);
}

mod endpoints {
    use super::*;

    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use axum::Router;
    use http_body_util::BodyExt;
    use serde_json::{json, Value};
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;
    use tokenizers::Tokenizer;
    use tower::ServiceExt;

    use msh_node::chat::{ChatTemplate, Codec};
    use msh_node::http::{router, HttpState};

    fn codec() -> Codec {
        let vocab = (0..64).map(|i| (format!("w{i}"), i)).collect();
        let model = WordLevel::builder().vocab(vocab).unk_token("w0".into()).build().unwrap();
        let mut tok = Tokenizer::new(model);
        tok.with_pre_tokenizer(Some(Whitespace {}));
        Codec::from_tokenizer(tok, ChatTemplate::Chatml)
    }

    async fn call(app: &Router, method: &str, path: &str, body: Option<Value>) -> (StatusCode, Value) {
        let req = Request::builder()
            .method(method)
            .uri(path)
            .header("content-type", "application/json")
            .body(body.map_or(Body::empty(), |b| Body::from(b.to_string())))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn endpoints_report_and_manage_the_workers() {
        let sched = SchedulerConfig { health_interval_ms: 20, ..Default::default() };
        let (ls, addrs) = listeners(2);
        let cfg = config(&addrs, sched);
        let engine = || Slow(MockEngine::new(0..LAYERS, LAYERS, DIM).with_vocab(64), Duration::from_millis(2));
        for (k, l) in ls.into_iter().enumerate() {
            worker(&cfg, k + 1, l, engine(), 1);
        }
        let (client, _sched) = requests::spawn(&cfg).unwrap();
        assert_eq!(board(&client).wait_ready(2, Duration::from_secs(5)), 2);
        let b = client.board().unwrap().clone();
        let app = router(Arc::new(HttpState::new(client, codec(), "mock".into(), 1 << 20, 64)));

        let (status, all) = call(&app, "GET", "/v1/workers", None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(all["ready"], 2);
        let names: Vec<&str> = all["workers"].as_array().unwrap().iter().map(|w| w["name"].as_str().unwrap()).collect();
        assert_eq!(names, ["n1", "n2"]);
        assert_eq!(all["workers"][0]["state"], "ready");
        assert_eq!(call(&app, "GET", "/v1/workers/nope", None).await.0, StatusCode::NOT_FOUND);

        let (status, n1) = call(&app, "POST", "/v1/workers/n1/drain", None).await;
        assert_eq!((status, &n1["state"]), (StatusCode::OK, &json!("draining")));
        assert_eq!(call(&app, "GET", "/v1/workers/n1", None).await.1["state"], "draining");

        // A long streaming completion, held open: it runs on n2.
        let chat = json!({ "messages": [{ "role": "user", "content": "w5 w6 w7" }], "max_tokens": 500_000, "stream": true });
        let req = Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(chat.to_string()))
            .unwrap();
        let stream = app.clone().oneshot(req).await.unwrap();
        assert_eq!(stream.status(), StatusCode::OK);
        let b2 = b.clone();
        tokio::task::spawn_blocking(move || until("tokens from n2", || b2.requests().first().is_some_and(|r| r.tokens_sent > 2)))
            .await
            .unwrap();
        let (_, reqs) = call(&app, "GET", "/v1/requests", None).await;
        let r = &reqs["requests"][0];
        assert_eq!((&r["id"], &r["worker"], &r["phase"]), (&json!(1), &json!("n2"), &json!("decode")));
        assert_eq!(r["max_tokens"], 500_000);
        let (_, n2) = call(&app, "GET", "/v1/workers/n2", None).await;
        assert_eq!((&n2["running"], &n2["requests"][0]["id"]), (&json!(1), &json!(1)));
        assert!(n2["ping_ms"].as_f64().is_some() && n2["last_pong"].as_u64().is_some(), "{n2}");
        assert!(n2["tokens_per_s"].as_f64().unwrap() > 0.0, "{n2}");
        let series = b.series();
        assert_eq!((series[0].state, series[1].state, series[1].running), ("draining", "ready", 1));

        let (status, cancelled) = call(&app, "POST", "/v1/requests/chatcmpl-1/cancel", None).await;
        assert_eq!((status, &cancelled["cancelled"]["id"]), (StatusCode::OK, &json!(1)));
        assert_eq!(call(&app, "POST", "/v1/requests/1/cancel", None).await.0, StatusCode::NOT_FOUND);
        assert_eq!(call(&app, "POST", "/v1/requests/x/cancel", None).await.0, StatusCode::BAD_REQUEST);
        drop(stream);
        let (status, n1) = call(&app, "POST", "/v1/workers/n1/resume", None).await;
        assert_eq!((status, &n1["state"]), (StatusCode::OK, &json!("ready")));
        let b2 = b.clone();
        tokio::task::spawn_blocking(move || until("the request gone", || b2.requests().is_empty())).await.unwrap();
        assert_eq!(call(&app, "GET", "/v1/requests", None).await.1["requests"], json!([]));
    }
}

#[test]
fn small_chunks_give_the_same_tokens() {
    // 4 KiB chunks: a `Start` with 1500 prompt ids (6 KiB) or more goes in two frames or more,
    // beside the other link traffic.
    let prompts: Vec<Vec<u32>> =
        (0..3).map(|i| (0..1500 + 300 * i).map(|k| (13 * k + 7 * i) % 5000).collect()).collect();
    let want = reference(&prompts, 6);
    let sched = SchedulerConfig { max_in_flight: 2, prefill_chunk: 64, ..Default::default() };
    let (ls, addrs) = listeners(1);
    let cfg = Config { net: NetConfig { max_chunk_kib: 4, ..Default::default() }, ..config(&addrs, sched) };
    let w = worker(&cfg, 1, ls.into_iter().next().unwrap(), mock(), 1);
    let (client, sched) = requests::spawn(&cfg).unwrap();
    assert_eq!(board(&client).wait_ready(1, Duration::from_secs(5)), 1);
    let rxs: Vec<_> = prompts.iter().map(|p| client.submit(p.clone(), 6)).collect();
    let got: Vec<_> = rxs.into_iter().map(|rx| collect(rx).unwrap().tokens).collect();
    assert_eq!(got, want);
    client.shutdown();
    sched.join().unwrap();
    assert_eq!(w.join().unwrap().live_requests(), 0);
}

#[test]
fn a_worker_drops_a_silent_controller_and_takes_the_next() {
    let sched = SchedulerConfig { health_interval_ms: 20, health_timeout_ms: 200, ..Default::default() };
    let (ls, addrs) = listeners(1);
    let cfg = config(&addrs, sched);
    let l = ls.into_iter().next().unwrap();
    let worker_cfg = cfg.clone();
    thread::spawn(move || {
        let mut w = RequestWorker::new(mock(), "n1", sched.prefill_chunk, None);
        w.serve_forever(&worker_cfg, 1, &l).unwrap();
    });
    // A controller that says `Hello`, then nothing, as a frozen machine would.
    let mut frozen = Connection::connect(addrs[0].as_str(), "n0", Duration::from_secs(1)).unwrap();
    let t0 = Instant::now();
    match frozen.recv_frame(&mut Vec::new()) {
        Err(FrameError::Io(e)) if e.kind() == io::ErrorKind::UnexpectedEof => {}
        other => panic!("expected the worker to drop the link, got {other:?}"),
    }
    let took = t0.elapsed();
    assert!(took >= Duration::from_millis(150) && took < Duration::from_secs(3), "{took:?}");
    // The next controller, pinging, is served, and keeps its link.
    let (client, sched_thread) = requests::spawn(&cfg).unwrap();
    assert_eq!(board(&client).wait_ready(1, Duration::from_secs(5)), 1);
    assert_eq!(client.generate(prompt(1), 6).unwrap().tokens, reference(&[prompt(1)], 6)[0]);
    thread::sleep(Duration::from_millis(500));
    assert_eq!(state(&client, "n1"), WorkerState::Ready);
    assert_eq!(client.generate(prompt(2), 6).unwrap().tokens, reference(&[prompt(2)], 6)[0]);
    client.shutdown();
    sched_thread.join().unwrap();
}

#[test]
fn reconnect_tries_back_off_as_configured() {
    // Nothing listens at n1's address yet.
    let (ls, addrs) = listeners(1);
    drop(ls);
    let net = NetConfig { reconnect_min_ms: 50, reconnect_max_ms: 200, ..Default::default() };
    let cfg = Config { net, ..config(&addrs, SchedulerConfig::default()) };
    let (client, sched) = requests::spawn(&cfg).unwrap();
    let b = board(&client);
    // When each try count is first seen.
    let mut seen: Vec<(u32, Instant)> = Vec::new();
    let t0 = Instant::now();
    while seen.len() < 6 {
        assert!(t0.elapsed() < Duration::from_secs(5), "tries seen: {seen:?}");
        let v = b.worker("n1").unwrap();
        if v.reconnect_tries > seen.last().map_or(0, |s| s.0) {
            assert!(v.next_try.is_some(), "no next try time while down");
            seen.push((v.reconnect_tries, Instant::now()));
        }
        thread::sleep(Duration::from_millis(1));
    }
    // Each wait is the nominal one (50, 100, 200, 200, ... ms) shortened by up to a quarter.
    for w in seen.windows(2) {
        assert_eq!(w[1].0, w[0].0 + 1, "{seen:?}");
        let nominal = Duration::from_millis((50u64 << (w[0].0 - 1)).min(200));
        let gap = w[1].1 - w[0].1;
        let ok = gap >= nominal.mul_f64(0.7) && gap <= nominal + Duration::from_millis(150);
        assert!(ok, "after try {}: {gap:?} for {nominal:?}", w[0].0);
    }
    // Started on that address, the worker is found and the count cleared.
    let w = worker(&cfg, 1, TcpListener::bind(addrs[0].as_str()).unwrap(), mock(), 1);
    assert_eq!(b.wait_ready(1, Duration::from_secs(5)), 1);
    let v = b.worker("n1").unwrap();
    assert_eq!((v.reconnect_tries, v.next_try), (0, None));
    assert_eq!(client.generate(prompt(0), 6).unwrap().tokens, reference(&[prompt(0)], 6)[0]);
    client.shutdown();
    sched.join().unwrap();
    assert_eq!(w.join().unwrap().live_requests(), 0);
}
