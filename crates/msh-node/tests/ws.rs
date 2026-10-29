//! `GET /v1/ws` over a real socket, on the layer split (one mock node) and the request split
//! (workers on loopback), and `POST /v1/workers/{name}/command`.

use std::collections::{HashMap, HashSet};
use std::net::{SocketAddr, TcpListener};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use futures_util::{SinkExt, StreamExt};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::Tokenizer;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};
use tower::ServiceExt;

use msh_node::chat::{ChatTemplate, Codec};
use msh_node::config::{CacheConfig, CacheMode, Config, NodeConfig, Split};
use msh_node::engine::{Engine, MockEngine, Sample, StageIn};
use msh_node::http::{router, HttpState};
use msh_node::node::{accept_controller, RequestWorker};
use msh_node::requests::{self, Board};
use msh_node::scheduler::{self, Client, SchedulerConfig};
use msh_node::Error;

const VOCAB: u32 = 64;
const DIM: usize = 64;
const LAYERS: u32 = 8;

type Ws = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;

/// Words `w0`..`w63`; anything else is `w0`.
fn codec() -> Codec {
    let vocab = (0..VOCAB).map(|i| (format!("w{i}"), i)).collect();
    let model = WordLevel::builder().vocab(vocab).unk_token("w0".into()).build().unwrap();
    let mut tok = Tokenizer::new(model);
    tok.with_pre_tokenizer(Some(Whitespace {}));
    Codec::from_tokenizer(tok, ChatTemplate::Chatml)
}

/// The mock, `delay` slower per step, keeping the ids it holds state for in `live`.
struct Tracked {
    engine: MockEngine,
    delay: Duration,
    live: Arc<Mutex<HashSet<u64>>>,
}

impl Engine for Tracked {
    fn hidden_dim(&self) -> usize {
        self.engine.hidden_dim()
    }
    fn has_head(&self) -> bool {
        self.engine.has_head()
    }
    fn step(&mut self, req: u64, pos: u32, input: StageIn<'_>, s: Sample, h: &mut Vec<f32>, t: &mut Vec<u32>) -> Result<(), Error> {
        thread::sleep(self.delay);
        self.live.lock().unwrap().insert(req);
        self.engine.step(req, pos, input, s, h, t)
    }
    fn free(&mut self, req: u64) {
        self.live.lock().unwrap().remove(&req);
        self.engine.free(req)
    }
}

fn tracked(ms: u64) -> (Tracked, Arc<Mutex<HashSet<u64>>>) {
    let live: Arc<Mutex<HashSet<u64>>> = Arc::default();
    let engine = MockEngine::new(0..LAYERS, LAYERS, DIM).with_vocab(VOCAB);
    (Tracked { engine, delay: Duration::from_millis(ms), live: live.clone() }, live)
}

fn app(client: Client) -> Router {
    router(Arc::new(HttpState::new(client, codec(), "mock".into(), 1 << 20, 64)))
}

/// One process, the default (layer) split.
fn local(ms: u64) -> (Router, Arc<Mutex<HashSet<u64>>>) {
    let (engine, live) = tracked(ms);
    let (client, _) = scheduler::spawn(engine, None, SchedulerConfig::default());
    (app(client), live)
}

/// A request-split config: controller `n0`, then a worker `n1`, `n2`, ... at each address.
fn config(addrs: &[String]) -> Config {
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
        scheduler: SchedulerConfig { health_interval_ms: 20, ..Default::default() },
        engine: Default::default(),
        http: None,
        metrics: None,
        log: Default::default(),
        service: Default::default(),
        net: Default::default(),
        // Off, so every chat runs on a worker instead of coming back from the cache.
        cache: CacheConfig { mode: CacheMode::Off, ..Default::default() },
        nodes,
    };
    cfg.validate().unwrap();
    cfg
}

/// A controller and `n` workers on loopback, all ready; the workers' engines are slowed by `ms`.
fn split(n: usize, ms: u64) -> (Router, Arc<Board>) {
    let ls: Vec<TcpListener> = (0..n).map(|_| TcpListener::bind("127.0.0.1:0").unwrap()).collect();
    let addrs: Vec<String> = ls.iter().map(|l| l.local_addr().unwrap().to_string()).collect();
    let cfg = config(&addrs);
    for (k, l) in ls.into_iter().enumerate() {
        let cfg = cfg.clone();
        thread::spawn(move || {
            let mut w = RequestWorker::new(tracked(ms).0, &cfg.nodes[k + 1].name, cfg.scheduler.prefill_chunk, None);
            let link = accept_controller(&cfg, k + 1, l).unwrap();
            let _ = w.serve(link);
        });
    }
    let (client, _) = requests::spawn(&cfg).unwrap();
    let b = client.board().unwrap().clone();
    assert_eq!(b.wait_ready(n, Duration::from_secs(5)), n);
    (app(client), b)
}

/// Serves `app` on a loopback port.
async fn serve(app: Router) -> SocketAddr {
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, app).await.unwrap() });
    addr
}

async fn connect(addr: SocketAddr) -> Ws {
    tokio_tungstenite::connect_async(format!("ws://{addr}/v1/ws")).await.unwrap().0
}

async fn send(ws: &mut Ws, v: Value) {
    ws.send(Message::Text(v.to_string().into())).await.unwrap();
}

/// The next text message, as JSON.
async fn recv(ws: &mut Ws) -> Value {
    loop {
        let msg = tokio::time::timeout(Duration::from_secs(10), ws.next()).await.expect("no message in 10 s");
        match msg.expect("socket closed").unwrap() {
            Message::Text(t) => return serde_json::from_str(t.as_str()).unwrap(),
            _ => continue,
        }
    }
}

fn messages(content: &str) -> Value {
    json!([{ "role": "user", "content": content }])
}

fn chat(id: &str, content: &str, max_tokens: usize) -> Value {
    json!({ "type": "chat", "id": id, "messages": messages(content), "max_tokens": max_tokens })
}

/// Reads until every chat in `ids` is done; returns each one's text and `done`, and the ids of
/// the token messages in the order they came.
async fn finish(ws: &mut Ws, ids: &[&str]) -> (HashMap<String, (String, Value)>, Vec<String>) {
    let mut out: HashMap<String, (String, Value)> = HashMap::new();
    let mut order = Vec::new();
    let mut open: HashSet<String> = ids.iter().map(|s| s.to_string()).collect();
    while !open.is_empty() {
        let m = recv(ws).await;
        let id = m["id"].as_str().unwrap().to_string();
        assert!(open.contains(&id), "message for a chat already done or unknown: {m}");
        let entry = out.entry(id.clone()).or_insert_with(|| (String::new(), Value::Null));
        match m["type"].as_str().unwrap() {
            "token" => {
                entry.0.push_str(m["text"].as_str().unwrap());
                order.push(id);
            }
            "done" => {
                entry.1 = m;
                open.remove(&id);
            }
            _ => panic!("unexpected {m}"),
        }
    }
    (out, order)
}

/// The `/v1/chat/completions` text for `content`.
async fn completion(app: &Router, content: &str, max_tokens: usize) -> String {
    let body = json!({ "model": "mock", "messages": messages(content), "max_tokens": max_tokens });
    let req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v: Value = serde_json::from_slice(&resp.into_body().collect().await.unwrap().to_bytes()).unwrap();
    v["choices"][0]["message"]["content"].as_str().unwrap().to_string()
}

/// Waits off the runtime until `f` holds, for at most 5 s.
async fn until(what: &'static str, f: impl Fn() -> bool + Send + 'static) {
    tokio::task::spawn_blocking(move || {
        let t0 = Instant::now();
        while !f() {
            assert!(t0.elapsed() < Duration::from_secs(5), "timed out waiting for {what}");
            thread::sleep(Duration::from_millis(5));
        }
    })
    .await
    .unwrap();
}

/// One chat streams to `done`, with the text and usage `/v1/chat/completions` gives.
async fn one_chat_matches_completion(app: &Router, addr: SocketAddr) {
    let want = completion(app, "w5 w6 w7", 12).await;
    let mut ws = connect(addr).await;
    send(&mut ws, chat("a", "w5 w6 w7", 12)).await;
    let (out, order) = finish(&mut ws, &["a"]).await;
    let (text, done) = &out["a"];
    assert_eq!(text.trim(), want.trim());
    assert_eq!(order.len(), 12);
    assert_eq!(done["finish_reason"], "length");
    assert_eq!(done["usage"]["completion_tokens"], 12);
    assert!(done["usage"]["prompt_tokens"].as_u64().unwrap() > 3, "{done}");
}

/// Three chats at once on one socket, interleaved, each the same as on its own.
async fn chats_interleave(app: &Router, addr: SocketAddr) {
    let prompts = [("a", "w5 w6 w7"), ("b", "w9 w1"), ("c", "w30 w31 w32 w33")];
    let mut want = HashMap::new();
    for (id, p) in prompts {
        want.insert(id, completion(app, p, 20).await);
    }
    let mut ws = connect(addr).await;
    for (id, p) in prompts {
        send(&mut ws, chat(id, p, 20)).await;
    }
    let (out, order) = finish(&mut ws, &["a", "b", "c"]).await;
    for (id, _) in prompts {
        assert_eq!(out[id].0.trim(), want[id].trim(), "chat {id}");
        assert_eq!(out[id].1["usage"]["completion_tokens"], 20, "chat {id}");
    }
    // Three contiguous runs would be one chat after another.
    let switches = order.windows(2).filter(|w| w[0] != w[1]).count();
    assert!(switches > 2, "not interleaved: {order:?}");
}

/// Cancel mid-chat: `done` "cancelled", nothing more for it, and the socket keeps working.
async fn cancel_mid_chat(addr: SocketAddr) {
    let mut ws = connect(addr).await;
    send(&mut ws, chat("long", "w5 w6 w7", 100_000)).await;
    for _ in 0..3 {
        assert_eq!(recv(&mut ws).await["type"], "token");
    }
    send(&mut ws, json!({ "type": "cancel", "id": "long" })).await;
    // Tokens already on the way may come first.
    let done = loop {
        let m = recv(&mut ws).await;
        assert_eq!(m["id"], "long", "{m}");
        if m["type"] != "token" {
            break m;
        }
    };
    assert_eq!((&done["type"], &done["finish_reason"]), (&json!("done"), &json!("cancelled")), "{done}");
    assert!(done["usage"]["completion_tokens"].as_u64().unwrap() >= 3, "{done}");
    // A short chat after it: only its own messages arrive.
    send(&mut ws, chat("next", "w9 w1", 4)).await;
    let (out, _) = finish(&mut ws, &["next"]).await;
    assert_eq!(out["next"].1["finish_reason"], "length");
    // Nothing late for "long": the next message is the answer to this bad cancel.
    send(&mut ws, json!({ "type": "cancel", "id": "long" })).await;
    let m = recv(&mut ws).await;
    assert_eq!((&m["type"], &m["id"]), (&json!("error"), &json!("long")), "{m}");
    // An id is free again once its chat is cancelled; the new chat gets only its own tokens.
    for _ in 0..5 {
        send(&mut ws, chat("again", "w5 w6 w7", 100_000)).await;
        assert_eq!(recv(&mut ws).await["type"], "token");
        send(&mut ws, json!({ "type": "cancel", "id": "again" })).await;
        send(&mut ws, chat("again", "w9 w1", 4)).await;
        let (out, order) = finish(&mut ws, &["again"]).await;
        assert_eq!(out["again"].1["finish_reason"], "cancelled");
        let (out, order2) = finish(&mut ws, &["again"]).await;
        assert_eq!((order2.len(), &out["again"].1["finish_reason"]), (4, &json!("length")), "{order:?} {order2:?}");
    }
}

/// Duplicate id, unknown cancel id, bad JSON and a chat too long for the context.
async fn errors(addr: SocketAddr) {
    let mut ws = connect(addr).await;
    send(&mut ws, chat("x", "w5 w6 w7", 100_000)).await;
    assert_eq!(recv(&mut ws).await["type"], "token");
    send(&mut ws, chat("x", "w1", 3)).await;
    let m = loop {
        let m = recv(&mut ws).await;
        if m["type"] != "token" {
            break m;
        }
    };
    assert_eq!((&m["type"], &m["id"]), (&json!("error"), &json!("x")), "{m}");
    assert!(m["message"].as_str().unwrap().contains("duplicate"), "{m}");
    send(&mut ws, json!({ "type": "cancel", "id": "x" })).await;
    let done = loop {
        let m = recv(&mut ws).await;
        if m["type"] != "token" {
            break m;
        }
    };
    assert_eq!(done["finish_reason"], "cancelled", "{done}");

    send(&mut ws, json!({ "type": "cancel", "id": "nope" })).await;
    let m = recv(&mut ws).await;
    assert_eq!((&m["type"], &m["id"]), (&json!("error"), &json!("nope")), "{m}");
    for bad in ["{not json", r#"{"type":"dance","id":"y"}"#, r#"{"type":"chat","id":"z"}"#] {
        ws.send(Message::Text(bad.into())).await.unwrap();
        let m = recv(&mut ws).await;
        assert_eq!((&m["type"], &m["id"]), (&json!("error"), &Value::Null), "{bad}: {m}");
    }
    send(&mut ws, chat("big", "w5", 1 << 21)).await;
    let m = recv(&mut ws).await;
    assert_eq!((&m["type"], &m["id"]), (&json!("error"), &json!("big")), "{m}");
    assert!(m["message"].as_str().unwrap().contains("context"), "{m}");
}

#[tokio::test(flavor = "multi_thread")]
async fn ws_on_the_layer_split() {
    let (app, live) = local(1);
    let addr = serve(app.clone()).await;
    one_chat_matches_completion(&app, addr).await;
    chats_interleave(&app, addr).await;
    cancel_mid_chat(addr).await;
    errors(addr).await;
    until("cancelled chats freed", {
        let live = live.clone();
        move || live.lock().unwrap().is_empty()
    })
    .await;

    // Closing the socket cancels its chats.
    let mut ws = connect(addr).await;
    send(&mut ws, chat("a", "w5 w6 w7", 100_000)).await;
    send(&mut ws, chat("b", "w9 w1", 100_000)).await;
    for _ in 0..4 {
        recv(&mut ws).await;
    }
    assert_eq!(live.lock().unwrap().len(), 2);
    ws.close(None).await.unwrap();
    drop(ws);
    until("chats on a closed socket freed", move || live.lock().unwrap().is_empty()).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn ws_on_the_request_split() {
    let (app, b) = split(2, 1);
    let addr = serve(app.clone()).await;
    one_chat_matches_completion(&app, addr).await;
    chats_interleave(&app, addr).await;
    cancel_mid_chat(addr).await;
    errors(addr).await;
    let b2 = b.clone();
    until("cancelled chats gone", move || b2.requests().is_empty()).await;

    // Closing the socket cancels its chats: dropped without a close frame, as a lost client.
    let mut ws = connect(addr).await;
    for id in ["a", "b", "c"] {
        send(&mut ws, chat(id, "w5 w6 w7", 100_000)).await;
    }
    for _ in 0..6 {
        recv(&mut ws).await;
    }
    assert_eq!(b.requests().len(), 3);
    drop(ws);
    let b2 = b.clone();
    until("chats on a closed socket gone", move || b2.requests().is_empty()).await;
    assert!(b.workers().iter().all(|w| w.running == 0));
}

async fn command(app: &Router, worker: &str, body: Value) -> (StatusCode, Value) {
    let req = Request::post(format!("/v1/workers/{worker}/command"))
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

#[tokio::test(flavor = "multi_thread")]
async fn worker_commands() {
    let (up, _b) = split(1, 0);
    let (status, v) = command(&up, "n1", json!({ "command": "status" })).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["ok"], true, "{v}");
    assert!(!v["message"].as_str().unwrap().is_empty(), "{v}");
    let (status, v) = command(&up, "n1", json!({ "command": "reset_stats" })).await;
    assert_eq!((status, v["message"].is_string()), (StatusCode::OK, true), "{v}");
    assert_eq!(command(&up, "n1", json!({ "command": "dance" })).await.0, StatusCode::BAD_REQUEST);
    assert_eq!(command(&up, "n9", json!({ "command": "status" })).await.0, StatusCode::NOT_FOUND);

    // A worker nobody listens for is never connected.
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap().to_string();
    drop(l);
    let (client, _) = requests::spawn(&config(&[addr])).unwrap();
    let down = app(client);
    let (status, v) = command(&down, "n1", json!({ "command": "status" })).await;
    assert_eq!(status, StatusCode::CONFLICT, "{v}");
    assert_eq!(command(&down, "n9", json!({ "command": "status" })).await.0, StatusCode::NOT_FOUND);

    // Not on the other splits.
    let (local, _) = local(0);
    assert_eq!(command(&local, "n1", json!({ "command": "status" })).await.0, StatusCode::NOT_FOUND);
}
