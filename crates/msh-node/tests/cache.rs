//! The controller's answer cache on the request split, on the mock engine: what it answers
//! without a worker and what it does not, its limits, its scope, its endpoints and series.

use std::net::TcpListener;
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use msh_node::cache;
use msh_node::chat::ChatTemplate;
use msh_node::config::{CacheConfig, CacheMode, Config, HttpConfig, KvFormat, NodeConfig, Split};
use msh_node::engine::MockEngine;
use msh_node::node::{accept_controller, RequestWorker};
use msh_node::requests::{self, Board};
use msh_node::scheduler::{self, Client, SchedulerConfig};
use msh_node::Error;

const DIM: usize = 64;
const LAYERS: u32 = 8;

fn mock() -> MockEngine {
    MockEngine::new(0..LAYERS, LAYERS, DIM).with_vocab(64)
}

fn config(addrs: &[String], cache: CacheConfig) -> Config {
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
        scheduler: SchedulerConfig { prefill_chunk: 16, ..Default::default() },
        engine: Default::default(),
        http: None,
        metrics: None,
        log: Default::default(),
        service: Default::default(),
        net: Default::default(),
        cache,
        nodes,
    };
    cfg.validate().unwrap();
    cfg
}

struct Cluster {
    client: Client,
    sched: JoinHandle<()>,
    workers: Vec<JoinHandle<()>>,
}

impl Cluster {
    fn start(workers: usize, cache: CacheConfig) -> Cluster {
        let ls: Vec<TcpListener> = (0..workers).map(|_| TcpListener::bind("127.0.0.1:0").unwrap()).collect();
        let addrs: Vec<String> = ls.iter().map(|l| l.local_addr().unwrap().to_string()).collect();
        let cfg = config(&addrs, cache);
        let workers = ls
            .into_iter()
            .enumerate()
            .map(|(k, l)| {
                let cfg = cfg.clone();
                thread::spawn(move || {
                    let mut w = RequestWorker::new(mock(), &cfg.nodes[k + 1].name, cfg.scheduler.prefill_chunk, None);
                    w.serve(accept_controller(&cfg, k + 1, l).unwrap()).unwrap();
                })
            })
            .collect();
        let (client, sched) = requests::spawn(&cfg).unwrap();
        let n = cfg.nodes.len() - 1;
        assert_eq!(client.board().unwrap().wait_ready(n, Duration::from_secs(5)), n);
        Cluster { client, sched, workers }
    }

    fn board(&self) -> &Board {
        self.client.board().unwrap()
    }

    /// Requests the workers finished.
    fn served(&self) -> u64 {
        self.board().workers().iter().map(|w| w.done).sum()
    }

    fn generate(&self, prompt: &[u32], max_new: usize) -> Vec<u32> {
        self.client.generate(prompt.to_vec(), max_new).unwrap().tokens
    }

    fn stop(self) {
        self.client.shutdown();
        self.sched.join().unwrap();
        for w in self.workers {
            w.join().unwrap();
        }
    }
}

fn mode(mode: CacheMode) -> CacheConfig {
    CacheConfig { mode, ..Default::default() }
}

/// One process, the reference answer.
fn reference(prompt: &[u32], max_new: usize) -> Vec<u32> {
    let (client, _) = scheduler::spawn(mock(), None, SchedulerConfig::default());
    client.generate(prompt.to_vec(), max_new).unwrap().tokens
}

fn long_prompt(seed: u32) -> Vec<u32> {
    (0..300).map(|k| (k * 7919 + seed * 104_729) % 50_000).collect()
}

#[test]
fn an_exact_repeat_is_answered_without_a_worker() {
    let c = Cluster::start(2, mode(CacheMode::Exact));
    let p = [11, 22, 33, 44, 55];
    let first = c.generate(&p, 8);
    assert_eq!(first, reference(&p, 8));
    assert_eq!(c.served(), 1);
    assert_eq!(c.generate(&p, 8), first);
    assert_eq!(c.served(), 1, "the repeat reached a worker");
    let s = c.board().cache();
    assert_eq!((s.mode, s.entries, s.hits.exact, s.misses.exact, s.hits.similar), ("exact", 1, 1, 1, 0));
    // Either worker's name is two bytes.
    assert_eq!(s.bytes, cache::Cache::entry_bytes(8, "n1", None));
    // The table shows it as the controller's, answered from the cache.
    let shown = c.board().requests();
    assert_eq!(shown.len(), 1);
    let r = &shown[0];
    assert_eq!((r.id, r.phase, r.worker.as_deref(), r.tokens_sent, r.prompt_tokens), (2, "cached", Some("controller"), 8, 5));
    let how = r.cached.as_ref().unwrap();
    assert_eq!((how.kind, how.similarity), ("exact", 1.0));
    assert!(how.made_by == "n1" || how.made_by == "n2");

    // With every worker drained a repeat is still answered; a new prompt is not.
    c.board().drain("n1").unwrap();
    c.board().drain("n2").unwrap();
    assert_eq!(c.generate(&p, 8), first);
    let err = c.client.generate(vec![9, 9, 9], 8).unwrap_err();
    assert!(matches!(&err, Error::Rejected(m) if m.contains("cluster down")), "{err}");
    c.board().resume("n1").unwrap();
    c.board().resume("n2").unwrap();

    // Another max_tokens is another answer: a worker makes it.
    assert_eq!(c.generate(&p, 7), first[..7]);
    assert_eq!(c.served(), 2);
    assert_eq!(c.board().cache().entries, 2);
    c.stop();
}

#[test]
fn similar_mode_answers_near_prompts_above_the_threshold() {
    let c = Cluster::start(1, mode(CacheMode::Similar));
    let p = long_prompt(1);
    let answer = c.generate(&p, 6);
    // One token changed: close enough, and answered with the cached prompt's answer.
    let mut near = p.clone();
    near[150] = 7;
    assert_ne!(reference(&near, 6), answer, "the mock should answer the near prompt differently");
    assert_eq!(c.generate(&near, 6), answer);
    assert_eq!(c.served(), 1);
    // Ten changed: below the threshold, so a worker answers it.
    let mut far = p.clone();
    for k in 0..10 {
        far[20 + 25 * k] = 7 + k as u32;
    }
    assert_eq!(c.generate(&far, 6), reference(&far, 6));
    assert_eq!(c.served(), 2);
    // Close, but another max_tokens.
    assert_eq!(c.generate(&near, 5), reference(&near, 5));
    assert_eq!(c.served(), 3);
    let s = c.board().cache();
    assert_eq!((s.hits.exact, s.hits.similar, s.misses.exact, s.misses.similar), (0, 1, 4, 3));
    assert_eq!((s.entries, s.bits, s.dim, s.similar_threshold), (3, 4, cache::VECTOR_DIM, 0.98));
    assert!(s.last_scan_us.is_some());
    c.stop();
}

#[test]
fn exact_mode_does_not_answer_near_prompts() {
    let c = Cluster::start(1, mode(CacheMode::Exact));
    let p = long_prompt(2);
    c.generate(&p, 4);
    let mut near = p.clone();
    near[150] = 7;
    assert_eq!(c.generate(&near, 4), reference(&near, 4));
    assert_eq!(c.served(), 2);
    assert_eq!(c.board().cache().misses.similar, 0);
    c.stop();
}

#[test]
fn off_never_answers_from_the_cache() {
    let c = Cluster::start(1, mode(CacheMode::Off));
    let p = [5, 6, 7];
    let want = c.generate(&p, 5);
    for _ in 0..3 {
        assert_eq!(c.generate(&p, 5), want);
    }
    assert_eq!(c.served(), 4);
    let s = c.board().cache();
    assert_eq!((s.mode, s.entries, s.hits.exact, s.misses.exact), ("off", 0, 0, 0));
    c.stop();
}

#[test]
fn eviction_keeps_to_the_limits() {
    let c = Cluster::start(1, CacheConfig { max_entries: 2, ..mode(CacheMode::Similar) });
    let ps: Vec<Vec<u32>> = (0..3).map(|i| vec![100 + i, 200 + i, 300 + i, 400 + i]).collect();
    for p in &ps {
        c.generate(p, 4);
    }
    let s = c.board().cache();
    assert_eq!((s.entries, s.evicted, s.inserted), (2, 1, 3));
    // The oldest went: a worker answers it again, and it pushes out the next oldest.
    c.generate(&ps[0], 4);
    assert_eq!(c.served(), 4);
    c.generate(&ps[2], 4);
    assert_eq!(c.served(), 4);
    c.generate(&ps[1], 4);
    assert_eq!(c.served(), 5);
    c.stop();

    // By bytes: room for two answers of 40 tokens.
    let per = cache::Cache::entry_bytes(40, "n1", Some(4));
    let max_mib = (2 * per + per / 2) as f64 / (1u64 << 20) as f64;
    let c = Cluster::start(1, CacheConfig { max_mib, ..mode(CacheMode::Similar) });
    for p in &ps {
        c.generate(p, 40);
        let s = c.board().cache();
        assert!(s.bytes <= s.max_bytes, "{} over {}", s.bytes, s.max_bytes);
    }
    let s = c.board().cache();
    assert_eq!((s.entries, s.evicted, s.bytes), (2, 1, 2 * per));
    c.stop();
}

#[test]
fn the_scope_covers_the_model_template_and_engine() {
    let base = config(&["127.0.0.1:2".into()], CacheConfig::default());
    let http = |model: &str, t| HttpConfig { listen: "x".into(), tokenizer: "t".into(), chat_template: t, model: model.into() };
    let with = |f: &dyn Fn(&mut Config)| {
        let mut c = base.clone();
        c.http = Some(http("olmoe", ChatTemplate::Olmoe));
        f(&mut c);
        cache::scope(&c)
    };
    let a = with(&|_| {});
    assert_eq!(a, with(&|_| {}));
    assert_ne!(a, cache::scope(&base));
    assert_ne!(a, with(&|c| c.http = Some(http("other", ChatTemplate::Olmoe))));
    assert_ne!(a, with(&|c| c.http = Some(http("olmoe", ChatTemplate::Chatml))));
    assert_ne!(a, with(&|c| c.engine.model_dir = "/m2".into()));
    assert_ne!(a, with(&|c| c.engine.kv_format = KvFormat::Turbo3));
    // What does not change an answer leaves it.
    assert_eq!(a, with(&|c| c.scheduler.max_in_flight = 7));
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

    use msh_node::chat::Codec;
    use msh_node::http::{router, HttpState};

    fn codec() -> Codec {
        let vocab = (0..64).map(|i| (format!("w{i}"), i)).collect();
        let model = WordLevel::builder().vocab(vocab).unk_token("w0".into()).build().unwrap();
        let mut tok = Tokenizer::new(model);
        tok.with_pre_tokenizer(Some(Whitespace {}));
        Codec::from_tokenizer(tok, ChatTemplate::Chatml)
    }

    async fn call(app: &Router, method: &str, path: &str, body: Option<Value>) -> (StatusCode, String) {
        let req = Request::builder()
            .method(method)
            .uri(path)
            .header("content-type", "application/json")
            .body(body.map_or(Body::empty(), |b| Body::from(b.to_string())))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        (status, String::from_utf8(bytes.to_vec()).unwrap())
    }

    async fn json_of(app: &Router, method: &str, path: &str) -> Value {
        let (status, body) = call(app, method, path, None).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        serde_json::from_str(&body).unwrap()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_cached_answer_streams_like_a_live_one_and_the_endpoints_report_it() {
        let c = Cluster::start(1, mode(CacheMode::Exact));
        let app = router(Arc::new(HttpState::new(c.client.clone(), codec(), "mock".into(), 1 << 20, 64)));
        let chat = |stream| json!({ "messages": [{ "role": "user", "content": "w5 w6 w7" }], "max_tokens": 12, "stream": stream });

        let (status, live) = call(&app, "POST", "/v1/chat/completions", Some(chat(false))).await;
        assert_eq!(status, StatusCode::OK, "{live}");
        let live: Value = serde_json::from_str(&live).unwrap();
        let text = live["choices"][0]["message"]["content"].as_str().unwrap().to_string();
        // The same request again, plain and streamed: from the cache, the same answer.
        let (_, cached) = call(&app, "POST", "/v1/chat/completions", Some(chat(false))).await;
        let cached: Value = serde_json::from_str(&cached).unwrap();
        assert_eq!(cached["choices"], live["choices"]);
        assert_eq!(cached["usage"], live["usage"]);
        let (status, body) = call(&app, "POST", "/v1/chat/completions", Some(chat(true))).await;
        assert_eq!(status, StatusCode::OK);
        let events: Vec<&str> = body.lines().filter_map(|l| l.strip_prefix("data: ")).collect();
        assert_eq!(events.last(), Some(&"[DONE]"));
        let chunks: Vec<Value> = events[..events.len() - 1].iter().map(|e| serde_json::from_str(e).unwrap()).collect();
        let streamed: String = chunks.iter().filter_map(|c| c["choices"][0]["delta"]["content"].as_str()).collect();
        assert_eq!(streamed.trim(), text.trim());
        assert_eq!(chunks.last().unwrap()["choices"][0]["finish_reason"], live["choices"][0]["finish_reason"]);
        assert_eq!(c.served(), 1);

        let s = json_of(&app, "GET", "/v1/cache").await;
        assert_eq!((&s["mode"], &s["entries"], &s["similar_threshold"]), (&json!("exact"), &json!(1), &json!(0.98)));
        assert_eq!((&s["hits"]["exact"], &s["hits"]["similar"], &s["misses"]["exact"]), (&json!(2), &json!(0), &json!(1)));
        assert!(s["bytes"].as_u64().unwrap() > 0 && s["max_bytes"] == json!(32 << 20), "{s}");

        // The series: this process's every cache, so at least what this test did.
        let text = msh_node::stats::render("n0", true);
        let value = |line: &str| -> f64 {
            let l = text.lines().find(|l| l.starts_with(line)).unwrap_or_else(|| panic!("{line} missing from\n{text}"));
            l.rsplit(' ').next().unwrap().parse().unwrap()
        };
        assert!(value("msh_cache_hits_total{node=\"n0\",kind=\"exact\"}") >= 2.0);
        assert!(value("msh_cache_misses_total{node=\"n0\",kind=\"exact\"}") >= 1.0);
        value("msh_cache_hits_total{node=\"n0\",kind=\"similar\"}");
        value("msh_cache_misses_total{node=\"n0\",kind=\"similar\"}");
        value("msh_cache_entries{node=\"n0\"}");
        value("msh_cache_bytes{node=\"n0\"}");

        let shown = json_of(&app, "GET", "/v1/requests").await;
        let phases: Vec<(&Value, &Value)> = shown["requests"].as_array().unwrap().iter().map(|r| (&r["phase"], &r["worker"])).collect();
        assert_eq!(phases, [(&json!("cached"), &json!("controller")); 2]);
        assert_eq!(shown["requests"][0]["cached"]["kind"], "exact");

        assert_eq!(json_of(&app, "POST", "/v1/cache/clear").await["cleared"], 1);
        assert_eq!(json_of(&app, "GET", "/v1/cache").await["entries"], 0);
        // Cleared: the next one goes to the worker.
        assert_eq!(call(&app, "POST", "/v1/chat/completions", Some(chat(false))).await.0, StatusCode::OK);
        assert_eq!(c.served(), 2);
        drop(app);
        c.stop();
    }
}
