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

use msh_node::config::{expand_home, Config, KvFormat, NodeConfig, PrefetchGuess};
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
///
/// Always f32 KV, whatever the deployment configs use: these tests check the split against the
/// MLX reference token for token, and a TurboQuant KV cache changes the wording.
fn open(layers: std::ops::Range<u32>, max_requests: usize, mem_cap: Option<u64>) -> StreamMoeEngine {
    open_kv(layers, max_requests, mem_cap, KvFormat::F32)
}

fn open_kv(layers: std::ops::Range<u32>, max_requests: usize, mem_cap: Option<u64>, kv_format: KvFormat) -> StreamMoeEngine {
    let dir = model_dir();
    let spec = OpenSpec { n_ctx: 512, mem_cap, max_requests, kv_format, ..OpenSpec::new(&dir, layers) };
    StreamMoeEngine::open(&spec)
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
    start_kv(splits, sched, KvFormat::F32)
}

fn start_kv(splits: &[[u32; 2]], sched: SchedulerConfig, kv_format: KvFormat) -> Cluster {
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
        net: Default::default(),
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
    let mut listeners = listeners.into_iter();
    let coord_listener = listeners.next().unwrap();
    let mut threads = Vec::new();
    for (k, l) in listeners.enumerate() {
        let (i, cfg) = (k + 1, cfg.clone());
        let engine = open_kv(cfg.nodes[i].layer_range(), sched.engine_slots(), cap, kv_format);
        threads.push(thread::spawn(move || {
            let link = accept_controller(&cfg, i, l).unwrap();
            Worker::new(engine, link, &cfg.nodes[i].name).run().unwrap();
        }));
    }
    let engine = open_kv(cfg.nodes[0].layer_range(), sched.engine_slots(), cap, kv_format);
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
        net: Default::default(),
        // The same prompt four times: every one is for the workers.
        cache: msh_node::config::CacheConfig { mode: msh_node::config::CacheMode::Off, ..Default::default() },
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

#[test]
#[ignore = "needs the OLMoE model on disk"]
fn three_node_cluster_matches_one_node_on_turbo4_kv() {
    // TurboQuant changes the tokens against f32, but each node quantizes its own layers' KV as
    // one process would, so a layer split still matches one process on the same format.
    let sched = SchedulerConfig { prefill_chunk: 8, ..Default::default() };
    let one = start_kv(&[[0, 16]], sched, KvFormat::Turbo4).generate(PROMPT.to_vec(), 32).unwrap().tokens;
    let three = start_kv(&[[0, 8], [8, 12], [12, 16]], sched, KvFormat::Turbo4).generate(PROMPT.to_vec(), 32).unwrap().tokens;
    assert_eq!(three, one);
}

/// Set in the child processes of [`kv_formats_compared`]: the one format that process runs.
const KV_BENCH_FORMAT: &str = "MSH_KV_BENCH_FORMAT";
const KV_FORMATS: [(&str, KvFormat); 4] =
    [("f32", KvFormat::F32), ("f16", KvFormat::F16), ("turbo4", KvFormat::Turbo4), ("turbo3", KvFormat::Turbo3)];
const KV_BENCH_PROMPTS: [&str; 4] = [
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain how a refrigerator keeps food cold, step by step.",
    "Write a Python function that checks whether a string is a palindrome, then explain it.",
    "List ten facts about the planet Mars.",
];
const KV_BENCH_TOKENS: usize = 128;

/// The lifetime peak of the process's physical footprint, in bytes.
fn peak_footprint() -> u64 {
    let mut info = std::mem::MaybeUninit::<libc::rusage_info_v4>::zeroed();
    // SAFETY: `info` is a valid buffer for the V4 flavour, which the kernel fills.
    let rc = unsafe {
        libc::proc_pid_rusage(std::process::id() as i32, libc::RUSAGE_INFO_V4, info.as_mut_ptr().cast())
    };
    assert_eq!(rc, 0, "proc_pid_rusage");
    // SAFETY: filled on success.
    unsafe { info.assume_init() }.ri_lifetime_max_phys_footprint
}

/// KV formats on one whole-model worker, as in the request split: OLMoE, n_ctx 2048, the default
/// request slots, a 4 GiB cap (the smallest whole GiB f32 KV fits in). Prints the budget, the peak footprint, prefill and decode speed
/// and greedy-token agreement with f32 on short prompts (one pass each, from a cold hot tier).
/// Each format runs in its own process: the cap and the peak count the whole process.
#[test]
#[ignore = "needs the OLMoE model on disk; a bench that prints a table"]
fn kv_formats_compared() {
    if let Ok(name) = std::env::var(KV_BENCH_FORMAT) {
        let (_, kv_format) = *KV_FORMATS.iter().find(|(n, _)| *n == name).expect("format");
        return kv_bench_one(kv_format);
    }
    let mut rows = Vec::new();
    for (name, _) in KV_FORMATS {
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["kv_formats_compared", "--exact", "--ignored", "--nocapture", "--test-threads=1"])
            .env(KV_BENCH_FORMAT, name)
            .output()
            .unwrap();
        let (stdout, stderr) = (String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
        assert!(out.status.success(), "{name}:\n{stdout}\n{stderr}");
        // The test harness may print its "test … " prefix on the same line.
        let row = stdout.lines().find_map(|l| Some(l.split_once("kvbench ")?.1)).expect("kvbench line");
        let mut row: serde_json::Value = serde_json::from_str(row).unwrap();
        // streammoe's own line for the tier it opened, beside the plan's figure.
        let hot = stderr.lines().chain(stdout.lines()).find(|l| l.contains("hot_tier")).unwrap_or("");
        row["hot_line"] = hot.trim().into();
        rows.push((name, row));
    }
    let gib = |v: &serde_json::Value| v.as_f64().unwrap() / (1u64 << 30) as f64;
    let tokens = |r: &serde_json::Value| -> Vec<Vec<u64>> { serde_json::from_value(r["tokens"].clone()).unwrap() };
    let reference = tokens(&rows[0].1);
    println!(
        "\nOLMoE-1B-7B, one whole-model worker, 4 GiB cap, n_ctx 2048, {} request slots, {} prompts, up to {KV_BENCH_TOKENS} tokens",
        rows[0].1["slots"],
        KV_BENCH_PROMPTS.len()
    );
    println!(
        "| format | KV GiB | KV scratch GiB | fixed GiB | hot tier GiB (plan) | peak MiB | prefill ms (mean) | decode tok/s | tokens before first difference | share matching f32 |"
    );
    println!("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |");
    for (name, r) in &rows {
        let (mut same, mut total, mut firsts) = (0, 0, Vec::new());
        for (a, b) in reference.iter().zip(&tokens(r)) {
            let n = a.len().max(b.len());
            same += (0..n).filter(|&i| a.get(i).is_some() && a.get(i) == b.get(i)).count();
            total += n;
            firsts.push((0..n).find(|&i| a.get(i) != b.get(i)).map_or("all".to_string(), |i| i.to_string()));
        }
        println!(
            "| {name} | {:.3} | {:.3} | {:.2} | {:.2} | {:.0} | {:.0} | {:.1} | {} | {:.0}% |",
            gib(&r["kv"]),
            gib(&r["kv_scratch"]),
            gib(&r["fixed"]),
            gib(&r["hot_tier"]),
            r["peak"].as_f64().unwrap() / (1u64 << 20) as f64,
            r["prefill_ms"].as_f64().unwrap(),
            r["decode_tps"].as_f64().unwrap(),
            firsts.join(" / "),
            100.0 * same as f64 / total as f64
        );
    }
    for (name, r) in &rows {
        println!("{name}: {} tokens; {}", tokens(r).iter().map(Vec::len).sum::<usize>(), r["hot_line"].as_str().unwrap());
    }
}

/// One format of [`kv_formats_compared`], in this process; prints one `kvbench` JSON line.
fn kv_bench_one(kv_format: KvFormat) {
    use std::time::Instant;

    use msh_node::chat::{ChatTemplate, Codec, Message};
    use msh_node::engine::{Engine, Sample, StageIn};
    use msh_node::streammoe_engine::Budgeter;

    let dir = model_dir();
    let (cap, slots) = (4u64 << 30, SchedulerConfig::default().engine_slots());
    let spec = |mem_cap| OpenSpec { n_ctx: 2048, mem_cap, max_requests: slots, kv_format, ..OpenSpec::new(&dir, 0..16) };
    let budget = Budgeter::new(&spec(None)).unwrap().budget(0..16, cap);
    let codec = Codec::load(&dir.join("hf/tokenizer.json"), ChatTemplate::Olmoe).unwrap();
    let mut engine = StreamMoeEngine::open(&spec(Some(cap))).expect("open OLMoE under the cap");
    let eos = engine.eos();
    let (mut hidden, mut out) = (Vec::new(), Vec::new());
    let (mut prefill, mut decode, mut decoded, mut all) = (0.0, 0.0, 0, Vec::new());
    for (k, text) in KV_BENCH_PROMPTS.iter().enumerate() {
        let req = k as u64 + 1;
        let prompt = codec.encode_chat(&[Message { role: "user".into(), content: text.to_string() }]).unwrap();
        let t0 = Instant::now();
        engine.step(req, 0, StageIn::Tokens(&prompt), Sample::Last, &mut hidden, &mut out).unwrap();
        prefill += t0.elapsed().as_secs_f64();
        let mut tokens = vec![out[0]];
        let t0 = Instant::now();
        while tokens.len() < KV_BENCH_TOKENS && tokens.last().copied() != eos {
            let (pos, last) = ((prompt.len() + tokens.len() - 1) as u32, tokens[tokens.len() - 1]);
            engine.step(req, pos, StageIn::Tokens(&[last]), Sample::Last, &mut hidden, &mut out).unwrap();
            tokens.push(out[0]);
        }
        decode += t0.elapsed().as_secs_f64();
        decoded += tokens.len() - 1;
        engine.free(req);
        all.push(tokens);
    }
    let row = serde_json::json!({
        "slots": slots,
        "kv": budget.kv,
        "kv_scratch": budget.kv_scratch,
        "fixed": budget.fixed,
        "hot_tier": budget.hot_tier,
        "peak": peak_footprint(),
        "prefill_ms": 1e3 * prefill / KV_BENCH_PROMPTS.len() as f64,
        "decode_tps": decoded as f64 / decode,
        "tokens": all,
    });
    println!("kvbench {row}");
}
