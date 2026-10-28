//! The HTTP API over a one-node mock cluster, with a tiny word-level tokenizer.

use std::sync::Arc;

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
use msh_node::engine::MockEngine;
use msh_node::http::{router, HttpState};
use msh_node::scheduler::{self, SchedulerConfig};

const VOCAB: u32 = 64;

/// Words `w0`..`w63`; anything else is `w0`.
fn codec() -> Codec {
    let vocab = (0..VOCAB).map(|i| (format!("w{i}"), i)).collect();
    let model = WordLevel::builder().vocab(vocab).unk_token("w0".into()).build().unwrap();
    let mut tok = Tokenizer::new(model);
    tok.with_pre_tokenizer(Some(Whitespace {}));
    Codec::from_tokenizer(tok, ChatTemplate::Chatml)
}

fn app(sched: SchedulerConfig, n_ctx: usize) -> Router {
    let engine = MockEngine::new(0..4, 4, 64).with_vocab(VOCAB);
    let (client, _) = scheduler::spawn(engine, None, sched);
    router(Arc::new(HttpState { client, codec: codec(), model: "mock".into(), n_ctx }))
}

async fn post(app: &Router, body: Value) -> (StatusCode, String) {
    let req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

fn chat(max_tokens: usize, stream: bool) -> Value {
    json!({ "model": "mock", "messages": [{ "role": "user", "content": "w5 w6 w7" }], "max_tokens": max_tokens, "stream": stream })
}

#[tokio::test]
async fn lists_the_model() {
    let req = Request::get("/v1/models").body(Body::empty()).unwrap();
    let resp = app(SchedulerConfig::default(), 256).oneshot(req).await.unwrap();
    let body: Value = serde_json::from_slice(&resp.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["data"][0]["id"], "mock");
}

#[tokio::test]
async fn completion_and_stream_produce_the_same_text() {
    let app = app(SchedulerConfig::default(), 256);
    let (status, body) = post(&app, chat(12, false)).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let full: Value = serde_json::from_str(&body).unwrap();
    // The scheduler's request id, as in the coordinator's logs.
    assert_eq!(full["id"], "chatcmpl-1");
    assert_eq!(full["usage"]["completion_tokens"], 12);
    assert_eq!(full["choices"][0]["finish_reason"], "length");
    let text = full["choices"][0]["message"]["content"].as_str().unwrap().to_string();
    assert!(!text.is_empty());

    let (status, body) = post(&app, chat(12, true)).await;
    assert_eq!(status, StatusCode::OK);
    let events: Vec<&str> = body.lines().filter_map(|l| l.strip_prefix("data: ")).collect();
    assert_eq!(events.last(), Some(&"[DONE]"));
    let chunks: Vec<Value> = events[..events.len() - 1].iter().map(|e| serde_json::from_str(e).unwrap()).collect();
    let streamed: String = chunks.iter().filter_map(|c| c["choices"][0]["delta"]["content"].as_str()).collect();
    // Word-level pieces after the first carry a leading space; the joined text is the same.
    assert_eq!(streamed.trim(), text.trim());
    assert_eq!(chunks.last().unwrap()["choices"][0]["finish_reason"], "length");
    assert!(chunks.iter().all(|c| c["id"] == "chatcmpl-2"));
}

#[tokio::test]
async fn refuses_requests_that_do_not_fit_the_context() {
    let app = app(SchedulerConfig::default(), 16);
    let (status, body) = post(&app, chat(100, false)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body.contains("context"));
}

#[tokio::test]
async fn full_scheduler_answers_429() {
    let sched = SchedulerConfig { max_in_flight: 1, max_queue: 0, ..Default::default() };
    let app = app(sched, 1 << 20);
    // Hold the only slot with a long streaming request, then ask again.
    let req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(chat(500_000, true).to_string()))
        .unwrap();
    let long = app.clone().oneshot(req).await.unwrap();
    assert_eq!(long.status(), StatusCode::OK);
    let (status, body) = post(&app, chat(4, false)).await;
    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS, "{body}");
    // Dropping the stream cancels the long request and frees the slot.
    drop(long);
    let mut status = StatusCode::TOO_MANY_REQUESTS;
    for _ in 0..200 {
        status = post(&app, chat(4, false)).await.0;
        if status == StatusCode::OK {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert_eq!(status, StatusCode::OK);
}
