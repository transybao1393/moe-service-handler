//! OpenAI-compatible HTTP endpoint on the coordinator (CL-2).
//!
//! - `GET /v1/models`
//! - `POST /v1/chat/completions`, with `stream: true` for server-sent events.
//!
//! Sampling is greedy; `temperature`, `top_p` and other sampling fields are accepted and
//! ignored. A request the scheduler refuses maps to 429 (queue full) or 503 (cluster down or
//! shutting down). Closing a streaming connection cancels the request. A completion's id is
//! `chatcmpl-<request id>`, the id in the coordinator's logs.
//!
//! Memory stays bounded (NF-2): bodies over [`HTTP_BODY_LIMIT`] get 413, and at most
//! `max_open` requests are read and set up at once; more get 429 before their body is read.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{DefaultBodyLimit, Request, State};
use axum::http::StatusCode;
use axum::response::sse::{Event as SseEvent, Sse};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::{mpsc, Semaphore};
use tokio_stream::wrappers::ReceiverStream;

use crate::chat::{Codec, Detokenizer, Message};
use crate::memory::HTTP_BODY_LIMIT;
use crate::scheduler::{Client, Event};

pub struct HttpState {
    pub client: Client,
    pub codec: Codec,
    /// Name reported by `/v1/models` and in responses.
    pub model: String,
    /// Context length: prompt plus completion must fit.
    pub n_ctx: usize,
    open: Semaphore,
}

impl HttpState {
    pub fn new(client: Client, codec: Codec, model: String, n_ctx: usize, max_open: usize) -> Self {
        HttpState { client, codec, model, n_ctx, open: Semaphore::new(max_open) }
    }
}

pub fn router(state: Arc<HttpState>) -> Router {
    Router::new()
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(middleware::from_fn_with_state(state.clone(), limit_open))
        .layer(DefaultBodyLimit::max(HTTP_BODY_LIMIT))
        .with_state(state)
}

/// Holds one of `max_open` permits while a request's body is read and it is set up.
async fn limit_open(State(s): State<Arc<HttpState>>, req: Request, next: Next) -> Response {
    let Ok(_permit) = s.open.try_acquire() else {
        tracing::warn!("chat completion refused: too many open requests");
        return error(StatusCode::TOO_MANY_REQUESTS, "too many open requests");
    };
    next.run(req).await
}

#[derive(Deserialize)]
struct ChatRequest {
    messages: Vec<Message>,
    #[serde(default)]
    stream: bool,
    max_tokens: Option<usize>,
    max_completion_tokens: Option<usize>,
}

async fn models(State(s): State<Arc<HttpState>>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{ "id": s.model, "object": "model", "created": 0, "owned_by": "moe-service-handler" }],
    }))
}

async fn chat_completions(State(s): State<Arc<HttpState>>, Json(req): Json<ChatRequest>) -> Response {
    let prompt = match s.codec.encode_chat(&req.messages) {
        Ok(p) if !p.is_empty() => p,
        Ok(_) => return bad_request("empty prompt"),
        Err(e) => return bad_request(&e.to_string()),
    };
    let room = s.n_ctx.saturating_sub(prompt.len());
    if room == 0 {
        let msg = format!("prompt of {} tokens fills the {}-token context", prompt.len(), s.n_ctx);
        return bad_request(&msg);
    }
    let max_new = match req.max_completion_tokens.or(req.max_tokens) {
        Some(n) if n > room => {
            let msg = format!("prompt ({}) plus max_tokens ({n}) exceeds the {}-token context", prompt.len(), s.n_ctx);
            return bad_request(&msg);
        }
        Some(n) => n,
        None => room,
    };

    let prompt_tokens = prompt.len();
    let (request, rx) = s.client.submit_with_id(prompt, max_new);
    tracing::info!(request, prompt = prompt_tokens, max_new, stream = req.stream, "chat completion");
    let mut events = forward(rx);
    // Wait for the first event so a refused request gets a proper status, even when streaming.
    let first = match events.recv().await {
        Some(Event::Rejected(m) | Event::Failed(m)) => return error(status_for(&m), &m),
        Some(ev) => ev,
        None => return error(StatusCode::SERVICE_UNAVAILABLE, "scheduler stopped"),
    };
    let id = format!("chatcmpl-{request}");
    if req.stream {
        stream(s, id, first, events, max_new).into_response()
    } else {
        complete(s, id, first, events, prompt_tokens, max_new).await
    }
}

/// Moves a request's events from the scheduler's blocking channel onto an async one. When the
/// async side is dropped (the client went away), the blocking receiver is dropped too, which
/// cancels the request in the scheduler.
fn forward(rx: std::sync::mpsc::Receiver<Event>) -> mpsc::Receiver<Event> {
    let (tx, arx) = mpsc::channel(64);
    tokio::task::spawn_blocking(move || {
        for ev in rx {
            if tx.blocking_send(ev).is_err() {
                return;
            }
        }
    });
    arx
}

async fn complete(
    s: Arc<HttpState>,
    id: String,
    first: Event,
    mut events: mpsc::Receiver<Event>,
    prompt_tokens: usize,
    max_new: usize,
) -> Response {
    let mut tokens = Vec::new();
    let mut ev = Some(first);
    while let Some(e) = ev {
        match e {
            Event::Token { token, .. } => tokens.push(token),
            Event::Done => break,
            Event::Failed(m) | Event::Rejected(m) => return error(status_for(&m), &m),
        }
        ev = events.recv().await;
    }
    let content = match s.codec.decode(&tokens) {
        Ok(t) => t,
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    };
    Json(json!({
        "id": id,
        "object": "chat.completion",
        "created": unix_secs(),
        "model": s.model,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": content },
            "finish_reason": finish_reason(tokens.len(), max_new),
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": tokens.len(),
            "total_tokens": prompt_tokens + tokens.len(),
        },
    }))
    .into_response()
}

fn stream(
    s: Arc<HttpState>,
    id: String,
    first: Event,
    mut events: mpsc::Receiver<Event>,
    max_new: usize,
) -> Sse<ReceiverStream<Result<SseEvent, Infallible>>> {
    let (tx, rx) = mpsc::channel(64);
    tokio::spawn(async move {
        let created = unix_secs();
        let chunk = |delta: Value, finish: Option<&str>| {
            let body = json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": s.model,
                "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }],
            });
            Ok(SseEvent::default().data(body.to_string()))
        };
        // Every send fails once the client has gone; returning then drops `events`, which
        // cancels the request.
        if tx.send(chunk(json!({ "role": "assistant", "content": "" }), None)).await.is_err() {
            return;
        }
        let mut detok = Detokenizer::new(&s.codec);
        let mut generated = 0;
        let mut ev = Some(first);
        while let Some(e) = ev {
            match e {
                Event::Token { token, .. } => {
                    generated += 1;
                    let piece = match detok.push(token) {
                        Ok(p) => p,
                        Err(e) => return send_error(&tx, &e.to_string()).await,
                    };
                    if let Some(text) = piece {
                        if tx.send(chunk(json!({ "content": text }), None)).await.is_err() {
                            return;
                        }
                    }
                }
                Event::Done => break,
                Event::Failed(m) | Event::Rejected(m) => return send_error(&tx, &m).await,
            }
            ev = events.recv().await;
        }
        let _ = tx.send(chunk(json!({}), Some(finish_reason(generated, max_new)))).await;
        let _ = tx.send(Ok(SseEvent::default().data("[DONE]"))).await;
    });
    Sse::new(ReceiverStream::new(rx))
}

async fn send_error(tx: &mpsc::Sender<Result<SseEvent, Infallible>>, msg: &str) {
    let body = json!({ "error": { "message": msg, "type": "server_error" } });
    let _ = tx.send(Ok(SseEvent::default().data(body.to_string()))).await;
}

fn finish_reason(generated: usize, max_new: usize) -> &'static str {
    if generated >= max_new {
        "length"
    } else {
        "stop"
    }
}

/// Maps a scheduler refusal or failure to a status code.
fn status_for(msg: &str) -> StatusCode {
    if msg.starts_with("queue full") {
        StatusCode::TOO_MANY_REQUESTS
    } else if msg.contains("cluster down") || msg.contains("shutting down") || msg.contains("scheduler stopped") {
        StatusCode::SERVICE_UNAVAILABLE
    } else if msg.starts_with("empty prompt") {
        StatusCode::BAD_REQUEST
    } else {
        StatusCode::INTERNAL_SERVER_ERROR
    }
}

fn bad_request(msg: &str) -> Response {
    tracing::info!(error = msg, "chat completion refused");
    error(StatusCode::BAD_REQUEST, msg)
}

fn error(status: StatusCode, msg: &str) -> Response {
    let kind = if status.is_client_error() { "invalid_request_error" } else { "server_error" };
    (status, Json(json!({ "error": { "message": msg, "type": kind } }))).into_response()
}

fn unix_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}
