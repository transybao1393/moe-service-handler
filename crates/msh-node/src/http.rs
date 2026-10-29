//! OpenAI-compatible HTTP endpoint on the coordinator (CL-2).
//!
//! - `GET /v1/models`
//! - `POST /v1/chat/completions`, with `stream: true` for server-sent events.
//! - `GET /v1/ws`: a WebSocket of JSON text messages carrying several chats at once. The client
//!   sends `{"type":"chat","id":"<its id>","messages":[...],"max_tokens":N}` (fields and checks
//!   as for chat completions) and `{"type":"cancel","id":"..."}`. The server answers
//!   `{"type":"token","id":"...","text":"..."}` per piece of text, then
//!   `{"type":"done","id":"...","finish_reason":"length"|"stop"|"cancelled","usage":{"prompt_tokens":n,"completion_tokens":n}}`,
//!   or `{"type":"error","id":"..."|null,"message":"..."}`. An error for a chat that was
//!   running ends it. Closing the socket cancels every chat on it.
//!
//! On the request split (docs/request-split.md), worker management, all JSON:
//!
//! - `GET /v1/workers`: every worker's row of the table; `GET /v1/workers/{name}`: one.
//! - `GET /v1/requests`: requests in flight and queued, then the last 16 answered from the
//!   controller's cache (phase `cached`, worker `controller`).
//! - `POST /v1/workers/{name}/drain` and `/resume`: stop or restart new requests to a worker.
//! - `POST /v1/workers/{name}/command` with `{"command": "status" | "reset_stats" | "restart"}`:
//!   sends it to the worker and answers `{"ok": bool, "message": str}` once the worker does.
//!   400 for an unknown command, 404 for an unknown worker, 409 when the worker is not
//!   connected, 504 after 10 s.
//! - `POST /v1/requests/{id}/cancel`: cancel one request (`7` or `chatcmpl-7`).
//! - `GET /v1/cache`: the answer cache's entries, bytes, hits and misses by kind, and limits;
//!   `POST /v1/cache/clear` empties it.
//!
//! An unknown worker or request gets 404, and so does all of it on the other splits.
//!
//! Sampling is greedy; `temperature`, `top_p` and other sampling fields are accepted and
//! ignored. A request the scheduler refuses maps to 429 (queue full) or 503 (cluster down or
//! shutting down). Closing a streaming connection cancels the request. A completion's id is
//! `chatcmpl-<request id>`, the id in the coordinator's logs.
//!
//! Memory stays bounded (NF-2): bodies over [`HTTP_BODY_LIMIT`] get 413, and at most
//! `max_open` requests are read and set up at once; more get 429 before their body is read.

use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::extract::ws::{Message as WsMessage, WebSocket, WebSocketUpgrade};
use axum::extract::{DefaultBodyLimit, Path, Request, State};
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
        .route("/v1/workers", get(workers))
        .route("/v1/workers/{name}", get(worker))
        .route("/v1/workers/{name}/drain", post(drain))
        .route("/v1/workers/{name}/resume", post(resume))
        .route("/v1/workers/{name}/command", post(command))
        .route("/v1/ws", get(ws))
        .route("/v1/requests", get(requests))
        .route("/v1/requests/{id}/cancel", post(cancel))
        .route("/v1/cache", get(cache))
        .route("/v1/cache/clear", post(clear_cache))
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

/// The request split's worker table, or 404 on the other splits.
fn board(s: &HttpState) -> Result<&crate::requests::Board, Response> {
    s.client
        .board()
        .map(|b| &**b)
        .ok_or_else(|| error(StatusCode::NOT_FOUND, "worker management needs split = \"requests\""))
}

fn found<T: serde::Serialize>(r: Result<T, String>) -> Response {
    match r {
        Ok(v) => Json(v).into_response(),
        Err(m) => error(StatusCode::NOT_FOUND, &m),
    }
}

async fn workers(State(s): State<Arc<HttpState>>) -> Response {
    match board(&s) {
        Ok(b) => Json(json!({ "ready": b.ready(), "workers": b.workers() })).into_response(),
        Err(r) => r,
    }
}

async fn worker(State(s): State<Arc<HttpState>>, Path(name): Path<String>) -> Response {
    match board(&s) {
        Ok(b) => found(b.worker(&name).ok_or_else(|| format!("no worker named {name}"))),
        Err(r) => r,
    }
}

async fn drain(State(s): State<Arc<HttpState>>, Path(name): Path<String>) -> Response {
    match board(&s) {
        Ok(b) => found(b.drain(&name)),
        Err(r) => r,
    }
}

async fn resume(State(s): State<Arc<HttpState>>, Path(name): Path<String>) -> Response {
    match board(&s) {
        Ok(b) => found(b.resume(&name)),
        Err(r) => r,
    }
}

async fn requests(State(s): State<Arc<HttpState>>) -> Response {
    match board(&s) {
        Ok(b) => Json(json!({ "requests": b.requests() })).into_response(),
        Err(r) => r,
    }
}

async fn cache(State(s): State<Arc<HttpState>>) -> Response {
    match board(&s) {
        Ok(b) => Json(b.cache()).into_response(),
        Err(r) => r,
    }
}

async fn clear_cache(State(s): State<Arc<HttpState>>) -> Response {
    match board(&s) {
        Ok(b) => Json(json!({ "cleared": b.clear_cache() })).into_response(),
        Err(r) => r,
    }
}

/// Takes the request id as the logs and completions show it: `7` or `chatcmpl-7`.
async fn cancel(State(s): State<Arc<HttpState>>, Path(id): Path<String>) -> Response {
    let b = match board(&s) {
        Ok(b) => b,
        Err(r) => return r,
    };
    match id.strip_prefix("chatcmpl-").unwrap_or(&id).parse::<u64>() {
        Ok(n) => found(b.cancel(n).map(|r| json!({ "cancelled": r }))),
        Err(_) => bad_request(&format!("bad request id {id:?}")),
    }
}

#[derive(Deserialize)]
struct CommandRequest {
    command: String,
}

/// Sends a command to a worker and waits, off the runtime, for its answer.
async fn command(
    State(s): State<Arc<HttpState>>,
    Path(name): Path<String>,
    Json(req): Json<CommandRequest>,
) -> Response {
    let b = match board(&s) {
        Ok(b) => b,
        Err(r) => return r,
    };
    let Some(kind) = msh_net::CommandKind::parse(&req.command) else {
        return bad_request(&format!("unknown command {:?}", req.command));
    };
    let rx = match b.command(&name, kind, Duration::from_secs(10)) {
        Ok(rx) => rx,
        Err(m) if m.starts_with("no worker") => return error(StatusCode::NOT_FOUND, &m),
        Err(m) => return error(StatusCode::CONFLICT, &m),
    };
    let outcome = tokio::task::spawn_blocking(move || rx.recv()).await;
    let (ok, message) = match outcome {
        Ok(Ok(o)) => (o.ok, o.message),
        _ => (false, "worker down".to_string()),
    };
    tracing::info!(worker = name, command = kind.as_str(), ok, message, "worker command");
    let status = if !ok && message == "timeout" { StatusCode::GATEWAY_TIMEOUT } else { StatusCode::OK };
    (status, Json(json!({ "ok": ok, "message": message }))).into_response()
}

/// Turns a chat into a prompt and a completion budget that fits the context.
fn prepare(s: &HttpState, messages: &[Message], max_tokens: Option<usize>) -> Result<(Vec<u32>, usize), String> {
    let prompt = match s.codec.encode_chat(messages) {
        Ok(p) if !p.is_empty() => p,
        Ok(_) => return Err("empty prompt".to_string()),
        Err(e) => return Err(e.to_string()),
    };
    let room = s.n_ctx.saturating_sub(prompt.len());
    if room == 0 {
        return Err(format!("prompt of {} tokens fills the {}-token context", prompt.len(), s.n_ctx));
    }
    match max_tokens {
        Some(n) if n > room => {
            Err(format!("prompt ({}) plus max_tokens ({n}) exceeds the {}-token context", prompt.len(), s.n_ctx))
        }
        Some(n) => Ok((prompt, n)),
        None => Ok((prompt, room)),
    }
}

async fn chat_completions(State(s): State<Arc<HttpState>>, Json(req): Json<ChatRequest>) -> Response {
    let (prompt, max_new) = match prepare(&s, &req.messages, req.max_completion_tokens.or(req.max_tokens)) {
        Ok(p) => p,
        Err(m) => return bad_request(&m),
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
        let mut pieces = Pieces::new(&s.codec, max_new);
        let mut ev = Some(first);
        while let Some(e) = ev {
            match e {
                Event::Token { token, .. } => {
                    let piece = match pieces.push(token) {
                        Ok(p) => p,
                        Err(m) => return send_error(&tx, &m).await,
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
        let _ = tx.send(chunk(json!({}), Some(pieces.finish_reason()))).await;
        let _ = tx.send(Ok(SseEvent::default().data("[DONE]"))).await;
    });
    Sse::new(ReceiverStream::new(rx))
}

/// Turns a request's tokens into text as they come, and counts them.
struct Pieces<'a> {
    detok: Detokenizer<'a>,
    generated: usize,
    max_new: usize,
}

impl<'a> Pieces<'a> {
    fn new(codec: &'a Codec, max_new: usize) -> Self {
        Pieces { detok: Detokenizer::new(codec), generated: 0, max_new }
    }

    fn push(&mut self, token: u32) -> Result<Option<String>, String> {
        self.generated += 1;
        self.detok.push(token).map_err(|e| e.to_string())
    }

    fn finish_reason(&self) -> &'static str {
        finish_reason(self.generated, self.max_new)
    }
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WsRequest {
    Chat {
        id: String,
        messages: Vec<Message>,
        max_tokens: Option<usize>,
        max_completion_tokens: Option<usize>,
    },
    Cancel {
        id: String,
    },
}

/// A message from one chat's task to its socket. `last` ends the chat. `request` tells a
/// cancelled chat's late messages from those of a new chat that reuses its id.
struct WsOut {
    id: String,
    request: u64,
    body: Value,
    last: bool,
}

/// A chat running on a socket: its request, and the task relaying its events.
struct WsChat {
    request: u64,
    task: tokio::task::AbortHandle,
    prompt_tokens: usize,
    generated: Arc<AtomicUsize>,
}

async fn ws(State(s): State<Arc<HttpState>>, upgrade: WebSocketUpgrade) -> Response {
    upgrade.max_message_size(HTTP_BODY_LIMIT).on_upgrade(move |socket| ws_serve(s, socket))
}

async fn ws_serve(s: Arc<HttpState>, mut socket: WebSocket) {
    let (tx, mut rx) = mpsc::channel::<WsOut>(64);
    let mut chats: HashMap<String, WsChat> = HashMap::new();
    loop {
        tokio::select! {
            msg = socket.recv() => {
                let text = match msg {
                    Some(Ok(WsMessage::Text(t))) => t,
                    Some(Ok(WsMessage::Close(_))) | Some(Err(_)) | None => break,
                    Some(Ok(_)) => continue,
                };
                let reply = match serde_json::from_str::<WsRequest>(text.as_str()) {
                    Ok(req) => ws_request(&s, &tx, &mut chats, req),
                    Err(e) => Some(ws_error(None, &format!("bad message: {e}"))),
                };
                if let Some(body) = reply {
                    if socket.send(WsMessage::Text(body.to_string().into())).await.is_err() {
                        break;
                    }
                }
            }
            Some(out) = rx.recv() => {
                // A chat cancelled from the socket may still have messages on the way; drop them.
                if chats.get(&out.id).map(|c| c.request) != Some(out.request) {
                    continue;
                }
                if out.last {
                    chats.remove(&out.id);
                }
                if socket.send(WsMessage::Text(out.body.to_string().into())).await.is_err() {
                    break;
                }
            }
        }
    }
    for (_, chat) in chats.drain() {
        ws_cancel(&s, &chat);
    }
}

/// Starts or cancels a chat; returns the message to send back at once, if any.
fn ws_request(
    s: &Arc<HttpState>,
    tx: &mpsc::Sender<WsOut>,
    chats: &mut HashMap<String, WsChat>,
    req: WsRequest,
) -> Option<Value> {
    match req {
        WsRequest::Cancel { id } => match chats.remove(&id) {
            Some(chat) => {
                ws_cancel(s, &chat);
                let generated = chat.generated.load(Ordering::Relaxed);
                Some(ws_done(&id, "cancelled", chat.prompt_tokens, generated))
            }
            None => Some(ws_error(Some(&id), &format!("no chat with id {id:?}"))),
        },
        WsRequest::Chat { id, messages, max_tokens, max_completion_tokens } => {
            if chats.contains_key(&id) {
                return Some(ws_error(Some(&id), &format!("duplicate id {id:?}")));
            }
            let (prompt, max_new) = match prepare(s, &messages, max_completion_tokens.or(max_tokens)) {
                Ok(p) => p,
                Err(m) => return Some(ws_error(Some(&id), &m)),
            };
            let prompt_tokens = prompt.len();
            let (request, rx) = s.client.submit_with_id(prompt, max_new);
            tracing::info!(request, prompt = prompt_tokens, max_new, client_id = id, "websocket chat");
            let generated = Arc::new(AtomicUsize::new(0));
            let relay =
                ws_chat(s.clone(), id.clone(), request, forward(rx), tx.clone(), generated.clone(), prompt_tokens, max_new);
            let task = tokio::spawn(relay).abort_handle();
            chats.insert(id, WsChat { request, task, prompt_tokens, generated });
            None
        }
    }
}

/// Relays one chat's events to its socket. `generated` counts its tokens for a cancel's usage.
async fn ws_chat(
    s: Arc<HttpState>,
    id: String,
    request: u64,
    mut events: mpsc::Receiver<Event>,
    tx: mpsc::Sender<WsOut>,
    generated: Arc<AtomicUsize>,
    prompt_tokens: usize,
    max_new: usize,
) {
    let mut pieces = Pieces::new(&s.codec, max_new);
    let last = loop {
        let text = match events.recv().await {
            Some(Event::Token { token, .. }) => {
                generated.fetch_add(1, Ordering::Relaxed);
                match pieces.push(token) {
                    Ok(Some(text)) => text,
                    Ok(None) => continue,
                    Err(m) => break ws_error(Some(&id), &m),
                }
            }
            Some(Event::Done) => break ws_done(&id, pieces.finish_reason(), prompt_tokens, pieces.generated),
            Some(Event::Failed(m) | Event::Rejected(m)) => break ws_error(Some(&id), &m),
            None => break ws_error(Some(&id), "scheduler stopped"),
        };
        let body = json!({ "type": "token", "id": id, "text": text });
        // The socket is gone; returning drops `events`, which cancels the request.
        if tx.send(WsOut { id: id.clone(), request, body, last: false }).await.is_err() {
            return;
        }
    };
    let _ = tx.send(WsOut { id, request, body: last, last: true }).await;
}

/// Stops a chat's relay, which cancels its request; on the request split, cancels it at once.
fn ws_cancel(s: &HttpState, chat: &WsChat) {
    chat.task.abort();
    if let Some(b) = s.client.board() {
        let _ = b.cancel(chat.request);
    }
}

fn ws_done(id: &str, finish_reason: &str, prompt_tokens: usize, completion_tokens: usize) -> Value {
    json!({
        "type": "done",
        "id": id,
        "finish_reason": finish_reason,
        "usage": { "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens },
    })
}

fn ws_error(id: Option<&str>, msg: &str) -> Value {
    json!({ "type": "error", "id": id, "message": msg })
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
