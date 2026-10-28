//! The request split's controller (`split = "requests"`, docs/request-split.md): no engine,
//! a queue, and the management of every worker.
//!
//! Each worker runs the whole model and serves whole requests ([`crate::node::RequestWorker`]).
//! The controller queues requests and gives each to a ready worker with the fewest running,
//! then the best recent tokens per second, up to `max_in_flight` per worker. It sends `Start`
//! with the prompt and the token limit; the worker sends each `Token` as it is made and a
//! `Done` when it stops at the limit or the end-of-sequence token. To end a request early
//! (cancel, stall, the client went away) the controller tells the client at once and sends
//! `Drop`; the worker's slot is taken until its `Done` comes back.
//!
//! The [`Board`] is the worker table, kept by the scheduler thread and read by the HTTP API:
//! per worker its state (`connecting`, `ready`, `draining`, `stuck`, `down`) and since when,
//! the health ping, what it runs (per request its phase, position, tokens sent, start and last
//! progress), done and failed counts, recent tokens and prompt positions per second and
//! errors in the last minute, its footprint and cap from its status report, and its last
//! error. Its actions ([`Board::drain`], [`Board::resume`], [`Board::cancel`],
//! [`Board::mark_stuck`], [`Board::mark_down`]) are plain calls, for the HTTP API or a policy.
//!
//! Health: every `health_interval_ms` each connected worker gets a `Ping`, and its `Pong`
//! carries its status report. A ping not back within `health_timeout_ms` makes the worker
//! stuck: it gets no new requests until a ping is answered, and its requests show what they
//! were on. A request without progress (a token, or prompt positions in a status report) for
//! `stall_timeout_ms` fails alone and is dropped on its worker. A worker whose connection
//! breaks is down: its requests that have sent no token yet go back to the front of the queue,
//! the others fail, and a background thread reconnects, forever, waiting `[net]
//! reconnect_min_ms` doubling to `reconnect_max_ms` between tries (the table shows the tries
//! and when the next is due); after its `Hello` it is ready again. A failed write on a link
//! counts as the link broken.
//! The cluster takes requests while at least one worker is ready; with none, new and queued
//! requests are refused as "cluster down". Every state change is one log line: worker, from,
//! to, reason.
//!
//! The controller keeps its own answer cache ([`crate::cache`], `[cache]`), behind its own
//! lock. A submitted request is looked up on the caller's thread (the HTTP handler), after its
//! prompt is tokenized and before the queue: a prompt it has answered (or in similar mode, one
//! close to it) gets that answer streamed back at once, as if from a worker, skipping the
//! queue, the capacity checks and every worker, even while no worker is ready. Such a request
//! is one log line with its kind and similarity, and shows in the table with phase `cached`
//! and worker `controller`. The scheduler thread fills the cache when a worker's `Done` ends a
//! request at the limit or the end-of-sequence token; failed, cancelled, stalled and dropped
//! requests are not kept.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use msh_net::{
    decode_done, decode_status, decode_token, encode_start, Connection, DoneReason, FrameError, Header, MsgType, Status,
    Writer,
};
use serde::Serialize;
use tracing::{debug, info, warn};

use crate::backoff::Backoff;
use crate::cache::{Cache, CacheStats, Hit, Probe};
use crate::config::{Config, NetConfig, Split};
use crate::scheduler::{Client, Event, Msg, SchedulerConfig};
use crate::stats::{WorkerSeries, STATS};
use crate::Error;

/// What a worker's link tells the scheduler. `link` numbers the worker's connections, so
/// news from an old one is ignored.
pub(crate) enum Note {
    Frame { worker: usize, link: u64, h: Header, payload: Vec<u8> },
    Lost { worker: usize, link: u64, reason: String },
    Joined { worker: usize, conn: Connection },
    /// A reconnect try failed; the next comes after `wait`.
    Retry { worker: usize, tries: u32, wait: Duration },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum WorkerState {
    /// Not connected yet since the controller started.
    Connecting,
    /// Takes new requests.
    Ready,
    /// Finishes what it runs and takes nothing new.
    Draining,
    /// Connected, but a health ping is overdue; takes nothing new.
    Stuck,
    /// Connection lost; reconnecting.
    Down,
}

impl WorkerState {
    pub fn as_str(self) -> &'static str {
        match self {
            WorkerState::Connecting => "connecting",
            WorkerState::Ready => "ready",
            WorkerState::Draining => "draining",
            WorkerState::Stuck => "stuck",
            WorkerState::Down => "down",
        }
    }
}

/// One worker's row of the table.
#[derive(Clone, Debug, Serialize)]
pub struct WorkerView {
    pub name: String,
    pub addr: String,
    pub state: WorkerState,
    /// When it entered the state (Unix ms), how long ago, and why.
    pub since: u64,
    pub state_ms: u64,
    pub reason: String,
    pub ping_ms: Option<f64>,
    /// When the last ping came back (Unix ms).
    pub last_pong: Option<u64>,
    pub running: usize,
    pub max_in_flight: usize,
    /// What it is doing now.
    pub requests: Vec<RequestView>,
    pub done: u64,
    pub failed: u64,
    /// Recent tokens made and prompt positions taken per second, and errors in the last minute.
    pub tokens_per_s: f64,
    pub prefill_per_s: f64,
    pub errors_last_min: u64,
    /// From its last status report, in bytes; cap 0 is no cap.
    pub footprint: u64,
    pub cap: u64,
    pub last_error: Option<LastError>,
    /// While connecting or down: failed reconnect tries so far, and when the next is due
    /// (Unix ms).
    pub reconnect_tries: u32,
    pub next_try: Option<u64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct LastError {
    /// Unix ms.
    pub at: u64,
    pub request: Option<u64>,
    pub message: String,
}

/// A request in flight or queued.
#[derive(Clone, Debug, Serialize)]
pub struct RequestView {
    pub id: u64,
    pub worker: Option<String>,
    /// `queued`, `prefill`, `decode`, or `dropping` (over for the client, the worker's slot
    /// not yet free).
    pub phase: &'static str,
    /// Positions the worker holds.
    pub position: u32,
    pub prompt_tokens: usize,
    pub tokens_sent: u32,
    pub max_tokens: usize,
    /// Unix ms.
    pub submitted: u64,
    pub started: Option<u64>,
    pub last_progress: Option<u64>,
    /// Time in the queue, so far or before it started.
    pub waiting_ms: u64,
    /// Time since its last progress, once started.
    pub idle_ms: Option<u64>,
    /// Times it went back to the queue after its worker was lost.
    pub retries: u32,
    /// Set while its worker is stuck: what it was on.
    pub stuck: Option<String>,
    /// Answered from the cache: how.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached: Option<CachedView>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CachedView {
    /// `exact` or `similar`.
    pub kind: &'static str,
    pub similarity: f32,
    /// The worker that made the answer.
    pub made_by: String,
}

/// Requests answered from the cache that the table keeps showing.
const SERVED_KEPT: usize = 16;

/// The answer cache and the requests it answered lately, behind their own lock so lookups
/// and reads never wait on the scheduler. Lock order: `Board::inner`, then this.
struct CacheState {
    cache: Cache,
    served: VecDeque<RequestView>,
    shutting_down: bool,
}

fn lock_cache(c: &Mutex<CacheState>) -> MutexGuard<'_, CacheState> {
    c.lock().unwrap_or_else(|e| e.into_inner())
}

/// Recent events, for rates and counts over a window.
#[derive(Default)]
struct Recent {
    events: VecDeque<(Instant, u64)>,
}

impl Recent {
    fn add(&mut self, now: Instant, n: u64, window: Duration) {
        self.events.push_back((now, n));
        self.trim(now, window);
    }

    fn trim(&mut self, now: Instant, window: Duration) {
        while self.events.front().is_some_and(|(t, _)| now.duration_since(*t) > window) {
            self.events.pop_front();
        }
    }

    fn sum(&self, now: Instant, window: Duration) -> u64 {
        self.events.iter().filter(|(t, _)| now.duration_since(*t) <= window).map(|(_, n)| n).sum()
    }

    /// Per second over `window`, or over the time since `since` if shorter.
    fn per_s(&self, now: Instant, window: Duration, since: Instant) -> f64 {
        let span = now.duration_since(since).min(window).as_secs_f64().max(0.5);
        self.sum(now, window) as f64 / span
    }
}

const RATE_WINDOW: Duration = Duration::from_secs(10);
const ERROR_WINDOW: Duration = Duration::from_secs(60);

struct Slot {
    name: String,
    addr: String,
    state: WorkerState,
    since: Instant,
    reason: String,
    /// Set by `drain`: the worker is draining whenever it is up.
    drain: bool,
    send: Option<Writer>,
    link: u64,
    /// Failed reconnect tries since the link was lost, and when the next is due.
    tries: u32,
    next_try: Option<Instant>,
    /// Since when the current link is up.
    up_since: Instant,
    /// Outstanding ping: sequence number and send time.
    ping: Option<(u64, Instant)>,
    next_ping_at: Instant,
    ping_ms: Option<f64>,
    last_pong: Option<Instant>,
    /// Requests it runs, in start order.
    running: Vec<u64>,
    done: u64,
    failed: u64,
    tokens: Recent,
    prefill: Recent,
    errors: Recent,
    footprint: u64,
    cap: u64,
    last_error: Option<(Instant, Option<u64>, String)>,
}

impl Slot {
    fn takes_work(&self, max_in_flight: usize) -> bool {
        self.state == WorkerState::Ready && self.send.is_some() && self.running.len() < max_in_flight
    }

    fn tokens_per_s(&self, now: Instant) -> f64 {
        self.tokens.per_s(now, RATE_WINDOW, self.up_since)
    }
}

struct Req {
    id: u64,
    prompt: Vec<u32>,
    max_new: usize,
    events: Sender<Event>,
    submitted: Instant,
    worker: Option<usize>,
    started: Option<Instant>,
    /// Time in the queue before its latest start.
    waited: Duration,
    decoding: bool,
    position: u32,
    sent: u32,
    last_progress: Instant,
    last_token: Option<Instant>,
    retries: u32,
    /// The client has its outcome; the worker's `Done` frees the slot.
    dropping: bool,
    /// Its cache key, and the tokens sent so far, for the cache once it is done.
    probe: Option<Probe>,
    answer: Vec<u32>,
}

impl Req {
    /// What it is on, e.g. "decode, token 212 of 500".
    fn doing(&self) -> String {
        match (self.worker, self.decoding) {
            (None, _) => "queued".into(),
            (Some(_), false) => format!("prefill, position {} of {}", self.position, self.prompt.len()),
            (Some(_), true) => format!("decode, token {} of {}", self.sent, self.max_new),
        }
    }
}

/// The worker table and the request split's state, shared by the scheduler thread and the
/// HTTP API. Every method takes the lock for its own duration.
pub struct Board {
    inner: Mutex<Inner>,
    cache: Arc<Mutex<CacheState>>,
}

struct Inner {
    me: String,
    cfg: SchedulerConfig,
    net: NetConfig,
    workers: Vec<Slot>,
    reqs: HashMap<u64, Req>,
    queue: VecDeque<u64>,
    tx: Sender<Msg>,
    span: tracing::Span,
    /// Tells the reconnecting threads to give up.
    stop: Arc<AtomicBool>,
    shutting_down: bool,
    ping_seq: u64,
    buf: Vec<u8>,
    tokens: Vec<u32>,
    stages: Vec<u32>,
    status: Status,
    cache: Arc<Mutex<CacheState>>,
}

/// Starts the request split's scheduler for `cfg`: node 0 is this controller, every other
/// node a worker, each connected (and reconnected) in the background. Use
/// [`Board::wait_ready`] on [`Client::board`] to wait for them.
pub fn spawn(cfg: &Config) -> Result<(Client, JoinHandle<()>), Error> {
    if cfg.split != Split::Requests {
        return Err(Error::Config("the request scheduler needs split = \"requests\"".into()));
    }
    let (tx, rx) = mpsc::channel();
    let span = tracing::Span::current();
    let now = Instant::now();
    let workers = cfg.nodes[1..]
        .iter()
        .map(|n| Slot {
            name: n.name.clone(),
            addr: n.addr.clone(),
            state: WorkerState::Connecting,
            since: now,
            reason: "controller started".into(),
            drain: false,
            send: None,
            link: 0,
            tries: 0,
            next_try: None,
            up_since: now,
            ping: None,
            next_ping_at: now,
            ping_ms: None,
            last_pong: None,
            running: Vec::new(),
            done: 0,
            failed: 0,
            tokens: Recent::default(),
            prefill: Recent::default(),
            errors: Recent::default(),
            footprint: 0,
            cap: 0,
            last_error: None,
        })
        .collect();
    let inner = Inner {
        me: cfg.nodes[0].name.clone(),
        cfg: cfg.scheduler,
        net: cfg.net,
        workers,
        reqs: HashMap::new(),
        queue: VecDeque::new(),
        tx: tx.clone(),
        span: span.clone(),
        stop: Arc::new(AtomicBool::new(false)),
        shutting_down: false,
        ping_seq: 0,
        buf: Vec::new(),
        tokens: Vec::new(),
        stages: Vec::new(),
        status: Status::default(),
        cache: Arc::new(Mutex::new(CacheState {
            cache: Cache::new(&cfg.cache, crate::cache::scope(cfg)),
            served: VecDeque::new(),
            shutting_down: false,
        })),
    };
    for w in 0..inner.workers.len() {
        inner.reconnect(w);
    }
    let cache = inner.cache.clone();
    let board = Arc::new(Board { inner: Mutex::new(inner), cache });
    let client = Client::new(tx, Some(board.clone()));
    let handle = thread::spawn(move || span.in_scope(|| run(board, rx)));
    Ok((client, handle))
}

fn run(board: Arc<Board>, rx: Receiver<Msg>) {
    let tick = {
        let c = board.lock().cfg;
        let shortest = c.health_interval_ms.min(c.health_timeout_ms).min(c.stall_timeout_ms).max(1);
        Duration::from_millis((shortest / 4).clamp(5, 100))
    };
    loop {
        let first = rx.recv_timeout(tick);
        let mut b = board.lock();
        match first {
            Ok(m) => {
                b.handle(m);
                while let Ok(m) = rx.try_recv() {
                    b.handle(m);
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => b.begin_shutdown(),
        }
        b.tick();
        if b.shutting_down && b.reqs.is_empty() {
            return b.stop();
        }
    }
}

fn unix_ms(at: Instant) -> u64 {
    let t = SystemTime::now().checked_sub(at.elapsed()).unwrap_or(UNIX_EPOCH);
    t.duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
}

/// A time to come as Unix ms.
fn unix_ms_ahead(at: Instant) -> u64 {
    let t = SystemTime::now() + at.saturating_duration_since(Instant::now());
    t.duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
}

fn ms(d: Duration) -> u64 {
    d.as_millis() as u64
}

impl Board {
    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Every worker's row, in config order.
    pub fn workers(&self) -> Vec<WorkerView> {
        let b = self.lock();
        (0..b.workers.len()).map(|w| b.view(w)).collect()
    }

    pub fn worker(&self, name: &str) -> Option<WorkerView> {
        let b = self.lock();
        b.find(name).ok().map(|w| b.view(w))
    }

    /// Requests in flight, then queued, each in id order.
    pub fn requests(&self) -> Vec<RequestView> {
        let b = self.lock();
        let mut ids: Vec<u64> = b.reqs.keys().copied().filter(|id| !b.queue.contains(id)).collect();
        ids.sort_unstable();
        ids.extend(b.queue.iter().copied());
        let mut views: Vec<RequestView> = ids.iter().map(|id| b.request_view(&b.reqs[id])).collect();
        views.extend(lock_cache(&self.cache).served.iter().cloned());
        views
    }

    pub fn request(&self, id: u64) -> Option<RequestView> {
        let b = self.lock();
        b.reqs.get(&id).map(|r| b.request_view(r))
    }

    /// Workers that take new requests now.
    pub fn ready(&self) -> usize {
        self.lock().ready()
    }

    /// Waits until `n` workers are ready or `timeout` passes; returns how many are.
    pub fn wait_ready(&self, n: usize, timeout: Duration) -> usize {
        let deadline = Instant::now() + timeout;
        loop {
            let ready = self.ready();
            if ready >= n || Instant::now() >= deadline {
                return ready;
            }
            thread::sleep(Duration::from_millis(10));
        }
    }

    /// Stops new requests to a worker; it finishes what it runs.
    pub fn drain(&self, name: &str) -> Result<WorkerView, String> {
        let mut b = self.lock();
        let w = b.find(name)?;
        b.workers[w].drain = true;
        if b.workers[w].state == WorkerState::Ready {
            b.set_state(w, WorkerState::Draining, "drain asked".into());
        }
        b.dispatch();
        Ok(b.view(w))
    }

    /// Lets a drained worker take new requests again.
    pub fn resume(&self, name: &str) -> Result<WorkerView, String> {
        let mut b = self.lock();
        let w = b.find(name)?;
        b.workers[w].drain = false;
        if b.workers[w].state == WorkerState::Draining {
            b.set_state(w, WorkerState::Ready, "resume asked".into());
        }
        b.dispatch();
        Ok(b.view(w))
    }

    /// Cancels a request in flight or queued: the client's request fails with "cancelled",
    /// and its worker drops it.
    pub fn cancel(&self, id: u64) -> Result<RequestView, String> {
        let mut b = self.lock();
        let view = match b.reqs.get(&id) {
            Some(r) if !r.dropping => b.request_view(r),
            _ => return Err(format!("no request {id} in flight or queued")),
        };
        info!(request = id, doing = view.phase, "cancelled by the operator");
        STATS.requests_cancelled.add(1);
        b.end_early(id, Some(Event::Failed("cancelled by the operator".into())));
        Ok(view)
    }

    /// Treats a worker as stuck until it next answers a ping: it gets no new requests.
    pub fn mark_stuck(&self, name: &str, reason: &str) -> Result<WorkerView, String> {
        let mut b = self.lock();
        let w = b.find(name)?;
        if b.workers[w].send.is_some() {
            b.set_state(w, WorkerState::Stuck, reason.into());
        }
        b.dispatch();
        Ok(b.view(w))
    }

    /// Closes a worker's link as if it were lost: its requests are retried or fail, and it
    /// reconnects.
    pub fn mark_down(&self, name: &str, reason: &str) -> Result<WorkerView, String> {
        let mut b = self.lock();
        let w = b.find(name)?;
        if b.workers[w].send.is_some() {
            b.lost(w, reason.into());
        }
        Ok(b.view(w))
    }

    /// Answers request `id` from the cache if it can, at once on the caller's thread.
    /// Otherwise returns the probe, if any, for the scheduler to keep with the request.
    pub(crate) fn from_cache(&self, id: u64, prompt: &[u32], max_new: usize, events: &Sender<Event>) -> Result<(), Option<Probe>> {
        if prompt.is_empty() || max_new == 0 {
            return Err(None);
        }
        let mut c = lock_cache(&self.cache);
        if c.shutting_down {
            return Err(None);
        }
        let Some(probe) = c.cache.probe(prompt, max_new) else { return Err(None) };
        let Some(hit) = c.cache.lookup(&probe) else { return Err(Some(probe)) };
        let now = unix_ms(Instant::now());
        let sent = hit.answer.len() as u32;
        if c.served.len() == SERVED_KEPT {
            c.served.pop_front();
        }
        c.served.push_back(RequestView {
            id,
            worker: Some("controller".into()),
            phase: "cached",
            position: prompt.len() as u32 + sent,
            prompt_tokens: prompt.len(),
            tokens_sent: sent,
            max_tokens: max_new,
            submitted: now,
            started: Some(now),
            last_progress: Some(now),
            waiting_ms: 0,
            idle_ms: None,
            retries: 0,
            stuck: None,
            cached: Some(CachedView { kind: hit.kind, similarity: hit.similarity, made_by: hit.worker.clone() }),
        });
        drop(c);
        answer_from_cache(id, hit, events);
        Ok(())
    }

    /// The per-worker Prometheus series.
    pub fn series(&self) -> Vec<WorkerSeries> {
        self.lock().series()
    }

    /// The answer cache's counts and limits.
    pub fn cache(&self) -> CacheStats {
        lock_cache(&self.cache).cache.stats()
    }

    /// Empties the answer cache; returns how many answers it held.
    pub fn clear_cache(&self) -> usize {
        let n = lock_cache(&self.cache).cache.clear();
        info!(entries = n, "answer cache cleared");
        n
    }
}

impl Inner {
    fn find(&self, name: &str) -> Result<usize, String> {
        self.workers.iter().position(|s| s.name == name).ok_or_else(|| format!("no worker named {name}"))
    }

    fn ready(&self) -> usize {
        self.workers.iter().filter(|s| s.state == WorkerState::Ready && s.send.is_some()).count()
    }

    fn view(&self, w: usize) -> WorkerView {
        let s = &self.workers[w];
        let now = Instant::now();
        WorkerView {
            name: s.name.clone(),
            addr: s.addr.clone(),
            state: s.state,
            since: unix_ms(s.since),
            state_ms: ms(s.since.elapsed()),
            reason: s.reason.clone(),
            ping_ms: s.ping_ms,
            last_pong: s.last_pong.map(unix_ms),
            running: s.running.len(),
            max_in_flight: self.cfg.max_in_flight.max(1),
            requests: s.running.iter().filter_map(|id| self.reqs.get(id)).map(|r| self.request_view(r)).collect(),
            done: s.done,
            failed: s.failed,
            tokens_per_s: (s.tokens_per_s(now) * 10.0).round() / 10.0,
            prefill_per_s: (s.prefill.per_s(now, RATE_WINDOW, s.up_since) * 10.0).round() / 10.0,
            errors_last_min: s.errors.sum(now, ERROR_WINDOW),
            footprint: s.footprint,
            cap: s.cap,
            last_error: s.last_error.as_ref().map(|(at, request, m)| LastError {
                at: unix_ms(*at),
                request: *request,
                message: m.clone(),
            }),
            reconnect_tries: s.tries,
            next_try: s.next_try.map(unix_ms_ahead),
        }
    }

    fn request_view(&self, r: &Req) -> RequestView {
        let worker = r.worker.map(|w| &self.workers[w]);
        let stuck = worker.filter(|s| s.state == WorkerState::Stuck && !r.dropping).map(|_| {
            format!("{}, no progress for {:.1} s", r.doing(), r.last_progress.elapsed().as_secs_f64())
        });
        let phase = match (r.dropping, r.worker, r.decoding) {
            (true, _, _) => "dropping",
            (false, None, _) => "queued",
            (false, Some(_), false) => "prefill",
            (false, Some(_), true) => "decode",
        };
        RequestView {
            id: r.id,
            worker: worker.map(|s| s.name.clone()),
            phase,
            position: r.position,
            prompt_tokens: r.prompt.len(),
            tokens_sent: r.sent,
            max_tokens: r.max_new,
            submitted: unix_ms(r.submitted),
            started: r.started.map(unix_ms),
            last_progress: r.started.map(|_| unix_ms(r.last_progress)),
            waiting_ms: ms(if r.started.is_some() { r.waited } else { r.submitted.elapsed() }),
            idle_ms: r.started.map(|_| ms(r.last_progress.elapsed())),
            retries: r.retries,
            stuck,
            cached: None,
        }
    }

    fn series(&self) -> Vec<WorkerSeries> {
        let now = Instant::now();
        self.workers
            .iter()
            .map(|s| WorkerSeries {
                name: s.name.clone(),
                state: s.state.as_str(),
                running: s.running.len(),
                tokens_per_s: s.tokens_per_s(now),
                ping_s: s.ping_ms.map(|m| m / 1e3),
                footprint: s.footprint,
                cap: s.cap,
                done: s.done,
                failed: s.failed,
            })
            .collect()
    }

    /// Changes a worker's state, with one log line.
    fn set_state(&mut self, w: usize, to: WorkerState, reason: String) {
        let s = &mut self.workers[w];
        if s.state == to {
            return;
        }
        let from = s.state.as_str();
        match to {
            WorkerState::Stuck | WorkerState::Down => {
                warn!(worker = s.name, from, to = to.as_str(), reason, "worker state")
            }
            _ => info!(worker = s.name, from, to = to.as_str(), reason, "worker state"),
        }
        s.state = to;
        s.since = Instant::now();
        s.reason = reason;
    }

    /// Ready, or draining if a drain was asked.
    fn up_state(&self, w: usize) -> WorkerState {
        if self.workers[w].drain {
            WorkerState::Draining
        } else {
            WorkerState::Ready
        }
    }

    fn handle(&mut self, m: Msg) {
        match m {
            Msg::Submit { id, prompt, max_new, events, probe } => self.submit(id, prompt, max_new, events, probe),
            Msg::Shutdown => self.begin_shutdown(),
            Msg::Requests(Note::Joined { worker, conn }) => self.joined(worker, conn),
            Msg::Requests(Note::Retry { worker, tries, wait }) => {
                let s = &mut self.workers[worker];
                if s.send.is_none() {
                    (s.tries, s.next_try) = (tries, Some(Instant::now() + wait));
                }
            }
            Msg::Requests(Note::Frame { worker, link, h, payload }) => {
                if self.workers[worker].link == link && self.workers[worker].send.is_some() {
                    self.on_frame(worker, h, payload);
                }
            }
            Msg::Requests(Note::Lost { worker, link, reason }) => {
                if self.workers[worker].link == link && self.workers[worker].send.is_some() {
                    self.lost(worker, reason);
                }
            }
            // The other splits' messages.
            Msg::Frame { .. } | Msg::Partial { .. } | Msg::Reply { .. } | Msg::ClusterDown { .. } | Msg::Rejoined(_) => {}
        }
    }

    fn begin_shutdown(&mut self) {
        if self.shutting_down {
            return;
        }
        self.shutting_down = true;
        lock_cache(&self.cache).shutting_down = true;
        info!(active = self.reqs.len() - self.queue.len(), waiting = self.queue.len(), "shutting down");
        while let Some(id) = self.queue.pop_front() {
            let r = self.reqs.remove(&id).unwrap();
            let _ = r.events.send(Event::Rejected("shutting down".into()));
        }
    }

    /// Closes every link, so the workers wait for the next controller.
    fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        for s in &mut self.workers {
            if let Some(send) = s.send.take() {
                send.close();
            }
        }
        crate::stats::WORKERS.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }

    /// A request the cache did not answer; `probe` is its cache key, to fill the cache with
    /// its answer.
    fn submit(&mut self, id: u64, prompt: Vec<u32>, max_new: usize, events: Sender<Event>, probe: Option<Probe>) {
        let ready = self.ready();
        let reject = if self.shutting_down {
            Some("shutting down".to_string())
        } else if prompt.is_empty() {
            Some("empty prompt".into())
        } else if ready == 0 {
            let states: Vec<String> = self.workers.iter().map(|s| format!("{} {}", s.name, s.state.as_str())).collect();
            Some(format!("cluster down: no worker ready ({})", states.join(", ")))
        } else if self.queue.len() >= self.cfg.max_queue && !self.workers.iter().any(|s| s.takes_work(self.cfg.max_in_flight.max(1))) {
            Some(format!("queue full ({} in flight, {} waiting)", self.reqs.len() - self.queue.len(), self.queue.len()))
        } else {
            None
        };
        if let Some(reason) = reject {
            warn!(request = id, prompt = prompt.len(), reason, "rejected");
            STATS.requests_rejected.add(1);
            let _ = events.send(Event::Rejected(reason));
            return;
        }
        if max_new == 0 {
            let _ = events.send(Event::Done);
            return;
        }
        debug!(request = id, prompt = prompt.len(), max_new, waiting = self.queue.len(), "queued");
        let now = Instant::now();
        let r = Req {
            id,
            prompt,
            max_new,
            events,
            submitted: now,
            worker: None,
            started: None,
            waited: Duration::ZERO,
            decoding: false,
            position: 0,
            sent: 0,
            last_progress: now,
            last_token: None,
            retries: 0,
            dropping: false,
            probe,
            answer: Vec::new(),
        };
        self.reqs.insert(id, r);
        self.queue.push_back(id);
        self.dispatch();
    }

    /// Starts queued requests on the workers that take work; with no worker ready, fails
    /// what is queued.
    fn dispatch(&mut self) {
        let limit = self.cfg.max_in_flight.max(1);
        while let Some(&id) = self.queue.front() {
            let now = Instant::now();
            let best = (0..self.workers.len()).filter(|&w| self.workers[w].takes_work(limit)).min_by(|&a, &b| {
                let (a, b) = (&self.workers[a], &self.workers[b]);
                a.running.len().cmp(&b.running.len()).then(b.tokens_per_s(now).total_cmp(&a.tokens_per_s(now)))
            });
            let Some(w) = best else { break };
            self.queue.pop_front();
            self.start_on(id, w);
        }
        if !self.queue.is_empty() && self.ready() == 0 {
            while let Some(id) = self.queue.pop_front() {
                let r = self.reqs.remove(&id).unwrap();
                warn!(request = id, "failed: no worker ready");
                STATS.requests_failed.add(1);
                let _ = r.events.send(Event::Failed("cluster down: no worker ready".into()));
            }
        }
    }

    fn start_on(&mut self, id: u64, w: usize) {
        let r = self.reqs.get_mut(&id).unwrap();
        encode_start(r.max_new.min(u32::MAX as usize) as u32, &r.prompt, &mut self.buf);
        let s = &mut self.workers[w];
        let sent = s.send.as_mut().map(|c| c.send(MsgType::Start, id, 0, &self.buf));
        if let Some(Err(e)) = sent {
            self.queue.push_front(id);
            return self.lost(w, format!("send: {e}"));
        }
        let now = Instant::now();
        r.worker = Some(w);
        r.started = Some(now);
        r.waited = r.submitted.elapsed();
        r.last_progress = now;
        r.position = 0;
        r.decoding = false;
        s.running.push(id);
        info!(
            request = id,
            worker = s.name,
            prompt = r.prompt.len(),
            max_new = r.max_new,
            queued_ms = ms(r.waited),
            retries = r.retries,
            "started"
        );
    }

    /// Ends a request before its worker does: the client gets `outcome` now (none when it
    /// went away), and the worker a `Drop`; its `Done` then frees the slot.
    fn end_early(&mut self, id: u64, outcome: Option<Event>) {
        let Some(r) = self.reqs.get_mut(&id) else { return };
        if let Some(ev) = outcome {
            let _ = r.events.send(ev);
        }
        let Some(w) = r.worker else {
            self.queue.retain(|&q| q != id);
            self.reqs.remove(&id);
            return;
        };
        r.dropping = true;
        let s = &mut self.workers[w];
        if let Some(Err(e)) = s.send.as_mut().map(|c| c.send(MsgType::Drop, id, 0, &[])) {
            self.lost(w, format!("send: {e}"));
        }
    }

    /// Removes a request its worker is done with.
    fn release(&mut self, w: usize, id: u64) -> Option<Req> {
        self.workers[w].running.retain(|&r| r != id);
        self.reqs.remove(&id)
    }

    fn fail(&mut self, w: usize, id: u64, msg: String) {
        let Some(r) = self.reqs.get(&id) else { return };
        warn!(request = id, worker = self.workers[w].name, generated = r.sent, error = msg, "failed");
        STATS.requests_failed.add(1);
        let s = &mut self.workers[w];
        s.failed += 1;
        s.errors.add(Instant::now(), 1, ERROR_WINDOW);
    }

    fn joined(&mut self, w: usize, conn: Connection) {
        if self.shutting_down {
            let mut conn = conn;
            conn.close_send();
            return;
        }
        let conn = conn.with_max_chunk(self.net.max_chunk());
        let peer = conn.peer_addr();
        let (recv, send) = match conn.split() {
            Ok(halves) => halves,
            Err(e) => {
                warn!(worker = self.workers[w].name, error = %e, "split link");
                return self.reconnect(w);
            }
        };
        let now = Instant::now();
        let s = &mut self.workers[w];
        s.link += 1;
        let (link, tx, span) = (s.link, self.tx.clone(), self.span.clone());
        // A failed write breaks the link as a failed read does.
        let lost = tx.clone();
        send.on_error(move |e| {
            let _ = lost.send(Msg::Requests(Note::Lost { worker: w, link, reason: format!("send: {e}") }));
        });
        s.send = Some(send);
        s.up_since = now;
        s.ping = None;
        s.next_ping_at = now;
        s.tries = 0;
        s.next_try = None;
        thread::spawn(move || span.in_scope(|| read_worker(w, link, recv, tx)));
        let to = self.up_state(w);
        self.set_state(w, to, format!("connected to {peer}"));
        self.dispatch();
    }

    /// The link to worker `w` is gone: requests that sent no token go back to the queue, the
    /// rest fail, and the worker reconnects in the background.
    fn lost(&mut self, w: usize, reason: String) {
        let s = &mut self.workers[w];
        // Nothing queued is worth waiting for: cut the link at once.
        if let Some(send) = s.send.take() {
            send.abort();
        }
        s.ping = None;
        s.last_error = Some((Instant::now(), None, format!("link lost: {reason}")));
        s.errors.add(Instant::now(), 1, ERROR_WINDOW);
        let running = std::mem::take(&mut s.running);
        let name = s.name.clone();
        let (mut retried, mut failed) = (0, 0);
        for &id in running.iter().rev() {
            let Some(r) = self.reqs.get_mut(&id) else { continue };
            if r.dropping {
                self.reqs.remove(&id);
            } else if r.sent == 0 {
                r.worker = None;
                r.started = None;
                r.retries += 1;
                r.position = 0;
                self.queue.push_front(id);
                retried += 1;
            } else {
                let msg = format!("worker {name} lost after {} tokens: {reason}", r.sent);
                self.fail(w, id, msg.clone());
                let r = self.reqs.remove(&id).unwrap();
                let _ = r.events.send(Event::Failed(msg));
                failed += 1;
            }
        }
        self.set_state(w, WorkerState::Down, format!("{reason}; {retried} requests retried, {failed} failed"));
        if !self.shutting_down {
            self.reconnect(w);
        }
        self.dispatch();
    }

    /// Connects to worker `w` in the background until it answers `Hello` with its name, with
    /// backoff between tries, forever.
    fn reconnect(&self, w: usize) {
        let s = &self.workers[w];
        let (name, addr, me) = (s.name.clone(), s.addr.clone(), self.me.clone());
        let (tx, stop, span, net) = (self.tx.clone(), self.stop.clone(), self.span.clone(), self.net);
        thread::spawn(move || {
            span.in_scope(|| {
                let mut backoff = Backoff::new(&net);
                while !stop.load(Ordering::Relaxed) {
                    let why = match Connection::connect(addr.as_str(), &me, Duration::from_secs(1)) {
                        Ok(c) if c.peer_name() == name => {
                            let _ = tx.send(Msg::Requests(Note::Joined { worker: w, conn: c }));
                            return;
                        }
                        Ok(c) => format!("a different node answered: {}", c.peer_name()),
                        Err(e) => e.to_string(),
                    };
                    let wait = backoff.failed();
                    let tries = backoff.tries();
                    info!(worker = name, addr, tries, wait_ms = ms(wait), error = why, "reconnect failed");
                    let _ = tx.send(Msg::Requests(Note::Retry { worker: w, tries, wait }));
                    // In short sleeps, so a shutdown is not held up.
                    let until = Instant::now() + wait;
                    while !stop.load(Ordering::Relaxed) && Instant::now() < until {
                        thread::sleep(until.saturating_duration_since(Instant::now()).min(Duration::from_millis(50)));
                    }
                }
            })
        });
    }

    fn on_frame(&mut self, w: usize, h: Header, payload: Vec<u8>) {
        let id = h.request_id;
        match h.msg_type {
            MsgType::Pong => self.on_pong(w, id, &payload),
            MsgType::Token => self.on_token(w, h, &payload),
            MsgType::Done => {
                let reason = match decode_done(&payload) {
                    Ok(DoneReason::Length) => "length",
                    Ok(DoneReason::Eos) => "eos",
                    Ok(DoneReason::Dropped) => "dropped",
                    Err(_) => "unknown",
                };
                let Some(r) = self.reqs.get(&id).filter(|r| r.worker == Some(w)) else { return };
                let dropping = r.dropping;
                let r = self.release(w, id).unwrap();
                if dropping {
                    debug!(request = id, worker = self.workers[w].name, "freed on the worker");
                } else {
                    let elapsed = r.submitted.elapsed();
                    info!(
                        request = id,
                        worker = self.workers[w].name,
                        generated = r.sent,
                        reason,
                        ms = ms(elapsed),
                        tok_s = (r.sent as f64 / elapsed.as_secs_f64() * 10.0).round() / 10.0,
                        "done"
                    );
                    STATS.requests_done.add(1);
                    self.workers[w].done += 1;
                    let _ = r.events.send(Event::Done);
                    let complete = matches!(reason, "length" | "eos") && r.answer.len() == r.sent as usize;
                    if let Some(p) = r.probe.filter(|_| complete) {
                        lock_cache(&self.cache).cache.insert(p, r.answer, &self.workers[w].name);
                    }
                }
                self.dispatch();
            }
            MsgType::Error => {
                let msg = String::from_utf8_lossy(&payload).into_owned();
                let s = &mut self.workers[w];
                s.last_error = Some((Instant::now(), (id != 0).then_some(id), msg.clone()));
                let Some(r) = self.reqs.get(&id).filter(|r| r.worker == Some(w)) else {
                    s.errors.add(Instant::now(), 1, ERROR_WINDOW);
                    return warn!(worker = s.name, error = msg, "worker error");
                };
                let dropping = r.dropping;
                if !dropping {
                    self.fail(w, id, msg.clone());
                }
                if let Some(r) = self.release(w, id).filter(|_| !dropping) {
                    let _ = r.events.send(Event::Failed(msg));
                }
                self.dispatch();
            }
            other => warn!(worker = self.workers[w].name, "unexpected {other:?} frame"),
        }
    }

    fn on_pong(&mut self, w: usize, seq: u64, payload: &[u8]) {
        let now = Instant::now();
        let s = &mut self.workers[w];
        let Some((_, sent)) = s.ping.filter(|&(q, _)| q == seq) else { return };
        let rtt = now.duration_since(sent);
        s.ping = None;
        s.ping_ms = Some(rtt.as_secs_f64() * 1e3);
        s.last_pong = Some(now);
        s.next_ping_at = sent + Duration::from_millis(self.cfg.health_interval_ms);
        STATS.ping_rtt_us.store(rtt.as_micros() as u64, Ordering::Relaxed);
        if let Err(e) = decode_status(payload, &mut self.status) {
            warn!(worker = s.name, error = %e, "bad status report");
        }
        s.footprint = self.status.footprint;
        s.cap = self.status.cap;
        for run in &self.status.requests {
            let Some(r) = self.reqs.get_mut(&run.request).filter(|r| r.worker == Some(w) && !r.dropping) else { continue };
            if run.position > r.position {
                if !r.decoding {
                    let prompt = (run.position as usize).min(r.prompt.len()) as u32;
                    s.prefill.add(now, prompt.saturating_sub(r.position) as u64, RATE_WINDOW);
                }
                r.position = run.position;
                r.last_progress = now;
            }
            r.decoding |= run.decoding;
        }
        if s.state == WorkerState::Stuck {
            let to = self.up_state(w);
            self.set_state(w, to, format!("ping answered after {} ms", ms(rtt)));
            self.dispatch();
        }
    }

    fn on_token(&mut self, w: usize, h: Header, payload: &[u8]) {
        let id = h.request_id;
        let Some(r) = self.reqs.get_mut(&id).filter(|r| r.worker == Some(w) && !r.dropping) else { return };
        if let Err(e) = decode_token(payload, &mut self.tokens, &mut self.stages) {
            let msg = format!("{}: bad token frame: {e}", self.workers[w].name);
            return self.end_early(id, Some(Event::Failed(msg)));
        }
        let now = Instant::now();
        let s = &mut self.workers[w];
        let mut gone = false;
        for &token in &self.tokens {
            let latency = now.duration_since(r.last_token.or(r.started).unwrap_or(now));
            if r.sent == 0 {
                STATS.time_to_first_token.observe(r.submitted.elapsed());
                STATS.prompt_positions.add(r.prompt.len() as u64);
                let rest = (r.prompt.len() as u32).saturating_sub(r.position);
                s.prefill.add(now, rest as u64, RATE_WINDOW);
                info!(request = id, worker = s.name, ttft_ms = ms(r.submitted.elapsed()), "first token");
            } else {
                STATS.decode_step.observe(latency);
            }
            r.sent += 1;
            r.position = h.position + 1;
            r.decoding = true;
            r.last_progress = now;
            r.last_token = Some(now);
            if r.probe.is_some() {
                r.answer.push(token);
            }
            s.tokens.add(now, 1, RATE_WINDOW);
            STATS.tokens_generated.add(1);
            if r.events.send(Event::Token { token, latency }).is_err() {
                info!(request = id, generated = r.sent, "cancelled by the client");
                STATS.requests_cancelled.add(1);
                gone = true;
                break;
            }
        }
        if gone {
            self.end_early(id, None);
        }
    }

    /// Health pings, stuck workers, stalled requests; publishes the series.
    fn tick(&mut self) {
        let now = Instant::now();
        let timeout = Duration::from_millis(self.cfg.health_timeout_ms);
        for w in 0..self.workers.len() {
            let s = &mut self.workers[w];
            if s.send.is_none() {
                continue;
            }
            match s.ping {
                Some((seq, sent)) if now.duration_since(sent) > timeout && s.state != WorkerState::Stuck => {
                    let on: Vec<String> = s.running.iter().filter_map(|id| self.reqs.get(id)).map(|r| format!("{} {}", r.id, r.doing())).collect();
                    let reason = format!("health ping {seq} not back after {} ms; running: [{}]", ms(timeout), on.join("; "));
                    self.set_state(w, WorkerState::Stuck, reason);
                }
                Some(_) => {}
                None if now >= s.next_ping_at => {
                    self.ping_seq += 1;
                    let seq = self.ping_seq;
                    match s.send.as_mut().unwrap().send(MsgType::Ping, seq, 0, &[]) {
                        Ok(()) => s.ping = Some((seq, now)),
                        Err(e) => self.lost(w, format!("send: {e}")),
                    }
                }
                None => {}
            }
        }
        let stall = Duration::from_millis(self.cfg.stall_timeout_ms);
        let stalled: Vec<u64> = self
            .reqs
            .values()
            .filter(|r| r.worker.is_some() && !r.dropping && now.duration_since(r.last_progress) > stall)
            .map(|r| r.id)
            .collect();
        for id in stalled {
            let (w, doing) = (self.reqs[&id].worker.unwrap(), self.reqs[&id].doing());
            let msg = format!("stalled: no progress for {} ms on {} ({doing})", ms(stall), self.workers[w].name);
            self.workers[w].last_error = Some((now, Some(id), msg.clone()));
            self.fail(w, id, msg.clone());
            self.end_early(id, Some(Event::Failed(msg)));
        }
        self.dispatch();
        STATS.requests_active.store((self.reqs.len() - self.queue.len()) as u64, Ordering::Relaxed);
        STATS.requests_waiting.store(self.queue.len() as u64, Ordering::Relaxed);
        *crate::stats::WORKERS.lock().unwrap_or_else(|e| e.into_inner()) = self.series();
    }
}

/// Streams a cached answer to the client, as a worker's tokens would come.
fn answer_from_cache(id: u64, hit: Hit, events: &Sender<Event>) {
    info!(
        request = id,
        phase = "cached",
        worker = "controller",
        kind = hit.kind,
        similarity = format_args!("{:.4}", hit.similarity),
        made_by = hit.worker,
        age_s = hit.age.as_secs(),
        hits = hit.hits,
        generated = hit.answer.len(),
        "cache hit"
    );
    for &token in &hit.answer {
        if events.send(Event::Token { token, latency: Duration::ZERO }).is_err() {
            info!(request = id, "cancelled by the client");
            STATS.requests_cancelled.add(1);
            return;
        }
    }
    STATS.tokens_generated.add(hit.answer.len() as u64);
    STATS.requests_done.add(1);
    let _ = events.send(Event::Done);
}

/// Receives what one worker sends and hands it to the scheduler; reports when the link ends.
fn read_worker(worker: usize, link: u64, mut recv: Connection, tx: Sender<Msg>) {
    let mut buf = Vec::new();
    loop {
        match recv.recv_frame(&mut buf) {
            Ok(h) => {
                let _ = tx.send(Msg::Requests(Note::Frame { worker, link, h, payload: std::mem::take(&mut buf) }));
            }
            Err(e) => {
                let reason = match e {
                    FrameError::Io(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => "connection closed".into(),
                    e => e.to_string(),
                };
                let _ = tx.send(Msg::Requests(Note::Lost { worker, link, reason }));
                return;
            }
        }
    }
}
