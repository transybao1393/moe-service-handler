//! Coordinator scheduler (CL-5): keeps up to `max_in_flight` requests in the cluster at once.
//!
//! A decoding request has one position in flight, because its next input is the token that
//! position produces. Throughput comes from interleaving requests: while request A's hidden
//! state is on the workers, the coordinator runs its own layers for request B.
//!
//! The prompt is known up front, so prefill goes in chunks of `prefill_chunk` positions (CL-6),
//! and up to `prefill_depth` chunks of one request may be in the cluster at once: the coordinator
//! runs chunk 2 while the workers run chunk 1. Every link is FIFO, so chunks reach each node in
//! order. Only the last chunk's token is used; it is the first generated token.
//!
//! Speculative decoding (`max_draft` > 0, engines that can rewind): a decode step carries the
//! next token plus up to `max_draft` drafted guesses, and the head samples after every
//! position. The coordinator keeps the guesses that match the head's tokens, emits those and
//! the head's next token, and sends `Rewind` through the workers for the rest. Rewind and the next
//! step follow each other on the same FIFO links, so rejecting costs no extra round trip.
//!
//! Prefix reuse (`prefix_cache` > 0): a finished request's state stays on every node. A new
//! prompt that extends it (the next turn of a chat) takes over its request id and prefills only
//! the new tokens. The least recently finished is dropped through the workers when more than
//! `prefix_cache` are kept; engines need `max_in_flight + prefix_cache` request slots.
//!
//! The scheduler thread owns the engine and the sending side of the first worker's connection;
//! it steps ready requests in FIFO order. One reader thread per worker receives what that
//! worker sends back and passes it on to the next worker; the last worker's reader turns every
//! frame into a message for the scheduler. Workers never talk to each other. Clients submit through a
//! [`Client`] and receive their tokens as [`Event`]s on their own channel; dropping that channel
//! cancels the request.
//!
//! Logs (NF-5): every line about a request carries `request`, the id the [`Client`] gave it,
//! and `state`, the id its state has on the nodes and in frames, metrics rows and worker logs.
//! The two differ only when a request continues a parked one. Lifecycle lines are `info`,
//! per-step lines `debug`.
//!
//! Health (CL-8): every `health_interval_ms` the scheduler sends a `Ping` through every worker.
//! If it has not come back within `health_timeout_ms`, or a connection breaks, the cluster is
//! down: in-flight requests fail and new ones are rejected until the cluster is restarted. The
//! ping queues behind hidden states on every node, so the timeout must exceed one full pass
//! through the workers, prefill chunks included.
//!
//! Expert split ([`spawn_experts`], docs/expert-parallel.md): the coordinator runs every layer
//! up to its routed experts, sends the hidden rows and the picked experts to every worker that
//! owns some of them at once, and finishes the layer when each has sent back its `Partial`;
//! after the last layer it runs the head. Throughput again comes from interleaving: while the
//! workers run a layer for request A, the coordinator routes a layer of request B. One step of
//! a request is in the cluster at a time (`prefill_depth` is 1), since a chunk's attention needs
//! the previous chunk's. Only the coordinator keeps request state, so `Drop` and `Rewind` stay
//! on it. It also keeps the expert ownership table ([`crate::owners`]): it sends each worker
//! its experts (`Own`) before serving, counts every pick, and every `rebalance_every`
//! positions sends a few experts to a less busy worker (`Move`); once the new owner confirms,
//! the table changes and the old owner gets `Release`. The ping goes to every worker and is
//! back when all have answered. A step's per-node times are summed over its layers, and the
//! workers' share of the step is the slowest worker of each layer, summed.
//!
//! Links (both splits): every link sends through a [`Writer`], so tokens, pings and drops go
//! ahead of chunked hidden states. A lost worker (a broken link, a write error, a ping not
//! back) takes the cluster down: in-flight requests fail, since their state on the workers is
//! gone, every link is cut and new requests are refused as "cluster down". A background
//! thread reconnects every worker with backoff, forever; once all are back (on the expert
//! split, once each holds its experts again: `Own` with the current table) the cluster serves
//! again by itself. Messages from links that were cut carry an older link epoch and are
//! ignored.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use msh_net::writer::{give_rows, take_rows};
use msh_net::{decode_move, decode_token, encode_move, encode_own, Connection, FrameError, Header, MsgType, Pick, Writer, FLAG_EVERY_TOKEN};
use serde::Deserialize;
use tracing::{debug, error, info, warn};

use crate::config::ExpertsConfig;
use crate::draft::{Drafter, PromptLookup};
use crate::engine::{Engine, MoeShape, Sample, StageIn};
use crate::metrics::{Metrics, StepKind, StepRecord};
use crate::node::{Plan, Star};
use crate::owners::{Move, Owners};
use crate::stats::STATS;
use crate::Error;

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SchedulerConfig {
    /// Requests stepping through the cluster at once.
    pub max_in_flight: usize,
    /// Requests waiting for a slot once all are taken; more are rejected.
    pub max_queue: usize,
    /// How often a health ping is sent through the workers.
    pub health_interval_ms: u64,
    /// How long a ping may take to come back before the cluster is declared down.
    pub health_timeout_ms: u64,
    /// Request split: how long a request may go without progress (a token, or prompt
    /// positions in the worker's status report) before it fails.
    pub stall_timeout_ms: u64,
    /// Request split: when a worker's link breaks, move its requests to another worker and
    /// carry on from the tokens already sent instead of failing them.
    pub resume_on_loss: bool,
    /// Prompt positions per prefill step. Bigger chunks make better use of the engine's
    /// batched path; smaller ones let other requests' decode steps in sooner.
    pub prefill_chunk: usize,
    /// Prefill chunks of one request in the cluster at once.
    pub prefill_depth: usize,
    /// Draft tokens per decode step (speculative decoding); 0 turns it off.
    pub max_draft: usize,
    /// Shortest repeat of the latest tokens that prompt lookup drafts from.
    pub draft_min_ngram: usize,
    /// Finished requests whose state stays on every node for a prompt that extends them.
    pub prefix_cache: usize,
}

impl SchedulerConfig {
    /// Request slots each node's engine needs.
    pub fn engine_slots(&self) -> usize {
        self.max_in_flight.max(1) + self.prefix_cache
    }
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        SchedulerConfig {
            max_in_flight: 3,
            max_queue: 64,
            health_interval_ms: 1000,
            health_timeout_ms: 5000,
            stall_timeout_ms: 30_000,
            resume_on_loss: true,
            prefill_chunk: 64,
            prefill_depth: 3,
            max_draft: 0,
            draft_min_ngram: 3,
            prefix_cache: 0,
        }
    }
}

/// What a client receives for one request. A request ends with exactly one of `Done`,
/// `Failed` or `Rejected`.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// A generated token. `latency` runs from the start of the step that produced it on the
    /// coordinator (for the first token, the last prompt chunk) to the token's arrival.
    Token { token: u32, latency: Duration },
    Done,
    Failed(String),
    /// Not admitted: queue full, empty prompt, cluster down or shutting down.
    Rejected(String),
}

#[derive(Debug)]
pub struct Generation {
    pub tokens: Vec<u32>,
    pub latencies: Vec<Duration>,
}

pub(crate) enum Msg {
    /// `probe`: the request split's cache key, from a lookup that missed.
    Submit { id: u64, prompt: Vec<u32>, max_new: usize, events: Sender<Event>, probe: Option<crate::cache::Probe> },
    /// Request split: from a worker's link (see [`crate::requests`]).
    Requests(crate::requests::Note),
    // `epoch` numbers the links a message came over; news from links already cut is ignored.
    /// Layer split: a frame back from the last worker.
    Frame { epoch: u64, h: Header, payload: Vec<u8> },
    /// Expert split: worker `from`'s outputs for one layer of a step.
    Partial { epoch: u64, from: usize, h: Header, layer: u16, compute_us: u32, values: Vec<f32> },
    /// Expert split: any other frame from worker `from`.
    Reply { epoch: u64, from: usize, h: Header, payload: Vec<u8> },
    ClusterDown { epoch: u64, reason: String },
    /// Every worker is connected again.
    Rejoined(Star),
    Shutdown,
}

/// Handle for submitting requests. Clones share one scheduler; when the last clone is dropped
/// the scheduler shuts down as if `shutdown` had been called.
#[derive(Clone)]
pub struct Client {
    inner: Arc<ClientInner>,
}

struct ClientInner {
    tx: Sender<Msg>,
    next_id: AtomicU64,
    /// Request split: the worker table and its actions.
    board: Option<Arc<crate::requests::Board>>,
}

impl Drop for ClientInner {
    fn drop(&mut self) {
        let _ = self.tx.send(Msg::Shutdown);
    }
}

impl Client {
    pub(crate) fn new(tx: Sender<Msg>, board: Option<Arc<crate::requests::Board>>) -> Client {
        Client { inner: Arc::new(ClientInner { tx, next_id: AtomicU64::new(1), board }) }
    }

    /// The request split's worker table and actions; `None` on the other splits.
    pub fn board(&self) -> Option<&Arc<crate::requests::Board>> {
        self.inner.board.as_ref()
    }

    /// Queues a request for greedy generation of up to `max_new` tokens.
    pub fn submit(&self, prompt: Vec<u32>, max_new: usize) -> Receiver<Event> {
        self.submit_with_id(prompt, max_new).1
    }

    /// [`Client::submit`], also returning the request id used in logs. Ids count up from 1 in
    /// submission order.
    pub fn submit_with_id(&self, prompt: Vec<u32>, max_new: usize) -> (u64, Receiver<Event>) {
        let id = self.inner.next_id.fetch_add(1, Ordering::Relaxed);
        let (events, rx) = mpsc::channel();
        // The request split answers from its cache here, on the caller's thread.
        let probe = match &self.inner.board {
            Some(b) => match b.from_cache(id, &prompt, max_new, &events) {
                Ok(()) => return (id, rx),
                Err(probe) => probe,
            },
            None => None,
        };
        let msg = Msg::Submit { id, prompt, max_new, events: events.clone(), probe };
        if self.inner.tx.send(msg).is_err() {
            let _ = events.send(Event::Rejected("scheduler stopped".into()));
        }
        (id, rx)
    }

    /// Submits and waits for the whole generation.
    pub fn generate(&self, prompt: Vec<u32>, max_new: usize) -> Result<Generation, Error> {
        collect(self.submit(prompt, max_new))
    }

    /// Stops accepting requests; the scheduler exits once in-flight requests finish.
    pub fn shutdown(&self) {
        let _ = self.inner.tx.send(Msg::Shutdown);
    }
}

/// Collects a request's events into a `Generation`, or the error that ended it.
pub fn collect(rx: Receiver<Event>) -> Result<Generation, Error> {
    let mut g = Generation { tokens: Vec::new(), latencies: Vec::new() };
    for ev in rx {
        match ev {
            Event::Token { token, latency } => {
                g.tokens.push(token);
                g.latencies.push(latency);
            }
            Event::Done => return Ok(g),
            Event::Failed(m) => return Err(Error::Failed(m)),
            Event::Rejected(m) => return Err(Error::Rejected(m)),
        }
    }
    Err(Error::Protocol("scheduler stopped mid-request".into()))
}

/// Starts the scheduler (and, with workers, a reader thread per worker). Returns the client handle and
/// the scheduler thread, which yields the engine back after `shutdown`.
pub fn spawn<E: Engine + 'static>(
    engine: E,
    star: Option<Star>,
    cfg: SchedulerConfig,
) -> (Client, JoinHandle<E>) {
    spawn_with_metrics(engine, star, cfg, None)
}

/// [`spawn`], also writing one metrics row per step to `metrics`.
pub fn spawn_with_metrics<E: Engine + 'static>(
    engine: E,
    star: Option<Star>,
    cfg: SchedulerConfig,
    metrics: Option<Metrics>,
) -> (Client, JoinHandle<E>) {
    let (tx, rx) = mpsc::channel();
    // The threads log inside the caller's span (the node's name).
    let span = tracing::Span::current();
    start(engine, star.map(Links::Layers), cfg, tx, rx, span, metrics)
}

/// The workers a scheduler starts with.
enum Links {
    Layers(Star),
    Experts(Fan, Arc<Plan>),
}

/// Starts the scheduler on the expert split: `engine` runs every layer but the routed experts
/// and the head, and the workers own the experts, split evenly to start. Sends every worker its
/// experts and waits until all hold them. Without workers, `engine` runs everything.
pub fn spawn_experts<E: Engine + 'static>(
    engine: E,
    star: Option<Star>,
    cfg: SchedulerConfig,
    experts: ExpertsConfig,
    metrics: Option<Metrics>,
) -> Result<(Client, JoinHandle<E>), Error> {
    let Some(shape) = engine.moe_shape() else {
        return Err(Error::Config("the engine cannot split by experts".into()));
    };
    if (shape.experts, shape.top_k) != (experts.count, experts.top_k) || !engine.has_head() {
        let want = format!("{} experts, top {}, and the head", experts.count, experts.top_k);
        return Err(Error::Config(format!("engine has {shape:?}, head {}; the config wants {want}", engine.has_head())));
    }
    // A chunk's attention needs the previous chunk's, and picks number rows in a u16.
    let cfg = SchedulerConfig { prefill_depth: 1, prefill_chunk: cfg.prefill_chunk.min(1 << 16), ..cfg };
    let (tx, rx) = mpsc::channel();
    let span = tracing::Span::current();
    let Some(star) = star else {
        return Ok(start(engine, None, cfg, tx, rx, span, metrics));
    };
    let plan = star.plan.clone();
    let fan = Fan::start(star, shape, experts, engine.hidden_dim(), &tx, &span)?;
    Ok(start(engine, Some(Links::Experts(fan, plan)), cfg, tx, rx, span, metrics))
}

fn start<E: Engine + 'static>(
    engine: E,
    links: Option<Links>,
    mut cfg: SchedulerConfig,
    tx: Sender<Msg>,
    rx: Receiver<Msg>,
    span: tracing::Span,
    metrics: Option<Metrics>,
) -> (Client, JoinHandle<E>) {
    if cfg.max_draft > 0 && !engine.can_speculate() {
        warn!("the engine cannot verify and rewind draft tokens; speculative decoding is off");
        cfg.max_draft = 0;
    }
    let drafter = PromptLookup { min_ngram: cfg.draft_min_ngram, max_ngram: cfg.draft_min_ngram.max(8) };
    let mut s = Scheduler {
        engine,
        next: None,
        fan: None,
        writers: Vec::new(),
        plan: None,
        epoch: 0,
        stop: Arc::new(AtomicBool::new(false)),
        tx: tx.clone(),
        span: span.clone(),
        cfg,
        rx,
        drafter: Box::new(drafter),
        hidden: Vec::new(),
        input: Vec::new(),
        tokens: Vec::new(),
        waiting: VecDeque::new(),
        active: HashMap::new(),
        ready: VecDeque::new(),
        down: None,
        shutting_down: false,
        ping: None,
        ping_left: 0,
        ping_seq: 0,
        next_ping_at: Instant::now(),
        ping_rtt: None,
        parked: VecDeque::new(),
        metrics,
        stages: Vec::new(),
        workers_us: None,
    };
    match links {
        Some(Links::Layers(star)) => {
            s.plan = Some(star.plan.clone());
            s.attach_layers(star);
        }
        Some(Links::Experts(fan, plan)) => {
            s.plan = Some(plan);
            s.writers = fan.sends.clone();
            for w in &s.writers {
                watch(w, &s.tx, 0);
            }
            s.fan = Some(fan);
        }
        None => {}
    }
    STATS.cluster_up.store(1, Ordering::Relaxed);
    let client = Client::new(tx, None);
    (client, thread::spawn(move || span.in_scope(|| s.run())))
}

/// The coordinator's side of the expert split.
struct Fan {
    /// Each worker's writer; empty while the cluster is down.
    sends: Vec<Writer>,
    /// Workers back after the cluster was down that have not confirmed their experts yet.
    joining: usize,
    dim: usize,
    owners: Owners,
    cfg: ExpertsConfig,
    layers: u32,
    /// Most experts a worker may own after moves.
    max_owned: usize,
    /// Moves sent to each worker and not confirmed yet, oldest first.
    moving: Vec<VecDeque<Move>>,
    /// Positions routed since the last rebalance.
    since_rebalance: u64,
    /// Scratch: one worker's picks, and a `Move` payload.
    picks: Vec<Pick>,
    weights: Vec<u8>,
    frame: Vec<u8>,
}

impl Fan {
    /// Sends every worker its share of the experts, waits until each holds them, and starts a
    /// reader per worker.
    fn start(
        star: Star,
        shape: MoeShape,
        cfg: ExpertsConfig,
        dim: usize,
        tx: &Sender<Msg>,
        span: &tracing::Span,
    ) -> Result<Fan, Error> {
        let n = star.workers.len();
        let owners = Owners::even(shape.layers, shape.experts, n);
        let (mut recvs, sends): (Vec<Connection>, Vec<Writer>) = star.workers.into_iter().unzip();
        let mut buf = Vec::new();
        // All at once, so the workers load their experts together.
        for (w, send) in sends.iter().enumerate() {
            let own: Vec<(u16, u16)> = owners.owned_by(w).into_iter().map(|(l, e)| (l as u16, e as u16)).collect();
            encode_own(&own, &mut buf);
            send.send(MsgType::Own, 0, 0, &buf)?;
        }
        for recv in &mut recvs {
            let h = recv.recv(&mut buf).map_err(|e| Error::Protocol(format!("{}: own: {e}", recv.peer_name())))?;
            if h.msg_type != MsgType::Own {
                return Err(Error::Protocol(format!("{}: expected Own back, got {:?}", recv.peer_name(), h.msg_type)));
            }
        }
        let total = shape.layers as usize * shape.experts as usize;
        info!(workers = n, layers = shape.layers, experts = shape.experts, top_k = shape.top_k, "experts assigned");
        for (w, recv) in recvs.into_iter().enumerate() {
            let (tx, span) = (tx.clone(), span.clone());
            thread::spawn(move || span.in_scope(|| read_expert_worker(w, recv, dim, tx, 0)));
        }
        Ok(Fan {
            sends,
            joining: 0,
            dim,
            owners,
            cfg,
            layers: shape.layers,
            max_owned: total.div_ceil(n) + cfg.extra_experts,
            moving: vec![VecDeque::new(); n],
            since_rebalance: 0,
            picks: Vec::new(),
            weights: Vec::new(),
            frame: Vec::new(),
        })
    }

    /// Takes the workers back after the cluster was down: sends each its experts from the
    /// current table and starts its reader; the cluster is up once all have confirmed.
    fn rejoin(&mut self, star: Star, tx: &Sender<Msg>, span: &tracing::Span, epoch: u64) -> Vec<Writer> {
        let (recvs, sends): (Vec<Connection>, Vec<Writer>) = star.workers.into_iter().unzip();
        let mut buf = Vec::new();
        for (w, (recv, send)) in recvs.into_iter().zip(&sends).enumerate() {
            watch(send, tx, epoch);
            let own: Vec<(u16, u16)> = self.owners.owned_by(w).into_iter().map(|(l, e)| (l as u16, e as u16)).collect();
            encode_own(&own, &mut buf);
            // A failed send shows as the reader's lost connection.
            let _ = send.send(MsgType::Own, 0, 0, &buf);
            let (tx, span, dim) = (tx.clone(), span.clone(), self.dim);
            thread::spawn(move || span.in_scope(|| read_expert_worker(w, recv, dim, tx, epoch)));
        }
        self.joining = sends.len();
        self.sends = sends.clone();
        sends
    }
}

/// Reports a failed write on `w` as the cluster down.
fn watch(w: &Writer, tx: &Sender<Msg>, epoch: u64) {
    let (tx, peer) = (tx.clone(), w.peer_name().to_string());
    w.on_error(move |e| {
        let _ = tx.send(Msg::ClusterDown { epoch, reason: format!("send to {peer}: {e}") });
    });
}

/// Receives what one worker sends back on the expert split: `Partial` outputs straight into
/// their own buffer (NET-6), anything else as it is.
fn read_expert_worker(from: usize, mut link: Connection, dim: usize, tx: Sender<Msg>, epoch: u64) {
    let mut buf = Vec::new();
    let lost = |link: &Connection, e: FrameError| Msg::ClusterDown {
        epoch,
        reason: format!("lost connection from {}: {e}", link.peer_name()),
    };
    loop {
        let h = match link.recv_header() {
            Ok(h) => h,
            Err(e) => {
                let _ = tx.send(lost(&link, e));
                break;
            }
        };
        let msg = if h.msg_type == MsgType::Partial {
            let mut values = take_rows();
            match link.recv_partial(&h, dim, &mut values) {
                Ok((layer, compute_us)) => Msg::Partial { epoch, from, h, layer, compute_us, values },
                // A malformed partial fails its request, like an error from the worker.
                Err(e @ FrameError::Remote(_)) => {
                    let payload = format!("{}: bad partial: {e}", link.peer_name()).into_bytes();
                    let h = Header::new(MsgType::Error, h.request_id, h.position, payload.len());
                    Msg::Reply { epoch, from, h, payload }
                }
                Err(e) => {
                    let _ = tx.send(lost(&link, e));
                    break;
                }
            }
        } else {
            if let Err(e) = link.recv_payload(&h, &mut buf) {
                let _ = tx.send(lost(&link, e));
                break;
            }
            Msg::Reply { epoch, from, h, payload: std::mem::take(&mut buf) }
        };
        // After the scheduler stops, keep reading until the worker closes, as `read_worker` does.
        let _ = tx.send(msg);
    }
}

/// Receives what one worker sends back and passes it on unchanged, the rows without a copy:
/// to the next worker, or to the scheduler after the last one.
fn read_worker(mut from: Connection, onward: Option<Writer>, tx: Sender<Msg>, epoch: u64) {
    loop {
        let m = match from.recv_message() {
            Ok(m) => m,
            Err(e) => {
                let reason = format!("lost connection from {}: {e}", from.peer_name());
                let _ = tx.send(Msg::ClusterDown { epoch, reason });
                break;
            }
        };
        match &onward {
            Some(to) => {
                if let Err(e) = to.send_message(m) {
                    let _ = tx.send(Msg::ClusterDown { epoch, reason: format!("send to {}: {e}", to.peer_name()) });
                    break;
                }
            }
            // After the scheduler stops, keep reading until the worker closes: closing with
            // its last replies unread would reset the connection under it.
            None => {
                let h = m.header;
                let _ = tx.send(Msg::Frame { epoch, h, payload: m.into_payload() });
            }
        }
    }
    // The next worker sees its connection end and finishes in turn.
    if let Some(to) = &onward {
        to.close();
    }
}

/// A step sent through the workers and not back yet.
struct Step {
    start: u32,
    positions: u32,
    /// Draft tokens after the real one; the head samples every position when there are any.
    drafts: Vec<u32>,
    started: Instant,
}

struct Request {
    /// The client's id for the request; the map key is its state id.
    id: u64,
    /// The prompt, then every generated token.
    tokens: Vec<u32>,
    prompt_len: usize,
    max_new: usize,
    /// Position of the next step on the coordinator. While decoding, `tokens[next_pos]` is the
    /// last generated token, the next step's input.
    next_pos: usize,
    events: Sender<Event>,
    /// Oldest first.
    in_flight: VecDeque<Step>,
    submitted: Instant,
    /// Time spent waiting for an in-flight slot.
    queued: Duration,
    /// `Drop` sent; this outcome is reported when it comes back through the workers.
    freeing: Option<Event>,
    /// Positions taken over from a finished request (prefix reuse).
    cached: usize,
    /// Whether the engines hold exactly `tokens[..next_pos]`, so the request can be parked
    /// when it finishes; not so while a verify step's drafts are still being accepted.
    parkable: bool,
    /// The client went away.
    cancelled: bool,
    /// Expert split: the step's progress through the layers.
    fan: FanStep,
}

/// A request's step on the expert split. The buffers are kept from step to step.
#[derive(Default)]
struct FanStep {
    /// The layer on the workers, or the number of layers once all are done.
    layer: u32,
    every: bool,
    /// The step's rows: each layer's output without its routed experts, then with them.
    hidden: Vec<f32>,
    /// The rows the experts take, and each row's picks.
    x: Vec<f32>,
    picks: Vec<Pick>,
    /// One output row per pick, filled in as `Partial`s come back.
    outputs: Vec<f32>,
    /// Per worker, the indexes in `picks` it was sent; cleared when its `Partial` is in.
    sent: Vec<Vec<usize>>,
    /// Workers whose `Partial` is still out.
    waiting: usize,
    /// Compute time on the coordinator, and on each worker, summed over the layers.
    coord_us: u64,
    worker_us: Vec<u64>,
    /// The slowest worker of each layer, summed, and of the layer on the workers.
    critical_us: u64,
    layer_us: u64,
}

/// A finished request whose state stays on every node.
struct Parked {
    /// State id.
    id: u64,
    /// Client id of the request that left it.
    request: u64,
    /// The positions the engines hold.
    tokens: Vec<u32>,
}

impl Request {
    fn generated(&self) -> usize {
        self.tokens.len() - self.prompt_len
    }

    /// Whether the coordinator can run this request's next step now.
    fn can_step(&self, cfg: &SchedulerConfig) -> bool {
        if self.freeing.is_some() {
            false
        } else if self.next_pos < self.prompt_len {
            self.in_flight.len() < cfg.prefill_depth.max(1)
        } else {
            self.in_flight.is_empty()
        }
    }
}

fn push_ready(ready: &mut VecDeque<u64>, id: u64) {
    if !ready.contains(&id) {
        ready.push_back(id);
    }
}

struct Scheduler<E: Engine> {
    engine: E,
    /// Layer split: the first worker's writer.
    next: Option<Writer>,
    /// Expert split with workers.
    fan: Option<Fan>,
    /// Every worker's writer; empty while the cluster is down.
    writers: Vec<Writer>,
    /// How to reconnect the workers; `None` without workers.
    plan: Option<Arc<Plan>>,
    /// Numbers the links; bumped when they are cut and when new ones come.
    epoch: u64,
    /// Tells the reconnecting thread to give up.
    stop: Arc<AtomicBool>,
    tx: Sender<Msg>,
    span: tracing::Span,
    cfg: SchedulerConfig,
    rx: Receiver<Msg>,
    drafter: Box<dyn Drafter>,
    hidden: Vec<f32>,
    /// Scratch for a step's input tokens and the head's output tokens.
    input: Vec<u32>,
    tokens: Vec<u32>,
    /// Keyed by client id until admitted; the state id is chosen then.
    waiting: VecDeque<(u64, Request)>,
    active: HashMap<u64, Request>,
    ready: VecDeque<u64>,
    /// Set while a worker is lost; requests are rejected with this reason.
    down: Option<String>,
    shutting_down: bool,
    /// Outstanding health ping: sequence number and send time.
    ping: Option<(u64, Instant)>,
    /// Workers the ping has not come back from: one on the layer split (the last), every
    /// worker on the expert split.
    ping_left: usize,
    ping_seq: u64,
    next_ping_at: Instant,
    /// Last health ping's trip through every worker (NET-5).
    ping_rtt: Option<Duration>,
    /// Least recently finished first.
    parked: VecDeque<Parked>,
    metrics: Option<Metrics>,
    /// Scratch for the stage times of a returning token.
    stages: Vec<u32>,
    /// The workers' share of a returning step when they ran at once (expert split).
    workers_us: Option<u64>,
}

impl<E: Engine> Scheduler<E> {
    fn run(mut self) -> E {
        loop {
            while let Ok(m) = self.rx.try_recv() {
                self.handle(m);
            }
            self.check_health();
            self.admit();
            STATS.requests_active.store(self.active.len() as u64, Ordering::Relaxed);
            STATS.requests_waiting.store(self.waiting.len() as u64, Ordering::Relaxed);
            if self.shutting_down && self.active.is_empty() {
                return self.stop();
            }
            if let Some(id) = self.ready.pop_front() {
                self.step(id);
                continue;
            }
            let msg = match self.health_deadline() {
                Some(at) => match self.rx.recv_timeout(at.saturating_duration_since(Instant::now())) {
                    Err(mpsc::RecvTimeoutError::Timeout) => continue,
                    other => other.ok(),
                },
                None => self.rx.recv().ok(),
            };
            match msg {
                Some(m) => self.handle(m),
                None => return self.stop(),
            }
        }
    }

    /// Closes the connections to the workers, so every worker exits (on the layer split, in
    /// turn), and hands back the engine.
    fn stop(self) -> E {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(next) = &self.next {
            next.close();
        }
        for send in self.fan.iter().flat_map(|f| f.sends.iter()) {
            send.close();
        }
        self.engine
    }

    /// The writers pings go to: the first worker's on the layer split, every worker's on the
    /// expert split. Empty without live workers.
    fn links(&self) -> &[Writer] {
        match (&self.next, &self.fan) {
            (Some(next), _) => std::slice::from_ref(next),
            (None, Some(fan)) => &fan.sends,
            (None, None) => &[],
        }
    }

    /// Layer split: starts a reader per worker of `star`, each passing what its worker sends
    /// on to the next worker's writer; steps go to the first worker.
    fn attach_layers(&mut self, star: Star) {
        let (recvs, sends): (Vec<Connection>, Vec<Writer>) = star.workers.into_iter().unzip();
        for (i, recv) in recvs.into_iter().enumerate() {
            watch(&sends[i], &self.tx, self.epoch);
            let (onward, tx, span, epoch) = (sends.get(i + 1).cloned(), self.tx.clone(), self.span.clone(), self.epoch);
            thread::spawn(move || span.in_scope(|| read_worker(recv, onward, tx, epoch)));
        }
        self.next = sends.first().cloned();
        self.writers = sends;
    }

    /// Every worker is connected again: the layer split serves at once, the expert split once
    /// each worker holds its experts.
    fn rejoined(&mut self, star: Star) {
        if self.shutting_down || !self.writers.is_empty() {
            for (_, w) in &star.workers {
                w.close();
            }
            return;
        }
        self.epoch += 1;
        self.ping = None;
        self.next_ping_at = Instant::now();
        let (tx, span, epoch) = (self.tx.clone(), self.span.clone(), self.epoch);
        match &mut self.fan {
            Some(fan) => {
                info!(workers = star.workers.len(), "workers back; sending them their experts");
                self.writers = fan.rejoin(star, &tx, &span, epoch);
            }
            None => {
                self.attach_layers(star);
                self.up();
            }
        }
    }

    /// The cluster serves again.
    fn up(&mut self) {
        if let Some(was) = self.down.take() {
            info!(workers = self.writers.len(), was, "cluster up");
            STATS.cluster_up.store(1, Ordering::Relaxed);
        }
    }

    /// Reconnects every worker in the background; the scheduler gets them back as one star.
    fn reconnect(&self) {
        let Some(plan) = self.plan.clone().filter(|_| !self.shutting_down) else { return };
        let (tx, stop, span) = (self.tx.clone(), self.stop.clone(), self.span.clone());
        thread::spawn(move || {
            span.in_scope(|| {
                if let Some(star) = crate::node::reconnect(&plan, &stop) {
                    let _ = tx.send(Msg::Rejoined(star));
                }
            })
        });
    }

    /// When the health check next needs attention, or `None` without live workers.
    fn health_deadline(&self) -> Option<Instant> {
        if self.links().is_empty() {
            return None;
        }
        Some(match self.ping {
            Some((_, sent)) => sent + Duration::from_millis(self.cfg.health_timeout_ms),
            None => self.next_ping_at,
        })
    }

    fn check_health(&mut self) {
        let now = Instant::now();
        if let Some((seq, sent)) = self.ping {
            if now.duration_since(sent) > Duration::from_millis(self.cfg.health_timeout_ms) {
                let reason = format!("health ping {seq} not back after {} ms", self.cfg.health_timeout_ms);
                self.cluster_down(reason);
            }
        } else if now >= self.next_ping_at && !self.links().is_empty() {
            self.ping_seq += 1;
            let seq = self.ping_seq;
            let mut failed = None;
            for link in self.links() {
                if let Err(e) = link.send(MsgType::Ping, seq, 0, &[]) {
                    failed = Some(format!("send to {}: {e}", link.peer_name()));
                    break;
                }
            }
            if let Some(reason) = failed {
                return self.cluster_down(reason);
            }
            // On the layer split the ping comes back from the last worker only.
            self.ping_left = if self.next.is_some() { 1 } else { self.links().len() };
            self.ping = Some((seq, now));
            self.next_ping_at = now + Duration::from_millis(self.cfg.health_interval_ms);
        }
    }

    /// A ping came back from a worker.
    fn on_ping(&mut self, seq: u64) {
        let Some((_, sent)) = self.ping.filter(|&(s, _)| s == seq) else { return };
        self.ping_left = self.ping_left.saturating_sub(1);
        if self.ping_left == 0 {
            self.ping_rtt = Some(sent.elapsed());
            STATS.ping_rtt_us.store(sent.elapsed().as_micros() as u64, Ordering::Relaxed);
            self.ping = None;
        }
    }

    fn handle(&mut self, m: Msg) {
        match m {
            Msg::Submit { id, prompt, max_new, events, .. } => self.submit(id, prompt, max_new, events),
            Msg::Frame { epoch, h, payload } if epoch == self.epoch => self.on_frame(h, payload),
            Msg::Partial { epoch, from, h, layer, compute_us, values } if epoch == self.epoch => {
                self.on_partial(from, h, layer, compute_us, values)
            }
            Msg::Reply { epoch, from, h, payload } if epoch == self.epoch => self.on_reply(from, h, payload),
            Msg::ClusterDown { epoch, reason } if epoch == self.epoch => self.cluster_down(reason),
            // From links already cut.
            Msg::Frame { .. } | Msg::Partial { .. } | Msg::Reply { .. } | Msg::ClusterDown { .. } => {}
            Msg::Rejoined(star) => self.rejoined(star),
            // Only the request split's scheduler gets these.
            Msg::Requests(_) => {}
            Msg::Shutdown => {
                self.shutting_down = true;
                info!(active = self.active.len(), waiting = self.waiting.len(), "shutting down");
                for (_, r) in self.waiting.drain(..) {
                    info!(request = r.id, "rejected: shutting down");
                    let _ = r.events.send(Event::Rejected("shutting down".into()));
                }
                while let Some(p) = self.parked.pop_front() {
                    self.evict(p);
                }
            }
        }
    }

    fn submit(&mut self, id: u64, prompt: Vec<u32>, max_new: usize, events: Sender<Event>) {
        let reject = if let Some(reason) = &self.down {
            Some(format!("cluster down: {reason}"))
        } else if self.shutting_down {
            Some("shutting down".into())
        } else if prompt.is_empty() {
            Some("empty prompt".into())
        } else if self.active.len() + self.waiting.len() >= self.cfg.max_in_flight.max(1) + self.cfg.max_queue {
            Some(format!("queue full ({} in flight, {} waiting)", self.active.len(), self.waiting.len()))
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
        debug!(request = id, prompt = prompt.len(), max_new, waiting = self.waiting.len(), "queued");
        let r = Request {
            id,
            prompt_len: prompt.len(),
            tokens: prompt,
            max_new,
            next_pos: 0,
            events,
            in_flight: VecDeque::new(),
            submitted: Instant::now(),
            queued: Duration::ZERO,
            freeing: None,
            cached: 0,
            parkable: true,
            cancelled: false,
            fan: FanStep::default(),
        };
        self.waiting.push_back((id, r));
    }

    fn admit(&mut self) {
        while self.active.len() < self.cfg.max_in_flight.max(1) {
            let Some((mut id, mut r)) = self.waiting.pop_front() else { break };
            let mut continues = None;
            r.queued = r.submitted.elapsed();
            // Continue the longest finished request this prompt extends; at least one prompt
            // token must be left to step.
            let reuse = (0..self.parked.len())
                .filter(|&i| {
                    let held = &self.parked[i].tokens;
                    held.len() < r.prompt_len && r.tokens.starts_with(held)
                })
                .max_by_key(|&i| self.parked[i].tokens.len());
            if let Some(p) = reuse.and_then(|i| self.parked.remove(i)) {
                id = p.id;
                r.next_pos = p.tokens.len();
                r.cached = p.tokens.len();
                continues = Some(p.request);
                STATS.cached_positions.add(r.cached as u64);
            }
            // Otherwise the state id is the client id. A continued request keeps an older
            // client's id, which no later request can have, so state ids never clash.
            info!(
                request = r.id,
                state = id,
                prompt = r.prompt_len,
                max_new = r.max_new,
                cached = r.cached,
                continues,
                queued_ms = r.queued.as_millis() as u64,
                "started"
            );
            self.active.insert(id, r);
            self.ready.push_back(id);
        }
    }

    /// Runs the coordinator's part of the request's next step (a prompt chunk, or a decode
    /// position with any draft tokens) and sends it on (or, with one node, finishes it).
    fn step(&mut self, id: u64) {
        let Some(r) = self.active.get_mut(&id).filter(|r| r.can_step(&self.cfg)) else { return };
        let started = Instant::now();
        let start = r.next_pos;
        let mut drafts = Vec::new();
        self.input.clear();
        if start < r.prompt_len {
            let end = r.prompt_len.min(start + self.cfg.prefill_chunk.max(1));
            self.input.extend_from_slice(&r.tokens[start..end]);
        } else {
            self.input.push(r.tokens[start]);
            // Never draft past the last token the request may still produce.
            let room = r.max_new - r.generated() - 1;
            self.drafter.draft(&r.tokens, self.cfg.max_draft.min(room), &mut drafts);
            self.input.extend_from_slice(&drafts);
        }
        let sample = if drafts.is_empty() { Sample::Last } else { Sample::Every };
        let workers = self.fan.as_ref().map(|f| f.sends.len());
        let result = match workers {
            // Expert split: the embedding now, the layers one by one as the workers answer.
            Some(_) => self.engine.embed(id, start as u32, &self.input, sample, &mut r.fan.hidden),
            None => {
                let input = StageIn::Tokens(&self.input);
                self.engine.step(id, start as u32, input, sample, &mut self.hidden, &mut self.tokens)
            }
        };
        if let Err(e) = result {
            return self.finish(id, Event::Failed(e.to_string()));
        }
        debug!(request = r.id, state = id, pos = start, positions = self.input.len(), drafted = drafts.len(), "step sent");
        let positions = self.input.len() as u32;
        r.next_pos += positions as usize;
        r.in_flight.push_back(Step { start: start as u32, positions, drafts, started });
        if let Some(workers) = workers {
            let f = &mut r.fan;
            f.layer = 0;
            f.every = sample == Sample::Every;
            f.coord_us = started.elapsed().as_micros() as u64;
            f.worker_us.clear();
            f.worker_us.resize(workers, 0);
            f.critical_us = 0;
            return self.route_layer(id);
        }
        STATS.compute.observe(started.elapsed());
        let coord_us = started.elapsed().as_micros().min(u32::MAX as u128) as u32;
        let next = match (self.engine.has_head(), &mut self.next) {
            // One node holds every layer and the head.
            (true, None) => {
                self.stages.clear();
                self.stages.push(coord_us);
                let tokens = std::mem::take(&mut self.tokens);
                self.on_tokens(id, start as u32, &tokens);
                self.tokens = tokens;
                return;
            }
            (false, Some(next)) => next,
            (true, Some(_)) | (false, None) => {
                let msg = "coordinator stage does not match the workers (head on the wrong node)";
                return self.finish(id, Event::Failed(msg.into()));
            }
        };
        let flags = if sample == Sample::Every { FLAG_EVERY_TOKEN } else { 0 };
        // The engine's output, handed over without a copy (NET-6).
        if let Err(e) = next.send_hidden(flags, id, start as u32, &[coord_us], &mut self.hidden) {
            let reason = format!("send to {}: {e}", next.peer_name());
            return self.cluster_down(reason);
        }
        // The next prompt chunk can follow right behind this one.
        if r.can_step(&self.cfg) {
            push_ready(&mut self.ready, id);
        }
    }

    /// Expert split: runs the next layer of `id`'s step up to its routed experts and sends each
    /// worker its picks; after the last layer, runs the head.
    fn route_layer(&mut self, id: u64) {
        let dim = self.engine.hidden_dim();
        let (Some(r), Some(fan)) = (self.active.get_mut(&id), self.fan.as_mut()) else { return };
        let Some(&Step { start, positions, .. }) = r.in_flight.front() else { return };
        let f = &mut r.fan;
        let t0 = Instant::now();
        if f.layer == fan.layers {
            let sample = if f.every { Sample::Every } else { Sample::Last };
            let result = self.engine.head(id, &f.hidden, sample, &mut self.tokens);
            f.coord_us += t0.elapsed().as_micros() as u64;
            if let Err(e) = result {
                return self.finish(id, Event::Failed(e.to_string()));
            }
            STATS.compute.observe(Duration::from_micros(f.coord_us));
            let us = |t: u64| t.min(u32::MAX as u64) as u32;
            self.stages.clear();
            self.stages.push(us(f.coord_us));
            self.stages.extend(f.worker_us.iter().map(|&t| us(t)));
            self.workers_us = Some(f.critical_us);
            fan.since_rebalance += positions as u64;
            let tokens = std::mem::take(&mut self.tokens);
            self.on_tokens(id, start, &tokens);
            self.tokens = tokens;
            self.workers_us = None;
            return self.rebalance();
        }
        let layer = f.layer;
        if let Err(e) = self.engine.route(id, layer, &mut f.hidden, &mut f.x, &mut f.picks) {
            return self.finish(id, Event::Failed(e.to_string()));
        }
        f.sent.resize_with(fan.sends.len(), Vec::new);
        f.sent.iter_mut().for_each(Vec::clear);
        for (k, p) in f.picks.iter().enumerate() {
            fan.owners.count(layer, p.expert as u32);
            f.sent[fan.owners.owner(layer, p.expert as u32)].push(k);
        }
        f.outputs.resize(f.picks.len() * dim, 0.0);
        f.waiting = 0;
        f.layer_us = 0;
        f.coord_us += t0.elapsed().as_micros() as u64;
        // To every worker that has picks, at once; the rows shared, not copied (NET-6).
        let x = Arc::new(std::mem::replace(&mut f.x, take_rows()));
        for (w, send) in fan.sends.iter().enumerate() {
            if f.sent[w].is_empty() {
                continue;
            }
            fan.picks.clear();
            fan.picks.extend(f.sent[w].iter().map(|&k| f.picks[k]));
            if let Err(e) = send.send_experts(id, start, layer as u16, &fan.picks, &x) {
                let reason = format!("send to {}: {e}", send.peer_name());
                return self.cluster_down(reason);
            }
            f.waiting += 1;
        }
    }

    /// Expert split: worker `from`'s outputs for a layer. Once every worker's are in, finishes
    /// the layer and goes on to the next.
    fn on_partial(&mut self, from: usize, h: Header, layer: u16, compute_us: u32, values: Vec<f32>) {
        let (id, dim) = (h.request_id, self.engine.hidden_dim());
        // Nothing waits for a step that already failed.
        let Some(r) = self.active.get_mut(&id) else { return };
        let f = &mut r.fan;
        let current = r.in_flight.front().is_some_and(|s| s.start == h.position) && f.layer == layer as u32;
        if !current || f.sent.get(from).is_none_or(Vec::is_empty) {
            return;
        }
        let mut sent = std::mem::take(&mut f.sent[from]);
        if values.len() != sent.len() * dim {
            let name = self.fan.as_ref().and_then(|f| f.sends.get(from)).map_or("a worker", |c| c.peer_name());
            let msg = format!("{name}: {} output rows for {} picks", values.len() / dim, sent.len());
            return self.finish(id, Event::Failed(msg));
        }
        for (out, &k) in values.chunks_exact(dim).zip(&sent) {
            f.outputs[k * dim..(k + 1) * dim].copy_from_slice(out);
        }
        give_rows(values);
        sent.clear();
        f.sent[from] = sent;
        f.worker_us[from] += compute_us as u64;
        f.layer_us = f.layer_us.max(compute_us as u64);
        f.waiting -= 1;
        if f.waiting > 0 {
            return;
        }
        f.critical_us += f.layer_us;
        let t0 = Instant::now();
        self.engine.combine(layer as u32, &f.picks, &f.outputs, &mut f.hidden);
        f.coord_us += t0.elapsed().as_micros() as u64;
        f.layer += 1;
        self.route_layer(id);
    }

    /// Expert split: any frame but `Partial` from worker `from`.
    fn on_reply(&mut self, from: usize, h: Header, payload: Vec<u8>) {
        let id = h.request_id;
        match h.msg_type {
            MsgType::Ping => self.on_ping(id),
            MsgType::Own => {
                let Some(fan) = self.fan.as_mut().filter(|f| f.joining > 0) else {
                    return self.cluster_down(format!("worker {from} confirmed experts not sent"));
                };
                fan.joining -= 1;
                if fan.joining == 0 {
                    self.up();
                }
            }
            MsgType::Error if id == 0 && self.fan.as_ref().is_some_and(|f| f.joining > 0) => {
                let msg = String::from_utf8_lossy(&payload);
                self.cluster_down(format!("a worker back after the cluster was down: {msg}"))
            }
            // Answers a control frame: a move the worker could not take keeps the old owner.
            MsgType::Error if id == 0 => {
                let msg = String::from_utf8_lossy(&payload);
                match self.fan.as_mut().and_then(|f| f.moving.get_mut(from)?.pop_front()) {
                    Some(m) => warn!(layer = m.layer, expert = m.expert, error = %msg, "expert not moved"),
                    None => warn!(error = %msg, "worker error"),
                }
            }
            MsgType::Error => {
                if self.active.get(&id).is_some_and(|r| !r.in_flight.is_empty()) {
                    self.finish(id, Event::Failed(String::from_utf8_lossy(&payload).into_owned()))
                }
            }
            MsgType::Move => self.on_moved(from, &payload),
            other => self.cluster_down(format!("unexpected {other:?} frame from the workers")),
        }
    }

    /// Worker `to` holds a moved expert: the table changes, then the old owner lets it go.
    /// Layers sent to the old owner before still find it there; the links are FIFO.
    fn on_moved(&mut self, to: usize, payload: &[u8]) {
        let Some(fan) = self.fan.as_mut().filter(|f| !f.sends.is_empty()) else { return };
        let got = decode_move(payload).ok().map(|(l, e, _)| (l as u32, e as u32));
        let Some(m) = fan.moving[to].pop_front().filter(|m| Some((m.layer, m.expert)) == got) else {
            return self.cluster_down(format!("worker {to} confirmed a move not asked for: {got:?}"));
        };
        fan.owners.set_owner(m.layer, m.expert, m.to);
        debug!(layer = m.layer, expert = m.expert, from = m.from, to = m.to, "expert moved");
        encode_move(m.layer as u16, m.expert as u16, &[], &mut fan.frame);
        let old = &fan.sends[m.from];
        if let Err(e) = old.send(MsgType::Release, 0, 0, &fan.frame) {
            let reason = format!("send to {}: {e}", old.peer_name());
            self.cluster_down(reason);
        }
    }

    /// Expert split: every `rebalance_every` positions, sends a few experts to a less busy
    /// worker, unless earlier moves are still out.
    fn rebalance(&mut self) {
        let Some(fan) = &mut self.fan else { return };
        let every = fan.cfg.rebalance_every;
        if every == 0 || fan.since_rebalance < every || fan.sends.is_empty() || fan.moving.iter().any(|m| !m.is_empty()) {
            return;
        }
        fan.since_rebalance = 0;
        let moves = fan.owners.plan_moves(fan.cfg.max_moves, fan.max_owned);
        if moves.is_empty() {
            return;
        }
        info!(moves = moves.len(), "moving experts");
        for m in moves {
            if let Err(e) = self.engine.expert_weights(m.layer, m.expert, &mut fan.weights) {
                warn!(layer = m.layer, expert = m.expert, error = %e, "expert not moved");
                continue;
            }
            encode_move(m.layer as u16, m.expert as u16, &fan.weights, &mut fan.frame);
            debug!(layer = m.layer, expert = m.expert, from = m.from, to = m.to, "moving expert");
            let new = &fan.sends[m.to];
            if let Err(e) = new.send_owned(MsgType::Move, 0, 0, 0, std::mem::take(&mut fan.frame)) {
                let reason = format!("send to {}: {e}", new.peer_name());
                return self.cluster_down(reason);
            }
            fan.moving[m.to].push_back(m);
        }
    }

    fn on_frame(&mut self, h: Header, payload: Vec<u8>) {
        let id = h.request_id;
        let in_flight = self.active.get(&id).is_some_and(|r| r.freeing.is_none() && !r.in_flight.is_empty());
        match h.msg_type {
            MsgType::Token if in_flight => match decode_token(&payload, &mut self.tokens, &mut self.stages) {
                Ok(()) => {
                    let tokens = std::mem::take(&mut self.tokens);
                    self.on_tokens(id, h.position, &tokens);
                    self.tokens = tokens;
                }
                Err(e) => self.finish(id, Event::Failed(format!("bad token frame: {e}"))),
            },
            MsgType::Error if in_flight => {
                self.finish(id, Event::Failed(String::from_utf8_lossy(&payload).into_owned()))
            }
            MsgType::Ping => self.on_ping(id),
            MsgType::Drop => {
                if let Some(outcome) = self.active.get_mut(&id).and_then(|r| r.freeing.take()) {
                    let r = self.active.remove(&id).unwrap();
                    let _ = r.events.send(outcome);
                }
            }
            // Frames for requests that already failed are expected and harmless; a `Rewind`
            // back from the workers needs nothing more.
            MsgType::Token | MsgType::Error | MsgType::Rewind => {}
            other => self.cluster_down(format!("unexpected {other:?} frame from the workers")),
        }
    }

    /// Handles the head's tokens for the step starting at `start`; `self.stages` holds the
    /// step's per-node times.
    fn on_tokens(&mut self, id: u64, start: u32, tokens: &[u32]) {
        let eos = self.engine.eos();
        let in_flight = self.active.len();
        let Some(r) = self.active.get_mut(&id) else { return };
        let Some(step) = r.in_flight.pop_front().filter(|s| s.start == start) else {
            let msg = format!("unexpected token for position {start}");
            return self.finish(id, Event::Failed(msg));
        };
        let want = if step.drafts.is_empty() { 1 } else { step.positions as usize };
        if tokens.len() != want {
            let msg = format!("{} tokens for a step that samples {want}", tokens.len());
            return self.finish(id, Event::Failed(msg));
        }
        let end = (step.start + step.positions) as usize;
        let kind = match end {
            n if n < r.prompt_len => StepKind::Prefill,
            n if n == r.prompt_len => StepKind::FirstToken,
            _ => StepKind::Decode,
        };
        // Drafts the head agrees with: tokens[i] is the head's token after draft i - 1.
        let accepted = step.drafts.iter().zip(tokens).take_while(|(d, t)| d == t).count();
        let step_us = step.started.elapsed().as_micros() as u64;
        debug!(
            request = r.id,
            state = id,
            pos = step.start,
            positions = step.positions,
            kind = kind.as_str(),
            drafted = step.drafts.len(),
            accepted,
            step_us,
            stage_us = ?self.stages,
            "step back"
        );
        match kind {
            StepKind::Decode => STATS.decode_step.observe(step.started.elapsed()),
            _ => STATS.prompt_positions.add(step.positions as u64),
        }
        STATS.drafted.add(step.drafts.len() as u64);
        STATS.accepted.add(accepted as u64);
        if kind == StepKind::FirstToken {
            let ttft = r.submitted.elapsed();
            STATS.time_to_first_token.observe(ttft);
            info!(request = r.id, state = id, ttft_ms = ttft.as_millis() as u64, "first token");
        }
        if let Some(m) = &mut self.metrics {
            let rec = StepRecord {
                request: id,
                pos: step.start,
                positions: step.positions,
                kind,
                tokens: &tokens[..=accepted],
                step_us,
                stage_us: &self.stages,
                workers_us: self.workers_us,
                queue_us: r.queued.as_micros() as u64,
                in_flight,
                ping_rtt_us: self.ping_rtt.map(|d| d.as_micros() as u64),
                cached: r.cached as u32,
                drafted: step.drafts.len() as u32,
                accepted: accepted as u32,
            };
            if let Err(e) = m.record(&rec) {
                error!(error = %e, "metrics: no more rows will be written");
                self.metrics = None;
            }
        }
        if kind == StepKind::Prefill {
            // A token predicted inside the prompt; the prompt's own next token wins.
            if r.can_step(&self.cfg) {
                push_ready(&mut self.ready, id);
            }
            return;
        }
        // Finishing before the rewind below would leave rejected drafts in the engines.
        r.parkable = step.drafts.is_empty();
        for &token in &tokens[..=accepted] {
            if Some(token) == eos {
                return self.finish(id, Event::Done);
            }
            r.tokens.push(token);
            let ev = Event::Token { token, latency: step.started.elapsed() };
            STATS.tokens_generated.add(1);
            if r.events.send(ev).is_err() {
                // The client went away: cancel. Nobody is listening for the outcome.
                info!(request = r.id, state = id, generated = r.generated(), "cancelled by the client");
                r.cancelled = true;
                STATS.requests_cancelled.add(1);
                return self.finish(id, Event::Done);
            }
            if r.generated() == r.max_new {
                return self.finish(id, Event::Done);
            }
        }
        // Positions up to the last accepted draft stay; the rest go on every node.
        r.parkable = true;
        r.next_pos = end - (step.drafts.len() - accepted);
        if accepted < step.drafts.len() {
            let keep = r.next_pos as u32;
            if let Err(e) = self.engine.rewind(id, keep) {
                return self.finish(id, Event::Failed(e.to_string()));
            }
            if let Some(next) = &self.next {
                if let Err(e) = next.send(MsgType::Rewind, id, keep, &[]) {
                    let reason = format!("send to {}: {e}", next.peer_name());
                    return self.cluster_down(reason);
                }
            }
        }
        push_ready(&mut self.ready, id);
    }

    /// Frees the request on this node and, with workers, sends `Drop` through them; the outcome is
    /// reported once `Drop` returns, so a finished request is freed on every node (CL-9). A
    /// request that finished well is parked instead when prefix reuse is on.
    fn finish(&mut self, id: u64, outcome: Event) {
        if let Some(Err(e)) = self.metrics.as_mut().map(Metrics::flush) {
            error!(error = %e, "metrics flush");
        }
        if let Some(r) = self.active.get(&id) {
            let elapsed = r.submitted.elapsed();
            let generated = r.generated();
            match &outcome {
                Event::Failed(e) => {
                    STATS.requests_failed.add(1);
                    warn!(request = r.id, state = id, generated, error = %e, "failed")
                }
                // Logged and counted where the client went away.
                _ if r.cancelled => {}
                _ => {
                    STATS.requests_done.add(1);
                    info!(
                        request = r.id,
                        state = id,
                        generated,
                        ms = elapsed.as_millis() as u64,
                        tok_s = (generated as f64 / elapsed.as_secs_f64() * 10.0).round() / 10.0,
                        "done"
                    )
                }
            }
        }
        self.ready.retain(|&r| r != id);
        let park = self.cfg.prefix_cache > 0
            && !self.shutting_down
            && outcome == Event::Done
            && self.active.get(&id).is_some_and(|r| r.parkable && r.in_flight.is_empty() && r.next_pos > 0);
        if park {
            let mut r = self.active.remove(&id).unwrap();
            r.tokens.truncate(r.next_pos);
            let _ = r.events.send(outcome);
            debug!(request = r.id, state = id, positions = r.tokens.len(), "parked");
            self.parked.push_back(Parked { id, request: r.id, tokens: r.tokens });
            if self.parked.len() > self.cfg.prefix_cache {
                let oldest = self.parked.pop_front().unwrap();
                self.evict(oldest);
            }
            return;
        }
        self.engine.free(id);
        let Some(next) = &self.next else {
            if let Some(r) = self.active.remove(&id) {
                let _ = r.events.send(outcome);
            }
            return;
        };
        if let Err(e) = next.send(MsgType::Drop, id, 0, &[]) {
            let reason = format!("send to {}: {e}", next.peer_name());
            return self.cluster_down(reason);
        }
        if let Some(r) = self.active.get_mut(&id) {
            r.freeing = Some(outcome);
        }
    }

    /// Frees a parked request on every node. Its `Drop` comes back to no active request and is
    /// ignored; any later step for a request reusing the slot follows it on the same links.
    fn evict(&mut self, p: Parked) {
        debug!(request = p.request, state = p.id, "dropping parked state");
        self.engine.free(p.id);
        if let Some(next) = &self.next {
            if let Err(e) = next.send(MsgType::Drop, p.id, 0, &[]) {
                let reason = format!("send to {}: {e}", next.peer_name());
                self.cluster_down(reason);
            }
        }
    }

    /// A worker is gone: fail every request, cut every link and refuse new requests until all
    /// workers are back. Workers' state is lost with them; this node's is freed.
    fn cluster_down(&mut self, reason: String) {
        if self.writers.is_empty() {
            return;
        }
        if self.down.is_some() {
            warn!(reason, "a worker lost again before the cluster was up");
        } else {
            error!(reason, active = self.active.len(), waiting = self.waiting.len(), "cluster down");
            STATS.cluster_downs.add(1);
            STATS.cluster_up.store(0, Ordering::Relaxed);
        }
        let failed = format!("cluster down: {reason}");
        for (id, r) in self.active.drain() {
            warn!(request = r.id, state = id, generated = r.generated(), "failed: cluster down");
            STATS.requests_failed.add(1);
            self.engine.free(id);
            let ev = match r.freeing {
                Some(outcome @ Event::Done) => outcome,
                _ => Event::Failed(failed.clone()),
            };
            let _ = r.events.send(ev);
        }
        for (_, r) in self.waiting.drain(..) {
            warn!(request = r.id, "failed: cluster down");
            STATS.requests_failed.add(1);
            let _ = r.events.send(Event::Failed(failed.clone()));
        }
        for p in self.parked.drain(..) {
            self.engine.free(p.id);
        }
        self.ready.clear();
        for w in self.writers.drain(..) {
            w.abort();
        }
        self.next = None;
        if let Some(fan) = &mut self.fan {
            fan.sends.clear();
            fan.joining = 0;
            fan.moving.iter_mut().for_each(VecDeque::clear);
        }
        self.epoch += 1;
        self.ping = None;
        self.down = Some(reason);
        self.reconnect();
    }
}
