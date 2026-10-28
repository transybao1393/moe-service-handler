//! Coordinator scheduler (CL-5): keeps up to `max_in_flight` requests in the ring at once.
//!
//! A decoding request has one position in flight, because its next input is the token that
//! position produces. Throughput comes from interleaving requests: while request A's hidden
//! state is on the workers, the coordinator runs its own layers for request B.
//!
//! The prompt is known up front, so prefill goes in chunks of `prefill_chunk` positions (CL-6),
//! and up to `prefill_depth` chunks of one request may be in the ring at once: the coordinator
//! runs chunk 2 while the workers run chunk 1. Every link is FIFO, so chunks reach each node in
//! order. Only the last chunk's token is used; it is the first generated token.
//!
//! Speculative decoding (`max_draft` > 0, engines that can rewind): a decode step carries the
//! next token plus up to `max_draft` drafted guesses, and the head samples after every
//! position. The coordinator keeps the guesses that match the head's tokens, emits those and
//! the head's next token, and sends `Rewind` round the ring for the rest. Rewind and the next
//! step follow each other on the same FIFO links, so rejecting costs no extra round trip.
//!
//! Two threads. The scheduler thread owns the engine and the connection to the next node; it
//! steps ready requests in FIFO order. A reader thread owns the connection from the previous
//! node and turns every frame into a message for the scheduler. Clients submit through a
//! [`Client`] and receive their tokens as [`Event`]s on their own channel; dropping that channel
//! cancels the request.
//!
//! Health (CL-8): every `health_interval_ms` the scheduler sends a `Ping` round the ring. If it
//! has not come back within `health_timeout_ms`, or a connection breaks, the ring is down:
//! in-flight requests fail and new ones are rejected until the cluster is restarted. The ping
//! queues behind hidden states on every node, so the timeout must exceed one full ring pass,
//! prefill chunks included.

use std::collections::{HashMap, VecDeque};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use msh_net::{decode_token, encode_hidden, Connection, Header, MsgType, FLAG_EVERY_TOKEN};
use serde::Deserialize;

use crate::draft::{Drafter, PromptLookup};
use crate::engine::{Engine, Sample, StageIn};
use crate::metrics::{Metrics, StepKind, StepRecord};
use crate::node::Ring;
use crate::Error;

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SchedulerConfig {
    /// Requests stepping through the ring at once.
    pub max_in_flight: usize,
    /// Requests waiting for a slot once all are taken; more are rejected.
    pub max_queue: usize,
    /// How often a health ping is sent round the ring.
    pub health_interval_ms: u64,
    /// How long a ping may take to come back before the ring is declared down.
    pub health_timeout_ms: u64,
    /// Prompt positions per prefill step. Bigger chunks make better use of the engine's
    /// batched path; smaller ones let other requests' decode steps in sooner.
    pub prefill_chunk: usize,
    /// Prefill chunks of one request in the ring at once.
    pub prefill_depth: usize,
    /// Draft tokens per decode step (speculative decoding); 0 turns it off.
    pub max_draft: usize,
    /// Shortest repeat of the latest tokens that prompt lookup drafts from.
    pub draft_min_ngram: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        SchedulerConfig {
            max_in_flight: 3,
            max_queue: 64,
            health_interval_ms: 1000,
            health_timeout_ms: 5000,
            prefill_chunk: 64,
            prefill_depth: 3,
            max_draft: 0,
            draft_min_ngram: 3,
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
    /// Not admitted: queue full, empty prompt, ring down or shutting down.
    Rejected(String),
}

#[derive(Debug)]
pub struct Generation {
    pub tokens: Vec<u32>,
    pub latencies: Vec<Duration>,
}

enum Msg {
    Submit { prompt: Vec<u32>, max_new: usize, events: Sender<Event> },
    Frame(Header, Vec<u8>),
    RingDown(String),
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
}

impl Drop for ClientInner {
    fn drop(&mut self) {
        let _ = self.tx.send(Msg::Shutdown);
    }
}

impl Client {
    /// Queues a request for greedy generation of up to `max_new` tokens.
    pub fn submit(&self, prompt: Vec<u32>, max_new: usize) -> Receiver<Event> {
        let (events, rx) = mpsc::channel();
        let msg = Msg::Submit { prompt, max_new, events: events.clone() };
        if self.inner.tx.send(msg).is_err() {
            let _ = events.send(Event::Rejected("scheduler stopped".into()));
        }
        rx
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

/// Starts the scheduler (and, for a ring, its reader thread). Returns the client handle and
/// the scheduler thread, which yields the engine back after `shutdown`.
pub fn spawn<E: Engine + 'static>(
    engine: E,
    ring: Option<Ring>,
    cfg: SchedulerConfig,
) -> (Client, JoinHandle<E>) {
    spawn_with_metrics(engine, ring, cfg, None)
}

/// [`spawn`], also writing one metrics row per step to `metrics`.
pub fn spawn_with_metrics<E: Engine + 'static>(
    engine: E,
    ring: Option<Ring>,
    cfg: SchedulerConfig,
    metrics: Option<Metrics>,
) -> (Client, JoinHandle<E>) {
    let (tx, rx) = mpsc::channel();
    let next = ring.map(|Ring { prev, next }| {
        let tx = tx.clone();
        thread::spawn(move || read_ring(prev, tx));
        next
    });
    let mut cfg = cfg;
    if cfg.max_draft > 0 && !engine.can_speculate() {
        eprintln!("scheduler: the engine cannot verify and rewind draft tokens; speculative decoding is off");
        cfg.max_draft = 0;
    }
    let drafter = PromptLookup { min_ngram: cfg.draft_min_ngram, max_ngram: cfg.draft_min_ngram.max(8) };
    let s = Scheduler {
        engine,
        next,
        cfg,
        rx,
        drafter: Box::new(drafter),
        hidden: Vec::new(),
        input: Vec::new(),
        tokens: Vec::new(),
        buf: Vec::new(),
        next_id: 1,
        waiting: VecDeque::new(),
        active: HashMap::new(),
        ready: VecDeque::new(),
        down: None,
        shutting_down: false,
        ping: None,
        ping_seq: 0,
        next_ping_at: Instant::now(),
        ring_rtt: None,
        metrics,
        stages: Vec::new(),
    };
    (Client { inner: Arc::new(ClientInner { tx }) }, thread::spawn(move || s.run()))
}

fn read_ring(mut prev: Connection, tx: Sender<Msg>) {
    let mut buf = Vec::new();
    loop {
        match prev.recv_frame(&mut buf) {
            Ok(h) => {
                if tx.send(Msg::Frame(h, std::mem::take(&mut buf))).is_err() {
                    return;
                }
            }
            Err(e) => {
                let _ = tx.send(Msg::RingDown(format!("lost connection from {}: {e}", prev.peer_name())));
                return;
            }
        }
    }
}

/// A step sent round the ring and not back yet.
struct Step {
    start: u32,
    positions: u32,
    /// Draft tokens after the real one; the head samples every position when there are any.
    drafts: Vec<u32>,
    started: Instant,
}

struct Request {
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
    /// `Drop` sent; this outcome is reported when it comes back round the ring.
    freeing: Option<Event>,
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
    next: Option<Connection>,
    cfg: SchedulerConfig,
    rx: Receiver<Msg>,
    drafter: Box<dyn Drafter>,
    hidden: Vec<f32>,
    /// Scratch for a step's input tokens and the head's output tokens.
    input: Vec<u32>,
    tokens: Vec<u32>,
    buf: Vec<u8>,
    next_id: u64,
    waiting: VecDeque<(u64, Request)>,
    active: HashMap<u64, Request>,
    ready: VecDeque<u64>,
    /// Set once the ring is lost; every later request is rejected with this reason.
    down: Option<String>,
    shutting_down: bool,
    /// Outstanding health ping: sequence number and send time.
    ping: Option<(u64, Instant)>,
    ping_seq: u64,
    next_ping_at: Instant,
    /// Last health ping's trip round the ring (NET-5).
    ring_rtt: Option<Duration>,
    metrics: Option<Metrics>,
    /// Scratch for the stage times of a returning token.
    stages: Vec<u32>,
}

impl<E: Engine> Scheduler<E> {
    fn run(mut self) -> E {
        loop {
            while let Ok(m) = self.rx.try_recv() {
                self.handle(m);
            }
            self.check_health();
            self.admit();
            if self.shutting_down && self.active.is_empty() {
                return self.engine;
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
                None => return self.engine,
            }
        }
    }

    /// When the health check next needs attention, or `None` without a live ring.
    fn health_deadline(&self) -> Option<Instant> {
        self.next.as_ref()?;
        Some(match self.ping {
            Some((_, sent)) => sent + Duration::from_millis(self.cfg.health_timeout_ms),
            None => self.next_ping_at,
        })
    }

    fn check_health(&mut self) {
        let Some(next) = &mut self.next else { return };
        let now = Instant::now();
        if let Some((seq, sent)) = self.ping {
            if now.duration_since(sent) > Duration::from_millis(self.cfg.health_timeout_ms) {
                let reason = format!("health ping {seq} not back after {} ms", self.cfg.health_timeout_ms);
                self.ring_down(reason);
            }
        } else if now >= self.next_ping_at {
            self.ping_seq += 1;
            if let Err(e) = next.send(MsgType::Ping, self.ping_seq, 0, &[]) {
                let reason = format!("send to {}: {e}", next.peer_name());
                return self.ring_down(reason);
            }
            self.ping = Some((self.ping_seq, now));
            self.next_ping_at = now + Duration::from_millis(self.cfg.health_interval_ms);
        }
    }

    fn handle(&mut self, m: Msg) {
        match m {
            Msg::Submit { prompt, max_new, events } => self.submit(prompt, max_new, events),
            Msg::Frame(h, payload) => self.on_frame(h, payload),
            Msg::RingDown(reason) => self.ring_down(reason),
            Msg::Shutdown => {
                self.shutting_down = true;
                for (_, r) in self.waiting.drain(..) {
                    let _ = r.events.send(Event::Rejected("shutting down".into()));
                }
            }
        }
    }

    fn submit(&mut self, prompt: Vec<u32>, max_new: usize, events: Sender<Event>) {
        let reject = if let Some(reason) = &self.down {
            Some(format!("ring down: {reason}"))
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
            let _ = events.send(Event::Rejected(reason));
            return;
        }
        if max_new == 0 {
            let _ = events.send(Event::Done);
            return;
        }
        let id = self.next_id;
        self.next_id += 1;
        let r = Request {
            prompt_len: prompt.len(),
            tokens: prompt,
            max_new,
            next_pos: 0,
            events,
            in_flight: VecDeque::new(),
            submitted: Instant::now(),
            queued: Duration::ZERO,
            freeing: None,
        };
        self.waiting.push_back((id, r));
    }

    fn admit(&mut self) {
        while self.active.len() < self.cfg.max_in_flight.max(1) {
            let Some((id, mut r)) = self.waiting.pop_front() else { break };
            r.queued = r.submitted.elapsed();
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
        let input = StageIn::Tokens(&self.input);
        if let Err(e) = self.engine.step(id, start as u32, input, sample, &mut self.hidden, &mut self.tokens) {
            return self.finish(id, Event::Failed(e.to_string()));
        }
        let coord_us = started.elapsed().as_micros().min(u32::MAX as u128) as u32;
        let positions = self.input.len() as u32;
        r.next_pos += positions as usize;
        r.in_flight.push_back(Step { start: start as u32, positions, drafts, started });
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
                let msg = "coordinator stage does not match the ring (head on the wrong node)";
                return self.finish(id, Event::Failed(msg.into()));
            }
        };
        encode_hidden(&[coord_us], &self.hidden, &mut self.buf);
        let flags = if sample == Sample::Every { FLAG_EVERY_TOKEN } else { 0 };
        if let Err(e) = next.send_flagged(MsgType::Hidden, flags, id, start as u32, &self.buf) {
            let reason = format!("send to {}: {e}", next.peer_name());
            return self.ring_down(reason);
        }
        // The next prompt chunk can follow right behind this one.
        if r.can_step(&self.cfg) {
            push_ready(&mut self.ready, id);
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
            MsgType::Ping => {
                if let Some((_, sent)) = self.ping.filter(|&(seq, _)| seq == id) {
                    self.ring_rtt = Some(sent.elapsed());
                    self.ping = None;
                }
            }
            MsgType::Drop => {
                if let Some(outcome) = self.active.get_mut(&id).and_then(|r| r.freeing.take()) {
                    let r = self.active.remove(&id).unwrap();
                    let _ = r.events.send(outcome);
                }
            }
            // Frames for requests that already failed are expected and harmless; a `Rewind`
            // back from the ring needs nothing more.
            MsgType::Token | MsgType::Error | MsgType::Rewind => {}
            other => self.ring_down(format!("unexpected {other:?} frame from the ring")),
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
        if let Some(m) = &mut self.metrics {
            let rec = StepRecord {
                request: id,
                pos: step.start,
                positions: step.positions,
                kind,
                tokens: &tokens[..=accepted],
                step_us: step.started.elapsed().as_micros() as u64,
                stage_us: &self.stages,
                queue_us: r.queued.as_micros() as u64,
                in_flight,
                ring_rtt_us: self.ring_rtt.map(|d| d.as_micros() as u64),
                drafted: step.drafts.len() as u32,
                accepted: accepted as u32,
            };
            if let Err(e) = m.record(&rec) {
                eprintln!("metrics: {e}; no more rows will be written");
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
        for &token in &tokens[..=accepted] {
            if Some(token) == eos {
                return self.finish(id, Event::Done);
            }
            r.tokens.push(token);
            let ev = Event::Token { token, latency: step.started.elapsed() };
            if r.events.send(ev).is_err() {
                // The client went away: cancel. Nobody is listening for the outcome.
                return self.finish(id, Event::Done);
            }
            if r.generated() == r.max_new {
                return self.finish(id, Event::Done);
            }
        }
        // Positions up to the last accepted draft stay; the rest go on every node.
        r.next_pos = end - (step.drafts.len() - accepted);
        if accepted < step.drafts.len() {
            let keep = r.next_pos as u32;
            if let Err(e) = self.engine.rewind(id, keep) {
                return self.finish(id, Event::Failed(e.to_string()));
            }
            if let Some(next) = &mut self.next {
                if let Err(e) = next.send(MsgType::Rewind, id, keep, &[]) {
                    let reason = format!("send to {}: {e}", next.peer_name());
                    return self.ring_down(reason);
                }
            }
        }
        push_ready(&mut self.ready, id);
    }

    /// Frees the request on this node and, for a ring, sends `Drop` round it; the outcome is
    /// reported once `Drop` returns, so a finished request is freed on every node (CL-9).
    fn finish(&mut self, id: u64, outcome: Event) {
        self.engine.free(id);
        if let Some(Err(e)) = self.metrics.as_mut().map(Metrics::flush) {
            eprintln!("metrics: {e}");
        }
        self.ready.retain(|&r| r != id);
        let Some(next) = &mut self.next else {
            if let Some(r) = self.active.remove(&id) {
                let _ = r.events.send(outcome);
            }
            return;
        };
        if let Err(e) = next.send(MsgType::Drop, id, 0, &[]) {
            let reason = format!("send to {}: {e}", next.peer_name());
            return self.ring_down(reason);
        }
        if let Some(r) = self.active.get_mut(&id) {
            r.freeing = Some(outcome);
        }
    }

    /// The ring is gone: fail every request and refuse new ones. Workers' state is lost with
    /// them; this node's is freed.
    fn ring_down(&mut self, reason: String) {
        if self.down.is_some() {
            return;
        }
        let failed = format!("ring down: {reason}");
        for (id, r) in self.active.drain() {
            self.engine.free(id);
            let ev = match r.freeing {
                Some(outcome @ Event::Done) => outcome,
                _ => Event::Failed(failed.clone()),
            };
            let _ = r.events.send(ev);
        }
        for (_, r) in self.waiting.drain(..) {
            let _ = r.events.send(Event::Failed(failed.clone()));
        }
        self.ready.clear();
        self.next = None;
        self.down = Some(reason);
    }
}
