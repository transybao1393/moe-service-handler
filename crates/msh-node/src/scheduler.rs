//! Coordinator scheduler (CL-5): keeps up to `max_in_flight` requests in the ring at once.
//!
//! Each request has at most one position in flight, because its next input is the token that
//! position produces. Throughput comes from interleaving requests: while request A's hidden
//! state is on the workers, the coordinator runs its own layers for request B.
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
//! queues behind hidden states on every node, so the timeout must exceed one full ring pass.

use std::collections::{HashMap, VecDeque};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use msh_net::{f32s_to_bytes, Connection, Header, MsgType};
use serde::Deserialize;

use crate::engine::{Engine, StageIn};
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
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        SchedulerConfig { max_in_flight: 3, max_queue: 64, health_interval_ms: 1000, health_timeout_ms: 5000 }
    }
}

/// What a client receives for one request. A request ends with exactly one of `Done`,
/// `Failed` or `Rejected`.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// A generated token. `latency` runs from the step's start on the coordinator to the
    /// token's arrival, network included.
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
    let (tx, rx) = mpsc::channel();
    let next = ring.map(|Ring { prev, next }| {
        let tx = tx.clone();
        thread::spawn(move || read_ring(prev, tx));
        next
    });
    let hidden = vec![0.0; engine.hidden_dim()];
    let s = Scheduler {
        engine,
        next,
        cfg,
        rx,
        hidden,
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

enum Phase {
    /// Waiting for its next step on the coordinator.
    Ready,
    /// A position is on the workers.
    InFlight,
    /// `Drop` sent; the outcome is reported when it comes back round the ring.
    Freeing(Event),
}

struct Request {
    prompt: Vec<u32>,
    max_new: usize,
    /// Position of the next step, and the token fed into it.
    pos: usize,
    input: u32,
    generated: usize,
    events: Sender<Event>,
    step_started: Instant,
    phase: Phase,
}

struct Scheduler<E: Engine> {
    engine: E,
    next: Option<Connection>,
    cfg: SchedulerConfig,
    rx: Receiver<Msg>,
    hidden: Vec<f32>,
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
        let input = prompt[0];
        let r = Request {
            prompt,
            max_new,
            pos: 0,
            input,
            generated: 0,
            events,
            step_started: Instant::now(),
            phase: Phase::Ready,
        };
        self.waiting.push_back((id, r));
    }

    fn admit(&mut self) {
        while self.active.len() < self.cfg.max_in_flight.max(1) {
            let Some((id, r)) = self.waiting.pop_front() else { break };
            self.active.insert(id, r);
            self.ready.push_back(id);
        }
    }

    /// Runs the coordinator's part of one position and sends it on (or, in mode A, finishes it).
    fn step(&mut self, id: u64) {
        let Some(r) = self.active.get_mut(&id) else { return };
        r.step_started = Instant::now();
        let (pos, input) = (r.pos as u32, r.input);
        let token = match self.engine.step(id, pos, StageIn::Token(input), &mut self.hidden) {
            Ok(token) => token,
            Err(e) => return self.finish(id, Event::Failed(e.to_string())),
        };
        let next = match (token, &mut self.next) {
            // Mode A: this node holds the head.
            (Some(token), None) => return self.on_token(id, pos, token),
            (None, Some(next)) => next,
            (Some(_), Some(_)) | (None, None) => {
                let msg = "coordinator stage does not match the ring (head on the wrong node)";
                return self.finish(id, Event::Failed(msg.into()));
            }
        };
        f32s_to_bytes(&self.hidden, &mut self.buf);
        if let Err(e) = next.send(MsgType::Hidden, id, pos, &self.buf) {
            let reason = format!("send to {}: {e}", next.peer_name());
            return self.ring_down(reason);
        }
        if let Some(r) = self.active.get_mut(&id) {
            r.phase = Phase::InFlight;
        }
    }

    fn on_frame(&mut self, h: Header, payload: Vec<u8>) {
        let id = h.request_id;
        let in_flight = matches!(self.active.get(&id).map(|r| &r.phase), Some(Phase::InFlight));
        match h.msg_type {
            MsgType::Token if in_flight && payload.len() == 4 => {
                self.on_token(id, h.position, u32::from_le_bytes(payload[..4].try_into().unwrap()))
            }
            MsgType::Error if in_flight => {
                self.finish(id, Event::Failed(String::from_utf8_lossy(&payload).into_owned()))
            }
            MsgType::Ping => {
                if self.ping.is_some_and(|(seq, _)| seq == id) {
                    self.ping = None;
                }
            }
            MsgType::Drop => {
                if let Some(Request { phase: Phase::Freeing(outcome), events, .. }) = self.active.remove(&id) {
                    let _ = events.send(outcome);
                }
            }
            // Frames for requests that already failed are expected and harmless.
            MsgType::Token | MsgType::Error => {}
            other => self.ring_down(format!("unexpected {other:?} frame from the ring")),
        }
    }

    fn on_token(&mut self, id: u64, pos: u32, token: u32) {
        let eos = self.engine.eos();
        let Some(r) = self.active.get_mut(&id) else { return };
        if pos as usize != r.pos {
            let msg = format!("token for position {pos}, expected {}", r.pos);
            return self.finish(id, Event::Failed(msg));
        }
        r.pos += 1;
        r.phase = Phase::Ready;
        if r.pos < r.prompt.len() {
            r.input = r.prompt[r.pos];
            self.ready.push_back(id);
            return;
        }
        if Some(token) == eos {
            return self.finish(id, Event::Done);
        }
        r.generated += 1;
        let ev = Event::Token { token, latency: r.step_started.elapsed() };
        if r.events.send(ev).is_err() {
            // The client went away: cancel. Nobody is listening for the outcome.
            return self.finish(id, Event::Done);
        }
        if r.generated == r.max_new {
            return self.finish(id, Event::Done);
        }
        r.input = token;
        self.ready.push_back(id);
    }

    /// Frees the request on this node and, for a ring, sends `Drop` round it; the outcome is
    /// reported once `Drop` returns, so a finished request is freed on every node (CL-9).
    fn finish(&mut self, id: u64, outcome: Event) {
        self.engine.free(id);
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
            r.phase = Phase::Freeing(outcome);
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
            let ev = match r.phase {
                Phase::Freeing(outcome @ Event::Done) => outcome,
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
