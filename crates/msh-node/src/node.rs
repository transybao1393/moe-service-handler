//! Star topology and the worker loop. The coordinator side lives in `scheduler`.
//!
//! The coordinator (the controller) holds one persistent TCP connection to each worker; workers
//! have no connections to one another. A step goes from the controller to the first worker and
//! comes back; the controller sends the result on to the next worker, and so on, until the
//! worker holding the head sends the sampled tokens back. `Drop` takes the same path, so when
//! it returns from the last worker every node has freed the request (CL-9); `Rewind` (rejected
//! draft tokens) too. An engine error on a worker comes back as an `Error` frame, which the
//! controller passes on until it reaches the scheduler; the cluster stays up.
//!
//! With the expert split the controller sends each layer to every worker that has picks at
//! once, and each answers on its own link: `Experts` gets back a `Partial`. A worker holds the
//! experts `Own` and `Move` give it and no state per request, so `Drop` and `Rewind` never
//! reach it. The same loop serves both splits; the controller decides which frames come.
//!
//! With the request split a worker runs [`RequestWorker`] instead: the whole model, whole
//! requests.
//!
//! Every link sends through a [`Writer`]: small frames go ahead of chunked hidden states. A
//! worker serves one controller connection after another: when the link ends, or nothing has
//! come from the controller for `health_timeout_ms` (it pings every `health_interval_ms`), it
//! frees every request it held and waits for the next controller.

use std::collections::{HashMap, HashSet, VecDeque};
use std::io::ErrorKind;
use std::net::TcpListener;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};

use msh_net::{
    decode_move, decode_own, decode_start, encode_done, encode_move, encode_status, encode_token, Connection,
    DoneReason, FrameError, Header, MsgType, RunStatus, Status, Writer, FLAG_EVERY_TOKEN,
};

use crate::backoff::Backoff;
use crate::config::{Config, NetConfig};
use crate::engine::{Engine, Sample, StageIn};
use crate::Error;
use tracing::{debug, info, trace, warn};

/// The controller's connections to its workers, in config order, each split into a receiving
/// half and a [`Writer`].
pub struct Star {
    pub(crate) workers: Vec<(Connection, Writer)>,
    /// How to connect to them again.
    pub(crate) plan: Arc<Plan>,
}

/// What reconnecting to the workers takes.
pub(crate) struct Plan {
    pub(crate) me: String,
    /// Each worker's name and address, in config order.
    pub(crate) workers: Vec<(String, String)>,
    pub(crate) net: NetConfig,
    /// How often a held link is pinged while others are still missing, so its worker keeps it.
    pub(crate) ping_every: Duration,
}

impl Plan {
    fn new(cfg: &Config) -> Plan {
        Plan {
            me: cfg.nodes[0].name.clone(),
            workers: cfg.nodes[1..].iter().map(|n| (n.name.clone(), n.addr.clone())).collect(),
            net: cfg.net,
            ping_every: Duration::from_millis(cfg.scheduler.health_interval_ms.clamp(1, 1000)),
        }
    }

    /// One try at worker `i`: connected, its name checked, split.
    fn connect(&self, i: usize, timeout: Duration) -> Result<(Connection, Writer), Error> {
        let (name, addr) = &self.workers[i];
        let conn = Connection::connect(addr.as_str(), &self.me, timeout)
            .map_err(|e| Error::Protocol(format!("connect to {addr}: {e}")))?
            .with_max_chunk(self.net.max_chunk());
        if conn.peer_name() != name {
            return Err(Error::Config(format!("{}: expected worker {name} at {addr}, got {}", self.me, conn.peer_name())));
        }
        Ok(conn.split()?)
    }
}

/// The controller's side: connects to every worker in config order. Returns `None` for a
/// one-node cluster.
pub fn connect_workers(cfg: &Config, timeout: Duration) -> Result<Option<Star>, Error> {
    let plan = Arc::new(Plan::new(cfg));
    let mut workers = Vec::new();
    for (name, addr) in &plan.workers {
        let conn = connect_with_retry(addr, &plan.me, timeout)?.with_max_chunk(cfg.net.max_chunk());
        if conn.peer_name() != name {
            return Err(Error::Config(format!(
                "{}: expected worker {name} at {addr}, got {}",
                plan.me,
                conn.peer_name()
            )));
        }
        workers.push(conn.split()?);
    }
    Ok((!workers.is_empty()).then_some(Star { workers, plan }))
}

/// Connects to every worker of `plan` again, trying each missing one with backoff until all
/// are back or `stop` is set. Links already back are pinged meanwhile, so their workers keep
/// them.
pub(crate) fn reconnect(plan: &Arc<Plan>, stop: &AtomicBool) -> Option<Star> {
    let n = plan.workers.len();
    let mut links: Vec<Option<(Connection, Writer)>> = (0..n).map(|_| None).collect();
    let mut backoff = Backoff::new(&plan.net);
    loop {
        if stop.load(Ordering::Relaxed) {
            for (_, w) in links.iter().flatten() {
                w.close();
            }
            return None;
        }
        let attempt = backoff.tries() + 1;
        for i in 0..n {
            if links[i].is_some() {
                continue;
            }
            let name = &plan.workers[i].0;
            match plan.connect(i, Duration::from_secs(1)) {
                Ok(link) => {
                    info!(worker = name, attempt, "worker reconnected");
                    links[i] = Some(link);
                }
                Err(e) => info!(worker = name, attempt, error = %e, "reconnect failed"),
            }
        }
        if links.iter().all(Option::is_some) {
            return Some(Star { workers: links.into_iter().flatten().collect(), plan: plan.clone() });
        }
        let wait = backoff.failed();
        let missing: Vec<&str> = (0..n).filter(|&i| links[i].is_none()).map(|i| plan.workers[i].0.as_str()).collect();
        info!(attempt, missing = missing.join(" "), wait_ms = wait.as_millis() as u64, "reconnecting");
        let until = Instant::now() + wait;
        while !stop.load(Ordering::Relaxed) {
            let left = until.saturating_duration_since(Instant::now());
            if left.is_zero() {
                break;
            }
            // A link that broke meanwhile is tried again.
            for l in &mut links {
                if l.as_ref().is_some_and(|(_, w)| w.send(MsgType::Ping, 0, 0, &[]).is_err()) {
                    *l = None;
                }
            }
            thread::sleep(left.min(plan.ping_every));
        }
    }
}

/// A worker's side: waits for the controller (node 0) to connect.
pub fn accept_controller(cfg: &Config, idx: usize, listener: TcpListener) -> Result<Connection, Error> {
    let me = &cfg.nodes[idx].name;
    let want = &cfg.nodes[0].name;
    let (stream, _) = listener.accept()?;
    let conn = Connection::accept(stream, me)?.with_max_chunk(cfg.net.max_chunk());
    if conn.peer_name() != want {
        return Err(Error::Config(format!(
            "{me}: expected the controller {want}, got {} at {}",
            conn.peer_name(),
            conn.peer_addr()
        )));
    }
    Ok(conn)
}

/// Retries while the worker is not listening yet.
fn connect_with_retry(addr: &str, me: &str, timeout: Duration) -> Result<Connection, Error> {
    let deadline = Instant::now() + timeout;
    loop {
        match Connection::connect(addr, me, Duration::from_secs(1)) {
            Ok(c) => return Ok(c),
            Err(FrameError::Io(e))
                if matches!(e.kind(), ErrorKind::ConnectionRefused | ErrorKind::TimedOut)
                    && Instant::now() < deadline =>
            {
                thread::sleep(Duration::from_millis(100));
            }
            Err(e) => return Err(Error::Protocol(format!("connect to {addr}: {e}"))),
        }
    }
}

/// Whether a read failed because nothing came within the read timeout.
fn timed_out(e: &FrameError) -> bool {
    matches!(e, FrameError::Io(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut))
}

/// Accepts the controller on `listener` again and again, calling `serve` on each connection
/// until it ends: a controller that lost the link reconnects here.
fn accept_forever(cfg: &Config, idx: usize, listener: &TcpListener, mut serve: impl FnMut(Connection) -> Result<(), Error>) -> Result<(), Error> {
    loop {
        match accept_controller(cfg, idx, listener.try_clone()?) {
            Ok(link) => {
                info!(controller = link.peer_name(), "connected");
                match serve(link) {
                    Ok(()) => info!("controller link ended; waiting for the next controller"),
                    Err(e) => warn!(error = %e, "controller link failed; waiting for the next controller"),
                }
            }
            Err(e) => warn!(error = %e, "controller not accepted"),
        }
    }
}

/// Nodes 1..n: run the layer range on each hidden state from the controller (one position, or
/// a prompt chunk) and send the result back; the node holding the head sends back the sampled
/// tokens instead.
pub struct Worker<E: Engine> {
    engine: E,
    link: Option<Connection>,
    name: String,
    /// Longest wait for a frame from the controller before the link counts as dead.
    keepalive: Option<Duration>,
}

impl<E: Engine> Worker<E> {
    pub fn new(engine: E, link: Connection, name: &str) -> Self {
        Worker { engine, link: Some(link), name: name.to_string(), keepalive: None }
    }

    /// A worker without a link yet, for [`Worker::serve_forever`].
    pub fn idle(engine: E, name: &str) -> Self {
        Worker { engine, link: None, name: name.to_string(), keepalive: None }
    }

    /// Drops a link that is quiet for longer than `timeout`; `None` waits forever.
    pub fn with_keepalive(mut self, timeout: Option<Duration>) -> Self {
        self.keepalive = timeout;
        self
    }

    /// Serves the link it was made with until the controller closes it; returns the engine for
    /// inspection.
    pub fn run(mut self) -> Result<E, Error> {
        let link = self.link.take().ok_or_else(|| Error::Protocol("worker has no link".into()))?;
        self.serve(link)?;
        Ok(self.engine)
    }

    /// Serves one controller connection after another, each from a clean state; drops a link
    /// quiet for `health_timeout_ms`.
    pub fn serve_forever(&mut self, cfg: &Config, idx: usize, listener: &TcpListener) -> Result<(), Error> {
        self.keepalive = Some(Duration::from_millis(cfg.scheduler.health_timeout_ms.max(1)));
        accept_forever(cfg, idx, listener, |link| self.serve(link))
    }

    pub fn into_engine(self) -> E {
        self.engine
    }

    /// Serves one connection until it ends, then frees every request it held.
    pub fn serve(&mut self, link: Connection) -> Result<(), Error> {
        let (mut recv, send) = link.split()?;
        recv.set_read_timeout(self.keepalive)?;
        let mut held = HashSet::new();
        let result = self.serve_link(&mut recv, &send, &mut held);
        if !held.is_empty() {
            info!(freed = held.len(), "freeing what the controller left");
        }
        for id in held {
            self.engine.free(id);
        }
        send.close();
        result
    }

    fn serve_link(&mut self, link: &mut Connection, back: &Writer, held: &mut HashSet<u64>) -> Result<(), Error> {
        let dim = self.engine.hidden_dim();
        let mut hidden = Vec::new();
        let mut hidden_out = Vec::new();
        let mut tokens = Vec::new();
        let mut buf = Vec::new();
        let mut out = Vec::new();
        let mut stages = Vec::new();
        let mut picks = Vec::new();
        let mut experts = Vec::new();
        loop {
            let h = match link.recv_header() {
                Ok(h) => h,
                Err(FrameError::Io(e)) if e.kind() == ErrorKind::UnexpectedEof => return Ok(()),
                Err(e) if timed_out(&e) => {
                    let ms = self.keepalive.unwrap_or_default().as_millis() as u64;
                    warn!(timeout_ms = ms, "nothing from the controller; dropping the link");
                    return Ok(());
                }
                Err(e) => return Err(e.into()),
            };
            if !matches!(h.msg_type, MsgType::Hidden | MsgType::Experts) {
                link.recv_payload(&h, &mut buf)?;
            }
            match h.msg_type {
                MsgType::Experts => {
                    // Rows straight into `hidden`, outputs handed to the writer (NET-6).
                    let decoded = link.recv_experts(&h, dim, &mut picks, &mut hidden).map_err(Error::from);
                    let t0 = Instant::now();
                    let result = decoded.and_then(|layer| {
                        self.engine.run_experts(h.request_id, layer as u32, &hidden, &picks, &mut hidden_out)?;
                        Ok(layer)
                    });
                    crate::stats::STATS.compute.observe(t0.elapsed());
                    let us = t0.elapsed().as_micros().min(u32::MAX as u128) as u32;
                    match result {
                        Ok(layer) => {
                            trace!(state = h.request_id, pos = h.position, layer, picks = picks.len(), us, "experts");
                            back.send_partial(h.request_id, h.position, layer, us, &mut hidden_out)?;
                        }
                        Err(e) => {
                            warn!(state = h.request_id, pos = h.position, error = %e, "experts failed");
                            let msg = format!("{}: {e}", self.name);
                            back.send(MsgType::Error, h.request_id, h.position, msg.as_bytes())?;
                        }
                    }
                }
                // Control frames of the expert split come with request id 0; so does an error
                // answering one.
                MsgType::Own => {
                    let result = decode_own(&buf, &mut experts).map_err(Error::from).and_then(|()| {
                        let list: Vec<(u32, u32)> = experts.iter().map(|&(l, e)| (l as u32, e as u32)).collect();
                        self.engine.own_experts(&list)
                    });
                    match result {
                        Ok(()) => {
                            info!(experts = experts.len(), "holding experts");
                            back.send(MsgType::Own, 0, 0, &[])?;
                        }
                        Err(e) => self.send_control_error(back, "own", &e)?,
                    }
                }
                MsgType::Move => {
                    let result = decode_move(&buf).map_err(Error::from).and_then(|(layer, expert, weights)| {
                        self.engine.load_expert(layer as u32, expert as u32, weights)?;
                        Ok((layer, expert))
                    });
                    match result {
                        Ok((layer, expert)) => {
                            debug!(layer, expert, "expert moved in");
                            encode_move(layer, expert, &[], &mut out);
                            back.send(MsgType::Move, 0, 0, &out)?;
                        }
                        Err(e) => self.send_control_error(back, "move", &e)?,
                    }
                }
                MsgType::Release => match decode_move(&buf) {
                    Ok((layer, expert, _)) => {
                        debug!(layer, expert, "expert released");
                        self.engine.release_expert(layer as u32, expert as u32);
                    }
                    Err(e) => self.send_control_error(back, "release", &e.into())?,
                },
                MsgType::Hidden => {
                    let sample = if h.flags & FLAG_EVERY_TOKEN != 0 { Sample::Every } else { Sample::Last };
                    // Straight into `hidden`, and `hidden_out` handed to the writer (NET-6).
                    let decoded = link.recv_hidden(&h, dim, &mut hidden, &mut stages).map_err(Error::from);
                    held.insert(h.request_id);
                    let result = decoded.and_then(|rows| {
                        let t0 = Instant::now();
                        let input = StageIn::Hidden(&hidden);
                        let r = self.engine.step(h.request_id, h.position, input, sample, &mut hidden_out, &mut tokens);
                        // This node's compute time travels with the positions (CL-10).
                        crate::stats::STATS.compute.observe(t0.elapsed());
                        let us = t0.elapsed().as_micros().min(u32::MAX as u128) as u32;
                        stages.push(us);
                        debug!(state = h.request_id, pos = h.position, positions = rows, us, "step");
                        r
                    });
                    match result {
                        Ok(()) if self.engine.has_head() => {
                            encode_token(&tokens, &stages, &mut out);
                            back.send(MsgType::Token, h.request_id, h.position, &out)?
                        }
                        Ok(()) => back.send_hidden(h.flags, h.request_id, h.position, &stages, &mut hidden_out)?,
                        Err(e) => {
                            warn!(state = h.request_id, pos = h.position, error = %e, "step failed");
                            let msg = format!("{}: {e}", self.name);
                            back.send(MsgType::Error, h.request_id, h.position, msg.as_bytes())?
                        }
                    }
                }
                MsgType::Rewind => {
                    debug!(state = h.request_id, keep = h.position, "rewind");
                    if let Err(e) = self.engine.rewind(h.request_id, h.position) {
                        warn!(state = h.request_id, error = %e, "rewind failed");
                        let msg = format!("{}: {e}", self.name);
                        back.send(MsgType::Error, h.request_id, h.position, msg.as_bytes())?;
                    }
                    back.send(MsgType::Rewind, h.request_id, h.position, &[])?;
                }
                // An earlier worker's failure, passed on by the controller: send it back
                // unchanged so the scheduler learns which request failed.
                MsgType::Error => back.send(MsgType::Error, h.request_id, h.position, &buf)?,
                // Health ping from the controller: answer it (CL-8).
                MsgType::Ping => back.send(MsgType::Ping, h.request_id, h.position, &buf)?,
                MsgType::Drop => {
                    debug!(state = h.request_id, "free");
                    self.engine.free(h.request_id);
                    held.remove(&h.request_id);
                    back.send(MsgType::Drop, h.request_id, 0, &[])?;
                }
                other => return Err(Error::Protocol(format!("{}: unexpected {other:?} frame", self.name))),
            }
        }
    }

    /// Reports a failed `Own`, `Move` or `Release` to the controller.
    fn send_control_error(&self, back: &Writer, what: &str, e: &Error) -> Result<(), Error> {
        warn!(error = %e, "{what} failed");
        let msg = format!("{}: {what}: {e}", self.name);
        Ok(back.send(MsgType::Error, 0, 0, msg.as_bytes())?)
    }
}

/// What the request split's worker loop hears from the controller.
enum FromController {
    Start(Header, Vec<u8>),
    Drop(u64),
    Other(Header),
    /// The connection ended, and why.
    Closed(String),
}

/// A request a request-split worker runs.
struct Run {
    /// The prompt, then every token made.
    tokens: Vec<u32>,
    prompt_len: usize,
    max_new: usize,
    /// Positions the engine holds.
    next_pos: usize,
    made: usize,
}

/// Request split (`split = "requests"`): a worker running every layer and the head, serving
/// whole requests. It interleaves the requests it holds one step at a time, round-robin: a
/// prompt chunk of up to `prefill_chunk` positions, or one decode position. It sends each
/// token as it is made and stops a request itself, with a `Done`, at its token limit or the
/// end-of-sequence token; on `Drop` it frees the request and answers `Done` too. An engine
/// error fails only its request (an `Error` frame, no `Done`). A reader thread answers every
/// `Ping` at once with a `Pong` carrying the status report, even while a long step runs: a slow
/// step shows as a request without progress, a frozen process as a ping not answered.
pub struct RequestWorker<E: Engine> {
    engine: E,
    name: String,
    prefill_chunk: usize,
    cap: Option<u64>,
    /// Longest wait for a frame from the controller before the link counts as dead.
    keepalive: Option<Duration>,
    hidden: Vec<f32>,
    out: Vec<u32>,
    frame: Vec<u8>,
}

impl<E: Engine> RequestWorker<E> {
    /// `cap` is reported in the status; keeping to it is the engine's job.
    pub fn new(engine: E, name: &str, prefill_chunk: usize, cap: Option<u64>) -> Self {
        let prefill_chunk = prefill_chunk.max(1);
        RequestWorker {
            engine,
            name: name.into(),
            prefill_chunk,
            cap,
            keepalive: None,
            hidden: Vec::new(),
            out: Vec::new(),
            frame: Vec::new(),
        }
    }

    /// Drops a link that is quiet for longer than `timeout`; `None` waits forever.
    pub fn with_keepalive(mut self, timeout: Option<Duration>) -> Self {
        self.keepalive = timeout;
        self
    }

    pub fn into_engine(self) -> E {
        self.engine
    }

    /// Accepts the controller on `listener` again and again, serving each connection until it
    /// ends: a controller that lost the link reconnects here. Drops a link quiet for
    /// `health_timeout_ms`.
    pub fn serve_forever(&mut self, cfg: &Config, idx: usize, listener: &TcpListener) -> Result<(), Error> {
        self.keepalive = Some(Duration::from_millis(cfg.scheduler.health_timeout_ms.max(1)));
        accept_forever(cfg, idx, listener, |link| self.serve(link))
    }

    /// Serves one connection until it ends, then frees every request it held.
    pub fn serve(&mut self, link: Connection) -> Result<(), Error> {
        let (recv, send) = link.split()?;
        recv.set_read_timeout(self.keepalive)?;
        let status = Arc::new(Mutex::new(Status { cap: self.cap.unwrap_or(0), ..Default::default() }));
        let (tx, rx) = mpsc::channel();
        let reader = {
            let (send, status, span) = (send.clone(), status.clone(), tracing::Span::current());
            let keepalive = self.keepalive;
            thread::spawn(move || span.in_scope(|| read_controller(recv, send, status, tx, keepalive)))
        };
        let mut running: HashMap<u64, Run> = HashMap::new();
        let mut order: VecDeque<u64> = VecDeque::new();
        let why = 'serve: loop {
            // Wait while idle; between steps take whatever has come in.
            let first = if order.is_empty() {
                match rx.recv() {
                    Ok(m) => Some(m),
                    Err(_) => break 'serve "reader stopped".to_string(),
                }
            } else {
                None
            };
            for m in first.into_iter().chain(rx.try_iter()) {
                let sent = match m {
                    FromController::Closed(why) => break 'serve why,
                    FromController::Start(h, payload) => self.start(h.request_id, &payload, &mut running, &mut order, &send),
                    FromController::Drop(id) => match running.remove(&id) {
                        Some(r) => {
                            debug!(state = id, made = r.made, "dropped");
                            self.engine.free(id);
                            order.retain(|&o| o != id);
                            self.send_done(&send, id, r.made, DoneReason::Dropped)
                        }
                        // Already over: its `Done` or `Error` is on the way.
                        None => Ok(()),
                    },
                    FromController::Other(h) => {
                        let msg = format!("{}: unexpected {:?} frame on the request split", self.name, h.msg_type);
                        warn!("{msg}");
                        send.send(MsgType::Error, h.request_id, h.position, msg.as_bytes())
                    }
                };
                if let Err(e) = sent {
                    break 'serve format!("send: {e}");
                }
            }
            if let Some(id) = order.pop_front() {
                let r = running.get_mut(&id).expect("every queued request runs");
                match self.step(id, r, &send) {
                    Ok(true) => order.push_back(id),
                    Ok(false) => {
                        running.remove(&id);
                        self.engine.free(id);
                    }
                    Err(e) => break 'serve format!("send: {e}"),
                }
            }
            lock(&status).requests = order
                .iter()
                .map(|id| {
                    let r = &running[id];
                    let (position, made) = (r.next_pos as u32, r.made as u32);
                    RunStatus { request: *id, decoding: r.next_pos >= r.prompt_len, position, made }
                })
                .collect();
        };
        info!(reason = why, freed = running.len(), "controller link ended");
        for id in running.keys() {
            self.engine.free(*id);
        }
        // The reader stops once the controller closes its side too.
        send.close();
        let _ = reader.join();
        Ok(())
    }

    fn start(
        &mut self,
        id: u64,
        payload: &[u8],
        running: &mut HashMap<u64, Run>,
        order: &mut VecDeque<u64>,
        send: &Writer,
    ) -> std::io::Result<()> {
        let mut prompt = Vec::new();
        let max_new = match decode_start(payload, &mut prompt) {
            Err(e) => Err(format!("bad start: {e}")),
            Ok(_) if running.contains_key(&id) => Err(format!("request {id} is already running")),
            Ok(_) if !self.engine.has_head() => Err("this worker's engine has no head".into()),
            Ok(n) => Ok(n as usize),
        };
        let max_new = match max_new {
            Ok(n) => n,
            Err(why) => {
                warn!(state = id, why, "start refused");
                let msg = format!("{}: {why}", self.name);
                return send.send(MsgType::Error, id, 0, msg.as_bytes());
            }
        };
        debug!(state = id, prompt = prompt.len(), max_new, "start");
        if max_new == 0 {
            return self.send_done(send, id, 0, DoneReason::Length);
        }
        let prompt_len = prompt.len();
        running.insert(id, Run { tokens: prompt, prompt_len, max_new, next_pos: 0, made: 0 });
        order.push_back(id);
        Ok(())
    }

    /// Runs one step of `r`; returns whether the request goes on.
    fn step(&mut self, id: u64, r: &mut Run, send: &Writer) -> std::io::Result<bool> {
        let start = r.next_pos;
        let end = if start < r.prompt_len { r.prompt_len.min(start + self.prefill_chunk) } else { start + 1 };
        let t0 = Instant::now();
        let input = StageIn::Tokens(&r.tokens[start..end]);
        let result = self.engine.step(id, start as u32, input, Sample::Last, &mut self.hidden, &mut self.out);
        crate::stats::STATS.compute.observe(t0.elapsed());
        let us = t0.elapsed().as_micros().min(u32::MAX as u128) as u32;
        trace!(state = id, pos = start, positions = end - start, us, "step");
        let token = match result {
            // A prompt chunk before the last: nothing to send.
            Ok(()) if end < r.prompt_len => {
                r.next_pos = end;
                return Ok(true);
            }
            Ok(()) => self.out.first().copied().ok_or_else(|| Error::Engine("the engine sampled no token".into())),
            Err(e) => Err(e),
        };
        let token = match token {
            Ok(t) => t,
            Err(e) => {
                warn!(state = id, pos = start, error = %e, "request failed");
                let msg = format!("{}: {e}", self.name);
                send.send(MsgType::Error, id, start as u32, msg.as_bytes())?;
                return Ok(false);
            }
        };
        r.next_pos = end;
        if Some(token) == self.engine.eos() {
            self.send_done(send, id, r.made, DoneReason::Eos)?;
            return Ok(false);
        }
        r.tokens.push(token);
        r.made += 1;
        // The token's own position in the sequence.
        encode_token(&[token], &[us], &mut self.frame);
        send.send(MsgType::Token, id, (r.tokens.len() - 1) as u32, &self.frame)?;
        if r.made == r.max_new {
            self.send_done(send, id, r.made, DoneReason::Length)?;
            return Ok(false);
        }
        Ok(true)
    }

    fn send_done(&mut self, send: &Writer, id: u64, made: usize, why: DoneReason) -> std::io::Result<()> {
        encode_done(why, &mut self.frame);
        send.send(MsgType::Done, id, made as u32, &self.frame)
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// The request-split worker's reader: answers pings with the status report and passes
/// everything else on to the worker loop.
fn read_controller(
    mut recv: Connection,
    send: Writer,
    status: Arc<Mutex<Status>>,
    tx: mpsc::Sender<FromController>,
    keepalive: Option<Duration>,
) {
    let mut buf = Vec::new();
    let mut out = Vec::new();
    loop {
        let h = match recv.recv_frame(&mut buf) {
            Ok(h) => h,
            Err(FrameError::Io(e)) if e.kind() == ErrorKind::UnexpectedEof => {
                let _ = tx.send(FromController::Closed("the controller closed the connection".into()));
                return;
            }
            Err(e) if timed_out(&e) => {
                let ms = keepalive.unwrap_or_default().as_millis();
                let _ = tx.send(FromController::Closed(format!("nothing from the controller in {ms} ms")));
                return;
            }
            Err(e) => {
                let _ = tx.send(FromController::Closed(e.to_string()));
                return;
            }
        };
        let msg = match h.msg_type {
            MsgType::Ping => {
                let mut s = lock(&status).clone();
                s.footprint = crate::memory::footprint()
                    .unwrap_or_else(|| crate::memory::LAST_FOOTPRINT.load(std::sync::atomic::Ordering::Relaxed));
                encode_status(&s, &mut out);
                // A broken link also fails the worker loop's next send.
                let _ = send.send(MsgType::Pong, h.request_id, 0, &out);
                continue;
            }
            MsgType::Start => FromController::Start(h, std::mem::take(&mut buf)),
            MsgType::Drop => FromController::Drop(h.request_id),
            _ => FromController::Other(h),
        };
        if tx.send(msg).is_err() {
            return;
        }
    }
}
