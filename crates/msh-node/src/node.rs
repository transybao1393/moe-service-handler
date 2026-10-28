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
//! requests, and a controller that may reconnect after losing the link.

use std::collections::{HashMap, VecDeque};
use std::io::ErrorKind;
use std::net::TcpListener;
use std::sync::{mpsc, Arc, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};

use msh_net::{
    decode_move, decode_own, decode_start, encode_done, encode_move, encode_status, encode_token, Connection,
    DoneReason, FrameError, Header, MsgType, RunStatus, Status, FLAG_EVERY_TOKEN,
};

use crate::config::Config;
use crate::engine::{Engine, Sample, StageIn};
use crate::Error;
use tracing::{debug, info, trace, warn};

/// The controller's connections to its workers, in config order, each split into a receiving
/// and a sending half.
pub struct Star {
    pub(crate) workers: Vec<(Connection, Connection)>,
}

/// The controller's side: connects to every worker in config order. Returns `None` for a
/// one-node cluster.
pub fn connect_workers(cfg: &Config, timeout: Duration) -> Result<Option<Star>, Error> {
    let me = &cfg.nodes[0].name;
    let mut workers = Vec::new();
    for node in &cfg.nodes[1..] {
        let conn = connect_with_retry(&node.addr, me, timeout)?;
        if conn.peer_name() != node.name {
            return Err(Error::Config(format!(
                "{me}: expected worker {} at {}, got {}",
                node.name,
                node.addr,
                conn.peer_name()
            )));
        }
        workers.push(conn.split()?);
    }
    Ok((!workers.is_empty()).then_some(Star { workers }))
}

/// A worker's side: waits for the controller (node 0) to connect.
pub fn accept_controller(cfg: &Config, idx: usize, listener: TcpListener) -> Result<Connection, Error> {
    let me = &cfg.nodes[idx].name;
    let want = &cfg.nodes[0].name;
    let (stream, _) = listener.accept()?;
    let conn = Connection::accept(stream, me)?;
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

/// Nodes 1..n: run the layer range on each hidden state from the controller (one position, or
/// a prompt chunk) and send the result back; the node holding the head sends back the sampled
/// tokens instead.
pub struct Worker<E: Engine> {
    engine: E,
    link: Connection,
    name: String,
}

impl<E: Engine> Worker<E> {
    pub fn new(engine: E, link: Connection, name: &str) -> Self {
        Worker { engine, link, name: name.to_string() }
    }

    /// Serves until the controller closes the connection; returns the engine for inspection.
    pub fn run(mut self) -> Result<E, Error> {
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
            let h = match self.link.recv_header() {
                Ok(h) => h,
                Err(FrameError::Io(e)) if e.kind() == ErrorKind::UnexpectedEof => return Ok(self.engine),
                Err(e) => return Err(e.into()),
            };
            if !matches!(h.msg_type, MsgType::Hidden | MsgType::Experts) {
                self.link.recv_payload(&h, &mut buf)?;
            }
            match h.msg_type {
                MsgType::Experts => {
                    // Rows straight into `hidden`, outputs sent straight from `hidden_out` (NET-6).
                    let decoded = self.link.recv_experts(&h, dim, &mut picks, &mut hidden).map_err(Error::from);
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
                            self.link.send_partial(h.request_id, h.position, layer, us, &hidden_out)?;
                        }
                        Err(e) => {
                            warn!(state = h.request_id, pos = h.position, error = %e, "experts failed");
                            let msg = format!("{}: {e}", self.name);
                            self.link.send(MsgType::Error, h.request_id, h.position, msg.as_bytes())?;
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
                            self.link.send(MsgType::Own, 0, 0, &[])?;
                        }
                        Err(e) => self.send_control_error("own", &e)?,
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
                            self.link.send(MsgType::Move, 0, 0, &out)?;
                        }
                        Err(e) => self.send_control_error("move", &e)?,
                    }
                }
                MsgType::Release => match decode_move(&buf) {
                    Ok((layer, expert, _)) => {
                        debug!(layer, expert, "expert released");
                        self.engine.release_expert(layer as u32, expert as u32);
                    }
                    Err(e) => self.send_control_error("release", &e.into())?,
                },
                MsgType::Hidden => {
                    let sample = if h.flags & FLAG_EVERY_TOKEN != 0 { Sample::Every } else { Sample::Last };
                    // Straight into `hidden`, and sent back straight from `hidden_out` (NET-6).
                    let decoded = self.link.recv_hidden(&h, dim, &mut hidden, &mut stages).map_err(Error::from);
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
                    let back = &mut self.link;
                    let sent = match result {
                        Ok(()) if self.engine.has_head() => {
                            encode_token(&tokens, &stages, &mut out);
                            back.send(MsgType::Token, h.request_id, h.position, &out)
                        }
                        Ok(()) => back.send_hidden(h.flags, h.request_id, h.position, &stages, &hidden_out),
                        Err(e) => {
                            warn!(state = h.request_id, pos = h.position, error = %e, "step failed");
                            let msg = format!("{}: {e}", self.name);
                            back.send(MsgType::Error, h.request_id, h.position, msg.as_bytes())
                        }
                    };
                    sent?;
                }
                MsgType::Rewind => {
                    debug!(state = h.request_id, keep = h.position, "rewind");
                    if let Err(e) = self.engine.rewind(h.request_id, h.position) {
                        warn!(state = h.request_id, error = %e, "rewind failed");
                        let msg = format!("{}: {e}", self.name);
                        self.link.send(MsgType::Error, h.request_id, h.position, msg.as_bytes())?;
                    }
                    self.link.send(MsgType::Rewind, h.request_id, h.position, &[])?;
                }
                // An earlier worker's failure, passed on by the controller: send it back
                // unchanged so the scheduler learns which request failed.
                MsgType::Error => self.link.send(MsgType::Error, h.request_id, h.position, &buf)?,
                // Health ping from the controller: answer it (CL-8).
                MsgType::Ping => self.link.send(MsgType::Ping, h.request_id, h.position, &buf)?,
                MsgType::Drop => {
                    debug!(state = h.request_id, "free");
                    self.engine.free(h.request_id);
                    self.link.send(MsgType::Drop, h.request_id, 0, &[])?;
                }
                other => return Err(Error::Protocol(format!("{}: unexpected {other:?} frame", self.name))),
            }
        }
    }

    /// Reports a failed `Own`, `Move` or `Release` to the controller.
    fn send_control_error(&mut self, what: &str, e: &Error) -> Result<(), Error> {
        warn!(error = %e, "{what} failed");
        let msg = format!("{}: {what}: {e}", self.name);
        Ok(self.link.send(MsgType::Error, 0, 0, msg.as_bytes())?)
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
    hidden: Vec<f32>,
    out: Vec<u32>,
    frame: Vec<u8>,
}

impl<E: Engine> RequestWorker<E> {
    /// `cap` is reported in the status; keeping to it is the engine's job.
    pub fn new(engine: E, name: &str, prefill_chunk: usize, cap: Option<u64>) -> Self {
        let prefill_chunk = prefill_chunk.max(1);
        RequestWorker { engine, name: name.into(), prefill_chunk, cap, hidden: Vec::new(), out: Vec::new(), frame: Vec::new() }
    }

    pub fn into_engine(self) -> E {
        self.engine
    }

    /// Accepts the controller on `listener` again and again, serving each connection until it
    /// ends: a controller that lost the link reconnects here.
    pub fn serve_forever(&mut self, cfg: &Config, idx: usize, listener: &TcpListener) -> Result<(), Error> {
        loop {
            match accept_controller(cfg, idx, listener.try_clone()?) {
                Ok(link) => {
                    info!(controller = link.peer_name(), "connected");
                    self.serve(link)?;
                }
                Err(e) => warn!(error = %e, "controller not accepted"),
            }
        }
    }

    /// Serves one connection until it ends, then frees every request it held.
    pub fn serve(&mut self, link: Connection) -> Result<(), Error> {
        let (recv, send) = link.split()?;
        let send = Arc::new(Mutex::new(send));
        let status = Arc::new(Mutex::new(Status { cap: self.cap.unwrap_or(0), ..Default::default() }));
        let (tx, rx) = mpsc::channel();
        let reader = {
            let (send, status, span) = (send.clone(), status.clone(), tracing::Span::current());
            thread::spawn(move || span.in_scope(|| read_controller(recv, send, status, tx)))
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
                        lock(&send).send(MsgType::Error, h.request_id, h.position, msg.as_bytes())
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
        lock(&send).close_send();
        let _ = reader.join();
        Ok(())
    }

    fn start(
        &mut self,
        id: u64,
        payload: &[u8],
        running: &mut HashMap<u64, Run>,
        order: &mut VecDeque<u64>,
        send: &Mutex<Connection>,
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
                return lock(send).send(MsgType::Error, id, 0, msg.as_bytes());
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
    fn step(&mut self, id: u64, r: &mut Run, send: &Mutex<Connection>) -> std::io::Result<bool> {
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
                lock(send).send(MsgType::Error, id, start as u32, msg.as_bytes())?;
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
        lock(send).send(MsgType::Token, id, (r.tokens.len() - 1) as u32, &self.frame)?;
        if r.made == r.max_new {
            self.send_done(send, id, r.made, DoneReason::Length)?;
            return Ok(false);
        }
        Ok(true)
    }

    fn send_done(&mut self, send: &Mutex<Connection>, id: u64, made: usize, why: DoneReason) -> std::io::Result<()> {
        encode_done(why, &mut self.frame);
        lock(send).send(MsgType::Done, id, made as u32, &self.frame)
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// The request-split worker's reader: answers pings with the status report and passes
/// everything else on to the worker loop.
fn read_controller(
    mut recv: Connection,
    send: Arc<Mutex<Connection>>,
    status: Arc<Mutex<Status>>,
    tx: mpsc::Sender<FromController>,
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
                let _ = lock(&send).send(MsgType::Pong, h.request_id, 0, &out);
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
