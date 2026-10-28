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

use std::io::ErrorKind;
use std::net::TcpListener;
use std::thread;
use std::time::{Duration, Instant};

use msh_net::{decode_move, decode_own, encode_move, encode_token, Connection, FrameError, MsgType, FLAG_EVERY_TOKEN};

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
