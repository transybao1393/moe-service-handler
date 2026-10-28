//! Ring topology and the worker loop. The coordinator side lives in `scheduler`.
//!
//! Nodes form a ring in config order: node i sends hidden states to node i+1, and the last
//! node sends sampled token ids back to node 0, the coordinator. `Drop` travels the whole ring,
//! so when it returns to the coordinator every node has freed the request (CL-9); `Rewind`
//! (rejected draft tokens) travels the same way. An engine error on a worker travels on to the
//! coordinator as an `Error` frame; the ring stays up.

use std::io::ErrorKind;
use std::net::TcpListener;
use std::thread;
use std::time::{Duration, Instant};

use msh_net::{encode_token, Connection, FrameError, MsgType, FLAG_EVERY_TOKEN};

use crate::config::Config;
use crate::engine::{Engine, Sample, StageIn};
use crate::Error;
use tracing::{debug, warn};

pub struct Ring {
    /// Receives from the previous node.
    pub prev: Connection,
    /// Sends to the next node.
    pub next: Connection,
}

/// Joins the ring as node `idx`. Returns `None` for a one-node cluster (no ring).
///
/// Accepting from the previous node runs on its own thread while this one connects to the next
/// node: doing them in sequence deadlocks, as every node would wait on its neighbour's handshake.
pub fn join_ring(
    cfg: &Config,
    idx: usize,
    listener: TcpListener,
    timeout: Duration,
) -> Result<Option<Ring>, Error> {
    let n = cfg.nodes.len();
    if n == 1 {
        return Ok(None);
    }
    let me = cfg.nodes[idx].name.clone();
    let prev_name = cfg.nodes[(idx + n - 1) % n].name.clone();
    let next_cfg = &cfg.nodes[(idx + 1) % n];

    let acceptor = {
        let me = me.clone();
        thread::spawn(move || -> Result<Connection, Error> {
            let (stream, _) = listener.accept()?;
            Ok(Connection::accept(stream, &me)?)
        })
    };
    let next = connect_with_retry(&next_cfg.addr, &me, timeout)?;
    let prev = acceptor
        .join()
        .map_err(|_| Error::Protocol("accept thread panicked".into()))??;

    for (conn, want) in [(&prev, &prev_name), (&next, &next_cfg.name)] {
        if conn.peer_name() != want {
            return Err(Error::Config(format!(
                "{me}: expected neighbour {want}, got {} at {}",
                conn.peer_name(),
                conn.peer_addr()
            )));
        }
    }
    Ok(Some(Ring { prev, next }))
}

/// Retries while the next node is not listening yet.
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

/// Nodes 1..n: run the layer range on each incoming hidden state (one position, or a prompt
/// chunk) and pass it on; the node holding the head sends the sampled token back to the
/// coordinator instead.
pub struct Worker<E: Engine> {
    engine: E,
    ring: Ring,
    name: String,
}

impl<E: Engine> Worker<E> {
    pub fn new(engine: E, ring: Ring, name: &str) -> Self {
        Worker { engine, ring, name: name.to_string() }
    }

    /// Serves until the previous node closes the connection; returns the engine for inspection.
    pub fn run(mut self) -> Result<E, Error> {
        let dim = self.engine.hidden_dim();
        let mut hidden = Vec::new();
        let mut hidden_out = Vec::new();
        let mut tokens = Vec::new();
        let mut buf = Vec::new();
        let mut out = Vec::new();
        let mut stages = Vec::new();
        loop {
            let h = match self.ring.prev.recv_header() {
                Ok(h) => h,
                Err(FrameError::Io(e)) if e.kind() == ErrorKind::UnexpectedEof => return Ok(self.engine),
                Err(e) => return Err(e.into()),
            };
            if h.msg_type != MsgType::Hidden {
                self.ring.prev.recv_payload(&h, &mut buf)?;
            }
            match h.msg_type {
                MsgType::Hidden => {
                    let sample = if h.flags & FLAG_EVERY_TOKEN != 0 { Sample::Every } else { Sample::Last };
                    // Straight into `hidden`, and sent on straight from `hidden_out` (NET-6).
                    let decoded = self.ring.prev.recv_hidden(&h, dim, &mut hidden, &mut stages).map_err(Error::from);
                    let result = decoded.and_then(|rows| {
                        let t0 = Instant::now();
                        let input = StageIn::Hidden(&hidden);
                        let r = self.engine.step(h.request_id, h.position, input, sample, &mut hidden_out, &mut tokens);
                        // This node's compute time travels on with the positions (CL-10).
                        crate::stats::STATS.compute.observe(t0.elapsed());
                        let us = t0.elapsed().as_micros().min(u32::MAX as u128) as u32;
                        stages.push(us);
                        debug!(ring = h.request_id, pos = h.position, positions = rows, us, "step");
                        r
                    });
                    let next = &mut self.ring.next;
                    let sent = match result {
                        Ok(()) if self.engine.has_head() => {
                            encode_token(&tokens, &stages, &mut out);
                            next.send(MsgType::Token, h.request_id, h.position, &out)
                        }
                        Ok(()) => next.send_hidden(h.flags, h.request_id, h.position, &stages, &hidden_out),
                        Err(e) => {
                            warn!(ring = h.request_id, pos = h.position, error = %e, "step failed");
                            let msg = format!("{}: {e}", self.name);
                            next.send(MsgType::Error, h.request_id, h.position, msg.as_bytes())
                        }
                    };
                    sent?;
                }
                MsgType::Rewind => {
                    debug!(ring = h.request_id, keep = h.position, "rewind");
                    if let Err(e) = self.engine.rewind(h.request_id, h.position) {
                        warn!(ring = h.request_id, error = %e, "rewind failed");
                        let msg = format!("{}: {e}", self.name);
                        self.ring.next.send(MsgType::Error, h.request_id, h.position, msg.as_bytes())?;
                    }
                    self.ring.next.send(MsgType::Rewind, h.request_id, h.position, &[])?;
                }
                // An upstream failure: pass it on unchanged so the coordinator learns which
                // request failed.
                MsgType::Error => self.ring.next.send(MsgType::Error, h.request_id, h.position, &buf)?,
                // Health ping from the coordinator: pass it on (CL-8).
                MsgType::Ping => self.ring.next.send(MsgType::Ping, h.request_id, h.position, &buf)?,
                MsgType::Drop => {
                    debug!(ring = h.request_id, "free");
                    self.engine.free(h.request_id);
                    self.ring.next.send(MsgType::Drop, h.request_id, 0, &[])?;
                }
                other => return Err(Error::Protocol(format!("{}: unexpected {other:?} frame", self.name))),
            }
        }
    }
}
