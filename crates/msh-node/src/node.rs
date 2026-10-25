//! Ring topology and the worker loop. The coordinator side lives in `scheduler`.
//!
//! Nodes form a ring in config order: node i sends hidden states to node i+1, and the last
//! node sends sampled token ids back to node 0, the coordinator. `Drop` travels the whole ring,
//! so when it returns to the coordinator every node has freed the request (CL-9). An engine
//! error on a worker travels on to the coordinator as an `Error` frame; the ring stays up.

use std::io::ErrorKind;
use std::net::TcpListener;
use std::thread;
use std::time::{Duration, Instant};

use msh_net::{bytes_to_f32s, f32s_to_bytes, Connection, FrameError, MsgType};

use crate::config::Config;
use crate::engine::Engine;
use crate::Error;

pub struct Ring {
    /// Receives from the previous node.
    pub prev: Connection,
    /// Sends to the next node.
    pub next: Connection,
}

/// Joins the ring as node `idx`. Returns `None` for a one-node cluster (mode A).
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

/// Nodes 1..n: run the layer range on each incoming hidden state and pass it on; the last node
/// applies the head and sends the token back to the coordinator.
pub struct Worker<E: Engine> {
    engine: E,
    ring: Ring,
    name: String,
    head: bool,
}

impl<E: Engine> Worker<E> {
    pub fn new(engine: E, ring: Ring, name: &str, head: bool) -> Self {
        Worker { engine, ring, name: name.to_string(), head }
    }

    /// Serves until the previous node closes the connection; returns the engine for inspection.
    pub fn run(mut self) -> Result<E, Error> {
        let mut hidden = vec![0.0; self.engine.hidden_dim()];
        let mut buf = Vec::new();
        let mut out = Vec::new();
        loop {
            let h = match self.ring.prev.recv_frame(&mut buf) {
                Ok(h) => h,
                Err(FrameError::Io(e)) if e.kind() == ErrorKind::UnexpectedEof => return Ok(self.engine),
                Err(e) => return Err(e.into()),
            };
            match h.msg_type {
                MsgType::Hidden => {
                    let result = bytes_to_f32s(&buf, &mut hidden)
                        .map_err(Error::from)
                        .and_then(|()| self.engine.forward(h.request_id, h.position, &mut hidden));
                    let sent = match result {
                        Ok(()) if self.head => {
                            let token = self.engine.head(&hidden)?;
                            self.ring.next.send(MsgType::Token, h.request_id, h.position, &token.to_le_bytes())
                        }
                        Ok(()) => {
                            f32s_to_bytes(&hidden, &mut out);
                            self.ring.next.send(MsgType::Hidden, h.request_id, h.position, &out)
                        }
                        Err(e) => {
                            let msg = format!("{}: {e}", self.name);
                            self.ring.next.send(MsgType::Error, h.request_id, h.position, msg.as_bytes())
                        }
                    };
                    sent?;
                }
                // An upstream failure: pass it on unchanged so the coordinator learns which
                // request failed.
                MsgType::Error => self.ring.next.send(MsgType::Error, h.request_id, h.position, &buf)?,
                MsgType::Drop => {
                    self.engine.free(h.request_id);
                    self.ring.next.send(MsgType::Drop, h.request_id, 0, &[])?;
                }
                other => return Err(Error::Protocol(format!("{}: unexpected {other:?} frame", self.name))),
            }
        }
    }
}
