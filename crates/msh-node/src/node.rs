//! Ring topology, coordinator and worker loops.
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

#[derive(Debug)]
pub struct Generation {
    pub tokens: Vec<u32>,
    /// Wall time of the step that produced each token.
    pub step_times: Vec<Duration>,
}

/// Node 0: embeds tokens, runs its layer range, and drives the ring one position at a time.
pub struct Coordinator<E: Engine> {
    engine: E,
    ring: Option<Ring>,
    hidden: Vec<f32>,
    buf: Vec<u8>,
}

impl<E: Engine> Coordinator<E> {
    pub fn new(engine: E, ring: Option<Ring>) -> Self {
        let hidden = vec![0.0; engine.hidden_dim()];
        Coordinator { engine, ring, hidden, buf: Vec::new() }
    }

    pub fn engine(&self) -> &E {
        &self.engine
    }

    /// Greedy generation of up to `max_new` tokens. The request's state is freed on every
    /// node afterwards, whether generation succeeded or not.
    pub fn generate(&mut self, req: u64, prompt: &[u32], max_new: usize) -> Result<Generation, Error> {
        if prompt.is_empty() {
            return Err(Error::Protocol("empty prompt".into()));
        }
        let result = self.run(req, prompt, max_new);
        let freed = self.free(req);
        let generation = result?;
        freed?;
        Ok(generation)
    }

    fn run(&mut self, req: u64, prompt: &[u32], max_new: usize) -> Result<Generation, Error> {
        let mut g = Generation { tokens: Vec::new(), step_times: Vec::new() };
        let mut input = prompt[0];
        let mut pos = 0;
        while g.tokens.len() < max_new {
            let t0 = Instant::now();
            let out = self.step(req, pos as u32, input)?;
            pos += 1;
            if pos < prompt.len() {
                input = prompt[pos];
            } else {
                g.tokens.push(out);
                g.step_times.push(t0.elapsed());
                input = out;
            }
        }
        Ok(g)
    }

    fn step(&mut self, req: u64, pos: u32, token: u32) -> Result<u32, Error> {
        self.engine.embed(token, &mut self.hidden)?;
        self.engine.forward(req, pos, &mut self.hidden)?;
        let Some(ring) = &mut self.ring else {
            return self.engine.head(&self.hidden);
        };
        f32s_to_bytes(&self.hidden, &mut self.buf);
        ring.next.send(MsgType::Hidden, req, pos, &self.buf)?;
        let h = ring.prev.recv(&mut self.buf)?;
        if h.msg_type != MsgType::Token || h.request_id != req || h.position != pos || self.buf.len() != 4 {
            return Err(Error::Protocol(format!(
                "expected token for request {req} position {pos}, got {:?} for {} at {}",
                h.msg_type, h.request_id, h.position
            )));
        }
        Ok(u32::from_le_bytes(self.buf[..4].try_into().unwrap()))
    }

    fn free(&mut self, req: u64) -> Result<(), Error> {
        self.engine.free(req);
        let Some(ring) = &mut self.ring else { return Ok(()) };
        ring.next.send(MsgType::Drop, req, 0, &[])?;
        // Skip any tokens still in flight from a failed step until our Drop comes back.
        loop {
            match ring.prev.recv(&mut self.buf) {
                Ok(h) if h.msg_type == MsgType::Drop && h.request_id == req => return Ok(()),
                Ok(_) | Err(FrameError::Remote(_)) => {}
                Err(e) => return Err(e.into()),
            }
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
