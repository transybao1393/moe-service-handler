//! Round-trip latency probe (NET-5, milestone M0).
//!
//! The client sends a `Ping` of N bytes and the server echoes it back as a `Pong` of the same
//! size, over one persistent `TCP_NODELAY` connection. An 8 KB round trip therefore costs about
//! two pipeline hops of one f32 hidden state (2048 × 4 bytes).

use std::net::TcpListener;
use std::thread;
use std::time::{Duration, Instant};

use crate::conn::Connection;
use crate::frame::{FrameError, MsgType};

/// Accepts connections forever, echoing every `Ping` as a `Pong`. One thread per connection.
pub fn serve(listener: TcpListener, name: &str) -> std::io::Result<()> {
    for stream in listener.incoming() {
        let stream = stream?;
        let name = name.to_string();
        thread::spawn(move || {
            let peer = stream.peer_addr().map(|a| a.to_string()).unwrap_or_default();
            match Connection::accept(stream, &name) {
                Ok(mut conn) => {
                    eprintln!("connected: {} ({peer})", conn.peer_name());
                    match echo(&mut conn) {
                        Ok(()) => eprintln!("closed: {} ({peer})", conn.peer_name()),
                        Err(e) => eprintln!("error from {peer}: {e}"),
                    }
                }
                Err(e) => eprintln!("handshake with {peer} failed: {e}"),
            }
        });
    }
    Ok(())
}

/// Echoes `Ping` frames until the peer closes the connection.
pub fn echo(conn: &mut Connection) -> Result<(), FrameError> {
    let mut buf = Vec::new();
    loop {
        let h = match conn.recv(&mut buf) {
            Ok(h) => h,
            Err(FrameError::Io(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(e),
        };
        match h.msg_type {
            MsgType::Ping => conn.send(MsgType::Pong, h.request_id, h.position, &buf)?,
            got => return Err(FrameError::Unexpected { expected: MsgType::Ping, got }),
        }
    }
}

/// Measures `iters` round trips of `size` bytes after `warmup` unrecorded ones.
pub fn measure(
    conn: &mut Connection,
    size: usize,
    warmup: usize,
    iters: usize,
) -> Result<Vec<Duration>, FrameError> {
    let payload: Vec<u8> = (0..size).map(|i| (i * 31 + 7) as u8).collect();
    let mut reply = Vec::with_capacity(size);
    let mut samples = Vec::with_capacity(iters);
    for i in 0..(warmup + iters) {
        let seq = i as u32;
        let t0 = Instant::now();
        conn.send(MsgType::Ping, 0, seq, &payload)?;
        let h = conn.recv(&mut reply)?;
        let rtt = t0.elapsed();
        if h.msg_type != MsgType::Pong {
            return Err(FrameError::Unexpected { expected: MsgType::Pong, got: h.msg_type });
        }
        if h.position != seq || reply.len() != size {
            return Err(FrameError::Remote(format!(
                "echo mismatch: seq {seq} got {}, size {size} got {}",
                h.position,
                reply.len()
            )));
        }
        if i >= warmup {
            samples.push(rtt);
        }
    }
    // Check the content once rather than on every iteration, to keep it out of the timing.
    if reply != payload {
        return Err(FrameError::Remote("echo payload corrupted".into()));
    }
    Ok(samples)
}

#[derive(Clone, Copy, Debug)]
pub struct Stats {
    pub n: usize,
    pub min: f64,
    pub p50: f64,
    pub p90: f64,
    pub p99: f64,
    pub max: f64,
    pub mean: f64,
    pub stddev: f64,
}

impl Stats {
    /// Summarises samples in microseconds. Percentiles use the nearest-rank method.
    pub fn from_samples(samples: &[Duration]) -> Stats {
        assert!(!samples.is_empty(), "no samples");
        let mut us: Vec<f64> = samples.iter().map(|d| d.as_secs_f64() * 1e6).collect();
        us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = us.len();
        let pct = |p: f64| us[((p / 100.0 * n as f64).ceil() as usize).clamp(1, n) - 1];
        let mean = us.iter().sum::<f64>() / n as f64;
        let var = us.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        Stats {
            n,
            min: us[0],
            p50: pct(50.0),
            p90: pct(90.0),
            p99: pct(99.0),
            max: us[n - 1],
            mean,
            stddev: var.sqrt(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_rank_percentiles() {
        let samples: Vec<Duration> = (1..=100).map(Duration::from_micros).collect();
        let s = Stats::from_samples(&samples);
        assert_eq!(s.n, 100);
        assert_eq!(s.min, 1.0);
        assert_eq!(s.p50, 50.0);
        assert_eq!(s.p90, 90.0);
        assert_eq!(s.p99, 99.0);
        assert_eq!(s.max, 100.0);
        assert!((s.mean - 50.5).abs() < 1e-9);
    }
}
