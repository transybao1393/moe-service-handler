//! Live counters for Prometheus (CL-12). A node with `prometheus = "ADDR"` in its `[[nodes]]`
//! entry serves them as text at `http://ADDR/metrics`; each line carries `node="NAME"`.
//!
//! Every node reports its compute time per step and its memory footprint. The coordinator also
//! reports requests by outcome, tokens, time to first token, round trip per decode step,
//! drafts, prefix reuse and the ring's health-ping time. The per-step CSV (CL-10) stays the
//! source for detailed analysis; these are for watching a running cluster.

use std::fmt::Write as _;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::thread;
use std::time::Duration;

use tracing::{info, warn};

pub struct Counter(AtomicU64);

impl Counter {
    const fn new() -> Self {
        Counter(AtomicU64::new(0))
    }

    pub fn add(&self, n: u64) {
        self.0.fetch_add(n, Relaxed);
    }

    pub fn get(&self) -> u64 {
        self.0.load(Relaxed)
    }
}

/// Upper bounds of the histogram buckets, in seconds.
const BOUNDS: [f64; 12] = [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 10.0];

pub struct Histogram {
    /// Observations at or below each bound (not cumulative; summed when rendered).
    buckets: [AtomicU64; BOUNDS.len()],
    count: AtomicU64,
    sum_us: AtomicU64,
}

impl Histogram {
    const fn new() -> Self {
        Histogram { buckets: [const { AtomicU64::new(0) }; BOUNDS.len()], count: AtomicU64::new(0), sum_us: AtomicU64::new(0) }
    }

    pub fn observe(&self, d: Duration) {
        let secs = d.as_secs_f64();
        if let Some(i) = BOUNDS.iter().position(|&b| secs <= b) {
            self.buckets[i].fetch_add(1, Relaxed);
        }
        self.count.fetch_add(1, Relaxed);
        self.sum_us.fetch_add(d.as_micros() as u64, Relaxed);
    }
}

pub struct Stats {
    pub requests_done: Counter,
    pub requests_failed: Counter,
    pub requests_cancelled: Counter,
    pub requests_rejected: Counter,
    pub requests_active: AtomicU64,
    pub requests_waiting: AtomicU64,
    pub tokens_generated: Counter,
    pub prompt_positions: Counter,
    pub cached_positions: Counter,
    pub drafted: Counter,
    pub accepted: Counter,
    pub time_to_first_token: Histogram,
    /// From the start of a decode step on the coordinator to its tokens' return.
    pub decode_step: Histogram,
    /// This node's own compute per step.
    pub compute: Histogram,
    /// Last health ping's trip round the ring, in microseconds.
    pub ring_rtt_us: AtomicU64,
    /// Bytes: cap + allowance, or 0 without a cap.
    pub footprint_limit: AtomicU64,
}

pub static STATS: Stats = Stats {
    requests_done: Counter::new(),
    requests_failed: Counter::new(),
    requests_cancelled: Counter::new(),
    requests_rejected: Counter::new(),
    requests_active: AtomicU64::new(0),
    requests_waiting: AtomicU64::new(0),
    tokens_generated: Counter::new(),
    prompt_positions: Counter::new(),
    cached_positions: Counter::new(),
    drafted: Counter::new(),
    accepted: Counter::new(),
    time_to_first_token: Histogram::new(),
    decode_step: Histogram::new(),
    compute: Histogram::new(),
    ring_rtt_us: AtomicU64::new(0),
    footprint_limit: AtomicU64::new(0),
};

/// The Prometheus text format for this node.
pub fn render(node: &str, coordinator: bool) -> String {
    let s = &STATS;
    let mut out = String::new();
    let l = format!("node=\"{node}\"");
    let mut metric = |name: &str, kind: &str, help: &str, values: &[(&str, f64)]| {
        let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} {kind}");
        for (labels, v) in values {
            let sep = if labels.is_empty() { "" } else { "," };
            let _ = writeln!(out, "{name}{{{l}{sep}{labels}}} {v}");
        }
    };
    let footprint = crate::memory::LAST_FOOTPRINT.load(Relaxed) as f64;
    metric("msh_footprint_bytes", "gauge", "Physical memory footprint of the node process.", &[("", footprint)]);
    let limit = s.footprint_limit.load(Relaxed);
    if limit > 0 {
        metric("msh_footprint_limit_bytes", "gauge", "Engine cap plus service allowance.", &[("", limit as f64)]);
    }
    if coordinator {
        let outcomes = [
            ("outcome=\"done\"", s.requests_done.get() as f64),
            ("outcome=\"failed\"", s.requests_failed.get() as f64),
            ("outcome=\"cancelled\"", s.requests_cancelled.get() as f64),
            ("outcome=\"rejected\"", s.requests_rejected.get() as f64),
        ];
        metric("msh_requests_total", "counter", "Requests by how they ended.", &outcomes);
        let states = [
            ("state=\"active\"", s.requests_active.load(Relaxed) as f64),
            ("state=\"waiting\"", s.requests_waiting.load(Relaxed) as f64),
        ];
        metric("msh_requests", "gauge", "Requests in the ring and waiting for a slot.", &states);
        for (name, help, c) in [
            ("msh_tokens_generated_total", "Tokens sent to clients.", &s.tokens_generated),
            ("msh_prompt_positions_total", "Prompt positions prefilled.", &s.prompt_positions),
            ("msh_cached_positions_total", "Prompt positions reused from a parked request.", &s.cached_positions),
            ("msh_draft_tokens_total", "Draft tokens sent for verification.", &s.drafted),
            ("msh_draft_tokens_accepted_total", "Draft tokens the head confirmed.", &s.accepted),
        ] {
            metric(name, "counter", help, &[("", c.get() as f64)]);
        }
        let rtt = s.ring_rtt_us.load(Relaxed) as f64 / 1e6;
        metric("msh_ring_rtt_seconds", "gauge", "Last health ping's trip round the ring.", &[("", rtt)]);
        histogram(&mut out, &l, "msh_time_to_first_token_seconds", "From submission to the first token.", &s.time_to_first_token);
        histogram(&mut out, &l, "msh_decode_step_seconds", "Decode step round trip through the ring.", &s.decode_step);
    }
    histogram(&mut out, &l, "msh_compute_seconds", "This node's compute per step.", &s.compute);
    out
}

fn histogram(out: &mut String, l: &str, name: &str, help: &str, h: &Histogram) {
    let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} histogram");
    let mut cumulative = 0;
    for (b, n) in BOUNDS.iter().zip(&h.buckets) {
        cumulative += n.load(Relaxed);
        let _ = writeln!(out, "{name}_bucket{{{l},le=\"{b}\"}} {cumulative}");
    }
    let count = h.count.load(Relaxed);
    let _ = writeln!(out, "{name}_bucket{{{l},le=\"+Inf\"}} {count}");
    let _ = writeln!(out, "{name}_sum{{{l}}} {}", h.sum_us.load(Relaxed) as f64 / 1e6);
    let _ = writeln!(out, "{name}_count{{{l}}} {count}");
}

/// Answers `GET /metrics` on `listener` from a background thread; anything else gets 404.
pub fn serve(listener: TcpListener, node: String, coordinator: bool) {
    let span = tracing::Span::current();
    thread::spawn(move || {
        let _span = span.entered();
        if let Ok(addr) = listener.local_addr() {
            info!(url = format!("http://{addr}/metrics"), "prometheus");
        }
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
            // Read the whole head before answering: closing with unread bytes resets the
            // connection. Only the request line matters.
            let mut head = Vec::new();
            let mut buf = [0u8; 1024];
            while !head.windows(4).any(|w| w == b"\r\n\r\n") && head.len() < 16 * 1024 {
                match stream.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => head.extend_from_slice(&buf[..n]),
                }
            }
            let line = String::from_utf8_lossy(&head);
            let resp = if line.starts_with("GET /metrics ") || line.starts_with("GET /metrics?") {
                let body = render(&node, coordinator);
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
            } else {
                "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_string()
            };
            if let Err(e) = stream.write_all(resp.as_bytes()) {
                warn!(error = %e, "prometheus: write");
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_buckets_are_cumulative() {
        let h = Histogram::new();
        h.observe(Duration::from_micros(500));
        h.observe(Duration::from_millis(30));
        h.observe(Duration::from_secs(20));
        let mut out = String::new();
        histogram(&mut out, "node=\"t\"", "x_seconds", "help", &h);
        assert!(out.contains("x_seconds_bucket{node=\"t\",le=\"0.001\"} 1\n"), "{out}");
        assert!(out.contains("x_seconds_bucket{node=\"t\",le=\"0.05\"} 2\n"));
        assert!(out.contains("x_seconds_bucket{node=\"t\",le=\"10\"} 2\n"));
        assert!(out.contains("x_seconds_bucket{node=\"t\",le=\"+Inf\"} 3\n"));
        assert!(out.contains("x_seconds_count{node=\"t\"} 3\n"));
    }

    #[test]
    fn serves_metrics_over_http() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        serve(listener, "m4".into(), true);
        let get = |path: &str| {
            let mut s = std::net::TcpStream::connect(addr).unwrap();
            write!(s, "GET {path} HTTP/1.1\r\nHost: x\r\n\r\n").unwrap();
            let mut body = String::new();
            s.read_to_string(&mut body).unwrap();
            body
        };
        let body = get("/metrics");
        assert!(body.starts_with("HTTP/1.1 200"), "{body}");
        assert!(body.contains("msh_requests_total{node=\"m4\",outcome=\"done\"}"));
        assert!(body.contains("# TYPE msh_compute_seconds histogram"));
        assert!(get("/other").starts_with("HTTP/1.1 404"));
    }
}
