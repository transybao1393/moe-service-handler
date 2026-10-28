//! The service's own memory (NF-2): network buffers, requests, HTTP and metrics stay inside
//! `[service] mem_allowance_mib`, counted apart from streammoe's cap, so a node's whole
//! footprint stays within cap + allowance.
//!
//! Before the engine opens, a node measures its footprint (binary, runtime, tokenizer) and adds
//! the most its buffers can grow to ([`growth`]). Over the allowance, it refuses to start. While
//! it serves, [`watch`] samples the footprint and warns when it passes cap + allowance.
//! The footprint is the kernel's physical footprint (what Activity Monitor shows as Memory):
//! memory the process wrote, not clean pages of mapped files.

use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::Duration;

use tracing::{info, warn};

use crate::config::{Config, Split};

/// Largest HTTP request body accepted.
pub const HTTP_BODY_LIMIT: usize = 1 << 20;

const MIB: f64 = (1u64 << 20) as f64;

/// The process's physical footprint in bytes, or `None` where it cannot be read.
pub fn footprint() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        let mut info = std::mem::MaybeUninit::<libc::rusage_info_v2>::zeroed();
        // SAFETY: `info` is a valid buffer for the V2 flavour, which the kernel fills.
        let rc = unsafe {
            libc::proc_pid_rusage(std::process::id() as i32, libc::RUSAGE_INFO_V2, info.as_mut_ptr().cast())
        };
        // SAFETY: filled on success.
        (rc == 0).then(|| unsafe { info.assume_init() }.ri_phys_footprint)
    }
    #[cfg(not(target_os = "macos"))]
    None
}

/// The last footprint [`watch`] read, for metrics; 0 before the first.
pub static LAST_FOOTPRINT: AtomicU64 = AtomicU64::new(0);

/// Named byte counts.
#[derive(Debug, Default)]
pub struct Budget {
    pub items: Vec<(&'static str, u64)>,
}

impl Budget {
    pub fn total(&self) -> u64 {
        self.items.iter().map(|(_, b)| b).sum()
    }

    pub fn describe(&self) -> String {
        let parts: Vec<String> = self.items.iter().map(|(n, b)| format!("{n} {:.1}", *b as f64 / MIB)).collect();
        format!("{:.1} MiB ({})", self.total() as f64 / MIB, parts.join(", "))
    }
}

/// The most node `idx`'s buffers can grow to beyond what it holds at start-up.
pub fn growth(cfg: &Config, idx: usize, serves_http: bool) -> Budget {
    if cfg.split == Split::Requests {
        return growth_requests(cfg, idx, serves_http);
    }
    let s = &cfg.scheduler;
    let row = cfg.hidden_dim as u64 * 4;
    // A step brings a prompt chunk, or a token and its drafts.
    let positions = s.prefill_chunk.max(1).max(1 + s.max_draft) as u64;
    let mut b = Budget::default();
    let workers = cfg.nodes.len() as u64 - 1;
    // A read and a write buffer per connection: one to each worker on the controller, one to
    // the controller on a worker.
    let links = if idx == 0 { workers } else { 1 };
    if workers > 0 {
        b.items.push(("connections", links * 2 * msh_net::conn::BUFFER_BYTES as u64));
    }
    // Expert split: each layer's picks come back as one output row per pick. A `Move` carries
    // one expert's weights, counted with the engine's.
    let picks = cfg.experts.top_k as u64;
    let experts = cfg.split == Split::Experts;
    if idx == 0 && experts {
        // Per request in flight: its rows, the rows the experts take, one output row per pick,
        // and the workers' partials on their way in (one row per pick in all).
        let per_request = (2 + 2 * picks) * positions * row;
        b.items.push(("frames", s.max_in_flight.max(1) as u64 * per_request));
    } else if idx == 0 {
        // The step's output, sent straight from it, and each worker's result passed on.
        b.items.push(("frames", (1 + workers) * positions * row));
    }
    if idx == 0 {
        // Every admitted, queued or parked request holds up to n_ctx token ids.
        let held = (s.max_in_flight.max(1) + s.max_queue + s.prefix_cache) as u64;
        b.items.push(("requests", held * cfg.engine.n_ctx as u64 * 4));
        if serves_http {
            b.items.push(("http bodies", http_max_open(cfg) as u64 * HTTP_BODY_LIMIT as u64));
        }
        if cfg.metrics.is_some() {
            b.items.push(("metrics", 64 << 10));
        }
    } else if experts {
        // The rows read straight off the wire, and an output row for each pick, all of them
        // at worst.
        b.items.push(("frames", (1 + picks) * positions * row));
    } else {
        // The hidden state read straight off the wire, and the step's output sent back.
        b.items.push(("frames", 2 * positions * row));
    }
    b
}

/// [`growth`] on the request split. The controller holds token ids, never hidden states: every
/// request in flight on any worker or queued keeps its prompt (for a retry) up to n_ctx ids,
/// and one `Start` payload is built at a time. A worker holds its requests' token ids, the
/// `Start` read off the wire, and the engine's output for a prompt chunk.
fn growth_requests(cfg: &Config, idx: usize, serves_http: bool) -> Budget {
    let s = &cfg.scheduler;
    let workers = cfg.nodes.len() as u64 - 1;
    let ids = cfg.engine.n_ctx as u64 * 4;
    let mut b = Budget::default();
    let links = if idx == 0 { workers } else { 1 };
    b.items.push(("connections", links * 2 * msh_net::conn::BUFFER_BYTES as u64));
    b.items.push(("frames", ids));
    if idx == 0 {
        let held = workers * s.max_in_flight.max(1) as u64 + s.max_queue as u64;
        b.items.push(("requests", held * ids));
        if serves_http {
            b.items.push(("http bodies", http_max_open(cfg) as u64 * HTTP_BODY_LIMIT as u64));
        }
    } else {
        b.items.push(("requests", s.max_in_flight.max(1) as u64 * ids));
        b.items.push(("hidden", s.prefill_chunk.max(1) as u64 * cfg.hidden_dim as u64 * 4));
    }
    b
}

/// HTTP requests open at once: as many as the scheduler can hold. More get 429 before their
/// body is read.
pub fn http_max_open(cfg: &Config) -> usize {
    let s = &cfg.scheduler;
    match cfg.split {
        // `max_in_flight` is per worker there.
        Split::Requests => (cfg.nodes.len() - 1) * s.max_in_flight.max(1) + s.max_queue,
        _ => s.max_in_flight.max(1) + s.max_queue,
    }
}

/// Checks that the footprint now plus `growth` fits the allowance. Call before the engine opens.
pub fn check_start(cfg: &Config, growth: &Budget) -> Result<(), String> {
    let allowance = cfg.service.mem_allowance();
    let Some(now) = footprint() else {
        info!(growth = growth.describe(), "service memory: footprint unavailable on this OS");
        return Ok(());
    };
    let need = now + growth.total();
    info!(
        start_mib = format_args!("{:.1}", now as f64 / MIB),
        growth = growth.describe(),
        allowance_mib = format_args!("{:.0}", allowance as f64 / MIB),
        "service memory"
    );
    if need > allowance {
        return Err(format!(
            "service memory: {:.1} MiB at start plus up to {} exceeds [service] mem_allowance_mib = {:.0}",
            now as f64 / MIB,
            growth.describe(),
            allowance as f64 / MIB
        ));
    }
    Ok(())
}

/// Samples the footprint every `every` on a background thread, keeping [`LAST_FOOTPRINT`]
/// current and warning when it first passes `limit` (cap + allowance) and when it is back.
pub fn watch(limit: Option<u64>, every: Duration) {
    if footprint().is_none() {
        return;
    }
    let span = tracing::Span::current();
    thread::spawn(move || {
        let _span = span.entered();
        let mut over = false;
        loop {
            let now = footprint().unwrap_or(0);
            LAST_FOOTPRINT.store(now, Ordering::Relaxed);
            if let Some(limit) = limit {
                let mib = |b: u64| format_args!("{:.0}", b as f64 / MIB).to_string();
                if now > limit && !over {
                    warn!(footprint_mib = mib(now), limit_mib = mib(limit), "footprint above cap + allowance");
                } else if now <= limit && over {
                    info!(footprint_mib = mib(now), limit_mib = mib(limit), "footprint back within cap + allowance");
                }
                over = now > limit;
            }
            thread::sleep(every);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_os = "macos")]
    fn footprint_counts_written_memory() {
        let before = footprint().unwrap();
        let block = vec![1u8; 64 << 20];
        let after = footprint().unwrap();
        assert!(after >= before + (60 << 20), "{before} -> {after}");
        drop(block);
    }

    #[test]
    fn growth_follows_the_config() {
        let cfg = Config::parse(
            "hidden_dim = 2048\nnum_layers = 40\n[scheduler]\nprefill_chunk = 64\nmax_in_flight = 3\nmax_queue = 5\n\
             [[nodes]]\nname = \"a\"\naddr = \"127.0.0.1:1\"\nlayers = [0, 20]\n\
             [[nodes]]\nname = \"b\"\naddr = \"127.0.0.1:2\"\nlayers = [20, 40]\n",
        )
        .unwrap();
        let coord = growth(&cfg, 0, true);
        let item = |b: &Budget, n: &str| b.items.iter().find(|(k, _)| *k == n).map(|(_, v)| *v);
        // Its own output and the one worker's result passed on.
        assert_eq!(item(&coord, "frames"), Some(2 * 64 * 2048 * 4));
        assert_eq!(item(&coord, "requests"), Some(8 * 2048 * 4));
        assert_eq!(item(&coord, "http bodies"), Some(8 << 20));
        assert_eq!(item(&coord, "connections"), Some(512 << 10));
        let worker = growth(&cfg, 1, false);
        assert_eq!(item(&worker, "connections"), Some(512 << 10));
        assert_eq!(item(&worker, "frames"), Some(2 * 64 * 2048 * 4));
        assert_eq!(item(&worker, "requests"), None);
    }

    #[test]
    fn growth_on_the_request_split() {
        let cfg = Config::parse(
            "hidden_dim = 2048\nnum_layers = 40\nsplit = \"requests\"\n\
             [scheduler]\nprefill_chunk = 64\nmax_in_flight = 2\nmax_queue = 5\n\
             [engine]\nn_ctx = 4096\n\
             [[nodes]]\nname = \"m4\"\naddr = \"127.0.0.1:1\"\n\
             [[nodes]]\nname = \"m1\"\naddr = \"127.0.0.1:2\"\n\
             [[nodes]]\nname = \"m4b\"\naddr = \"127.0.0.1:3\"\n",
        )
        .unwrap();
        let item = |b: &Budget, n: &str| b.items.iter().find(|(k, _)| *k == n).map(|(_, v)| *v);
        let ids = 4096 * 4;
        let coord = growth(&cfg, 0, true);
        assert_eq!(item(&coord, "connections"), Some(2 * (512 << 10)));
        // Two per worker in flight and five queued, each up to n_ctx ids; no hidden states.
        assert_eq!(item(&coord, "requests"), Some(9 * ids));
        assert_eq!(item(&coord, "frames"), Some(ids));
        assert_eq!(item(&coord, "hidden"), None);
        assert_eq!(item(&coord, "http bodies"), Some(9 << 20));
        let worker = growth(&cfg, 1, false);
        assert_eq!(item(&worker, "connections"), Some(512 << 10));
        assert_eq!(item(&worker, "requests"), Some(2 * ids));
        assert_eq!(item(&worker, "hidden"), Some(64 * 2048 * 4));
        // Well inside 256 MiB on both.
        assert!(coord.total() < 16 << 20 && worker.total() < 4 << 20);
    }

    #[test]
    fn growth_on_the_expert_split() {
        let cfg = Config::parse(
            "hidden_dim = 2048\nnum_layers = 40\nsplit = \"experts\"\n[experts]\ntop_k = 8\n\
             [scheduler]\nprefill_chunk = 64\nmax_in_flight = 3\nmax_queue = 5\n\
             [[nodes]]\nname = \"m4\"\naddr = \"127.0.0.1:1\"\n\
             [[nodes]]\nname = \"m1\"\naddr = \"127.0.0.1:2\"\n\
             [[nodes]]\nname = \"m4b\"\naddr = \"127.0.0.1:3\"\n",
        )
        .unwrap();
        let item = |b: &Budget, n: &str| b.items.iter().find(|(k, _)| *k == n).map(|(_, v)| *v);
        let row = 2048 * 4;
        let coord = growth(&cfg, 0, false);
        // A link to each worker; per request in flight 2 + 2 × 8 rows per position.
        assert_eq!(item(&coord, "connections"), Some(2 * (512 << 10)));
        assert_eq!(item(&coord, "frames"), Some(3 * 18 * 64 * row));
        assert_eq!(item(&coord, "requests"), Some(8 * 2048 * 4));
        for w in [1, 2] {
            let worker = growth(&cfg, w, false);
            assert_eq!(item(&worker, "connections"), Some(512 << 10));
            assert_eq!(item(&worker, "frames"), Some(9 * 64 * row));
        }
        // 27 MiB of frames on the controller, 4.5 MiB on a worker: well inside 256 MiB.
        assert!(coord.total() < 64 << 20 && growth(&cfg, 1, false).total() < 8 << 20);
    }
}
