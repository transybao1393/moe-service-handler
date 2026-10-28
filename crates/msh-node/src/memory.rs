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

use crate::config::Config;

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
    let s = &cfg.scheduler;
    let row = cfg.hidden_dim as u64 * 4;
    // A step brings a prompt chunk, or a token and its drafts.
    let positions = s.prefill_chunk.max(1).max(1 + s.max_draft) as u64;
    let mut b = Budget::default();
    if cfg.nodes.len() > 1 {
        b.items.push(("connections", 2 * 2 * msh_net::conn::BUFFER_BYTES as u64));
    }
    if idx == 0 {
        // Step input and output, and the frame encoded from it.
        b.items.push(("frames", 2 * positions * row));
        // Every admitted, queued or parked request holds up to n_ctx token ids.
        let held = (s.max_in_flight.max(1) + s.max_queue + s.prefix_cache) as u64;
        b.items.push(("requests", held * cfg.engine.n_ctx as u64 * 4));
        if serves_http {
            b.items.push(("http bodies", http_max_open(cfg) as u64 * HTTP_BODY_LIMIT as u64));
        }
        if cfg.metrics.is_some() {
            b.items.push(("metrics", 64 << 10));
        }
    } else {
        // Received frame, its hidden state, the step's output, and the frame sent on.
        b.items.push(("frames", 4 * positions * row));
    }
    b
}

/// HTTP requests open at once: as many as the scheduler can hold. More get 429 before their
/// body is read.
pub fn http_max_open(cfg: &Config) -> usize {
    cfg.scheduler.max_in_flight.max(1) + cfg.scheduler.max_queue
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
        assert_eq!(item(&coord, "frames"), Some(2 * 64 * 2048 * 4));
        assert_eq!(item(&coord, "requests"), Some(8 * 2048 * 4));
        assert_eq!(item(&coord, "http bodies"), Some(8 << 20));
        assert_eq!(item(&coord, "connections"), Some(1 << 20));
        let worker = growth(&cfg, 1, false);
        assert_eq!(item(&worker, "frames"), Some(4 * 64 * 2048 * 4));
        assert_eq!(item(&worker, "requests"), None);
    }
}
