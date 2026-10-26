//! M0 network round-trip probe.
//!
//!   msh-probe serve ADDR:PORT
//!   msh-probe run   HOST:PORT [LINK_LABEL] [CSV_FILE]
//!
//! `run` pings each size over one persistent TCP_NODELAY connection, prints a table and,
//! when CSV_FILE is given, appends one row per size with run metadata.

use std::fs::OpenOptions;
use std::io::Write;
use std::net::TcpListener;
use std::process::ExitCode;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use msh_net::host::{hardware, hostname, GIT_HASH};
use msh_net::probe::{self, Stats};
use msh_net::{Connection, PROTOCOL_VERSION};

/// Token id, then 1 / 8 / 128 / 2048-token f32 hidden states (2048 × 4 bytes per token).
const SIZES: &[usize] = &[4, 8_192, 65_536, 1_048_576, 16_777_216];
const ITERS: usize = 1000;
const WARMUP: usize = 100;
/// Large sizes run fewer iterations so each size moves at most ~256 MB each way.
const BYTES_PER_SIZE: usize = 256 << 20;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let result = match args.as_slice() {
        ["serve", addr] => serve(addr),
        ["run", addr, rest @ ..] if rest.len() <= 2 => {
            run(addr, rest.first().copied().unwrap_or("unspecified"), rest.get(1).copied())
        }
        _ => Err("usage: msh-probe serve ADDR:PORT | msh-probe run HOST:PORT [LINK] [CSV]".into()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn serve(addr: &str) -> Result<(), String> {
    let listener = TcpListener::bind(addr).map_err(|e| format!("bind {addr}: {e}"))?;
    eprintln!("msh-probe v{PROTOCOL_VERSION} serving on {addr}");
    probe::serve(listener, &hostname()).map_err(|e| e.to_string())
}

fn run(addr: &str, link: &str, csv: Option<&str>) -> Result<(), String> {
    let mut conn = Connection::connect(addr, &hostname(), Duration::from_secs(5))
        .map_err(|e| format!("connect {addr}: {e}"))?;
    let meta = format!(
        "{},{},{},{},{},{link}",
        unix_secs(),
        GIT_HASH,
        hostname(),
        hardware(),
        conn.peer_name()
    );
    println!("{} -> {} ({}) over {link}", hostname(), conn.peer_name(), conn.peer_addr());
    println!("{:>10} {:>5} {:>9} {:>9} {:>9} {:>9} {:>9} {:>8}",
        "bytes", "n", "min µs", "p50 µs", "p99 µs", "max µs", "hop µs", "MB/s");

    let mut rows = Vec::new();
    for &size in SIZES {
        let n = ITERS.min(BYTES_PER_SIZE / size).max(20);
        let samples = probe::measure(&mut conn, size, WARMUP.min(n), n)
            .map_err(|e| format!("size {size}: {e}"))?;
        let s = Stats::from_samples(&samples);
        // One direction's throughput at the median: the payload crosses the link twice.
        let mbps = 2.0 * size as f64 / s.p50;
        println!("{size:>10} {:>5} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {mbps:>8.0}",
            s.n, s.min, s.p50, s.p99, s.max, s.p50 / 2.0);
        rows.push(format!("{meta},{size},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{mbps:.1}",
            s.n, s.min, s.p50, s.p99, s.max, s.mean, s.p50 / 2.0));
    }

    if let Some(path) = csv {
        let mut f = OpenOptions::new().create(true).append(true).open(path)
            .map_err(|e| format!("{path}: {e}"))?;
        let new = f.metadata().map(|m| m.len() == 0).unwrap_or(true);
        let header = "timestamp,git_hash,local,local_hw,peer,link,bytes,n,\
                      min_us,p50_us,p99_us,max_us,mean_us,hop_p50_us,mb_per_s";
        let body = rows.join("\n");
        let text = if new { format!("{header}\n{body}\n") } else { format!("{body}\n") };
        f.write_all(text.as_bytes()).map_err(|e| format!("{path}: {e}"))?;
        println!("appended to {path}");
    }
    Ok(())
}

fn unix_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}
