//! Facts about this build and machine, recorded alongside measurements.

use std::process::Command;

/// Git commit the binary was built from, with `-dirty` for uncommitted changes.
pub const GIT_HASH: &str = env!("MSH_GIT_HASH");

fn sh(cmd: &str, args: &[&str]) -> String {
    Command::new(cmd)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().replace(',', " "))
        .unwrap_or_else(|| "?".into())
}

pub fn hostname() -> String {
    sh("hostname", &["-s"])
}

/// e.g. "Apple M4 / 16 GB / macOS 15.1".
pub fn hardware() -> String {
    let mem_gb = sh("sysctl", &["-n", "hw.memsize"]).parse::<u64>().map(|b| b >> 30).unwrap_or(0);
    format!(
        "{} / {mem_gb} GB / macOS {}",
        sh("sysctl", &["-n", "machdep.cpu.brand_string"]),
        sh("sw_vers", &["-productVersion"])
    )
}
