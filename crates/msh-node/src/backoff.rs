//! Waits between reconnect tries: exponential, from `[net] reconnect_min_ms` doubling to
//! `reconnect_max_ms`, each shortened by up to a quarter at random so workers lost together
//! are not all tried at the same moment.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::config::NetConfig;

pub struct Backoff {
    min: Duration,
    max: Duration,
    next: Duration,
    tries: u32,
    rng: u64,
}

impl Backoff {
    pub fn new(net: &NetConfig) -> Backoff {
        let min = Duration::from_millis(net.reconnect_min_ms);
        let seed = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64);
        let here = &min as *const Duration as u64;
        Backoff { min, max: Duration::from_millis(net.reconnect_max_ms), next: min, tries: 0, rng: (seed ^ here) | 1 }
    }

    /// Failed tries so far.
    pub fn tries(&self) -> u32 {
        self.tries
    }

    /// Counts a failed try and returns how long to wait before the next.
    pub fn failed(&mut self) -> Duration {
        self.tries += 1;
        // xorshift64
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        let jitter = (self.rng >> 11) as f64 / (1u64 << 53) as f64;
        let wait = self.next.mul_f64(1.0 - 0.25 * jitter);
        self.next = (self.next * 2).min(self.max);
        wait
    }

    pub fn reset(&mut self) {
        self.next = self.min;
        self.tries = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doubles_to_the_cap_with_jitter() {
        let net = NetConfig { reconnect_min_ms: 500, reconnect_max_ms: 10_000, ..Default::default() };
        let mut b = Backoff::new(&net);
        let ms: Vec<u128> = (0..8).map(|_| b.failed().as_millis()).collect();
        for (got, full) in ms.iter().zip([500, 1000, 2000, 4000, 8000, 10_000, 10_000, 10_000]) {
            assert!(*got <= full && *got >= full * 3 / 4, "{ms:?}");
        }
        assert_eq!(b.tries(), 8);
        b.reset();
        assert!(b.failed() <= Duration::from_millis(500));
    }
}
