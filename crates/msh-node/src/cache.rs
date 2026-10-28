//! The request split's answer cache, on the controller (`[cache]`): prompts it has answered,
//! with the answer and the worker that made it. It lives in the controller's RAM only, inside
//! the service allowance; nothing in it is shared with the workers, whose KV caches stay
//! their own.
//!
//! The key is everything that changes an answer: the prompt's tokens (a keyed 128-bit hash and
//! the length), `max_new`, and a scope hash of the model, its engine settings and the chat
//! template. Sampling is greedy, so the same key gets the same answer. The request split's
//! [`crate::requests::Board`] owns the cache: it looks a request up when it is submitted, before
//! the queue, and keeps its answer when a worker is done with it. A lookup tries the exact
//! key first. In `similar` mode it then scans every entry with the same scope and `max_new` for
//! the nearest prompt vector by estimated cosine and answers from it at `similar_threshold` or
//! above.
//!
//! The controller runs no model, so a prompt vector is made from the tokens alone
//! ([`prompt_vector`]): token 1-, 2- and 3-grams hashed with a sign into [`VECTOR_DIM`]
//! coordinates, then normalised. Prompts that share most of their n-grams score close to 1.
//! Entries keep the vector as a TurboQuant code ([`crate::turboquant`]), `bits` per
//! coordinate, and the scan estimates cosines on the codes directly. Exact mode keeps no codes.
//!
//! Past `max_entries` or `max_mib` the least recently used entries go. Counts: a lookup without
//! an exact match is an exact miss; in similar mode it then is a similar hit or miss.

use std::collections::hash_map::RandomState;
use std::collections::{BTreeMap, HashMap};
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::atomic::Ordering::Relaxed;
use std::time::{Duration, Instant};

use serde::Serialize;

use crate::config::{CacheConfig, CacheMode, Config};
use crate::stats::STATS;
use crate::turboquant::{Code, TurboQuant};

/// Coordinates of a prompt vector.
pub const VECTOR_DIM: usize = 256;
const SEED: u64 = 0x6d73_685f_6361_6368;

/// A prompt's vector without a model: signed hashed token n-grams (n = 1, 2, 3), normalised.
/// Deterministic; zero for an empty prompt.
pub fn prompt_vector(prompt: &[u32]) -> Vec<f32> {
    let mut v = vec![0.0f32; VECTOR_DIM];
    for n in 1..=3 {
        for gram in prompt.windows(n) {
            let mut h = n as u64;
            for &t in gram {
                h = mix(h ^ t as u64);
            }
            let sign = if h >> 63 == 0 { 1.0 } else { -1.0 };
            v[h as usize & (VECTOR_DIM - 1)] += sign;
        }
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

fn mix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// What besides the prompt and `max_new` changes an answer: the model and its engine settings,
/// and the chat template.
pub fn scope(cfg: &Config) -> u64 {
    let e = &cfg.engine;
    let http = cfg.http.as_ref().map(|h| (h.model.as_str(), format!("{:?}", h.chat_template)));
    let text = format!(
        "{:?}|{}|{:?}|{:?}|{}|{}|{http:?}",
        e.kind, e.model_dir, e.kv_format, e.experts_dir, cfg.num_layers, cfg.hidden_dim
    );
    text.bytes().fold(SEED, |h, b| mix(h ^ b as u64))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Key {
    scope: u64,
    max_new: usize,
    len: usize,
    hash: [u64; 2],
}

/// A prompt's key, and in similar mode its vector: made at submission, kept with the request
/// and stored with its answer.
pub struct Probe {
    key: Key,
    vector: Option<Vec<f32>>,
}

/// A cached answer for a prompt.
#[derive(Clone, Debug)]
pub struct Hit {
    /// `exact` or `similar`.
    pub kind: &'static str,
    /// Estimated cosine to the cached prompt; 1 for an exact hit.
    pub similarity: f32,
    pub answer: Vec<u32>,
    /// The worker that made it, how long ago, and the entry's hits so far, this one included.
    pub worker: String,
    pub age: Duration,
    pub hits: u64,
}

struct Entry {
    key: Key,
    code: Option<Code>,
    answer: Box<[u32]>,
    worker: String,
    at: Instant,
    hits: u64,
    /// Its place in `lru`.
    used: u64,
    bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Kinds {
    pub exact: u64,
    pub similar: u64,
}

/// `GET /v1/cache`.
#[derive(Clone, Debug, Serialize)]
pub struct CacheStats {
    pub mode: &'static str,
    pub entries: usize,
    pub bytes: u64,
    pub max_entries: usize,
    pub max_bytes: u64,
    pub bits: u32,
    pub dim: usize,
    pub similar_threshold: f32,
    pub hits: Kinds,
    pub misses: Kinds,
    pub inserted: u64,
    pub evicted: u64,
    /// The last similar scan, in microseconds.
    pub last_scan_us: Option<u64>,
}

pub struct Cache {
    cfg: CacheConfig,
    scope: u64,
    hashers: [RandomState; 2],
    /// Similar mode only.
    tq: Option<TurboQuant>,
    entries: Vec<Entry>,
    index: HashMap<Key, usize>,
    /// Entries by last use, oldest first.
    lru: BTreeMap<u64, usize>,
    tick: u64,
    bytes: u64,
    hits: Kinds,
    misses: Kinds,
    inserted: u64,
    evicted: u64,
    last_scan: Option<Duration>,
}

impl Cache {
    pub fn new(cfg: &CacheConfig, scope: u64) -> Cache {
        let tq = (cfg.mode == CacheMode::Similar).then(|| TurboQuant::new(VECTOR_DIM, cfg.bits, SEED));
        Cache {
            cfg: *cfg,
            scope,
            hashers: [RandomState::new(), RandomState::new()],
            tq,
            entries: Vec::new(),
            index: HashMap::new(),
            lru: BTreeMap::new(),
            tick: 0,
            bytes: 0,
            hits: Kinds::default(),
            misses: Kinds::default(),
            inserted: 0,
            evicted: 0,
            last_scan: None,
        }
    }

    /// The most an entry's bookkeeping takes beside its answer and code: the entry, its index
    /// slot and its place in the LRU, with room for the tables' spare capacity.
    pub fn entry_overhead() -> u64 {
        (2 * (std::mem::size_of::<Entry>() + std::mem::size_of::<(Key, usize)>() + 1) + 64) as u64
    }

    /// Bytes an entry takes, for an answer of `tokens` and a `bits`-wide code.
    pub fn entry_bytes(tokens: usize, worker: &str, bits: Option<u32>) -> u64 {
        let code = bits.map_or(0, |b| VECTOR_DIM * b as usize / 8 + std::mem::size_of::<Code>());
        Self::entry_overhead() + (tokens * 4 + worker.len() + code) as u64
    }

    /// The probe for a prompt, `None` with the cache off.
    pub fn probe(&self, prompt: &[u32], max_new: usize) -> Option<Probe> {
        if self.cfg.mode == CacheMode::Off {
            return None;
        }
        let hash = self.hashers.each_ref().map(|s| {
            let mut h = s.build_hasher();
            prompt.hash(&mut h);
            h.finish()
        });
        let key = Key { scope: self.scope, max_new, len: prompt.len(), hash };
        Some(Probe { key, vector: self.tq.as_ref().map(|_| prompt_vector(prompt)) })
    }

    pub fn lookup(&mut self, p: &Probe) -> Option<Hit> {
        if let Some(&i) = self.index.get(&p.key) {
            self.hits.exact += 1;
            STATS.cache_hits_exact.add(1);
            return Some(self.hit(i, "exact", 1.0));
        }
        self.misses.exact += 1;
        STATS.cache_misses_exact.add(1);
        let (Some(tq), Some(v)) = (&self.tq, &p.vector) else { return None };
        let t0 = Instant::now();
        let q = tq.query(v);
        let mut best: Option<(usize, f32)> = None;
        for (i, e) in self.entries.iter().enumerate() {
            let Some(code) = e.code.as_ref().filter(|_| e.key.scope == p.key.scope && e.key.max_new == p.key.max_new) else {
                continue;
            };
            let s = tq.cosine(&q, code);
            if best.map_or(true, |(_, b)| s > b) {
                best = Some((i, s));
            }
        }
        self.last_scan = Some(t0.elapsed());
        match best.filter(|&(_, s)| s >= self.cfg.similar_threshold) {
            Some((i, s)) => {
                self.hits.similar += 1;
                STATS.cache_hits_similar.add(1);
                Some(self.hit(i, "similar", s))
            }
            None => {
                self.misses.similar += 1;
                STATS.cache_misses_similar.add(1);
                None
            }
        }
    }

    fn hit(&mut self, i: usize, kind: &'static str, similarity: f32) -> Hit {
        self.touch(i);
        let e = &mut self.entries[i];
        e.hits += 1;
        Hit { kind, similarity, answer: e.answer.to_vec(), worker: e.worker.clone(), age: e.at.elapsed(), hits: e.hits }
    }

    fn touch(&mut self, i: usize) {
        self.tick += 1;
        let e = &mut self.entries[i];
        self.lru.remove(&e.used);
        e.used = self.tick;
        self.lru.insert(self.tick, i);
    }

    /// Keeps `answer` for the probe's prompt, replacing what that prompt had, then evicts the
    /// least recently used past the limits. An answer too big for `max_mib` alone is not kept.
    pub fn insert(&mut self, p: Probe, answer: Vec<u32>, worker: &str) {
        let code = match (&self.tq, &p.vector) {
            (Some(tq), Some(v)) => Some(tq.encode(v)),
            _ => None,
        };
        let bytes = Self::entry_overhead() + (answer.len() * 4 + worker.len()) as u64 + code.as_ref().map_or(0, |c| c.bytes() as u64);
        if bytes > self.cfg.max_bytes() {
            return;
        }
        if let Some(&i) = self.index.get(&p.key) {
            self.remove(i);
        }
        let entry = Entry {
            key: p.key,
            code,
            answer: answer.into_boxed_slice(),
            worker: worker.into(),
            at: Instant::now(),
            hits: 0,
            used: 0,
            bytes,
        };
        self.entries.push(entry);
        let i = self.entries.len() - 1;
        self.index.insert(p.key, i);
        self.touch(i);
        self.bytes += bytes;
        self.inserted += 1;
        while self.entries.len() > self.cfg.max_entries || self.bytes > self.cfg.max_bytes() {
            let (_, &oldest) = self.lru.first_key_value().expect("entries are in the lru");
            self.remove(oldest);
            self.evicted += 1;
        }
        self.publish();
    }

    fn remove(&mut self, i: usize) {
        let e = self.entries.swap_remove(i);
        self.index.remove(&e.key);
        self.lru.remove(&e.used);
        self.bytes -= e.bytes;
        // The last entry moved into `i`.
        if let Some(moved) = self.entries.get(i) {
            self.index.insert(moved.key, i);
            self.lru.insert(moved.used, i);
        }
    }

    /// Drops every entry; returns how many there were.
    pub fn clear(&mut self) -> usize {
        let n = self.entries.len();
        self.entries = Vec::new();
        self.index = HashMap::new();
        self.lru.clear();
        self.bytes = 0;
        self.publish();
        n
    }

    fn publish(&self) {
        STATS.cache_entries.store(self.entries.len() as u64, Relaxed);
        STATS.cache_bytes.store(self.bytes, Relaxed);
    }

    pub fn stats(&self) -> CacheStats {
        CacheStats {
            mode: self.cfg.mode.as_str(),
            entries: self.entries.len(),
            bytes: self.bytes,
            max_entries: self.cfg.max_entries,
            max_bytes: self.cfg.max_bytes(),
            bits: self.cfg.bits,
            dim: VECTOR_DIM,
            similar_threshold: self.cfg.similar_threshold,
            hits: self.hits,
            misses: self.misses,
            inserted: self.inserted,
            evicted: self.evicted,
            last_scan_us: self.last_scan.map(|d| d.as_micros() as u64),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(mode: CacheMode) -> CacheConfig {
        CacheConfig { mode, ..Default::default() }
    }

    fn prompt(seed: u32, len: usize) -> Vec<u32> {
        (0..len as u32).map(|i| (mix((seed as u64) << 32 | i as u64) % 50_000) as u32).collect()
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn prompt_vectors_follow_shared_ngrams() {
        let p = prompt(1, 300);
        assert_eq!(prompt_vector(&p), prompt_vector(&p));
        assert!(prompt_vector(&[]).iter().all(|&x| x == 0.0));
        let mut one = p.clone();
        one[150] = 7;
        let mut ten = p.clone();
        for k in 0..10 {
            ten[20 + 25 * k] = 7 + k as u32;
        }
        let v = prompt_vector(&p);
        let (c1, c10, other) = (cosine(&v, &prompt_vector(&one)), cosine(&v, &prompt_vector(&ten)), cosine(&v, &prompt_vector(&prompt(2, 300))));
        assert!(c1 > 0.98, "{c1}");
        assert!(c10 < 0.95 && c10 > 0.8, "{c10}");
        assert!(other.abs() < 0.2, "{other}");
    }

    #[test]
    fn exact_then_similar_lookups() {
        let mut c = Cache::new(&cfg(CacheMode::Similar), 1);
        let p = prompt(1, 300);
        assert!(c.lookup(&c.probe(&p, 20).unwrap()).is_none());
        c.insert(c.probe(&p, 20).unwrap(), vec![5, 6, 7], "m1");
        let hit = c.lookup(&c.probe(&p, 20).unwrap()).unwrap();
        assert_eq!((hit.kind, hit.similarity, hit.answer, hit.worker.as_str(), hit.hits), ("exact", 1.0, vec![5, 6, 7], "m1", 1));
        let mut near = p.clone();
        near[100] = 9;
        let hit = c.lookup(&c.probe(&near, 20).unwrap()).unwrap();
        assert_eq!((hit.kind, hit.hits), ("similar", 2));
        assert!(hit.similarity >= 0.98 && hit.similarity < 1.0, "{}", hit.similarity);
        // Another max_new, or another scope, is another answer.
        assert!(c.lookup(&c.probe(&p, 21).unwrap()).is_none());
        assert!(c.lookup(&c.probe(&near, 21).unwrap()).is_none());
        let other = Cache::new(&cfg(CacheMode::Similar), 2);
        assert!(c.lookup(&other.probe(&p, 20).unwrap()).is_none());
        let s = c.stats();
        assert_eq!((s.hits.exact, s.hits.similar, s.misses.exact, s.misses.similar), (1, 1, 5, 4));
        assert!(s.last_scan_us.is_some());
        assert_eq!(c.clear(), 1);
        assert_eq!((c.stats().entries, c.stats().bytes), (0, 0));
        assert!(c.lookup(&c.probe(&p, 20).unwrap()).is_none());
    }

    #[test]
    fn exact_mode_keeps_no_codes_and_off_keeps_nothing() {
        let mut c = Cache::new(&cfg(CacheMode::Exact), 1);
        let p = prompt(1, 300);
        c.insert(c.probe(&p, 20).unwrap(), vec![1], "m1");
        assert!(c.entries[0].code.is_none());
        let mut near = p.clone();
        near[100] = 9;
        assert!(c.lookup(&c.probe(&near, 20).unwrap()).is_none());
        assert_eq!(c.stats().misses.similar, 0);
        assert_eq!(c.stats().bytes, Cache::entry_bytes(1, "m1", None));
        assert!(Cache::new(&cfg(CacheMode::Off), 1).probe(&p, 20).is_none());
    }

    #[test]
    fn evicts_the_least_recently_used() {
        let mut c = Cache::new(&CacheConfig { max_entries: 3, ..cfg(CacheMode::Similar) }, 1);
        let ps: Vec<_> = (0..5).map(|i| prompt(i, 50)).collect();
        for p in &ps[..3] {
            c.insert(c.probe(p, 8).unwrap(), vec![1, 2], "m1");
        }
        // Use the first, so the second is the oldest.
        assert!(c.lookup(&c.probe(&ps[0], 8).unwrap()).is_some());
        c.insert(c.probe(&ps[3], 8).unwrap(), vec![3], "m1");
        let present = |c: &mut Cache, p: &[u32]| c.lookup(&c.probe(p, 8).unwrap()).is_some_and(|h| h.kind == "exact");
        assert!(!present(&mut c, &ps[1]));
        assert!(present(&mut c, &ps[0]) && present(&mut c, &ps[2]) && present(&mut c, &ps[3]));
        assert_eq!((c.stats().entries, c.stats().evicted), (3, 1));
        // Replacing an answer keeps one entry.
        c.insert(c.probe(&ps[3], 8).unwrap(), vec![4, 4], "m2");
        assert_eq!(c.lookup(&c.probe(&ps[3], 8).unwrap()).unwrap().answer, vec![4, 4]);
        assert_eq!(c.stats().entries, 3);

        // By bytes: room for about three entries of 1000 tokens.
        let per = Cache::entry_bytes(1000, "m1", Some(4));
        let max_mib = (3 * per + per / 2) as f64 / (1u64 << 20) as f64;
        let mut c = Cache::new(&CacheConfig { max_mib, ..cfg(CacheMode::Similar) }, 1);
        for p in &ps {
            c.insert(c.probe(p, 8).unwrap(), vec![9; 1000], "m1");
            assert!(c.stats().bytes <= c.stats().max_bytes);
        }
        assert_eq!((c.stats().entries, c.stats().evicted, c.stats().bytes), (3, 2, 3 * per));
        // Too big alone: not kept.
        c.insert(c.probe(&prompt(9, 5), 8).unwrap(), vec![0; 400_000], "m1");
        assert_eq!(c.stats().entries, 3);
    }

    /// Similar-scan time: `cargo test --release -p msh-node --lib cache::tests::scan_time --
    /// --ignored --nocapture`.
    #[test]
    #[ignore]
    fn scan_time() {
        for bits in [2, 3, 4] {
            for n in [1_000, 10_000, 100_000] {
                let mut c = Cache::new(&CacheConfig { max_entries: n, max_mib: 1024.0, bits, ..cfg(CacheMode::Similar) }, 1);
                for i in 0..n {
                    c.insert(c.probe(&prompt(i as u32, 64), 8).unwrap(), vec![1; 16], "m1");
                }
                let probe = c.probe(&prompt(u32::MAX, 64), 8).unwrap();
                let runs = 20;
                let t0 = Instant::now();
                for _ in 0..runs {
                    assert!(c.lookup(&probe).is_none());
                }
                let per = t0.elapsed() / runs;
                let mib = c.stats().bytes as f64 / (1u64 << 20) as f64;
                println!("{bits} bits, {n} entries: {per:?} per similar lookup, {mib:.1} MiB");
            }
        }
    }
}
