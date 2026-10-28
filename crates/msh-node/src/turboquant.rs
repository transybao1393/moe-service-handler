//! TurboQuant vector quantization (Zandieh et al., 2025, "TurboQuant: online vector
//! quantization with near-optimal distortion rate"), written from the paper's algorithm.
//!
//! A vector is split into its norm and its direction. The direction is rotated at random, which
//! makes every coordinate follow the same known distribution, that of one coordinate of a
//! uniform unit vector in `n` dimensions, density ∝ (1 − x²)^((n−3)/2). Each coordinate is then
//! quantized on its own to `b` bits with the Lloyd-Max codebook for that distribution, computed
//! once per quantizer. The rotation is a seeded randomized Hadamard transform, three rounds of
//! random signs then a fast Walsh-Hadamard transform; a dimension that is not a power of two
//! is padded with zeros (at least [`MIN_PADDED`]).
//!
//! The scalar codes minimise the reconstruction error (the paper's TurboQuant-mse), but the
//! inner product of a reconstruction is biased low. [`TurboQuant::with_qjl`] adds the paper's
//! 1-bit QJL residual (TurboQuant-prod): `b − 1` bits of scalar code, then the signs of a
//! Gaussian projection of what is left, for an unbiased inner product. Without it, each code
//! keeps `align`, the inner product of the unit direction with its reconstruction, and
//! [`TurboQuant::cosine`] divides by it: a vector scores exactly 1 against its own code.
//!
//! Inner products are estimated on the codes, without decoding: a [`Query`] holds the rotated
//! query and a table of its products with every centroid, so a code costs one table lookup
//! per byte (2 or 4 bits) or per coordinate (3 bits).

use std::f64::consts::PI;

/// Smallest padded dimension: the codebook needs a few coordinates to share the norm.
pub const MIN_PADDED: usize = 16;
const ROUNDS: usize = 3;

/// A quantizer for vectors of one dimension, bit width and seed.
#[derive(Clone, Debug)]
pub struct TurboQuant {
    dim: usize,
    /// The padded dimension, a power of two.
    n: usize,
    /// Bits per coordinate of the scalar code: `bits`, or `bits - 1` with the QJL residual.
    code_bits: u32,
    /// The rotation's random signs, `ROUNDS` rows of `n`.
    signs: Vec<f32>,
    /// Lloyd-Max centroids for one rotated coordinate of a unit vector, ascending, and the
    /// edges between them.
    centroids: Vec<f32>,
    edges: Vec<f32>,
    /// QJL: a Gaussian `n × n` projection, row-major.
    qjl: Option<Vec<f32>>,
}

/// A quantized vector.
#[derive(Clone, Debug, PartialEq)]
pub struct Code {
    pub norm: f32,
    /// The unit direction's inner product with its reconstruction; 1 with no error.
    pub align: f32,
    /// `code_bits` per padded coordinate, little-endian within each byte.
    codes: Box<[u8]>,
    /// QJL: the residual's norm and the signs of its projection, one bit each.
    residual: Option<(f32, Box<[u8]>)>,
}

impl Code {
    /// Heap and inline bytes this code takes.
    pub fn bytes(&self) -> usize {
        std::mem::size_of::<Code>() + self.codes.len() + self.residual.as_ref().map_or(0, |(_, s)| s.len())
    }
}

/// A vector prepared for estimates against many codes.
pub struct Query {
    norm: f32,
    /// Its rotated unit direction.
    rotated: Vec<f32>,
    /// Products with the centroids: per byte of code and byte value when a byte holds whole
    /// codes, else per coordinate and code.
    table: Vec<f32>,
    /// QJL: the projection of `rotated`.
    projected: Option<Vec<f32>>,
}

impl TurboQuant {
    /// The MSE quantizer: `bits` (1..=8) per coordinate.
    pub fn new(dim: usize, bits: u32, seed: u64) -> TurboQuant {
        assert!((1..=8).contains(&bits), "TurboQuant takes 1..=8 bits, not {bits}");
        let n = dim.next_power_of_two().max(MIN_PADDED);
        let mut rng = SplitMix(seed);
        let signs = (0..ROUNDS * n).map(|_| if rng.next() & 1 == 0 { 1.0 } else { -1.0 }).collect();
        let (centroids, edges) = codebook(n, 1 << bits);
        TurboQuant { dim, n, code_bits: bits, signs, centroids, edges, qjl: None }
    }

    /// The inner-product quantizer: `bits - 1` (1..=7) of scalar code and the 1-bit QJL
    /// residual, `bits` in all.
    pub fn with_qjl(dim: usize, bits: u32, seed: u64) -> TurboQuant {
        assert!((2..=8).contains(&bits), "TurboQuant with QJL takes 2..=8 bits, not {bits}");
        let mut q = TurboQuant::new(dim, bits - 1, seed);
        let mut rng = SplitMix(seed ^ 0x9e37_79b9_7f4a_7c15);
        q.qjl = Some((0..q.n * q.n).map(|_| rng.gaussian()).collect());
        q
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Bits per coordinate in all, the QJL bit included.
    pub fn bits(&self) -> u32 {
        self.code_bits + self.qjl.is_some() as u32
    }

    /// Bytes of code per vector, besides the norm and the fixed fields.
    pub fn code_bytes(&self) -> usize {
        self.n * self.bits() as usize / 8
    }

    pub fn encode(&self, x: &[f32]) -> Code {
        assert_eq!(x.len(), self.dim, "vector of {} for a {}-dimension quantizer", x.len(), self.dim);
        let norm = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let mut v = vec![0.0; self.n];
        if norm > 0.0 && norm.is_finite() {
            for (o, i) in v.iter_mut().zip(x) {
                *o = i / norm;
            }
        }
        self.rotate(&mut v);
        let idx: Vec<u8> = v.iter().map(|&c| self.edges.partition_point(|&e| e < c) as u8).collect();
        let mut align = 0.0;
        for (c, &i) in v.iter_mut().zip(&idx) {
            let q = self.centroids[i as usize];
            align += *c * q;
            // `v` becomes the residual.
            *c -= q;
        }
        let residual = self.qjl.as_ref().map(|s| {
            let gamma = v.iter().map(|r| r * r).sum::<f32>().sqrt();
            let mut signs = vec![0u8; self.n / 8];
            for (j, row) in s.chunks_exact(self.n).enumerate() {
                if dot(row, &v) >= 0.0 {
                    signs[j / 8] |= 1 << (j % 8);
                }
            }
            (gamma, signs.into_boxed_slice())
        });
        let codes = pack(&idx, self.code_bits);
        if !(norm > 0.0 && norm.is_finite()) {
            // Zero (or not finite): decodes to zero and scores 0 against anything.
            return Code { norm: 0.0, align: 1.0, codes, residual };
        }
        Code { norm, align, codes, residual }
    }

    pub fn decode(&self, c: &Code) -> Vec<f32> {
        let mut v: Vec<f32> = (0..self.n).map(|i| self.centroids[unpack(&c.codes, self.code_bits, i)]).collect();
        if let (Some(s), Some((gamma, signs))) = (&self.qjl, &c.residual) {
            // The paper's dequantization: sqrt(pi/2) / n · gamma · Sᵀ sign(S r).
            let scale = (PI / 2.0).sqrt() as f32 / self.n as f32 * gamma;
            for (j, row) in s.chunks_exact(self.n).enumerate() {
                let sign = if signs[j / 8] >> (j % 8) & 1 == 1 { scale } else { -scale };
                for (o, r) in v.iter_mut().zip(row) {
                    *o += sign * r;
                }
            }
        }
        self.unrotate(&mut v);
        v.truncate(self.dim);
        for o in &mut v {
            *o *= c.norm;
        }
        v
    }

    /// Prepares `q` for [`inner_product`](Self::inner_product) and [`cosine`](Self::cosine).
    pub fn query(&self, q: &[f32]) -> Query {
        assert_eq!(q.len(), self.dim, "vector of {} for a {}-dimension quantizer", q.len(), self.dim);
        let norm = q.iter().map(|v| v * v).sum::<f32>().sqrt();
        let mut rotated = vec![0.0; self.n];
        if norm > 0.0 && norm.is_finite() {
            for (o, i) in rotated.iter_mut().zip(q) {
                *o = i / norm;
            }
        }
        self.rotate(&mut rotated);
        let levels = self.centroids.len();
        let table = if 8 % self.code_bits == 0 {
            let per = (8 / self.code_bits) as usize;
            let mask = levels - 1;
            let mut t = vec![0.0; self.n / per * 256];
            for (j, row) in t.chunks_exact_mut(256).enumerate() {
                // A byte's value is its lower codes' value plus its top code's product, so
                // each entry is one add.
                row[..levels].iter_mut().zip(&self.centroids).for_each(|(o, c)| *o = rotated[j * per] * c);
                for k in 1..per {
                    let (low, rest) = row.split_at_mut(levels.pow(k as u32));
                    let q = rotated[j * per + k];
                    for (top, chunk) in rest.chunks_exact_mut(low.len()).take(mask).enumerate() {
                        let p = q * self.centroids[top + 1];
                        chunk.iter_mut().zip(low.iter()).for_each(|(o, l)| *o = l + p);
                    }
                    // Top code 0 last: it adds onto the lower codes in place.
                    let p = q * self.centroids[0];
                    low.iter_mut().for_each(|l| *l += p);
                }
            }
            t
        } else {
            rotated.iter().flat_map(|&r| self.centroids.iter().map(move |&c| r * c)).collect()
        };
        let projected = self.qjl.as_ref().map(|s| s.chunks_exact(self.n).map(|row| dot(row, &rotated)).collect());
        Query { norm: if norm.is_finite() { norm } else { 0.0 }, rotated, table, projected }
    }

    /// The inner product of the query's unit direction with the code's reconstruction (scalar
    /// part only).
    fn scalar_dot(&self, q: &Query, c: &Code) -> f32 {
        if 8 % self.code_bits == 0 {
            c.codes.iter().enumerate().map(|(j, &b)| q.table[j * 256 + b as usize]).sum()
        } else {
            let levels = self.centroids.len();
            (0..self.n).map(|i| q.table[i * levels + unpack(&c.codes, self.code_bits, i)]).sum()
        }
    }

    /// QJL's estimate of the query direction's inner product with the code's residual.
    fn residual_dot(&self, q: &Query, c: &Code) -> f32 {
        let (Some(p), Some((gamma, signs))) = (&q.projected, &c.residual) else { return 0.0 };
        let sum: f32 = p.iter().enumerate().map(|(j, &v)| if signs[j / 8] >> (j % 8) & 1 == 1 { v } else { -v }).sum();
        (PI / 2.0).sqrt() as f32 / self.n as f32 * gamma * sum
    }

    /// Estimates `<q, x>` for the vector `x` behind `c`: the reconstruction's inner product,
    /// unbiased with the QJL residual.
    pub fn inner_product(&self, q: &Query, c: &Code) -> f32 {
        q.norm * c.norm * (self.scalar_dot(q, c) + self.residual_dot(q, c))
    }

    /// Estimates the cosine between `q` and the vector behind `c`; 0 when either is zero.
    /// Without QJL the reconstruction's inner product is divided by the code's `align`, which
    /// removes the bias along the vector itself. An estimate: it can fall a little outside
    /// [-1, 1].
    pub fn cosine(&self, q: &Query, c: &Code) -> f32 {
        if q.norm == 0.0 || c.norm == 0.0 {
            return 0.0;
        }
        let d = self.scalar_dot(q, c);
        if self.qjl.is_some() {
            d + self.residual_dot(q, c)
        } else {
            d / c.align.max(f32::EPSILON)
        }
    }

    /// Estimates `<x, y>` from two codes alone: the inner product of their scalar
    /// reconstructions.
    pub fn inner_product_codes(&self, a: &Code, b: &Code) -> f32 {
        let levels = self.centroids.len();
        let table: Vec<f32> = self.centroids.iter().flat_map(|&x| self.centroids.iter().map(move |&y| x * y)).collect();
        let sum: f32 = (0..self.n)
            .map(|i| table[unpack(&a.codes, self.code_bits, i) * levels + unpack(&b.codes, self.code_bits, i)])
            .sum();
        a.norm * b.norm * sum
    }

    fn rotate(&self, v: &mut [f32]) {
        let scale = 1.0 / (self.n as f32).sqrt();
        for r in 0..ROUNDS {
            for (x, s) in v.iter_mut().zip(&self.signs[r * self.n..][..self.n]) {
                *x *= s;
            }
            fwht(v);
            v.iter_mut().for_each(|x| *x *= scale);
        }
    }

    fn unrotate(&self, v: &mut [f32]) {
        let scale = 1.0 / (self.n as f32).sqrt();
        for r in (0..ROUNDS).rev() {
            fwht(v);
            for (x, s) in v.iter_mut().zip(&self.signs[r * self.n..][..self.n]) {
                *x *= s * scale;
            }
        }
    }
}

impl Query {
    /// The query's rotated unit direction, for tests and tools.
    pub fn rotated(&self) -> &[f32] {
        &self.rotated
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// In-place fast Walsh-Hadamard transform, unnormalised; `v.len()` is a power of two.
fn fwht(v: &mut [f32]) {
    let n = v.len();
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(2 * h) {
            for j in i..i + h {
                let (a, b) = (v[j], v[j + h]);
                v[j] = a + b;
                v[j + h] = a - b;
            }
        }
        h *= 2;
    }
}

fn pack(idx: &[u8], bits: u32) -> Box<[u8]> {
    let mut out = vec![0u8; (idx.len() * bits as usize).div_ceil(8)];
    for (i, &c) in idx.iter().enumerate() {
        for k in 0..bits as usize {
            let bit = i * bits as usize + k;
            out[bit / 8] |= (c >> k & 1) << (bit % 8);
        }
    }
    out.into_boxed_slice()
}

fn unpack(codes: &[u8], bits: u32, i: usize) -> usize {
    let bit = i * bits as usize;
    let (byte, off) = (bit / 8, bit % 8);
    let lo = codes[byte] as usize;
    let hi = codes.get(byte + 1).copied().unwrap_or(0) as usize;
    ((lo | hi << 8) >> off) & ((1 << bits) - 1)
}

/// The Lloyd-Max codebook with `levels` centroids for one coordinate of a uniform unit vector
/// in `n` dimensions, and the edges between the centroids. Found by Lloyd's iteration on a fine
/// grid of the density, in units of `1 / sqrt(n)`.
fn codebook(n: usize, levels: usize) -> (Vec<f32>, Vec<f32>) {
    let nf = n as f64;
    let zmax = nf.sqrt().min(12.0);
    const GRID: usize = 20_000;
    let dz = 2.0 * zmax / GRID as f64;
    let density = |z: f64| {
        let t = 1.0 - z * z / nf;
        if t > 0.0 { t.powf((nf - 3.0) / 2.0) } else { 0.0 }
    };
    // Running mass and first moment from -zmax (trapezoids), so a bin's mean costs two lookups.
    let (mut mass, mut moment) = (vec![0.0; GRID + 1], vec![0.0; GRID + 1]);
    for i in 0..GRID {
        let (a, b) = (-zmax + i as f64 * dz, -zmax + (i + 1) as f64 * dz);
        let (fa, fb) = (density(a), density(b));
        mass[i + 1] = mass[i] + (fa + fb) / 2.0 * dz;
        moment[i + 1] = moment[i] + (a * fa + b * fb) / 2.0 * dz;
    }
    let at = |table: &[f64], z: f64| {
        let x = ((z + zmax) / dz).clamp(0.0, GRID as f64);
        let i = (x as usize).min(GRID - 1);
        table[i] + (x - i as f64) * (table[i + 1] - table[i])
    };
    // Start at the density's quantiles, the middle of each of `levels` equal masses.
    let total = mass[GRID];
    let mut c: Vec<f64> = (0..levels)
        .map(|j| {
            let want = (j as f64 + 0.5) / levels as f64 * total;
            -zmax + mass.partition_point(|&m| m < want) as f64 * dz
        })
        .collect();
    let mut edges = vec![0.0; levels - 1];
    for _ in 0..20_000 {
        for (e, w) in edges.iter_mut().zip(c.windows(2)) {
            *e = (w[0] + w[1]) / 2.0;
        }
        let mut moved: f64 = 0.0;
        for j in 0..levels {
            let lo = if j == 0 { -zmax } else { edges[j - 1] };
            let hi = if j == levels - 1 { zmax } else { edges[j] };
            let m = at(&mass, hi) - at(&mass, lo);
            if m > 0.0 {
                // Lloyd's step, over-relaxed: it converges far sooner.
                let step = (at(&moment, hi) - at(&moment, lo)) / m - c[j];
                c[j] += 1.8 * step;
                moved = moved.max(step.abs());
            }
        }
        if moved < 1e-12 {
            break;
        }
    }
    for (e, w) in edges.iter_mut().zip(c.windows(2)) {
        *e = (w[0] + w[1]) / 2.0;
    }
    let s = nf.sqrt();
    (c.iter().map(|z| (z / s) as f32).collect(), edges.iter().map(|z| (z / s) as f32).collect())
}

/// A small seeded generator (SplitMix64).
pub(crate) struct SplitMix(pub u64);

impl SplitMix {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in (0, 1).
    fn uniform(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// Standard normal (Box-Muller).
    pub fn gaussian(&mut self) -> f32 {
        let (u, v) = (self.uniform(), self.uniform());
        ((-2.0 * u.ln()).sqrt() * (2.0 * PI * v).cos()) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(rng: &mut SplitMix, d: usize) -> Vec<f32> {
        let v: Vec<f32> = (0..d).map(|_| rng.gaussian()).collect();
        let n = dot(&v, &v).sqrt();
        v.iter().map(|x| x / n).collect()
    }

    /// The Gaussian Lloyd-Max distortion per coordinate, which the paper's MSE bounds give for
    /// unit vectors: 0.36, 0.117, 0.03, 0.009 at 1..=4 bits.
    const D_MSE: [f32; 5] = [0.0, 0.3634, 0.1175, 0.03454, 0.009497];

    #[test]
    fn codebook_tends_to_the_gaussian_lloyd_max() {
        // In units of the coordinate's standard deviation, 1 / sqrt(n).
        let z = |n: usize, levels| codebook(n, levels).0.iter().map(|c| c * (n as f32).sqrt()).collect::<Vec<_>>();
        let close = |got: &[f32], want: &[f32]| got.iter().zip(want).all(|(a, b)| (a - b).abs() < 0.01);
        let two = z(4096, 4);
        assert!(close(&two, &[-1.5104, -0.4528, 0.4528, 1.5104]), "{two:?}");
        let three = z(4096, 8);
        assert!(close(&three[4..], &[0.2451, 0.7560, 1.3439, 2.1520]), "{three:?}");
        let four = z(4096, 16);
        assert!(close(&four[8..], &[0.1284, 0.3881, 0.6568, 0.9424, 1.2562, 1.6181, 2.0690, 2.7326]), "{four:?}");
        // Symmetric at every size.
        let (c, _) = codebook(16, 8);
        assert!(c.iter().zip(c.iter().rev()).all(|(a, b)| (a + b).abs() < 1e-5), "{c:?}");
    }

    #[test]
    fn reconstruction_error_is_within_the_papers_bounds() {
        for d in [256, 200] {
            for bits in [2, 3, 4] {
                let q = TurboQuant::new(d, bits, 7);
                let mut rng = SplitMix(bits as u64);
                let trials = 200;
                let mut err = 0.0;
                for _ in 0..trials {
                    let x = unit(&mut rng, d);
                    let y = q.decode(&q.encode(&x));
                    err += x.iter().zip(&y).map(|(a, b)| (a - b) * (a - b)).sum::<f32>();
                }
                let err = err / trials as f32;
                // Upper: the Lloyd-Max distortion, loosely; lower: the paper's 4^-b for any
                // quantizer.
                let bound = D_MSE[bits as usize];
                assert!(err < 1.2 * bound, "d {d}, {bits} bits: error {err} over {bound}");
                assert!(err > 0.25f32.powi(bits as i32), "d {d}, {bits} bits: error {err} below 4^-b");
            }
        }
    }

    #[test]
    fn inner_products_are_within_the_papers_bounds() {
        let d = 256;
        for bits in [2, 3, 4] {
            let mse = TurboQuant::new(d, bits, 11);
            let prod = TurboQuant::with_qjl(d, bits, 11);
            let mut rng = SplitMix(100 + bits as u64);
            let trials = 300;
            let (mut e_mse, mut e_prod, mut e_codes) = (0.0, 0.0, 0.0);
            for _ in 0..trials {
                let (x, y) = (unit(&mut rng, d), unit(&mut rng, d));
                let truth = dot(&x, &y);
                let (cx, cy) = (mse.encode(&x), mse.encode(&y));
                e_mse += (mse.inner_product(&mse.query(&y), &cx) - truth).powi(2);
                e_codes += (mse.inner_product_codes(&cx, &cy) - truth).powi(2);
                e_prod += (prod.inner_product(&prod.query(&y), &prod.encode(&x)) - truth).powi(2);
            }
            let n = trials as f32;
            let (e_mse, e_prod, e_codes) = (e_mse / n, e_prod / n, e_codes / n);
            // A random unit y sees about D_mse / d of the reconstruction error; two codes about
            // twice that. TurboQuant-prod: at most sqrt(3) pi^2 / d · 4^-b (the paper's bound).
            let dm = D_MSE[bits as usize] / d as f32;
            assert!(e_mse < 2.0 * dm, "{bits} bits: query error {e_mse} over {}", 2.0 * dm);
            assert!(e_codes < 4.0 * dm, "{bits} bits: code-code error {e_codes} over {}", 4.0 * dm);
            let bound = 3f32.sqrt() * (PI as f32).powi(2) / d as f32 * 0.25f32.powi(bits as i32);
            assert!(e_prod < bound, "{bits} bits: QJL error {e_prod} over {bound}");
        }
    }

    #[test]
    fn qjl_removes_the_bias_and_align_corrects_the_cosine() {
        let d = 256;
        let mut rng = SplitMix(5);
        let (mse, prod) = (TurboQuant::new(d, 2, 3), TurboQuant::with_qjl(d, 2, 3));
        let trials = 200;
        let (mut raw, mut unbiased) = (0.0, 0.0);
        for _ in 0..trials {
            let x = unit(&mut rng, d);
            let c = mse.encode(&x);
            let q = mse.query(&x);
            raw += mse.inner_product(&q, &c);
            // A vector against its own code: exactly 1 after the correction.
            assert!((mse.cosine(&q, &c) - 1.0).abs() < 1e-4);
            unbiased += prod.inner_product(&prod.query(&x), &prod.encode(&x));
        }
        let (raw, unbiased) = (raw / trials as f32, unbiased / trials as f32);
        // The 2-bit reconstruction is short by about D_mse; QJL's estimate is not.
        assert!((raw - (1.0 - D_MSE[2])).abs() < 0.02, "{raw}");
        assert!((unbiased - 1.0).abs() < 0.01, "{unbiased}");
        // A near copy scores close to its true cosine.
        let x = unit(&mut rng, d);
        let noise = unit(&mut rng, d);
        let y: Vec<f32> = x.iter().zip(&noise).map(|(a, b)| a + 0.15 * b).collect();
        let truth = dot(&x, &y) / dot(&y, &y).sqrt();
        let q4 = TurboQuant::new(d, 4, 3);
        let est = q4.cosine(&q4.query(&y), &q4.encode(&x));
        assert!((est - truth).abs() < 0.01, "{est} vs {truth}");
    }

    #[test]
    fn a_seed_gives_the_same_codes() {
        let mut rng = SplitMix(9);
        let x = unit(&mut rng, 128);
        for bits in [2, 3, 4] {
            let a = TurboQuant::new(128, bits, 42).encode(&x);
            assert_eq!(a, TurboQuant::new(128, bits, 42).encode(&x));
            assert_ne!(a, TurboQuant::new(128, bits, 43).encode(&x));
            assert_eq!(a.codes.len(), 128 * bits as usize / 8);
        }
        let q = TurboQuant::with_qjl(128, 3, 42);
        assert_eq!(q.encode(&x), TurboQuant::with_qjl(128, 3, 42).encode(&x));
        assert_eq!((q.bits(), q.code_bytes()), (3, 48));
    }

    #[test]
    fn zero_and_odd_sized_vectors() {
        for bits in [2, 3, 4] {
            let q = TurboQuant::new(100, bits, 1);
            let zero = q.encode(&[0.0; 100]);
            assert_eq!(zero.norm, 0.0);
            assert!(q.decode(&zero).iter().all(|&v| v == 0.0));
            let mut rng = SplitMix(2);
            let x = unit(&mut rng, 100);
            assert_eq!(q.cosine(&q.query(&x), &zero), 0.0);
            assert_eq!(q.cosine(&q.query(&[0.0; 100]), &q.encode(&x)), 0.0);
            assert_eq!(q.inner_product(&q.query(&x), &zero), 0.0);
            // Padded to 128: the decoded vector has the input's 100 values.
            let y = q.decode(&q.encode(&x));
            assert_eq!(y.len(), 100);
            let err: f32 = x.iter().zip(&y).map(|(a, b)| (a - b) * (a - b)).sum();
            assert!(err < 3.0 * D_MSE[bits as usize], "{bits} bits: {err}");
            // Scale is kept apart from the direction.
            let big: Vec<f32> = x.iter().map(|v| v * 1000.0).collect();
            let c = q.encode(&big);
            assert!((c.norm - 1000.0).abs() < 0.1);
            assert!((q.cosine(&q.query(&x), &c) - 1.0).abs() < 1e-4);
        }
        // Tiny vectors pad to MIN_PADDED.
        let q = TurboQuant::new(3, 4, 1);
        let y = q.decode(&q.encode(&[3.0, -4.0, 0.0]));
        assert!((y[0] - 3.0).abs() < 0.5 && (y[1] + 4.0).abs() < 0.5, "{y:?}");
        assert_eq!(q.code_bytes(), MIN_PADDED / 2);
    }

    /// The measured errors at 256 dimensions: `cargo test --release -p msh-node --lib
    /// turboquant::tests::measured -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn measured() {
        let d = 256;
        for bits in [2, 3, 4] {
            let (mse, prod) = (TurboQuant::new(d, bits, 1), TurboQuant::with_qjl(d, bits, 1));
            let mut rng = SplitMix(77);
            let trials = 2000;
            let (mut rec, mut ip, mut ip_prod, mut bias, mut bias_prod) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for _ in 0..trials {
                let (x, y) = (unit(&mut rng, d), unit(&mut rng, d));
                let c = mse.encode(&x);
                rec += x.iter().zip(mse.decode(&c)).map(|(a, b)| (a - b) * (a - b)).sum::<f32>();
                let truth = dot(&x, &y);
                ip += (mse.inner_product(&mse.query(&y), &c) - truth).powi(2);
                ip_prod += (prod.inner_product(&prod.query(&y), &prod.encode(&x)) - truth).powi(2);
                bias += mse.inner_product(&mse.query(&x), &c) - 1.0;
                bias_prod += prod.inner_product(&prod.query(&x), &prod.encode(&x)) - 1.0;
            }
            let n = trials as f32;
            println!(
                "{bits} bits: |x - x^|^2 {:.5}; inner product error^2 x d: mse {:.4}, qjl {:.4}; bias on <x, x>: mse {:+.4}, qjl {:+.4}",
                rec / n, ip / n * d as f32, ip_prod / n * d as f32, bias / n, bias_prod / n
            );
        }
    }
}
