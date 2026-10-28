//! Layer split (CL-3): which contiguous layers each node runs.
//!
//! A token passes every node in turn, so the slowest node sets the pace. The planner splits
//! `0..num_layers` in node order so that the slowest node's time (its layers divided by its
//! `speed`) is as small as possible; among equally fast splits it takes the most even one,
//! and of those the one giving later nodes more layers.
//! A node gets only a range it can hold (`fits`, from its memory cap) and at least one layer.
//! The result depends only on its inputs, so every node computes the same split from the same
//! config. A `layers` entry on every node overrides the planner (CL-4).
//!
//! `speed` is relative (layers per unit of time). Until streammoe has a per-layer benchmark
//! (DEP-8), it comes from the config or from a previous run's metrics ([`speeds_from_metrics`]).

use std::ops::Range;

/// Splits `0..num_layers` across nodes with these speeds. `fits(i, range)` says whether node `i`
/// can hold `range`.
pub fn plan(
    num_layers: u32,
    speeds: &[f64],
    fits: impl Fn(usize, Range<u32>) -> bool,
) -> Result<Vec<Range<u32>>, String> {
    let n = speeds.len();
    let l = num_layers as usize;
    if n == 0 || l < n {
        return Err(format!("{num_layers} layers cannot be split across {n} nodes"));
    }
    if let Some(s) = speeds.iter().find(|s| !(**s > 0.0 && s.is_finite())) {
        return Err(format!("node speed {s} must be positive"));
    }
    let time = |i: usize, a: usize, b: usize| (b - a) as f64 / speeds[i];
    let ok: Vec<Vec<Vec<bool>>> = (0..n)
        .map(|i| (0..=l).map(|a| (0..=l).map(|b| b > a && fits(i, a as u32..b as u32)).collect()).collect())
        .collect();

    // First pass: the smallest possible time of the slowest node.
    let mut best = vec![vec![f64::INFINITY; l + 1]; n + 1];
    best[0][0] = 0.0;
    for i in 0..n {
        for a in 0..l {
            if best[i][a].is_infinite() {
                continue;
            }
            for b in a + 1..=l {
                if ok[i][a][b] {
                    let t = best[i][a].max(time(i, a, b));
                    if t < best[i + 1][b] {
                        best[i + 1][b] = t;
                    }
                }
            }
        }
    }
    let limit = best[n][l];
    if limit.is_infinite() {
        return Err("the model does not fit: no split gives every node a range it can hold".into());
    }

    // Second pass: within that limit, the most even split (least sum of squared times).
    let within = limit * (1.0 + 1e-9);
    let mut cost = vec![vec![f64::INFINITY; l + 1]; n + 1];
    let mut from = vec![vec![0usize; l + 1]; n + 1];
    cost[0][0] = 0.0;
    for i in 0..n {
        for a in 0..l {
            if cost[i][a].is_infinite() {
                continue;
            }
            for b in a + 1..=l {
                let t = time(i, a, b);
                if ok[i][a][b] && t <= within && cost[i][a] + t * t < cost[i + 1][b] {
                    cost[i + 1][b] = cost[i][a] + t * t;
                    from[i + 1][b] = a;
                }
            }
        }
    }
    let mut ranges = vec![0..0; n];
    let mut b = l;
    for i in (0..n).rev() {
        let a = from[i + 1][b];
        ranges[i] = a as u32..b as u32;
        b = a;
    }
    Ok(ranges)
}

/// Per-node speeds measured in a metrics CSV (`--metrics`): for every node named in the run's
/// split, layers per millisecond of compute over its decode steps. Returns `(name, speed)`
/// in ring order.
pub fn speeds_from_metrics(csv: &str) -> Result<Vec<(String, f64)>, String> {
    let mut lines = csv.lines();
    let header = split_csv(lines.next().ok_or("empty metrics file")?);
    let col = |name: &str| header.iter().position(|h| h == name).ok_or(format!("no {name} column"));
    let (split_col, kind_col, stage_col) = (col("split")?, col("kind")?, col("stage_us")?);
    let mut nodes: Vec<(String, u32)> = Vec::new();
    let mut total_us: Vec<f64> = Vec::new();
    let mut steps = 0usize;
    for line in lines.filter(|l| !l.trim().is_empty()) {
        let row = split_csv(line);
        if row.get(kind_col).map(String::as_str) != Some("decode") {
            continue;
        }
        let split = parse_split(&row[split_col])?;
        if nodes.is_empty() {
            total_us = vec![0.0; split.len()];
            nodes = split;
        } else if split != nodes {
            return Err("rows from runs with different splits; use one run's rows".into());
        }
        let stages: Vec<f64> = row[stage_col].split(';').map(|t| t.parse().unwrap_or(f64::NAN)).collect();
        if stages.len() != nodes.len() || stages.iter().any(|t| t.is_nan()) {
            return Err(format!("stage_us {:?} does not match {} nodes", row[stage_col], nodes.len()));
        }
        for (sum, t) in total_us.iter_mut().zip(stages) {
            *sum += t;
        }
        steps += 1;
    }
    if steps == 0 {
        return Err("no decode rows".into());
    }
    Ok(nodes
        .into_iter()
        .zip(total_us)
        .map(|((name, layers), us)| (name, layers as f64 / (us / steps as f64 / 1000.0)))
        .collect())
}

/// `m4:0-8 m1:8-12` → `[("m4", 8), ("m1", 4)]`.
fn parse_split(s: &str) -> Result<Vec<(String, u32)>, String> {
    s.split_whitespace()
        .map(|part| {
            let bad = || format!("bad split entry {part:?}");
            let (name, range) = part.rsplit_once(':').ok_or_else(bad)?;
            let (a, b) = range.split_once('-').ok_or_else(bad)?;
            let (a, b): (u32, u32) = (a.parse().map_err(|_| bad())?, b.parse().map_err(|_| bad())?);
            Ok((name.to_string(), b.checked_sub(a).filter(|&n| n > 0).ok_or_else(bad)?))
        })
        .collect()
}

/// Splits one CSV line, honouring double-quoted fields.
fn split_csv(line: &str) -> Vec<String> {
    let (mut fields, mut cur, mut quoted) = (Vec::new(), String::new(), false);
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' if quoted && chars.peek() == Some(&'"') => {
                cur.push('"');
                chars.next();
            }
            '"' => quoted = !quoted,
            ',' if !quoted => fields.push(std::mem::take(&mut cur)),
            _ => cur.push(c),
        }
    }
    fields.push(cur);
    fields
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lens(r: &[Range<u32>]) -> Vec<u32> {
        r.iter().map(|r| r.end - r.start).collect()
    }

    #[test]
    fn equal_nodes_get_an_even_split() {
        let r = plan(40, &[1.0, 1.0, 1.0], |_, _| true).unwrap();
        // The leftover layer goes to the last node.
        assert_eq!(r, [0..13, 13..26, 26..40]);
        assert_eq!(lens(&plan(16, &[1.0, 1.0], |_, _| true).unwrap()), [8, 8]);
    }

    #[test]
    fn faster_nodes_get_more_layers() {
        // M4, M1 (about half the memory bandwidth), M4.
        assert_eq!(lens(&plan(40, &[2.0, 1.0, 2.0], |_, _| true).unwrap()), [16, 8, 16]);
        // 16 / 9 / 15 is as fast (the M4s take 9.1 time units either way), but less even.
        assert_eq!(lens(&plan(40, &[1.76, 1.0, 1.76], |_, _| true).unwrap()), [16, 8, 16]);
    }

    #[test]
    fn memory_limits_override_speed() {
        // The fast first node can hold at most 10 layers.
        let r = plan(40, &[2.0, 1.0, 2.0], |i, r| i != 0 || r.len() <= 10).unwrap();
        assert_eq!(lens(&r), [10, 10, 20]);
        assert!(plan(40, &[1.0, 1.0], |_, r| r.len() <= 19).unwrap_err().contains("does not fit"));
    }

    #[test]
    fn every_node_gets_a_layer() {
        assert_eq!(lens(&plan(3, &[100.0, 1.0, 1.0], |_, _| true).unwrap()), [1, 1, 1]);
        assert!(plan(2, &[1.0, 1.0, 1.0], |_, _| true).is_err());
        assert!(plan(4, &[1.0, 0.0], |_, _| true).is_err());
    }

    #[test]
    fn speeds_come_from_decode_rows() {
        let csv = "\
run_id,git_hash,host,hardware,split,engine,request,pos,positions,kind,tokens,step_us,coord_us,workers_us,network_us,stage_us,queue_us,in_flight,ring_rtt_us,cached,drafted,accepted
1,abc,h,\"Apple M4, 16 GB\",m4:0-8 m1:8-12 m4b:12-16,mock,1,0,25,first_token,5,900,400,500,0,400;200;300,0,1,,0,0,0
1,abc,h,\"Apple M4, 16 GB\",m4:0-8 m1:8-12 m4b:12-16,mock,1,25,1,decode,6,100,8,7,0,8;4;2,0,1,,0,0,0
1,abc,h,\"Apple M4, 16 GB\",m4:0-8 m1:8-12 m4b:12-16,mock,1,26,1,decode,7,100,8,7,0,8;4;6,0,1,,0,0,0
";
        let s = speeds_from_metrics(csv).unwrap();
        let names: Vec<&str> = s.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["m4", "m1", "m4b"]);
        // 8 layers in 8 µs, 4 in 4 µs, 4 in 4 µs on average: all 1 layer per µs.
        for (_, speed) in &s {
            assert!((speed - 1000.0).abs() < 1e-9, "{s:?}");
        }
        assert!(speeds_from_metrics("run_id\n").is_err());
    }
}
