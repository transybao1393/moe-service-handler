//! Run-level metrics (CL-10, CL-11): one CSV row per step of every request: a prompt chunk or
//! one decode position.
//!
//! Each row splits the step's wall time on the coordinator into each node's compute time, as
//! reported in the frames (no clock sync needed), and the remainder: network transfer and
//! queueing behind other requests on the nodes. Rows also carry the run's metadata, so files
//! from different runs can be concatenated.

use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::Path;

const HEADER: &str = "run_id,git_hash,host,hardware,split,engine,request,pos,positions,kind,tokens,\
                      step_us,coord_us,workers_us,network_us,stage_us,queue_us,in_flight,ring_rtt_us,\
                      cached,drafted,accepted";

/// Describes the run; repeated on every row.
pub struct RunInfo {
    pub run_id: u64,
    pub git_hash: String,
    pub host: String,
    pub hardware: String,
    /// Layer split, e.g. `m4:0-8 m1a:8-12 m1b:12-16`.
    pub split: String,
    /// Engine and model, e.g. `streammoe olmoe-1b-7b`.
    pub engine: String,
}

/// What a row describes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepKind {
    /// A prompt chunk whose output is discarded.
    Prefill,
    /// The last prompt chunk, which produces the first token.
    FirstToken,
    /// A generated token fed back in.
    Decode,
}

impl StepKind {
    pub fn as_str(self) -> &'static str {
        match self {
            StepKind::Prefill => "prefill",
            StepKind::FirstToken => "first_token",
            StepKind::Decode => "decode",
        }
    }
}

pub struct StepRecord<'a> {
    pub request: u64,
    /// First position of the step.
    pub pos: u32,
    /// Positions in the step: the chunk size in prefill, 1 when decoding.
    pub positions: u32,
    pub kind: StepKind,
    /// Tokens the step produced: one, or with draft tokens the accepted ones and the next.
    pub tokens: &'a [u32],
    pub step_us: u64,
    /// Compute time per node in ring order, coordinator first.
    pub stage_us: &'a [u32],
    /// Time the request waited for an in-flight slot.
    pub queue_us: u64,
    /// Requests in flight when the step finished.
    pub in_flight: usize,
    /// Last health ping's trip round the ring (NET-5), if one has completed.
    pub ring_rtt_us: Option<u64>,
    /// Prompt positions the request took over from a finished one (prefix reuse).
    pub cached: u32,
    /// Draft tokens the step carried, and how many the head confirmed.
    pub drafted: u32,
    pub accepted: u32,
}

pub struct Metrics {
    out: BufWriter<File>,
    /// The metadata columns, already joined.
    run: String,
}

impl Metrics {
    /// Appends to `path`, writing the header if the file is new or empty.
    pub fn create(path: &Path, run: &RunInfo) -> io::Result<Metrics> {
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)?;
        }
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        let empty = file.metadata()?.len() == 0;
        let mut out = BufWriter::new(file);
        if empty {
            writeln!(out, "{HEADER}")?;
        }
        let run = [
            run.run_id.to_string(),
            csv(&run.git_hash),
            csv(&run.host),
            csv(&run.hardware),
            csv(&run.split),
            csv(&run.engine),
        ]
        .join(",");
        Ok(Metrics { out, run })
    }

    pub fn record(&mut self, r: &StepRecord) -> io::Result<()> {
        let compute: u64 = r.stage_us.iter().map(|&t| t as u64).sum();
        let coord = r.stage_us.first().copied().unwrap_or(0) as u64;
        let stages: Vec<String> = r.stage_us.iter().map(u32::to_string).collect();
        let kind = r.kind.as_str();
        let tokens = match r.kind {
            StepKind::Prefill => String::new(),
            _ => r.tokens.iter().map(u32::to_string).collect::<Vec<_>>().join(";"),
        };
        writeln!(
            self.out,
            "{},{},{},{},{kind},{tokens},{},{coord},{},{},{},{},{},{},{},{},{}",
            self.run,
            r.request,
            r.pos,
            r.positions,
            r.step_us,
            compute - coord,
            r.step_us.saturating_sub(compute),
            stages.join(";"),
            r.queue_us,
            r.in_flight,
            r.ring_rtt_us.map(|t| t.to_string()).unwrap_or_default(),
            r.cached,
            r.drafted,
            r.accepted,
        )
    }

    pub fn flush(&mut self) -> io::Result<()> {
        self.out.flush()
    }
}

fn csv(s: &str) -> String {
    if s.contains([',', '"', '\n']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_split_step_time_into_compute_and_network() {
        let dir = std::env::temp_dir().join(format!("msh-metrics-{}", std::process::id()));
        let path = dir.join("run.csv");
        let _ = std::fs::remove_file(&path);
        let run = RunInfo {
            run_id: 7,
            git_hash: "abc".into(),
            host: "h".into(),
            hardware: "Apple M4, 16 GB".into(),
            split: "m4:0-8 m1a:8-16".into(),
            engine: "mock".into(),
        };
        let mut m = Metrics::create(&path, &run).unwrap();
        let rec = |kind, stage_us| StepRecord {
            request: 1,
            pos: 3,
            positions: 1,
            kind,
            tokens: &[42, 43],
            step_us: 1000,
            stage_us,
            queue_us: 5,
            in_flight: 2,
            ring_rtt_us: Some(300),
            cached: 0,
            drafted: 3,
            accepted: 1,
        };
        m.record(&rec(StepKind::Prefill, &[400, 350])).unwrap();
        m.record(&rec(StepKind::Decode, &[400, 350])).unwrap();
        m.flush().unwrap();
        drop(m);
        // Appending to an existing file adds no second header.
        Metrics::create(&path, &run).unwrap();

        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0], HEADER);
        assert_eq!(lines[1], "7,abc,h,\"Apple M4, 16 GB\",m4:0-8 m1a:8-16,mock,1,3,1,prefill,,1000,400,350,250,400;350,5,2,300,0,3,1");
        assert!(lines[2].contains(",decode,42;43,"));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
