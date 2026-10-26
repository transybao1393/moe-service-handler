//! Cluster config (CL-1, CL-4, CL-7). One TOML file, shared by every node:
//!
//! ```toml
//! hidden_dim = 2048
//! num_layers = 40
//!
//! [[nodes]]                # first node = coordinator, runs the embedding
//! name = "m4"
//! addr = "192.168.1.10:7100"
//! layers = [0, 20]         # half-open layer range
//!
//! [[nodes]]                # last node runs the head and returns tokens to the first
//! name = "m1a"
//! addr = "192.168.1.11:7100"
//! layers = [20, 40]
//! ```
//!
//! A single node with `layers = [0, 40]` is mode A. An optional `[scheduler]` table sets
//! `max_in_flight` (default 3), `max_queue` (64), `health_interval_ms` (1000) and
//! `health_timeout_ms` (5000).
//!
//! An optional `[engine]` table picks the engine (default: the mock):
//!
//! ```toml
//! [engine]
//! kind = "streammoe"                     # needs a build with a model feature, e.g. --features olmoe
//! model_dir = "~/models/olmoe-1b-7b"   # artifacts/ and root/ as streammoe's converters write them
//! n_ctx = 2048
//! mem_cap_gib = 3.0                    # per node; a node can override it with its own mem_cap_gib
//! ```

use std::ops::Range;

use serde::Deserialize;

use crate::scheduler::SchedulerConfig;
use crate::Error;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub hidden_dim: usize,
    pub num_layers: u32,
    #[serde(default)]
    pub scheduler: SchedulerConfig,
    #[serde(default)]
    pub engine: EngineConfig,
    pub nodes: Vec<NodeConfig>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EngineKind {
    #[default]
    Mock,
    StreamMoe,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EngineConfig {
    pub kind: EngineKind,
    pub model_dir: String,
    pub n_ctx: usize,
    /// Hard memory cap for streammoe on each node, in GiB. `None` = uncapped.
    pub mem_cap_gib: Option<f64>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig { kind: EngineKind::Mock, model_dir: String::new(), n_ctx: 2048, mem_cap_gib: None }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NodeConfig {
    pub name: String,
    /// Address the other nodes connect to.
    pub addr: String,
    /// Address to bind; defaults to `addr`. Use `0.0.0.0:PORT` to listen on every interface.
    pub listen: Option<String>,
    pub layers: [u32; 2],
    /// Overrides `engine.mem_cap_gib` on this node.
    pub mem_cap_gib: Option<f64>,
}

impl NodeConfig {
    pub fn layer_range(&self) -> Range<u32> {
        self.layers[0]..self.layers[1]
    }

    pub fn listen_addr(&self) -> &str {
        self.listen.as_deref().unwrap_or(&self.addr)
    }

    /// This node's streammoe memory cap in bytes.
    pub fn mem_cap(&self, engine: &EngineConfig) -> Option<u64> {
        self.mem_cap_gib.or(engine.mem_cap_gib).map(|g| (g * (1u64 << 30) as f64) as u64)
    }
}

impl Config {
    pub fn parse(text: &str) -> Result<Config, Error> {
        let cfg: Config = toml::from_str(text).map_err(|e| Error::Config(e.to_string()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// The layer ranges must cover `0..num_layers` contiguously, in node order.
    pub fn validate(&self) -> Result<(), Error> {
        let bad = |m: String| Err(Error::Config(m));
        if self.nodes.is_empty() {
            return bad("no nodes".into());
        }
        let mut next = 0;
        for (i, n) in self.nodes.iter().enumerate() {
            if self.nodes[..i].iter().any(|m| m.name == n.name) {
                return bad(format!("duplicate node name {}", n.name));
            }
            let [lo, hi] = n.layers;
            if lo != next || hi <= lo {
                return bad(format!("node {}: layers [{lo}, {hi}) must start at {next} and be non-empty", n.name));
            }
            next = hi;
        }
        if next != self.num_layers {
            return bad(format!("layer ranges end at {next}, model has {}", self.num_layers));
        }
        if self.engine.kind == EngineKind::StreamMoe && self.engine.model_dir.is_empty() {
            return bad("engine.model_dir is required for kind = \"streammoe\"".into());
        }
        Ok(())
    }

    pub fn index_of(&self, name: &str) -> Result<usize, Error> {
        self.nodes
            .iter()
            .position(|n| n.name == name)
            .ok_or_else(|| Error::Config(format!("no node named {name}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const THREE: &str = r#"
        hidden_dim = 2048
        num_layers = 40
        [[nodes]]
        name = "m4"
        addr = "10.0.0.1:7100"
        listen = "0.0.0.0:7100"
        layers = [0, 20]
        [[nodes]]
        name = "m1a"
        addr = "10.0.0.2:7100"
        layers = [20, 30]
        [[nodes]]
        name = "m1b"
        addr = "10.0.0.3:7100"
        layers = [30, 40]
    "#;

    #[test]
    fn parses_three_node_split() {
        let c = Config::parse(THREE).unwrap();
        assert_eq!(c.nodes[1].layer_range(), 20..30);
        assert_eq!(c.nodes[0].listen_addr(), "0.0.0.0:7100");
        assert_eq!(c.nodes[2].listen_addr(), "10.0.0.3:7100");
        assert_eq!(c.index_of("m1b").unwrap(), 2);
        assert_eq!(c.scheduler.max_in_flight, 3);
    }

    #[test]
    fn scheduler_section_overrides_defaults() {
        let text = THREE.replace("num_layers = 40", "num_layers = 40\n[scheduler]\nmax_in_flight = 5");
        let c = Config::parse(&text).unwrap();
        assert_eq!((c.scheduler.max_in_flight, c.scheduler.max_queue), (5, 64));
    }

    #[test]
    fn engine_section_and_per_node_cap() {
        let text = THREE
            .replace("num_layers = 40", "num_layers = 40\n[engine]\nkind = \"streammoe\"\nmodel_dir = \"/m\"\nmem_cap_gib = 2.0")
            .replace("layers = [20, 30]", "layers = [20, 30]\nmem_cap_gib = 0.5");
        let c = Config::parse(&text).unwrap();
        assert_eq!(c.engine.kind, EngineKind::StreamMoe);
        assert_eq!(c.nodes[0].mem_cap(&c.engine), Some(2 << 30));
        assert_eq!(c.nodes[1].mem_cap(&c.engine), Some(1 << 29));
        assert_eq!(Config::parse(THREE).unwrap().engine.kind, EngineKind::Mock);
        assert!(Config::parse(&text.replace("model_dir = \"/m\"", "")).is_err());
    }

    #[test]
    fn rejects_gaps_overlaps_and_short_coverage() {
        for (from, to) in [("[20, 30]", "[21, 30]"), ("[20, 30]", "[19, 30]"), ("[30, 40]", "[30, 39]")] {
            assert!(Config::parse(&THREE.replace(from, to)).is_err(), "{to} accepted");
        }
    }
}
