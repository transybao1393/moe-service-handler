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
//! A single node with `layers = [0, 40]` is mode A.

use std::ops::Range;

use serde::Deserialize;

use crate::Error;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub hidden_dim: usize,
    pub num_layers: u32,
    pub nodes: Vec<NodeConfig>,
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
}

impl NodeConfig {
    pub fn layer_range(&self) -> Range<u32> {
        self.layers[0]..self.layers[1]
    }

    pub fn listen_addr(&self) -> &str {
        self.listen.as_deref().unwrap_or(&self.addr)
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
    }

    #[test]
    fn rejects_gaps_overlaps_and_short_coverage() {
        for (from, to) in [("[20, 30]", "[21, 30]"), ("[20, 30]", "[19, 30]"), ("[30, 40]", "[30, 39]")] {
            assert!(Config::parse(&THREE.replace(from, to)).is_err(), "{to} accepted");
        }
    }
}
