//! `msh-node`: coordinator and worker roles for moe-service-handler.
//!
//! A controller and its workers, each running a contiguous layer range, or (`split =
//! "experts"`) the controller running everything but the routed experts and the workers
//! owning those. The controller talks to every worker; workers never talk to each other. Its
//! scheduler keeps several requests in flight.

pub mod chat;
pub mod config;
pub mod draft;
pub mod engine;
pub mod http;
pub mod memory;
pub mod metrics;
#[cfg(feature = "streammoe")]
pub mod streammoe_engine;
pub mod node;
pub mod owners;
pub mod planner;
pub mod scheduler;
pub mod stats;

use std::fmt;

use msh_net::FrameError;

#[derive(Debug)]
pub enum Error {
    Config(String),
    Net(FrameError),
    Engine(String),
    Protocol(String),
    /// A request failed after being admitted (engine error on some node, cluster down).
    Failed(String),
    /// The scheduler did not admit the request (queue full, cluster down, ...).
    Rejected(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Config(m) => write!(f, "config: {m}"),
            Error::Net(e) => write!(f, "network: {e}"),
            Error::Engine(m) => write!(f, "engine: {m}"),
            Error::Protocol(m) => write!(f, "protocol: {m}"),
            Error::Failed(m) => write!(f, "request failed: {m}"),
            Error::Rejected(m) => write!(f, "rejected: {m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<FrameError> for Error {
    fn from(e: FrameError) -> Self {
        Error::Net(e)
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Net(FrameError::Io(e))
    }
}
