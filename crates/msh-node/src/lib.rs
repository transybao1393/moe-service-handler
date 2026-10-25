//! `msh-node`: coordinator and worker roles for moe-service-handler.
//!
//! A ring of nodes, each running a contiguous layer range, with the coordinator's scheduler
//! keeping several requests in flight. Planner and HTTP come later in M2.

pub mod config;
pub mod engine;
pub mod node;
pub mod scheduler;

use std::fmt;

use msh_net::FrameError;

#[derive(Debug)]
pub enum Error {
    Config(String),
    Net(FrameError),
    Engine(String),
    Protocol(String),
    /// The scheduler did not admit the request (queue full, ring down, ...).
    Rejected(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Config(m) => write!(f, "config: {m}"),
            Error::Net(e) => write!(f, "network: {e}"),
            Error::Engine(m) => write!(f, "engine: {m}"),
            Error::Protocol(m) => write!(f, "protocol: {m}"),
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
