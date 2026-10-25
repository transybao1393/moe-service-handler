//! `msh-net`: node-to-node transport for moe-service-handler.
//!
//! M0 scope: frame format, persistent connections with a version handshake, and the
//! round-trip latency probe (NET-5). Hidden-state and token routing come in M1.

pub mod conn;
pub mod frame;
pub mod probe;

pub use conn::Connection;
pub use frame::{bytes_to_f32s, f32s_to_bytes, FrameError, Header, MsgType, PROTOCOL_VERSION};
