//! `msh-net`: node-to-node transport for moe-service-handler.
//!
//! Frame format and payloads, persistent connections with a version handshake, the
//! round-trip latency probe (NET-5), and host facts recorded with every measurement.

pub mod conn;
pub mod frame;
pub mod host;
pub mod probe;

pub use conn::Connection;
pub use frame::{
    decode_hidden, decode_token, encode_hidden, encode_token, FrameError, Header, MsgType, FLAG_EVERY_TOKEN,
    PROTOCOL_VERSION,
};
