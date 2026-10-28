//! `msh-net`: node-to-node transport for moe-service-handler.
//!
//! Frame format and payloads, persistent connections with a version handshake, the
//! round-trip latency probe (NET-5), and host facts recorded with every measurement.

// Frames carry little-endian values, and hidden states are sent straight from memory.
#[cfg(not(target_endian = "little"))]
compile_error!("msh-net needs a little-endian host (Apple Silicon, NF-3)");

pub mod conn;
pub mod frame;
pub mod host;
pub mod probe;

pub use conn::Connection;
pub use frame::{
    decode_hidden, decode_token, encode_hidden, encode_token, FrameError, Header, MsgType, FLAG_EVERY_TOKEN,
    PROTOCOL_VERSION,
};
