//! `msh-net`: node-to-node transport for moe-service-handler.
//!
//! Frame format and payloads, persistent connections with a version handshake, chunked
//! messages and a prioritising writer per link, the round-trip latency probe (NET-5), and
//! host facts recorded with every measurement.

// Frames carry little-endian values, and hidden states are sent straight from memory.
#[cfg(not(target_endian = "little"))]
compile_error!("msh-net needs a little-endian host (Apple Silicon, NF-3)");

pub mod conn;
pub mod frame;
pub mod host;
pub mod probe;
pub mod writer;

pub use conn::{Connection, Message};
pub use writer::Writer;
pub use frame::{
    decode_done, decode_experts, decode_hidden, decode_move, decode_own, decode_partial, decode_start,
    decode_status, decode_token, encode_done, encode_experts, encode_hidden, encode_move, encode_own,
    encode_partial, encode_start, encode_status, encode_token, DoneReason, FrameError, Header, MsgType, Pick,
    RunStatus, Status, DEFAULT_MAX_CHUNK, FLAG_EVERY_TOKEN, FLAG_MORE, MAX_PAYLOAD, PROTOCOL_VERSION,
};
