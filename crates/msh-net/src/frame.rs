//! Wire frame format (NET-1, NET-2).
//!
//! Every frame is a fixed 24-byte little-endian header followed by the payload:
//!
//! | offset | size | field                         |
//! | ------ | ---- | ----------------------------- |
//! | 0      | 4    | magic `b"MSHN"`               |
//! | 4      | 2    | protocol version              |
//! | 6      | 1    | message type                  |
//! | 7      | 1    | flags (reserved, must be 0)   |
//! | 8      | 8    | request id                    |
//! | 16     | 4    | position                      |
//! | 20     | 4    | payload length in bytes       |

use std::fmt;
use std::io::{self, Read, Write};

pub const MAGIC: [u8; 4] = *b"MSHN";
pub const PROTOCOL_VERSION: u16 = 1;
pub const HEADER_LEN: usize = 24;
/// Upper bound on a payload; guards against a corrupt length field allocating gigabytes.
/// 64 MiB covers an 8k-token f32 prefill chunk (8192 × 2048 × 4 = 64 MiB).
pub const MAX_PAYLOAD: usize = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum MsgType {
    /// f32 hidden state, `seq_len × hidden_dim` values. f32 rather than BF16 because the engine
    /// works in f32 and a split pipeline must match a single process exactly (NF-1).
    Hidden = 1,
    /// Sampled token id (u32).
    Token = 2,
    /// Free all state for a request on the receiving node.
    Drop = 3,
    Ping = 4,
    Pong = 5,
    /// Sent once by each side on connect; payload is the node name (UTF-8).
    Hello = 6,
    /// Fatal error; payload is a UTF-8 message.
    Error = 7,
}

impl TryFrom<u8> for MsgType {
    type Error = FrameError;

    fn try_from(v: u8) -> Result<Self, FrameError> {
        Ok(match v {
            1 => MsgType::Hidden,
            2 => MsgType::Token,
            3 => MsgType::Drop,
            4 => MsgType::Ping,
            5 => MsgType::Pong,
            6 => MsgType::Hello,
            7 => MsgType::Error,
            other => return Err(FrameError::UnknownType(other)),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub version: u16,
    pub msg_type: MsgType,
    pub request_id: u64,
    pub position: u32,
    pub payload_len: u32,
}

impl Header {
    pub fn new(msg_type: MsgType, request_id: u64, position: u32, payload_len: usize) -> Self {
        Header {
            version: PROTOCOL_VERSION,
            msg_type,
            request_id,
            position,
            payload_len: payload_len as u32,
        }
    }

    pub fn encode(&self) -> [u8; HEADER_LEN] {
        let mut b = [0u8; HEADER_LEN];
        b[0..4].copy_from_slice(&MAGIC);
        b[4..6].copy_from_slice(&self.version.to_le_bytes());
        b[6] = self.msg_type as u8;
        b[7] = 0;
        b[8..16].copy_from_slice(&self.request_id.to_le_bytes());
        b[16..20].copy_from_slice(&self.position.to_le_bytes());
        b[20..24].copy_from_slice(&self.payload_len.to_le_bytes());
        b
    }

    /// Decodes a header. The version is returned as-is; callers decide whether it is acceptable.
    pub fn decode(b: &[u8; HEADER_LEN]) -> Result<Self, FrameError> {
        if b[0..4] != MAGIC {
            return Err(FrameError::BadMagic([b[0], b[1], b[2], b[3]]));
        }
        if b[7] != 0 {
            return Err(FrameError::BadFlags(b[7]));
        }
        let payload_len = u32::from_le_bytes(b[20..24].try_into().unwrap());
        if payload_len as usize > MAX_PAYLOAD {
            return Err(FrameError::TooLarge(payload_len as usize));
        }
        Ok(Header {
            version: u16::from_le_bytes(b[4..6].try_into().unwrap()),
            msg_type: MsgType::try_from(b[6])?,
            request_id: u64::from_le_bytes(b[8..16].try_into().unwrap()),
            position: u32::from_le_bytes(b[16..20].try_into().unwrap()),
            payload_len,
        })
    }
}

#[derive(Debug)]
pub enum FrameError {
    Io(io::Error),
    BadMagic([u8; 4]),
    BadFlags(u8),
    UnknownType(u8),
    TooLarge(usize),
    VersionMismatch { local: u16, remote: u16 },
    Unexpected { expected: MsgType, got: MsgType },
    Remote(String),
}

impl fmt::Display for FrameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FrameError::Io(e) => write!(f, "i/o error: {e}"),
            FrameError::BadMagic(m) => write!(f, "bad magic {m:?}"),
            FrameError::BadFlags(v) => write!(f, "reserved flags set: {v:#04x}"),
            FrameError::UnknownType(t) => write!(f, "unknown message type {t}"),
            FrameError::TooLarge(n) => write!(f, "payload of {n} bytes exceeds limit of {MAX_PAYLOAD}"),
            FrameError::VersionMismatch { local, remote } => {
                write!(f, "protocol version mismatch: local {local}, remote {remote}")
            }
            FrameError::Unexpected { expected, got } => {
                write!(f, "expected {expected:?} frame, got {got:?}")
            }
            FrameError::Remote(msg) => write!(f, "remote error: {msg}"),
        }
    }
}

impl std::error::Error for FrameError {}

impl From<io::Error> for FrameError {
    fn from(e: io::Error) -> Self {
        FrameError::Io(e)
    }
}

/// Writes header then payload. Wrap the writer in a `BufWriter` and flush after each frame so a
/// small frame leaves as a single segment under `TCP_NODELAY`.
pub fn write_frame<W: Write>(w: &mut W, header: &Header, payload: &[u8]) -> io::Result<()> {
    debug_assert_eq!(header.payload_len as usize, payload.len());
    w.write_all(&header.encode())?;
    w.write_all(payload)
}

/// Reads one frame into `payload` (resized to fit; capacity is reused).
pub fn read_frame<R: Read>(r: &mut R, payload: &mut Vec<u8>) -> Result<Header, FrameError> {
    let mut hb = [0u8; HEADER_LEN];
    r.read_exact(&mut hb)?;
    let header = Header::decode(&hb)?;
    payload.resize(header.payload_len as usize, 0);
    r.read_exact(payload)?;
    Ok(header)
}

/// Appends `values` to `out` as little-endian f32.
pub fn f32s_to_bytes(values: &[f32], out: &mut Vec<u8>) {
    out.clear();
    out.reserve(values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

/// Decodes little-endian f32 values into `out`, which must hold exactly `bytes.len() / 4` values.
pub fn bytes_to_f32s(bytes: &[u8], out: &mut [f32]) -> Result<(), FrameError> {
    if bytes.len() != out.len() * 4 {
        return Err(FrameError::Remote(format!(
            "hidden payload of {} bytes, expected {}",
            bytes.len(),
            out.len() * 4
        )));
    }
    for (v, c) in out.iter_mut().zip(bytes.chunks_exact(4)) {
        *v = f32::from_le_bytes(c.try_into().unwrap());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_payload_round_trip_is_exact() {
        let v = [0.0f32, -1.5, f32::MIN_POSITIVE, 1e-40, f32::MAX, std::f32::consts::PI];
        let mut bytes = Vec::new();
        f32s_to_bytes(&v, &mut bytes);
        let mut out = [0f32; 6];
        bytes_to_f32s(&bytes, &mut out).unwrap();
        assert_eq!(v.map(f32::to_bits), out.map(f32::to_bits));
        assert!(bytes_to_f32s(&bytes[..20], &mut out).is_err());
    }

    #[test]
    fn header_round_trip() {
        let h = Header::new(MsgType::Hidden, 0xDEAD_BEEF_0123_4567, 42, 4096);
        assert_eq!(Header::decode(&h.encode()).unwrap(), h);
    }

    #[test]
    fn header_is_little_endian() {
        let b = Header::new(MsgType::Token, 1, 2, 4).encode();
        assert_eq!(&b[0..4], b"MSHN");
        assert_eq!(&b[4..6], &[1, 0]);
        assert_eq!(b[6], 2);
        assert_eq!(&b[8..16], &[1, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(&b[16..20], &[2, 0, 0, 0]);
        assert_eq!(&b[20..24], &[4, 0, 0, 0]);
    }

    #[test]
    fn rejects_bad_magic_type_and_size() {
        let mut b = Header::new(MsgType::Ping, 0, 0, 0).encode();
        b[0] = b'X';
        assert!(matches!(Header::decode(&b), Err(FrameError::BadMagic(_))));

        let mut b = Header::new(MsgType::Ping, 0, 0, 0).encode();
        b[6] = 99;
        assert!(matches!(Header::decode(&b), Err(FrameError::UnknownType(99))));

        let mut b = Header::new(MsgType::Ping, 0, 0, 0).encode();
        b[20..24].copy_from_slice(&(MAX_PAYLOAD as u32 + 1).to_le_bytes());
        assert!(matches!(Header::decode(&b), Err(FrameError::TooLarge(_))));
    }

    #[test]
    fn frame_round_trip_small_and_large() {
        for len in [0usize, 4096, 200_000] {
            let payload: Vec<u8> = (0..len).map(|i| i as u8).collect();
            let mut wire = Vec::new();
            let h = Header::new(MsgType::Ping, 7, 9, len);
            write_frame(&mut wire, &h, &payload).unwrap();
            assert_eq!(wire.len(), HEADER_LEN + len);

            let mut out = Vec::new();
            let got = read_frame(&mut wire.as_slice(), &mut out).unwrap();
            assert_eq!(got, h);
            assert_eq!(out, payload);
        }
    }
}
