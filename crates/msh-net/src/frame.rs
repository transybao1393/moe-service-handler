//! Wire frame format (NET-1, NET-2).
//!
//! Every frame is a fixed 24-byte little-endian header followed by the payload:
//!
//! | offset | size | field                         |
//! | ------ | ---- | ----------------------------- |
//! | 0      | 4    | magic `b"MSHN"`               |
//! | 4      | 2    | protocol version              |
//! | 6      | 1    | message type                  |
//! | 7      | 1    | flags (see [`FLAG_EVERY_TOKEN`]) |
//! | 8      | 8    | request id                    |
//! | 16     | 4    | position (first, for a chunk) |
//! | 20     | 4    | payload length in bytes       |
//!
//! A `Hidden` frame carries one or more consecutive positions starting at `position` (a
//! prefill chunk); the matching `Token` frame repeats that start position and holds the token
//! sampled after the chunk's last position, or with [`FLAG_EVERY_TOKEN`] one token per position
//! (verifying draft tokens in speculative decoding).

use std::fmt;
use std::io::{self, Read, Write};

pub const MAGIC: [u8; 4] = *b"MSHN";
pub const PROTOCOL_VERSION: u16 = 4;

/// On a `Hidden` frame: the head returns the token sampled after every position, not just the
/// last. Workers pass it on.
pub const FLAG_EVERY_TOKEN: u8 = 1;
const KNOWN_FLAGS: u8 = FLAG_EVERY_TOKEN;
pub const HEADER_LEN: usize = 24;
/// Upper bound on a payload; guards against a corrupt length field allocating gigabytes.
/// 64 MiB covers an 8k-token f32 prefill chunk (8192 × 2048 × 4 = 64 MiB).
pub const MAX_PAYLOAD: usize = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum MsgType {
    /// Stage times, then the f32 hidden state, `seq_len × hidden_dim` values (see
    /// [`encode_hidden`]). f32 rather than BF16 because the engine works in f32 and a split
    /// pipeline must match a single process exactly (NF-1).
    Hidden = 1,
    /// Sampled token ids, then stage times (see [`encode_token`]).
    Token = 2,
    /// Free all state for a request on the receiving node.
    Drop = 3,
    Ping = 4,
    Pong = 5,
    /// Sent once by each side on connect; payload is the node name (UTF-8).
    Hello = 6,
    /// Fatal error; payload is a UTF-8 message.
    Error = 7,
    /// Keep only the first `position` positions of a request (rejected draft tokens). Travels
    /// the ring like `Drop`; no payload.
    Rewind = 8,
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
            8 => MsgType::Rewind,
            other => return Err(FrameError::UnknownType(other)),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub version: u16,
    pub msg_type: MsgType,
    pub flags: u8,
    pub request_id: u64,
    pub position: u32,
    pub payload_len: u32,
}

impl Header {
    pub fn new(msg_type: MsgType, request_id: u64, position: u32, payload_len: usize) -> Self {
        Header {
            version: PROTOCOL_VERSION,
            msg_type,
            flags: 0,
            request_id,
            position,
            payload_len: payload_len as u32,
        }
    }

    pub fn with_flags(mut self, flags: u8) -> Self {
        self.flags = flags;
        self
    }

    pub fn encode(&self) -> [u8; HEADER_LEN] {
        let mut b = [0u8; HEADER_LEN];
        b[0..4].copy_from_slice(&MAGIC);
        b[4..6].copy_from_slice(&self.version.to_le_bytes());
        b[6] = self.msg_type as u8;
        b[7] = self.flags;
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
        if b[7] & !KNOWN_FLAGS != 0 {
            return Err(FrameError::BadFlags(b[7]));
        }
        let payload_len = u32::from_le_bytes(b[20..24].try_into().unwrap());
        if payload_len as usize > MAX_PAYLOAD {
            return Err(FrameError::TooLarge(payload_len as usize));
        }
        Ok(Header {
            version: u16::from_le_bytes(b[4..6].try_into().unwrap()),
            msg_type: MsgType::try_from(b[6])?,
            flags: b[7],
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

// Hidden and Token payloads carry the compute time of every stage the positions have passed
// through, in microseconds, in ring order. The coordinator splits a step's wall time into
// per-node compute and the rest (network and queueing) from durations alone, so the nodes'
// clocks need not agree.
//
//   Hidden: stage_count u16, stage_us u32 × stage_count, f32 × (positions × hidden_dim)
//   Token:  token_count u16, token u32 × token_count, stage_count u16, stage_us u32 × stage_count

/// Encodes a hidden-state payload.
pub fn encode_hidden(stage_us: &[u32], values: &[f32], out: &mut Vec<u8>) {
    out.clear();
    out.reserve(2 + 4 * stage_us.len() + 4 * values.len());
    write_stages(stage_us, out);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

/// Decodes a hidden-state payload of `hidden_dim`-wide rows into `values`, resized to fit, and
/// `stage_us`. Returns the number of rows (positions).
pub fn decode_hidden(
    bytes: &[u8],
    hidden_dim: usize,
    values: &mut Vec<f32>,
    stage_us: &mut Vec<u32>,
) -> Result<usize, FrameError> {
    let rest = read_stages(bytes, stage_us)?;
    let row = hidden_dim * 4;
    if row == 0 || rest.is_empty() || rest.len() % row != 0 {
        return Err(bad_payload(format!("hidden of {} bytes is not whole rows of {row}", rest.len())));
    }
    values.clear();
    values.extend(rest.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())));
    Ok(rest.len() / row)
}

/// Encodes a token payload: one token, or one per position of a verify step.
pub fn encode_token(tokens: &[u32], stage_us: &[u32], out: &mut Vec<u8>) {
    out.clear();
    out.extend_from_slice(&(tokens.len() as u16).to_le_bytes());
    for t in tokens {
        out.extend_from_slice(&t.to_le_bytes());
    }
    write_stages(stage_us, out);
}

/// Decodes a token payload into `tokens` and `stage_us`.
pub fn decode_token(bytes: &[u8], tokens: &mut Vec<u32>, stage_us: &mut Vec<u32>) -> Result<(), FrameError> {
    let short = || bad_payload("token payload too short".into());
    let (n, rest) = bytes.split_at_checked(2).ok_or_else(short)?;
    let n = u16::from_le_bytes(n.try_into().unwrap()) as usize;
    let (ids, rest) = rest.split_at_checked(4 * n).ok_or_else(short)?;
    if n == 0 {
        return Err(bad_payload("token payload without tokens".into()));
    }
    if !read_stages(rest, stage_us)?.is_empty() {
        return Err(bad_payload("trailing bytes after token stage times".into()));
    }
    tokens.clear();
    tokens.extend(ids.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())));
    Ok(())
}

/// `values` as the little-endian bytes the wire carries: on a little-endian host (every Apple
/// Silicon Mac, NF-3) that is their own memory, so hidden states need no copy (NET-6).
pub(crate) fn f32_bytes(values: &[f32]) -> &[u8] {
    // SAFETY: any f32 is 4 valid bytes, u8 needs no alignment, and the length covers the slice.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

/// [`f32_bytes`] for writing into: every byte pattern is a valid f32.
pub(crate) fn f32_bytes_mut(values: &mut [f32]) -> &mut [u8] {
    // SAFETY: as in `f32_bytes`; any bytes written form valid f32s.
    unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), std::mem::size_of_val(values)) }
}

pub(crate) fn write_stages(stage_us: &[u32], out: &mut Vec<u8>) {
    out.extend_from_slice(&(stage_us.len() as u16).to_le_bytes());
    for t in stage_us {
        out.extend_from_slice(&t.to_le_bytes());
    }
}

/// Reads the stage times and returns the bytes after them.
fn read_stages<'a>(bytes: &'a [u8], stage_us: &mut Vec<u32>) -> Result<&'a [u8], FrameError> {
    let short = || bad_payload("stage times truncated".into());
    let (n, rest) = bytes.split_at_checked(2).ok_or_else(short)?;
    let n = u16::from_le_bytes(n.try_into().unwrap()) as usize;
    let (times, rest) = rest.split_at_checked(4 * n).ok_or_else(short)?;
    stage_us.clear();
    stage_us.extend(times.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())));
    Ok(rest)
}

pub(crate) fn bad_payload(msg: String) -> FrameError {
    FrameError::Remote(msg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hidden_payload_round_trip_is_exact() {
        let v = [0.0f32, -1.5, f32::MIN_POSITIVE, 1e-40, f32::MAX, std::f32::consts::PI];
        let mut bytes = Vec::new();
        encode_hidden(&[120, 95], &v, &mut bytes);
        let (mut out, mut stages) = (Vec::new(), Vec::new());
        assert_eq!(decode_hidden(&bytes, 3, &mut out, &mut stages).unwrap(), 2);
        assert_eq!(v.map(f32::to_bits)[..], out.iter().map(|x| x.to_bits()).collect::<Vec<_>>()[..]);
        assert_eq!(stages, [120, 95]);
        assert_eq!(decode_hidden(&bytes, 6, &mut out, &mut stages).unwrap(), 1);
        assert!(decode_hidden(&bytes, 4, &mut out, &mut stages).is_err());
        assert!(decode_hidden(&bytes[..bytes.len() - 4], 3, &mut out, &mut stages).is_err());
        assert!(decode_hidden(&bytes[..3], 3, &mut out, &mut stages).is_err());
        assert!(decode_hidden(&bytes[..10], 3, &mut out, &mut stages).is_err());
    }

    #[test]
    fn token_payload_round_trip() {
        let mut bytes = Vec::new();
        let (mut tokens, mut stages) = (Vec::new(), Vec::new());
        for want in [&[50_279][..], &[1, 2, 3, 4]] {
            encode_token(want, &[7, 8, 9], &mut bytes);
            decode_token(&bytes, &mut tokens, &mut stages).unwrap();
            assert_eq!((&tokens[..], &stages[..]), (want, &[7, 8, 9][..]));
        }
        assert!(decode_token(&bytes[..5], &mut tokens, &mut stages).is_err());
        bytes.push(0);
        assert!(decode_token(&bytes, &mut tokens, &mut stages).is_err());
        encode_token(&[], &[], &mut bytes);
        assert!(decode_token(&bytes, &mut tokens, &mut stages).is_err());
    }

    #[test]
    fn header_round_trip() {
        let h = Header::new(MsgType::Hidden, 0xDEAD_BEEF_0123_4567, 42, 4096);
        assert_eq!(Header::decode(&h.encode()).unwrap(), h);
        let h = h.with_flags(FLAG_EVERY_TOKEN);
        assert_eq!(Header::decode(&h.encode()).unwrap(), h);
    }

    #[test]
    fn header_is_little_endian() {
        let b = Header::new(MsgType::Token, 1, 2, 4).encode();
        assert_eq!(&b[0..4], b"MSHN");
        assert_eq!(&b[4..6], &PROTOCOL_VERSION.to_le_bytes());
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

        let mut b = Header::new(MsgType::Ping, 0, 0, 0).encode();
        b[7] = 0x80;
        assert!(matches!(Header::decode(&b), Err(FrameError::BadFlags(0x80))));
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
