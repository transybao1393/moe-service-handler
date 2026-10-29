//! Wire frame format (NET-1, NET-2).
//!
//! Every frame is a fixed 24-byte little-endian header followed by the payload:
//!
//! | offset | size | field                         |
//! | ------ | ---- | ----------------------------- |
//! | 0      | 4    | magic `b"MSHN"`               |
//! | 4      | 2    | protocol version              |
//! | 6      | 1    | message type                  |
//! | 7      | 1    | flags (see [`FLAG_EVERY_TOKEN`], [`FLAG_MORE`]) |
//! | 8      | 8    | request id                    |
//! | 16     | 4    | position (first, for a chunk) |
//! | 20     | 4    | payload length in bytes       |
//!
//! A `Hidden` frame carries one or more consecutive positions starting at `position` (a
//! prefill chunk); the matching `Token` frame repeats that start position and holds the token
//! sampled after the chunk's last position, or with [`FLAG_EVERY_TOKEN`] one token per position
//! (verifying draft tokens in speculative decoding).
//!
//! The expert split (docs/expert-parallel.md) adds `Own`, `Experts`, `Partial`, `Move` and
//! `Release`. An `Experts` frame and its `Partial` carry the step's request id and start
//! position, and the layer in the payload; the other three use request id 0.
//!
//! The request split (docs/request-split.md) adds `Start` and `Done`: the controller sends a
//! worker a whole request, the worker sends back each `Token` as it is made and a `Done` when
//! it stops. A worker answers `Ping` with a `Pong` carrying its status report (see
//! [`encode_status`]).
//!
//! Version 8 adds `Event`, `Command` and `Result`. A worker sends an `Event` on its own (a
//! request's prefill ended, its memory passed or fell back under its cap); the controller sends
//! a `Command` with its own command id as the request id, and the worker answers with a
//! `Result` carrying the same id. Unknown event or command kinds decode to
//! [`FrameError::UnknownKind`]; the receiver skips the frame and keeps the link.
//!
//! Chunking: a message whose payload is larger than the link's `max_chunk` goes as several
//! frames with the same type, request id and position, so a small frame never waits behind a
//! whole prompt chunk. Every frame but the last has [`FLAG_MORE`]; the first frame's payload
//! starts with the message's whole payload length (u32), then its first bytes. No frame's
//! payload is larger than `max_chunk`. Chunks of messages of different requests may
//! interleave; a request's own messages never do. Small messages are one frame, as before.

use std::fmt;
use std::io::{self, Read, Write};

pub const MAGIC: [u8; 4] = *b"MSHN";
pub const PROTOCOL_VERSION: u16 = 8;

/// On a `Hidden` frame: the head returns the token sampled after every position, not just the
/// last. Workers pass it on.
pub const FLAG_EVERY_TOKEN: u8 = 1;
/// More chunks of this message follow (see the module docs). The top bit, apart from the
/// flags messages carry, which take the low bits.
pub const FLAG_MORE: u8 = 0x80;
const KNOWN_FLAGS: u8 = FLAG_EVERY_TOKEN | FLAG_MORE;
pub const HEADER_LEN: usize = 24;
/// Upper bound on a payload; guards against a corrupt length field allocating gigabytes.
/// 64 MiB covers an 8k-token f32 prefill chunk (8192 × 2048 × 4 = 64 MiB). A chunked
/// message's whole payload has the same limit.
pub const MAX_PAYLOAD: usize = 64 * 1024 * 1024;
/// Largest frame payload a link sends by default; bigger messages go in chunks.
pub const DEFAULT_MAX_CHUNK: usize = 64 * 1024;

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
    /// Keep only the first `position` positions of a request (rejected draft tokens). Goes through
    /// every worker like `Drop`; no payload.
    Rewind = 8,
    /// The experts a worker owns, replacing any it owned before (see [`encode_own`]). The worker
    /// sends back an empty `Own` once it holds them all.
    Own = 9,
    /// One layer's expert work for a step: this worker's picks, then the hidden rows the experts
    /// take (see [`encode_experts`]).
    Experts = 10,
    /// A worker's answer to `Experts`: each pick's output, in the order asked (see
    /// [`encode_partial`]).
    Partial = 11,
    /// An expert's weights for its new owner (see [`encode_move`]). The worker sends back a
    /// `Move` without weights once it holds the expert.
    Move = 12,
    /// The receiving worker no longer owns an expert; layer and expert as in `Move`, no weights.
    Release = 13,
    /// A whole request for a worker (request split): the request id in the header, the token
    /// limit and the prompt's token ids in the payload (see [`encode_start`]).
    Start = 14,
    /// A worker stopped a request (request split): the request id in the header, the tokens it
    /// made as the position, and why in the payload (see [`encode_done`]).
    Done = 15,
    /// Something a worker tells its controller on its own (request split): the request id (0
    /// when not about one) and a kind-specific position in the header, then kind, value and text
    /// (see [`encode_event`]).
    Event = 16,
    /// An order for a worker: the command id in the header, the kind in the payload (see
    /// [`encode_command`]).
    Command = 17,
    /// A worker's answer to one `Command`, with the same id (see [`encode_result`]).
    Result = 18,
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
            9 => MsgType::Own,
            10 => MsgType::Experts,
            11 => MsgType::Partial,
            12 => MsgType::Move,
            13 => MsgType::Release,
            14 => MsgType::Start,
            15 => MsgType::Done,
            16 => MsgType::Event,
            17 => MsgType::Command,
            18 => MsgType::Result,
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
    /// The connection ended in the middle of a chunked message.
    Incomplete { msg_type: MsgType, request_id: u64 },
    /// An `Event` or `Command` of a kind this build does not know.
    UnknownKind { msg_type: MsgType, kind: u8 },
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
            FrameError::UnknownKind { msg_type, kind } => write!(f, "unknown {msg_type:?} kind {kind}"),
            FrameError::Incomplete { msg_type, request_id } => {
                write!(f, "connection ended inside a chunked {msg_type:?} of request {request_id}")
            }
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
// through, in microseconds, in pipeline order. The coordinator splits a step's wall time into
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

// Expert-split payloads. Layer and expert ids are u16, and so is a pick's row (in the step):
//
//   Own:     count u32, (layer u16, expert u16) × count
//   Experts: layer u16, pick_count u32, (row u16, expert u16, gate f32) × pick_count,
//            f32 × (rows × hidden_dim)
//   Partial: layer u16, compute_us u32, f32 × (pick_count × hidden_dim)
//   Move:    layer u16, expert u16, weights (bytes; none in the reply and in `Release`)
//
// `Experts` carries every row of the step, so picks keep their row numbers; `Partial` returns
// one row per pick. `compute_us` is the worker's compute time for the frame, like a stage time.

/// One expert picked for one row; its gate weight scales the expert's output.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pick {
    pub row: u16,
    pub expert: u16,
    pub gate: f32,
}

/// Bytes of a pick on the wire.
pub const PICK_LEN: usize = 8;
/// Bytes before the picks of an `Experts` payload, and before the rows of a `Partial` payload.
pub const EXPERTS_PREFIX: usize = 6;
pub const PARTIAL_PREFIX: usize = 6;

/// Encodes an `Own` payload.
pub fn encode_own(experts: &[(u16, u16)], out: &mut Vec<u8>) {
    out.clear();
    out.extend_from_slice(&(experts.len() as u32).to_le_bytes());
    for &(layer, expert) in experts {
        out.extend_from_slice(&layer.to_le_bytes());
        out.extend_from_slice(&expert.to_le_bytes());
    }
}

/// Decodes an `Own` payload into `experts`, `(layer, expert)` pairs.
pub fn decode_own(bytes: &[u8], experts: &mut Vec<(u16, u16)>) -> Result<(), FrameError> {
    let (n, rest) = bytes.split_at_checked(4).ok_or_else(|| bad_payload("own payload too short".into()))?;
    let n = u32::from_le_bytes(n.try_into().unwrap()) as usize;
    if rest.len() != 4 * n {
        return Err(bad_payload(format!("own payload of {} bytes for {n} experts", rest.len())));
    }
    experts.clear();
    experts.extend(rest.chunks_exact(4).map(|c| (u16_at(c, 0), u16_at(c, 2))));
    Ok(())
}

/// What an `Experts` payload has before its rows.
pub(crate) fn write_experts_prefix(layer: u16, picks: &[Pick], out: &mut Vec<u8>) {
    out.extend_from_slice(&layer.to_le_bytes());
    out.extend_from_slice(&(picks.len() as u32).to_le_bytes());
    for p in picks {
        out.extend_from_slice(&p.row.to_le_bytes());
        out.extend_from_slice(&p.expert.to_le_bytes());
        out.extend_from_slice(&p.gate.to_le_bytes());
    }
}

pub(crate) fn read_picks(bytes: &[u8], picks: &mut Vec<Pick>) {
    picks.clear();
    picks.extend(bytes.chunks_exact(PICK_LEN).map(|c| Pick {
        row: u16_at(c, 0),
        expert: u16_at(c, 2),
        gate: f32::from_le_bytes(c[4..8].try_into().unwrap()),
    }));
}

/// Checks that `len` bytes are whole rows and that every pick names one of them; returns the
/// rows.
pub(crate) fn check_rows(len: usize, hidden_dim: usize, picks: &[Pick]) -> Result<usize, FrameError> {
    let row = hidden_dim * 4;
    if row == 0 || len == 0 || len % row != 0 {
        return Err(bad_payload(format!("rows of {len} bytes are not whole rows of {row}")));
    }
    let rows = len / row;
    if let Some(p) = picks.iter().find(|p| p.row as usize >= rows) {
        return Err(bad_payload(format!("pick for row {} of {rows}", p.row)));
    }
    Ok(rows)
}

/// Encodes an `Experts` payload: this worker's picks and every row of the step.
pub fn encode_experts(layer: u16, picks: &[Pick], rows: &[f32], out: &mut Vec<u8>) {
    out.clear();
    out.reserve(EXPERTS_PREFIX + PICK_LEN * picks.len() + 4 * rows.len());
    write_experts_prefix(layer, picks, out);
    out.extend_from_slice(f32_bytes(rows));
}

/// Decodes an `Experts` payload of `hidden_dim`-wide rows into `picks` and `rows`; returns the
/// layer.
pub fn decode_experts(
    bytes: &[u8],
    hidden_dim: usize,
    picks: &mut Vec<Pick>,
    rows: &mut Vec<f32>,
) -> Result<u16, FrameError> {
    let short = || bad_payload("experts payload too short".into());
    let (head, rest) = bytes.split_at_checked(EXPERTS_PREFIX).ok_or_else(short)?;
    let n = u32::from_le_bytes(head[2..6].try_into().unwrap()) as usize;
    let (p, rest) = rest.split_at_checked(PICK_LEN * n).ok_or_else(short)?;
    read_picks(p, picks);
    check_rows(rest.len(), hidden_dim, picks)?;
    rows.clear();
    rows.extend(rest.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())));
    Ok(u16_at(head, 0))
}

/// Encodes a `Partial` payload: one output row per pick.
pub fn encode_partial(layer: u16, compute_us: u32, values: &[f32], out: &mut Vec<u8>) {
    out.clear();
    out.extend_from_slice(&layer.to_le_bytes());
    out.extend_from_slice(&compute_us.to_le_bytes());
    out.extend_from_slice(f32_bytes(values));
}

/// Decodes a `Partial` payload into `values`; returns the layer and the worker's compute time.
pub fn decode_partial(bytes: &[u8], hidden_dim: usize, values: &mut Vec<f32>) -> Result<(u16, u32), FrameError> {
    let (head, rest) =
        bytes.split_at_checked(PARTIAL_PREFIX).ok_or_else(|| bad_payload("partial payload too short".into()))?;
    check_rows(rest.len(), hidden_dim, &[])?;
    values.clear();
    values.extend(rest.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())));
    Ok((u16_at(head, 0), u32::from_le_bytes(head[2..6].try_into().unwrap())))
}

/// Encodes a `Move` payload; empty `weights` for the reply and for `Release`.
pub fn encode_move(layer: u16, expert: u16, weights: &[u8], out: &mut Vec<u8>) {
    out.clear();
    out.extend_from_slice(&layer.to_le_bytes());
    out.extend_from_slice(&expert.to_le_bytes());
    out.extend_from_slice(weights);
}

/// Decodes a `Move` (or `Release`) payload: layer, expert and the weights.
pub fn decode_move(bytes: &[u8]) -> Result<(u16, u16, &[u8]), FrameError> {
    let (head, weights) = bytes.split_at_checked(4).ok_or_else(|| bad_payload("move payload too short".into()))?;
    Ok((u16_at(head, 0), u16_at(head, 2), weights))
}

// Request-split payloads:
//
//   Start: max_new u32, count u32, token u32 × count
//   Done:  reason u8
//   Pong:  sections of (tag u16, len u32, len bytes); unknown tags are skipped, so a worker
//          can report more than its controller reads
//     tag 1, memory:   footprint u64, cap u64 (0: no cap)
//     tag 2, requests: count u32, entry_len u16, count entries of entry_len bytes, each
//                      request u64, decoding u8, position u32, made u32, then any later fields
//   Event:   kind u8, value u64, text (UTF-8, rest)
//   Command: kind u8 (arguments may follow in a later version)
//   Result:  ok u8 (1 or 0), message (UTF-8, rest)

/// Encodes a `Start` payload: generate up to `max_new` tokens after `prompt`.
pub fn encode_start(max_new: u32, prompt: &[u32], out: &mut Vec<u8>) {
    out.clear();
    out.reserve(8 + 4 * prompt.len());
    out.extend_from_slice(&max_new.to_le_bytes());
    out.extend_from_slice(&(prompt.len() as u32).to_le_bytes());
    for t in prompt {
        out.extend_from_slice(&t.to_le_bytes());
    }
}

/// Decodes a `Start` payload into `prompt`; returns the token limit.
pub fn decode_start(bytes: &[u8], prompt: &mut Vec<u32>) -> Result<u32, FrameError> {
    let (head, rest) = bytes.split_at_checked(8).ok_or_else(|| bad_payload("start payload too short".into()))?;
    let n = u32::from_le_bytes(head[4..8].try_into().unwrap()) as usize;
    if n == 0 || rest.len() != 4 * n {
        return Err(bad_payload(format!("start payload of {} bytes for {n} prompt tokens", rest.len())));
    }
    prompt.clear();
    prompt.extend(rest.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())));
    Ok(u32::from_le_bytes(head[0..4].try_into().unwrap()))
}

/// Why a worker stopped a request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum DoneReason {
    /// It made the token limit `Start` gave.
    Length = 0,
    /// The model made its end-of-sequence token, which is not sent.
    Eos = 1,
    /// The controller sent `Drop`, and the worker freed it.
    Dropped = 2,
}

/// Encodes a `Done` payload.
pub fn encode_done(reason: DoneReason, out: &mut Vec<u8>) {
    out.clear();
    out.push(reason as u8);
}

pub fn decode_done(bytes: &[u8]) -> Result<DoneReason, FrameError> {
    match bytes {
        [0] => Ok(DoneReason::Length),
        [1] => Ok(DoneReason::Eos),
        [2] => Ok(DoneReason::Dropped),
        _ => Err(bad_payload(format!("done payload {bytes:?}"))),
    }
}

/// What an `Event` reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum EventKind {
    /// A request's prompt is prefilled; position = prompt length, value = prefill microseconds.
    PrefillDone = 1,
    /// Footprint passed 95 % of the cap; value = footprint bytes, position = 0, text = cap in
    /// bytes as decimal.
    MemoryHigh = 2,
    /// Footprint back under 90 % of the cap; value = footprint bytes.
    MemoryOk = 3,
    /// Anything else worth a log line on the controller; text says what.
    Note = 4,
}

impl EventKind {
    pub const ALL: [EventKind; 4] = [EventKind::PrefillDone, EventKind::MemoryHigh, EventKind::MemoryOk, EventKind::Note];

    pub fn as_str(self) -> &'static str {
        match self {
            EventKind::PrefillDone => "prefill_done",
            EventKind::MemoryHigh => "memory_high",
            EventKind::MemoryOk => "memory_ok",
            EventKind::Note => "note",
        }
    }
}

/// Encodes an `Event` payload.
pub fn encode_event(kind: EventKind, value: u64, text: &str, out: &mut Vec<u8>) {
    out.clear();
    out.reserve(9 + text.len());
    out.push(kind as u8);
    out.extend_from_slice(&value.to_le_bytes());
    out.extend_from_slice(text.as_bytes());
}

/// Decodes an `Event` payload: kind, value and text.
pub fn decode_event(bytes: &[u8]) -> Result<(EventKind, u64, &str), FrameError> {
    let (head, text) = bytes.split_at_checked(9).ok_or_else(|| bad_payload("event payload too short".into()))?;
    let kind = match head[0] {
        1 => EventKind::PrefillDone,
        2 => EventKind::MemoryHigh,
        3 => EventKind::MemoryOk,
        4 => EventKind::Note,
        kind => return Err(FrameError::UnknownKind { msg_type: MsgType::Event, kind }),
    };
    let text = std::str::from_utf8(text).map_err(|_| bad_payload("event text is not UTF-8".into()))?;
    Ok((kind, u64::from_le_bytes(head[1..9].try_into().unwrap()), text))
}

/// What a `Command` asks a worker to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum CommandKind {
    /// Answer with a one-line status (requests held, footprint, cap, uptime).
    Status = 1,
    /// Clear the worker's own counters (and the engine's, if the Engine has a way; else say so).
    ResetStats = 2,
    /// Leave the serve loop so the process exits and launchd starts it again. Refused
    /// (ok = 0, "busy: N requests") while the worker holds any request.
    Restart = 3,
}

impl CommandKind {
    pub fn as_str(self) -> &'static str {
        match self {
            CommandKind::Status => "status",
            CommandKind::ResetStats => "reset_stats",
            CommandKind::Restart => "restart",
        }
    }

    pub const ALL: [CommandKind; 3] = [CommandKind::Status, CommandKind::ResetStats, CommandKind::Restart];

    /// The kind [`as_str`](Self::as_str) names, if any.
    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.as_str() == s)
    }
}

/// Encodes a `Command` payload.
pub fn encode_command(kind: CommandKind, out: &mut Vec<u8>) {
    out.clear();
    out.push(kind as u8);
}

/// Decodes a `Command` payload. Bytes after the kind are left for later arguments.
pub fn decode_command(bytes: &[u8]) -> Result<CommandKind, FrameError> {
    match bytes.first() {
        Some(1) => Ok(CommandKind::Status),
        Some(2) => Ok(CommandKind::ResetStats),
        Some(3) => Ok(CommandKind::Restart),
        Some(&kind) => Err(FrameError::UnknownKind { msg_type: MsgType::Command, kind }),
        None => Err(bad_payload("empty command payload".into())),
    }
}

/// Encodes a `Result` payload.
pub fn encode_result(ok: bool, message: &str, out: &mut Vec<u8>) {
    out.clear();
    out.reserve(1 + message.len());
    out.push(ok as u8);
    out.extend_from_slice(message.as_bytes());
}

/// Decodes a `Result` payload: whether the command worked, and the worker's message.
pub fn decode_result(bytes: &[u8]) -> Result<(bool, &str), FrameError> {
    let (ok, message) = match bytes.split_first() {
        Some((0, m)) => (false, m),
        Some((1, m)) => (true, m),
        _ => return Err(bad_payload(format!("result payload starts {:?}", bytes.first()))),
    };
    let message = std::str::from_utf8(message).map_err(|_| bad_payload("result message is not UTF-8".into()))?;
    Ok((ok, message))
}

/// A worker's status report, the payload of its `Pong`. New fields go in new sections (or at
/// the end of a request entry), so older controllers keep reading the report.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Status {
    /// Physical footprint in bytes; 0 where it cannot be read.
    pub footprint: u64,
    /// Memory cap in bytes; 0 without one.
    pub cap: u64,
    pub requests: Vec<RunStatus>,
}

/// One request a worker holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RunStatus {
    pub request: u64,
    /// Past the prompt.
    pub decoding: bool,
    /// Positions the engine holds.
    pub position: u32,
    /// Tokens made.
    pub made: u32,
}

const STATUS_MEMORY: u16 = 1;
const STATUS_REQUESTS: u16 = 2;
const RUN_STATUS_LEN: usize = 17;

fn write_section(tag: u16, body: impl FnOnce(&mut Vec<u8>), out: &mut Vec<u8>) {
    out.extend_from_slice(&tag.to_le_bytes());
    let at = out.len();
    out.extend_from_slice(&[0; 4]);
    body(out);
    let len = (out.len() - at - 4) as u32;
    out[at..at + 4].copy_from_slice(&len.to_le_bytes());
}

/// Encodes a `Pong` status report.
pub fn encode_status(s: &Status, out: &mut Vec<u8>) {
    out.clear();
    write_section(
        STATUS_MEMORY,
        |o| {
            o.extend_from_slice(&s.footprint.to_le_bytes());
            o.extend_from_slice(&s.cap.to_le_bytes());
        },
        out,
    );
    write_section(
        STATUS_REQUESTS,
        |o| {
            o.extend_from_slice(&(s.requests.len() as u32).to_le_bytes());
            o.extend_from_slice(&(RUN_STATUS_LEN as u16).to_le_bytes());
            for r in &s.requests {
                o.extend_from_slice(&r.request.to_le_bytes());
                o.push(r.decoding as u8);
                o.extend_from_slice(&r.position.to_le_bytes());
                o.extend_from_slice(&r.made.to_le_bytes());
            }
        },
        out,
    );
}

/// Decodes a `Pong` status report into `s`. Sections it does not know are skipped; an empty
/// payload leaves `s` empty.
pub fn decode_status(bytes: &[u8], s: &mut Status) -> Result<(), FrameError> {
    let short = || bad_payload("status report truncated".into());
    *s = Status::default();
    let mut rest = bytes;
    while !rest.is_empty() {
        let (head, tail) = rest.split_at_checked(6).ok_or_else(short)?;
        let len = u32::from_le_bytes(head[2..6].try_into().unwrap()) as usize;
        let (body, tail) = tail.split_at_checked(len).ok_or_else(short)?;
        rest = tail;
        match u16_at(head, 0) {
            STATUS_MEMORY => {
                let b = body.get(..16).ok_or_else(short)?;
                s.footprint = u64::from_le_bytes(b[0..8].try_into().unwrap());
                s.cap = u64::from_le_bytes(b[8..16].try_into().unwrap());
            }
            STATUS_REQUESTS => {
                let (b, entries) = body.split_at_checked(6).ok_or_else(short)?;
                let n = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
                let entry = u16_at(b, 4) as usize;
                if entry < RUN_STATUS_LEN || entries.len() != n * entry {
                    return Err(bad_payload(format!("{n} request entries of {entry} bytes in {}", entries.len())));
                }
                s.requests.extend(entries.chunks_exact(entry).map(|e| RunStatus {
                    request: u64::from_le_bytes(e[0..8].try_into().unwrap()),
                    decoding: e[8] != 0,
                    position: u32::from_le_bytes(e[9..13].try_into().unwrap()),
                    made: u32::from_le_bytes(e[13..17].try_into().unwrap()),
                }));
            }
            _ => {}
        }
    }
    Ok(())
}

fn u16_at(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
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
pub(crate) fn read_stages<'a>(bytes: &'a [u8], stage_us: &mut Vec<u32>) -> Result<&'a [u8], FrameError> {
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

/// Appends the header of the frame carrying the next chunk of a message with header `h`
/// (`h.payload_len` the whole payload), from payload byte `off`, to `out`; on the first of
/// several chunks the whole length too. Returns the payload bytes the frame carries.
pub(crate) fn chunk_header(h: &Header, off: usize, max_chunk: usize, out: &mut Vec<u8>) -> usize {
    let total = h.payload_len as usize;
    if total <= max_chunk {
        out.extend_from_slice(&h.encode());
        return total;
    }
    let first = off == 0;
    let n = (if first { max_chunk - 4 } else { max_chunk }).min(total - off);
    let mut c = *h;
    c.payload_len = (n + if first { 4 } else { 0 }) as u32;
    if off + n < total {
        c.flags |= FLAG_MORE;
    }
    out.extend_from_slice(&c.encode());
    if first {
        out.extend_from_slice(&(total as u32).to_le_bytes());
    }
    n
}

/// Bytes `off..off + n` of the payload `head` then `rows`, as the part in each.
pub(crate) fn payload_range<'a>(head: &'a [u8], rows: &'a [u8], off: usize, n: usize) -> (&'a [u8], &'a [u8]) {
    let end = off + n;
    let h = &head[off.min(head.len())..end.min(head.len())];
    let r = &rows[off.saturating_sub(head.len())..end.saturating_sub(head.len())];
    (h, r)
}

/// Writes every byte of `parts`, one `writev` at a time.
pub(crate) fn write_parts<W: Write>(w: &mut W, parts: &mut [&[u8]]) -> io::Result<()> {
    let mut i = 0;
    while i < parts.len() {
        if parts[i].is_empty() {
            i += 1;
            continue;
        }
        let mut slices = [io::IoSlice::new(&[]); 4];
        for (s, p) in slices.iter_mut().zip(&parts[i..]) {
            *s = io::IoSlice::new(p);
        }
        let mut n = w.write_vectored(&slices[..(parts.len() - i).min(4)])?;
        if n == 0 {
            return Err(io::ErrorKind::WriteZero.into());
        }
        while n > 0 {
            let take = n.min(parts[i].len());
            parts[i] = &parts[i][take..];
            n -= take;
            if parts[i].is_empty() {
                i += 1;
            }
        }
    }
    Ok(())
}

/// Writes one message as one frame, or in chunks of at most `max_chunk` payload bytes.
pub(crate) fn write_message<W: Write>(w: &mut W, h: &Header, head: &[u8], rows: &[u8], max_chunk: usize) -> io::Result<()> {
    debug_assert_eq!(h.payload_len as usize, head.len() + rows.len());
    let mut hb = Vec::with_capacity(HEADER_LEN + 4);
    let mut off = 0;
    loop {
        hb.clear();
        let n = chunk_header(h, off, max_chunk, &mut hb);
        let (a, b) = payload_range(head, rows, off, n);
        write_parts(w, &mut [&hb, a, b])?;
        off += n;
        if off >= h.payload_len as usize {
            return Ok(());
        }
    }
}

/// For a message type whose payload ends in f32 rows (`Hidden`, `Experts`, `Partial`): the
/// payload bytes needed to know the length of what comes before the rows.
pub(crate) fn rows_base(t: MsgType) -> Option<usize> {
    match t {
        MsgType::Hidden => Some(2),
        MsgType::Experts | MsgType::Partial => Some(6),
        _ => None,
    }
}

/// The payload bytes before the rows, from the first [`rows_base`] bytes.
pub(crate) fn rows_prefix(t: MsgType, head: &[u8]) -> usize {
    match t {
        MsgType::Hidden => 2 + 4 * u16_at(head, 0) as usize,
        MsgType::Experts => EXPERTS_PREFIX + PICK_LEN * u32::from_le_bytes(head[2..6].try_into().unwrap()) as usize,
        _ => PARTIAL_PREFIX,
    }
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
    fn own_payload_round_trip() {
        let mut bytes = Vec::new();
        let mut got = Vec::new();
        for want in [&[][..], &[(0, 0), (39, 255), (u16::MAX, 7)]] {
            encode_own(want, &mut bytes);
            decode_own(&bytes, &mut got).unwrap();
            assert_eq!(got, want);
        }
        assert!(decode_own(&bytes[..bytes.len() - 1], &mut got).is_err());
        assert!(decode_own(&bytes[..3], &mut got).is_err());
    }

    #[test]
    fn experts_payload_round_trip_is_exact() {
        let picks = [
            Pick { row: 0, expert: 3, gate: 0.25 },
            Pick { row: 1, expert: 255, gate: f32::MIN_POSITIVE },
            Pick { row: 1, expert: 0, gate: 1.0 - f32::EPSILON },
        ];
        let rows = [0.5f32, -0.0, 1e-40, f32::MAX, -3.25, std::f32::consts::E];
        let mut bytes = Vec::new();
        encode_experts(17, &picks, &rows, &mut bytes);
        assert_eq!(bytes.len(), EXPERTS_PREFIX + 3 * PICK_LEN + 6 * 4);
        let (mut p, mut r) = (Vec::new(), Vec::new());
        assert_eq!(decode_experts(&bytes, 3, &mut p, &mut r).unwrap(), 17);
        assert_eq!(p, picks);
        assert_eq!(rows.map(f32::to_bits)[..], r.iter().map(|x| x.to_bits()).collect::<Vec<_>>()[..]);
        // Rows must be whole, and every pick must name one.
        assert!(decode_experts(&bytes, 4, &mut p, &mut r).is_err());
        assert!(decode_experts(&bytes, 6, &mut p, &mut r).is_err(), "pick for row 1 of 1");
        assert!(decode_experts(&bytes[..bytes.len() - 4], 3, &mut p, &mut r).is_err());
        assert!(decode_experts(&bytes[..10], 3, &mut p, &mut r).is_err());
        assert!(decode_experts(&bytes[..5], 3, &mut p, &mut r).is_err());
    }

    #[test]
    fn partial_and_move_payloads_round_trip() {
        let values = [1.0f32, -2.5, 0.125, 9.0];
        let mut bytes = Vec::new();
        encode_partial(39, 1234, &values, &mut bytes);
        let mut got = Vec::new();
        assert_eq!(decode_partial(&bytes, 2, &mut got).unwrap(), (39, 1234));
        assert_eq!(got, values);
        assert!(decode_partial(&bytes, 3, &mut got).is_err());
        assert!(decode_partial(&bytes[..PARTIAL_PREFIX], 2, &mut got).is_err(), "no rows");

        encode_move(5, 200, b"weights", &mut bytes);
        assert_eq!(decode_move(&bytes).unwrap(), (5, 200, &b"weights"[..]));
        encode_move(5, 200, &[], &mut bytes);
        assert_eq!(decode_move(&bytes).unwrap(), (5, 200, &[][..]));
        assert!(decode_move(&bytes[..3]).is_err());
    }

    #[test]
    fn expert_split_types_decode() {
        for t in [MsgType::Own, MsgType::Experts, MsgType::Partial, MsgType::Move, MsgType::Release] {
            let h = Header::new(t, 3, 4, 0);
            assert_eq!(Header::decode(&h.encode()).unwrap().msg_type, t);
        }
    }

    #[test]
    fn start_and_done_payloads_round_trip() {
        let mut bytes = Vec::new();
        let mut prompt = Vec::new();
        encode_start(500, &[1, 2, u32::MAX], &mut bytes);
        assert_eq!(decode_start(&bytes, &mut prompt).unwrap(), 500);
        assert_eq!(prompt, [1, 2, u32::MAX]);
        assert!(decode_start(&bytes[..bytes.len() - 1], &mut prompt).is_err());
        assert!(decode_start(&bytes[..7], &mut prompt).is_err());
        encode_start(5, &[], &mut bytes);
        assert!(decode_start(&bytes, &mut prompt).is_err(), "empty prompt");
        for r in [DoneReason::Length, DoneReason::Eos, DoneReason::Dropped] {
            encode_done(r, &mut bytes);
            assert_eq!(decode_done(&bytes).unwrap(), r);
        }
        assert!(decode_done(&[3]).is_err());
        assert!(decode_done(&[]).is_err());
    }

    #[test]
    fn status_report_round_trips_and_skips_what_it_does_not_know() {
        let s = Status {
            footprint: 7 << 30,
            cap: 6 << 30,
            requests: vec![
                RunStatus { request: 3, decoding: false, position: 64, made: 0 },
                RunStatus { request: u64::MAX, decoding: true, position: 212, made: 190 },
            ],
        };
        let mut bytes = Vec::new();
        encode_status(&s, &mut bytes);
        let mut got = Status::default();
        decode_status(&bytes, &mut got).unwrap();
        assert_eq!(got, s);
        // A later worker's extra section is skipped.
        let mut newer = bytes.clone();
        write_section(99, |o| o.extend_from_slice(&[1, 2, 3]), &mut newer);
        decode_status(&newer, &mut got).unwrap();
        assert_eq!(got, s);
        // So are later fields at the end of a request entry.
        let mut wide = Vec::new();
        write_section(
            STATUS_REQUESTS,
            |o| {
                o.extend_from_slice(&1u32.to_le_bytes());
                o.extend_from_slice(&20u16.to_le_bytes());
                o.extend_from_slice(&5u64.to_le_bytes());
                o.extend_from_slice(&[1, 9, 0, 0, 0, 4, 0, 0, 0, 0xAA, 0xAA, 0xAA]);
            },
            &mut wide,
        );
        decode_status(&wide, &mut got).unwrap();
        assert_eq!(got.requests, [RunStatus { request: 5, decoding: true, position: 9, made: 4 }]);
        decode_status(&[], &mut got).unwrap();
        assert_eq!(got, Status::default());
        assert!(decode_status(&bytes[..bytes.len() - 1], &mut got).is_err());
        assert!(decode_status(&bytes[..4], &mut got).is_err());
    }

    #[test]
    fn event_command_result_payloads_round_trip() {
        let mut b = Vec::new();
        let events = [
            (EventKind::PrefillDone, 1234, ""),
            (EventKind::MemoryHigh, u64::MAX, "17179869184"),
            (EventKind::MemoryOk, 0, ""),
            (EventKind::Note, 5, "héllo"),
        ];
        for (kind, value, text) in events {
            encode_event(kind, value, text, &mut b);
            assert_eq!(decode_event(&b).unwrap(), (kind, value, text));
        }
        for kind in [CommandKind::Status, CommandKind::ResetStats, CommandKind::Restart] {
            encode_command(kind, &mut b);
            assert_eq!(decode_command(&b).unwrap(), kind);
            assert_eq!(CommandKind::parse(kind.as_str()), Some(kind));
        }
        assert_eq!(CommandKind::parse("reboot"), None);
        for (ok, message) in [(true, "ok"), (false, "busy: 2 requests"), (true, "")] {
            encode_result(ok, message, &mut b);
            assert_eq!(decode_result(&b).unwrap(), (ok, message));
        }
    }

    #[test]
    fn unknown_kinds_and_short_payloads_are_rejected() {
        let mut b = Vec::new();
        encode_event(EventKind::Note, 7, "x", &mut b);
        b[0] = 9;
        assert!(matches!(decode_event(&b), Err(FrameError::UnknownKind { msg_type: MsgType::Event, kind: 9 })));
        b[0] = 0;
        assert!(matches!(decode_event(&b), Err(FrameError::UnknownKind { kind: 0, .. })));
        encode_event(EventKind::Note, 7, "", &mut b);
        assert!(decode_event(&b[..8]).is_err());
        assert!(decode_event(&[]).is_err());
        encode_event(EventKind::Note, 7, "x", &mut b);
        *b.last_mut().unwrap() = 0xFF;
        assert!(decode_event(&b).is_err());
        assert!(matches!(decode_command(&[4]), Err(FrameError::UnknownKind { msg_type: MsgType::Command, kind: 4 })));
        assert!(decode_command(&[]).is_err());
        assert!(decode_result(&[]).is_err());
        assert!(decode_result(&[2, b'x']).is_err());
        assert!(decode_result(&[1, 0xFF]).is_err());
    }

    #[test]
    fn request_split_types_decode() {
        for t in [MsgType::Start, MsgType::Done, MsgType::Event, MsgType::Command, MsgType::Result] {
            let h = Header::new(t, 3, 4, 0);
            assert_eq!(Header::decode(&h.encode()).unwrap().msg_type, t);
        }
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
        b[7] = 0x40;
        assert!(matches!(Header::decode(&b), Err(FrameError::BadFlags(0x40))));
    }

    #[test]
    fn chunked_messages_split_and_rejoin() {
        let head = b"head".to_vec();
        let rows: Vec<u8> = (0..1000u32).map(|i| i as u8).collect();
        for max in [16, 64, 1004, 2000] {
            let h = Header::new(MsgType::Hidden, 3, 9, head.len() + rows.len()).with_flags(FLAG_EVERY_TOKEN);
            let mut wire = Vec::new();
            write_message(&mut wire, &h, &head, &rows, max).unwrap();
            let mut r = wire.as_slice();
            let (mut payload, mut joined, mut frames) = (Vec::new(), Vec::new(), 0);
            loop {
                let f = read_frame(&mut r, &mut payload).unwrap();
                assert!(f.payload_len as usize <= max && (f.request_id, f.position) == (3, 9));
                assert_eq!(f.flags & FLAG_EVERY_TOKEN, FLAG_EVERY_TOKEN);
                // The first of several chunks starts with the whole length.
                let body = if frames == 0 && max < 1004 { &payload[4..] } else { &payload[..] };
                joined.extend_from_slice(body);
                frames += 1;
                if f.flags & FLAG_MORE == 0 {
                    break;
                }
            }
            assert!(r.is_empty());
            assert_eq!(joined, [&head[..], &rows[..]].concat(), "max {max}");
            assert_eq!(frames > 1, max < 1004, "max {max}");
        }
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
