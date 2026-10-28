//! Persistent TCP connection between the controller and a worker (NET-3, NET-4, NET-7).
//!
//! A message larger than the link's `max_chunk` goes in chunks (see [`crate::frame`]). The
//! receiving side joins them per request: a message's chunks are read straight into the
//! buffer it ends up in, the rows of `Hidden`, `Experts` and `Partial` into f32s (NET-6), and
//! a message is handed out once its last chunk is in, so chunks of several requests may
//! interleave on one link. A malformed chunk sequence fails only its request: it comes out as
//! an `Error` frame for that request id.

use std::collections::{HashMap, VecDeque};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::time::Duration;

use crate::frame::{
    bad_payload, check_rows, f32_bytes, f32_bytes_mut, read_frame, read_picks, read_stages, rows_base, rows_prefix,
    write_experts_prefix, write_message, write_stages, FrameError, Header, MsgType, Pick, DEFAULT_MAX_CHUNK,
    EXPERTS_PREFIX, FLAG_MORE, HEADER_LEN, MAX_PAYLOAD, PARTIAL_PREFIX, PICK_LEN, PROTOCOL_VERSION,
};
use crate::writer::{self, Writer};

/// Bytes of each connection's read buffer, and of the write buffer of one that is not split.
/// Chunk data beyond what is buffered is read straight into its destination.
pub const BUFFER_BYTES: usize = 32 * 1024;
/// How long the `Hello` exchange may take once connected.
const HELLO_TIMEOUT: Duration = Duration::from_secs(10);

/// A complete message off the wire.
#[derive(Debug)]
pub struct Message {
    /// The whole payload's length; [`FLAG_MORE`] cleared.
    pub header: Header,
    /// The payload before its rows: for `Hidden`, `Experts` and `Partial` what comes before the
    /// f32 rows, for every other type the whole payload.
    pub head: Vec<u8>,
    /// `Hidden`, `Experts` and `Partial`: the rows, read straight off the wire.
    pub rows: Vec<f32>,
    /// Bytes of `rows` received; a malformed payload may end inside a value.
    rows_len: usize,
}

impl Message {
    fn new(header: Header) -> Message {
        Message { header, head: Vec::new(), rows: Vec::new(), rows_len: 0 }
    }

    /// The rows' bytes as they came.
    pub fn rows_bytes(&self) -> &[u8] {
        &f32_bytes(&self.rows)[..self.rows_len]
    }

    /// The rows; their byte count need not be a whole number of f32s.
    pub fn rows_len(&self) -> usize {
        self.rows_len
    }

    /// The whole payload in one buffer (a copy only for a message with rows).
    pub fn into_payload(mut self) -> Vec<u8> {
        if self.rows_len > 0 {
            self.head.extend_from_slice(&f32_bytes(&self.rows)[..self.rows_len]);
            writer::give_rows(std::mem::take(&mut self.rows));
        }
        self.head
    }
}

/// A message whose chunks are coming in.
struct Assembly {
    msg: Message,
    total: usize,
    got: usize,
    /// Bytes before the rows, once known.
    prefix: Option<usize>,
    /// Why the message is refused; its remaining chunks are skipped.
    broken: Option<String>,
}

impl Assembly {
    fn new(h: Header, total: usize) -> Assembly {
        let mut header = h;
        header.flags &= !FLAG_MORE;
        header.payload_len = total.min(u32::MAX as usize) as u32;
        let mut a = Assembly { msg: Message::new(header), total, got: 0, prefix: None, broken: None };
        if total > MAX_PAYLOAD {
            a.broken = Some(format!("chunked payload of {total} bytes exceeds the limit of {MAX_PAYLOAD}"));
        } else if rows_base(h.msg_type).is_none() {
            a.msg.head.reserve_exact(total);
        }
        a
    }

    /// Reads the next `n` payload bytes from `r` into place.
    fn feed(&mut self, r: &mut impl Read, mut n: usize) -> io::Result<()> {
        if self.broken.is_none() && self.got + n > self.total {
            self.broken = Some(format!("chunks carry more than the {} bytes announced", self.total));
        }
        self.got += n;
        if self.broken.is_some() {
            return skip(r, n);
        }
        let t = self.msg.header.msg_type;
        let m = &mut self.msg;
        while n > 0 {
            let want = match (self.prefix, rows_base(t)) {
                (None, None) => n,
                (None, Some(base)) if m.head.len() < base => base - m.head.len(),
                (None, Some(_)) => {
                    let p = rows_prefix(t, &m.head);
                    self.prefix = Some(p);
                    // Grows without zeroing what is already there; every byte is overwritten.
                    m.rows = writer::take_rows();
                    let len = self.total.saturating_sub(p).div_ceil(4);
                    m.rows.truncate(len);
                    m.rows.resize(len, 0.0);
                    continue;
                }
                (Some(p), _) if m.head.len() < p => p - m.head.len(),
                (Some(_), _) => {
                    let at = m.rows_len;
                    r.read_exact(&mut f32_bytes_mut(&mut m.rows)[at..at + n])?;
                    m.rows_len += n;
                    return Ok(());
                }
            };
            let k = want.min(n);
            let at = m.head.len();
            m.head.resize(at + k, 0);
            r.read_exact(&mut m.head[at..])?;
            n -= k;
        }
        Ok(())
    }

    /// The message, or an `Error` for its request when it was refused.
    fn finish(self, peer: &str) -> Message {
        let why = match self.broken {
            Some(why) => why,
            None if self.got != self.total => format!("chunks ended at {} of {} bytes", self.got, self.total),
            None => return self.msg,
        };
        writer::give_rows(self.msg.rows);
        refused(&self.msg.header, peer, &why)
    }
}

/// An `Error` message for `h`'s request: the message was refused.
fn refused(h: &Header, peer: &str, why: &str) -> Message {
    let text = format!("{peer}: bad chunked {:?} at position {}: {why}", h.msg_type, h.position).into_bytes();
    let mut m = Message::new(Header::new(MsgType::Error, h.request_id, h.position, text.len()));
    m.head = text;
    m
}

fn skip(r: &mut impl Read, n: usize) -> io::Result<()> {
    let copied = io::copy(&mut r.take(n as u64), &mut io::sink())?;
    if copied < n as u64 {
        return Err(io::ErrorKind::UnexpectedEof.into());
    }
    Ok(())
}

/// Whether frames of this type stand apart from request state: a ping's id is a sequence
/// number, not a request.
fn is_ping(t: MsgType) -> bool {
    matches!(t, MsgType::Ping | MsgType::Pong)
}

/// Turns on TCP keepalive, a backstop against a peer that vanished without closing.
fn keepalive(s: &TcpStream) {
    use std::os::fd::AsRawFd;
    let fd = s.as_raw_fd();
    let set = |level, name, v: libc::c_int| {
        // SAFETY: `fd` is an open socket and the value a c_int, as setsockopt expects.
        unsafe {
            libc::setsockopt(fd, level, name, (&v as *const libc::c_int).cast(), std::mem::size_of::<libc::c_int>() as libc::socklen_t)
        }
    };
    set(libc::SOL_SOCKET, libc::SO_KEEPALIVE, 1);
    // Probe after 10 s idle, every 5 s, 3 times: a dead peer shows within about 25 s.
    #[cfg(target_os = "macos")]
    set(libc::IPPROTO_TCP, libc::TCP_KEEPALIVE, 10);
    #[cfg(target_os = "linux")]
    set(libc::IPPROTO_TCP, libc::TCP_KEEPIDLE, 10);
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    {
        set(libc::IPPROTO_TCP, libc::TCP_KEEPINTVL, 5);
        set(libc::IPPROTO_TCP, libc::TCP_KEEPCNT, 3);
    }
}

pub struct Connection {
    reader: BufReader<TcpStream>,
    writer: BufWriter<TcpStream>,
    peer_name: String,
    peer_addr: SocketAddr,
    max_chunk: usize,
    /// Messages coming in chunks, by request id (pings apart).
    open: HashMap<(u64, bool), Assembly>,
    /// Complete messages not handed out yet, oldest first.
    done: VecDeque<Message>,
    /// The message [`Connection::recv_header`] returned the header of.
    current: Option<Message>,
    /// Scratch for a frame's prefix.
    prefix: Vec<u8>,
}

impl Connection {
    /// Connects and performs the `Hello` exchange. Works the same over Ethernet or a
    /// Thunderbolt bridge: the OS picks the interface from the destination address.
    pub fn connect<A: ToSocketAddrs>(
        addr: A,
        local_name: &str,
        timeout: Duration,
    ) -> Result<Self, FrameError> {
        let mut last_err = None;
        for sa in addr.to_socket_addrs()? {
            match TcpStream::connect_timeout(&sa, timeout) {
                Ok(s) => return Self::handshake(s, local_name),
                Err(e) => last_err = Some(e),
            }
        }
        Err(last_err
            .unwrap_or_else(|| io::Error::new(io::ErrorKind::NotFound, "address resolved to nothing"))
            .into())
    }

    /// Wraps an accepted stream and performs the `Hello` exchange.
    pub fn accept(stream: TcpStream, local_name: &str) -> Result<Self, FrameError> {
        Self::handshake(stream, local_name)
    }

    fn handshake(stream: TcpStream, local_name: &str) -> Result<Self, FrameError> {
        stream.set_nodelay(true)?;
        keepalive(&stream);
        // A peer that accepts but never says `Hello` (a frozen process) fails the exchange.
        stream.set_read_timeout(Some(HELLO_TIMEOUT))?;
        let peer_addr = stream.peer_addr()?;
        let mut conn = Connection {
            reader: BufReader::with_capacity(BUFFER_BYTES, stream.try_clone()?),
            writer: BufWriter::with_capacity(BUFFER_BYTES, stream),
            peer_name: String::new(),
            peer_addr,
            max_chunk: DEFAULT_MAX_CHUNK,
            open: HashMap::new(),
            done: VecDeque::new(),
            current: None,
            prefix: Vec::new(),
        };
        conn.send(MsgType::Hello, 0, 0, local_name.as_bytes())?;

        let mut payload = Vec::new();
        let h = read_frame(&mut conn.reader, &mut payload)?;
        if h.version != PROTOCOL_VERSION {
            let msg = format!("protocol version {} not supported (want {PROTOCOL_VERSION})", h.version);
            let _ = conn.send(MsgType::Error, 0, 0, msg.as_bytes());
            return Err(FrameError::VersionMismatch { local: PROTOCOL_VERSION, remote: h.version });
        }
        match h.msg_type {
            MsgType::Hello => {}
            MsgType::Error => return Err(FrameError::Remote(String::from_utf8_lossy(&payload).into())),
            got => return Err(FrameError::Unexpected { expected: MsgType::Hello, got }),
        }
        conn.peer_name = String::from_utf8_lossy(&payload).into_owned();
        conn.reader.get_ref().set_read_timeout(None)?;
        Ok(conn)
    }

    pub fn peer_name(&self) -> &str {
        &self.peer_name
    }

    pub fn peer_addr(&self) -> SocketAddr {
        self.peer_addr
    }

    /// Sets the largest frame payload this side sends; bigger messages go in chunks.
    pub fn with_max_chunk(mut self, bytes: usize) -> Self {
        self.max_chunk = bytes.clamp(64, MAX_PAYLOAD);
        self
    }

    pub fn max_chunk(&self) -> usize {
        self.max_chunk
    }

    /// Makes a read that waits longer than `timeout` fail (with `WouldBlock` or `TimedOut`):
    /// a peer that should ping and goes quiet is gone. `None` waits forever.
    pub fn set_read_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
        self.reader.get_ref().set_read_timeout(timeout)
    }

    /// Tells the peer this side sends no more: it reads end of stream once it has read what
    /// was sent. Needed after [`Connection::split`], since the socket stays open while the
    /// other half lives.
    pub fn close_send(&mut self) {
        let _ = self.writer.flush();
        let _ = self.writer.get_ref().shutdown(std::net::Shutdown::Write);
    }

    /// Splits into a receiving half and a [`Writer`], a thread that owns the sending side and
    /// sends small frames ahead of chunked ones. Use the receiving half to receive only.
    pub fn split(mut self) -> io::Result<(Connection, Writer)> {
        self.writer.flush()?;
        let stream = self.writer.get_ref().try_clone()?;
        let w = Writer::start(stream, self.peer_name.clone(), self.peer_addr, self.max_chunk)?;
        Ok((self, w))
    }

    /// Sends one frame and flushes it.
    pub fn send(
        &mut self,
        msg_type: MsgType,
        request_id: u64,
        position: u32,
        payload: &[u8],
    ) -> io::Result<()> {
        self.send_flagged(msg_type, 0, request_id, position, payload)
    }

    /// [`Connection::send`] with header flags.
    pub fn send_flagged(
        &mut self,
        msg_type: MsgType,
        flags: u8,
        request_id: u64,
        position: u32,
        payload: &[u8],
    ) -> io::Result<()> {
        let h = Header::new(msg_type, request_id, position, payload.len()).with_flags(flags);
        write_message(&mut self.writer, &h, payload, &[], self.max_chunk)?;
        self.writer.flush()
    }

    /// Sends a `Hidden` frame (the [`crate::encode_hidden`] payload) straight from `values`,
    /// without building the payload first (NET-6).
    pub fn send_hidden(
        &mut self,
        flags: u8,
        request_id: u64,
        position: u32,
        stage_us: &[u32],
        values: &[f32],
    ) -> io::Result<()> {
        self.prefix.clear();
        write_stages(stage_us, &mut self.prefix);
        let bytes = f32_bytes(values);
        let h = Header::new(MsgType::Hidden, request_id, position, self.prefix.len() + bytes.len()).with_flags(flags);
        write_message(&mut self.writer, &h, &self.prefix, bytes, self.max_chunk)?;
        self.writer.flush()
    }

    /// Sends an `Experts` frame (the [`crate::encode_experts`] payload), the rows straight from
    /// `rows` (NET-6).
    pub fn send_experts(
        &mut self,
        request_id: u64,
        position: u32,
        layer: u16,
        picks: &[Pick],
        rows: &[f32],
    ) -> io::Result<()> {
        self.prefix.clear();
        write_experts_prefix(layer, picks, &mut self.prefix);
        let bytes = f32_bytes(rows);
        let h = Header::new(MsgType::Experts, request_id, position, self.prefix.len() + bytes.len());
        write_message(&mut self.writer, &h, &self.prefix, bytes, self.max_chunk)?;
        self.writer.flush()
    }

    /// Sends a `Partial` frame (the [`crate::encode_partial`] payload) straight from `values`.
    pub fn send_partial(
        &mut self,
        request_id: u64,
        position: u32,
        layer: u16,
        compute_us: u32,
        values: &[f32],
    ) -> io::Result<()> {
        self.prefix.clear();
        self.prefix.extend_from_slice(&layer.to_le_bytes());
        self.prefix.extend_from_slice(&compute_us.to_le_bytes());
        let bytes = f32_bytes(values);
        let h = Header::new(MsgType::Partial, request_id, position, PARTIAL_PREFIX + bytes.len());
        write_message(&mut self.writer, &h, &self.prefix, bytes, self.max_chunk)?;
        self.writer.flush()
    }

    /// Reads frames until a message is complete and returns its header (the whole payload's
    /// length); its payload must be read next, with [`Connection::recv_payload`],
    /// [`Connection::recv_hidden`], [`Connection::recv_experts`] or
    /// [`Connection::recv_partial`]. A refused chunked message comes as an `Error` for its
    /// request.
    pub fn recv_header(&mut self) -> Result<Header, FrameError> {
        if let Some(m) = self.current.take() {
            writer::give_rows(m.rows);
        }
        while self.done.is_empty() {
            self.read_one()?;
        }
        let m = self.done.pop_front().unwrap();
        let h = m.header;
        self.current = Some(m);
        Ok(h)
    }

    /// Reads one frame: a whole message, or a chunk of one.
    fn read_one(&mut self) -> Result<(), FrameError> {
        let mut hb = [0u8; HEADER_LEN];
        if let Err(e) = self.reader.read_exact(&mut hb) {
            if let (io::ErrorKind::UnexpectedEof, Some(a)) = (e.kind(), self.open.values().next()) {
                let h = a.msg.header;
                return Err(FrameError::Incomplete { msg_type: h.msg_type, request_id: h.request_id });
            }
            return Err(e.into());
        }
        let h = Header::decode(&hb)?;
        let more = h.flags & FLAG_MORE != 0;
        let mut len = h.payload_len as usize;
        let key = (h.request_id, is_ping(h.msg_type));
        if let Some(a) = self.open.get_mut(&key) {
            let o = a.msg.header;
            if (o.msg_type, o.position) == (h.msg_type, h.position) {
                a.feed(&mut self.reader, len)?;
                if !more {
                    let a = self.open.remove(&key).unwrap();
                    self.done.push_back(a.finish(&self.peer_name));
                }
                return Ok(());
            }
            // Another message of the request while one is open: the open one is refused.
            let mut a = self.open.remove(&key).unwrap();
            a.broken = Some(format!("interrupted by a {:?} at position {}", h.msg_type, h.position));
            self.done.push_back(a.finish(&self.peer_name));
        }
        if !more {
            let mut a = Assembly::new(h, len);
            a.feed(&mut self.reader, len)?;
            self.done.push_back(a.finish(&self.peer_name));
            return Ok(());
        }
        // The first chunk starts with the whole payload's length.
        let mut a = if len < 4 {
            let mut a = Assembly::new(h, 0);
            a.broken = Some("first chunk without the payload length".into());
            a
        } else {
            let mut n = [0u8; 4];
            self.reader.read_exact(&mut n)?;
            len -= 4;
            Assembly::new(h, u32::from_le_bytes(n) as usize)
        };
        a.feed(&mut self.reader, len)?;
        self.open.insert(key, a);
        Ok(())
    }

    fn take_current(&mut self) -> Result<Message, FrameError> {
        self.current.take().ok_or_else(|| bad_payload("no message read: call recv_header first".into()))
    }

    pub fn recv_payload(&mut self, _h: &Header, payload: &mut Vec<u8>) -> Result<(), FrameError> {
        *payload = self.take_current()?.into_payload();
        Ok(())
    }

    /// Reads a `Hidden` frame's payload: the rows into `values` (whole rows of `hidden_dim`),
    /// without a copy (NET-6), and its stage times into `stage_us`, as
    /// [`crate::decode_hidden`] would; returns the rows. `values` is swapped with the buffer
    /// the rows were read into.
    pub fn recv_hidden(
        &mut self,
        _h: &Header,
        hidden_dim: usize,
        values: &mut Vec<f32>,
        stage_us: &mut Vec<u32>,
    ) -> Result<usize, FrameError> {
        let mut m = self.take_current()?;
        if !read_stages(&m.head, stage_us)?.is_empty() {
            return Err(bad_payload("hidden payload without rows".into()));
        }
        let (rest, row) = (m.rows_len, hidden_dim * 4);
        if row == 0 || rest == 0 || rest % row != 0 {
            writer::give_rows(m.rows);
            return Err(bad_payload(format!("hidden of {rest} bytes is not whole rows of {row}")));
        }
        m.rows.truncate(rest / 4);
        std::mem::swap(values, &mut m.rows);
        writer::give_rows(m.rows);
        Ok(rest / row)
    }

    /// Reads an `Experts` frame's payload: the picks into `picks` and the rows into `rows`, as
    /// [`crate::decode_experts`] would, without a copy; returns the layer.
    pub fn recv_experts(
        &mut self,
        _h: &Header,
        hidden_dim: usize,
        picks: &mut Vec<Pick>,
        rows: &mut Vec<f32>,
    ) -> Result<u16, FrameError> {
        let mut m = self.take_current()?;
        let short = || bad_payload("experts payload too short".into());
        let head = m.head.get(..EXPERTS_PREFIX).ok_or_else(short)?;
        let n = u32::from_le_bytes(head[2..6].try_into().unwrap()) as usize;
        let layer = u16::from_le_bytes([head[0], head[1]]);
        if m.head.len() != EXPERTS_PREFIX + PICK_LEN * n {
            return Err(short());
        }
        read_picks(&m.head[EXPERTS_PREFIX..], picks);
        if let Err(e) = check_rows(m.rows_len, hidden_dim, picks) {
            writer::give_rows(m.rows);
            return Err(e);
        }
        m.rows.truncate(m.rows_len / 4);
        std::mem::swap(rows, &mut m.rows);
        writer::give_rows(m.rows);
        Ok(layer)
    }

    /// Reads a `Partial` frame's payload into `values`, as [`crate::decode_partial`] would,
    /// without a copy; returns the layer and the worker's compute time.
    pub fn recv_partial(
        &mut self,
        _h: &Header,
        hidden_dim: usize,
        values: &mut Vec<f32>,
    ) -> Result<(u16, u32), FrameError> {
        let mut m = self.take_current()?;
        let head = m.head.get(..PARTIAL_PREFIX).ok_or_else(|| bad_payload("partial payload too short".into()))?;
        let got = (u16::from_le_bytes([head[0], head[1]]), u32::from_le_bytes(head[2..6].try_into().unwrap()));
        if let Err(e) = check_rows(m.rows_len, hidden_dim, &[]) {
            writer::give_rows(m.rows);
            return Err(e);
        }
        m.rows.truncate(m.rows_len / 4);
        std::mem::swap(values, &mut m.rows);
        writer::give_rows(m.rows);
        Ok(got)
    }

    /// Receives one whole message, rows apart from the rest, to pass on as it is.
    pub fn recv_message(&mut self) -> Result<Message, FrameError> {
        self.recv_header()?;
        self.take_current()
    }

    /// Receives one frame into `payload`. A remote `Error` frame is surfaced as `FrameError::Remote`.
    pub fn recv(&mut self, payload: &mut Vec<u8>) -> Result<Header, FrameError> {
        let h = self.recv_frame(payload)?;
        if h.msg_type == MsgType::Error {
            return Err(FrameError::Remote(String::from_utf8_lossy(payload).into()));
        }
        Ok(h)
    }

    /// Receives one message of any type, `Error` included, so its request id and position are
    /// kept; its whole payload goes into `payload`.
    pub fn recv_frame(&mut self, payload: &mut Vec<u8>) -> Result<Header, FrameError> {
        let h = self.recv_header()?;
        self.recv_payload(&h, payload)?;
        Ok(h)
    }
}
