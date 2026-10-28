//! Persistent TCP connection between the controller and a worker (NET-3, NET-4, NET-7).

use std::io::{self, BufReader, BufWriter, Read, Write};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::time::Duration;

use crate::frame::{
    bad_payload, f32_bytes, f32_bytes_mut, read_frame, write_frame, write_stages, FrameError, Header, MsgType,
    HEADER_LEN, PROTOCOL_VERSION,
};

/// Bytes of each connection's read and write buffers.
pub const BUFFER_BYTES: usize = 256 * 1024;

pub struct Connection {
    reader: BufReader<TcpStream>,
    writer: BufWriter<TcpStream>,
    peer_name: String,
    peer_addr: SocketAddr,
    /// Scratch for a hidden frame's stage times.
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
        let peer_addr = stream.peer_addr()?;
        let mut conn = Connection {
            reader: BufReader::with_capacity(BUFFER_BYTES, stream.try_clone()?),
            writer: BufWriter::with_capacity(BUFFER_BYTES, stream),
            peer_name: String::new(),
            peer_addr,
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
        Ok(conn)
    }

    pub fn peer_name(&self) -> &str {
        &self.peer_name
    }

    pub fn peer_addr(&self) -> SocketAddr {
        self.peer_addr
    }

    /// Tells the peer this side sends no more: it reads end of stream once it has read what
    /// was sent. Needed after [`Connection::split`], since the socket stays open while the
    /// other half lives.
    pub fn close_send(&mut self) {
        let _ = self.writer.flush();
        let _ = self.writer.get_ref().shutdown(std::net::Shutdown::Write);
    }

    /// Splits into a receiving half and a sending half over the same socket, so one thread can
    /// read while another writes. Use each half for its own direction only: the receiving half
    /// keeps the read buffer, the sending half the write buffer.
    pub fn split(self) -> io::Result<(Connection, Connection)> {
        let stream = self.reader.get_ref();
        let recv = Connection {
            writer: BufWriter::with_capacity(0, stream.try_clone()?),
            reader: self.reader,
            peer_name: self.peer_name.clone(),
            peer_addr: self.peer_addr,
            prefix: Vec::new(),
        };
        let send = Connection {
            reader: BufReader::with_capacity(0, recv.reader.get_ref().try_clone()?),
            writer: self.writer,
            peer_name: self.peer_name,
            peer_addr: self.peer_addr,
            prefix: self.prefix,
        };
        Ok((recv, send))
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
        write_frame(&mut self.writer, &h, payload)?;
        self.writer.flush()
    }

    /// Sends a `Hidden` frame (the [`crate::encode_hidden`] payload) straight from `values`,
    /// without building the payload first (NET-6). Large states bypass the write buffer too.
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
        self.writer.write_all(&h.encode())?;
        self.writer.write_all(&self.prefix)?;
        self.writer.write_all(bytes)?;
        self.writer.flush()
    }

    /// Reads the next frame's header; its payload must be read next, with
    /// [`Connection::recv_payload`] or [`Connection::recv_hidden`].
    pub fn recv_header(&mut self) -> Result<Header, FrameError> {
        let mut hb = [0u8; HEADER_LEN];
        self.reader.read_exact(&mut hb)?;
        Header::decode(&hb)
    }

    pub fn recv_payload(&mut self, h: &Header, payload: &mut Vec<u8>) -> Result<(), FrameError> {
        payload.resize(h.payload_len as usize, 0);
        self.reader.read_exact(payload)?;
        Ok(())
    }

    /// Reads a `Hidden` frame's payload straight into `values` (whole rows of `hidden_dim`)
    /// and its stage times into `stage_us`, as [`crate::decode_hidden`] would; returns the
    /// rows. A malformed payload is still read in full, so the next frame can follow.
    pub fn recv_hidden(
        &mut self,
        h: &Header,
        hidden_dim: usize,
        values: &mut Vec<f32>,
        stage_us: &mut Vec<u32>,
    ) -> Result<usize, FrameError> {
        let len = h.payload_len as usize;
        let mut n = [0u8; 2];
        if len < 2 {
            self.skip(len)?;
            return Err(bad_payload("stage times truncated".into()));
        }
        self.reader.read_exact(&mut n)?;
        let stages = 4 * u16::from_le_bytes(n) as usize;
        let row = hidden_dim * 4;
        let Some(rest) = (len - 2).checked_sub(stages) else {
            self.skip(len - 2)?;
            return Err(bad_payload("stage times truncated".into()));
        };
        self.prefix.resize(stages, 0);
        self.reader.read_exact(&mut self.prefix)?;
        stage_us.clear();
        stage_us.extend(self.prefix.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())));
        if row == 0 || rest == 0 || rest % row != 0 {
            self.skip(rest)?;
            return Err(bad_payload(format!("hidden of {rest} bytes is not whole rows of {row}")));
        }
        // Grows without zeroing what is already there; every byte is overwritten below.
        values.truncate(rest / 4);
        values.resize(rest / 4, 0.0);
        self.reader.read_exact(f32_bytes_mut(values))?;
        Ok(rest / row)
    }

    fn skip(&mut self, n: usize) -> io::Result<()> {
        io::copy(&mut (&mut self.reader).take(n as u64), &mut io::sink()).map(|_| ())
    }

    /// Receives one frame into `payload`. A remote `Error` frame is surfaced as `FrameError::Remote`.
    pub fn recv(&mut self, payload: &mut Vec<u8>) -> Result<Header, FrameError> {
        let h = self.recv_frame(payload)?;
        if h.msg_type == MsgType::Error {
            return Err(FrameError::Remote(String::from_utf8_lossy(payload).into()));
        }
        Ok(h)
    }

    /// Receives one frame of any type, `Error` included, so its request id and position are kept.
    pub fn recv_frame(&mut self, payload: &mut Vec<u8>) -> Result<Header, FrameError> {
        read_frame(&mut self.reader, payload)
    }
}
