//! Persistent TCP connection to a neighbouring node (NET-3, NET-4, NET-7).

use std::io::{self, BufReader, BufWriter, Write};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::time::Duration;

use crate::frame::{read_frame, write_frame, FrameError, Header, MsgType, PROTOCOL_VERSION};

pub struct Connection {
    reader: BufReader<TcpStream>,
    writer: BufWriter<TcpStream>,
    peer_name: String,
    peer_addr: SocketAddr,
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
            reader: BufReader::with_capacity(256 * 1024, stream.try_clone()?),
            writer: BufWriter::with_capacity(256 * 1024, stream),
            peer_name: String::new(),
            peer_addr,
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

    /// Sends one frame and flushes it.
    pub fn send(
        &mut self,
        msg_type: MsgType,
        request_id: u64,
        position: u32,
        payload: &[u8],
    ) -> io::Result<()> {
        let h = Header::new(msg_type, request_id, position, payload.len());
        write_frame(&mut self.writer, &h, payload)?;
        self.writer.flush()
    }

    /// Receives one frame into `payload`. A remote `Error` frame is surfaced as `FrameError::Remote`.
    pub fn recv(&mut self, payload: &mut Vec<u8>) -> Result<Header, FrameError> {
        let h = read_frame(&mut self.reader, payload)?;
        if h.msg_type == MsgType::Error {
            return Err(FrameError::Remote(String::from_utf8_lossy(payload).into()));
        }
        Ok(h)
    }
}
