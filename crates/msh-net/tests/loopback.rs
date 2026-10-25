use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;
use std::time::Duration;

use msh_net::frame::{Header, MsgType, HEADER_LEN};
use msh_net::{probe, Connection, FrameError};

#[test]
fn echo_round_trip_over_loopback() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    thread::spawn(move || probe::serve(listener, "server"));

    let mut conn = Connection::connect(addr, "client", Duration::from_secs(2)).unwrap();
    assert_eq!(conn.peer_name(), "server");
    for size in [4, 4096, 1 << 20] {
        assert_eq!(probe::measure(&mut conn, size, 2, 10).unwrap().len(), 10);
    }
}

#[test]
fn version_mismatch_is_refused_at_connect() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let server = thread::spawn(move || {
        let (s, _) = listener.accept().unwrap();
        Connection::accept(s, "server").err()
    });

    // A peer speaking protocol version 99.
    let mut s = TcpStream::connect(addr).unwrap();
    let mut h = Header::new(MsgType::Hello, 0, 0, 3);
    h.version = 99;
    s.write_all(&h.encode()).unwrap();
    s.write_all(b"old").unwrap();

    let err = server.join().unwrap().expect("server must refuse");
    assert!(matches!(err, FrameError::VersionMismatch { local: 1, remote: 99 }));

    // The server's reply is its Hello followed by an Error frame.
    let mut buf = Vec::new();
    s.read_to_end(&mut buf).unwrap();
    let hello_len = HEADER_LEN + "server".len();
    let err_frame = Header::decode(buf[hello_len..hello_len + HEADER_LEN].try_into().unwrap()).unwrap();
    assert_eq!(err_frame.msg_type, MsgType::Error);
}
