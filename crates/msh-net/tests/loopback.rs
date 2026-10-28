use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;
use std::time::Duration;

use msh_net::frame::{Header, MsgType, HEADER_LEN};
use msh_net::{probe, Connection, FrameError, PROTOCOL_VERSION};

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
    assert!(matches!(err, FrameError::VersionMismatch { local: PROTOCOL_VERSION, remote: 99 }));

    // The server's reply is its Hello followed by an Error frame.
    let mut buf = Vec::new();
    s.read_to_end(&mut buf).unwrap();
    let hello_len = HEADER_LEN + "server".len();
    let err_frame = Header::decode(buf[hello_len..hello_len + HEADER_LEN].try_into().unwrap()).unwrap();
    assert_eq!(err_frame.msg_type, MsgType::Error);
}

/// Two connected ends over loopback.
fn pair() -> (Connection, Connection) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let server = thread::spawn(move || Connection::accept(listener.accept().unwrap().0, "b").unwrap());
    let a = Connection::connect(addr, "a", Duration::from_secs(2)).unwrap();
    (a, server.join().unwrap())
}

#[test]
fn direct_hidden_frames_match_encoded_ones() {
    use msh_net::{decode_hidden, encode_hidden, FLAG_EVERY_TOKEN};
    let (mut a, mut b) = pair();
    let dim = 8;
    let values: Vec<f32> = (0..3 * dim).map(|i| i as f32 * 0.37 - 2.0).collect();
    let stages = [11, 22];

    // Sent directly, read the old way.
    a.send_hidden(FLAG_EVERY_TOKEN, 7, 40, &stages, &values).unwrap();
    let mut buf = Vec::new();
    let h = b.recv_frame(&mut buf).unwrap();
    assert_eq!((h.msg_type, h.flags, h.request_id, h.position), (MsgType::Hidden, FLAG_EVERY_TOKEN, 7, 40));
    let mut encoded = Vec::new();
    encode_hidden(&stages, &values, &mut encoded);
    assert_eq!(buf, encoded);

    // Sent the old way, read directly.
    a.send(MsgType::Hidden, 8, 41, &encoded).unwrap();
    let (mut got, mut got_stages) = (vec![9.0; 100], Vec::new());
    let h = b.recv_header().unwrap();
    assert_eq!(b.recv_hidden(&h, dim, &mut got, &mut got_stages).unwrap(), 3);
    assert_eq!((got.as_slice(), got_stages.as_slice()), (values.as_slice(), &stages[..]));
    let (mut old, mut old_stages) = (Vec::new(), Vec::new());
    decode_hidden(&encoded, dim, &mut old, &mut old_stages).unwrap();
    assert_eq!(got, old);
}

#[test]
fn malformed_hidden_payload_is_skipped() {
    let (mut a, mut b) = pair();
    // 2 stage-count bytes, then 5 bytes: not whole rows of 8 floats.
    a.send(MsgType::Hidden, 1, 0, &[0, 0, 1, 2, 3, 4, 5]).unwrap();
    a.send(MsgType::Ping, 2, 0, b"next").unwrap();
    let h = b.recv_header().unwrap();
    assert!(b.recv_hidden(&h, 8, &mut Vec::new(), &mut Vec::new()).is_err());
    let mut buf = Vec::new();
    let h = b.recv_frame(&mut buf).unwrap();
    assert_eq!((h.msg_type, h.request_id, buf.as_slice()), (MsgType::Ping, 2, &b"next"[..]));
}

/// `cargo test -p msh-net --release --test loopback -- --ignored --nocapture`
#[test]
#[ignore]
fn hidden_send_paths_timing() {
    use msh_net::{decode_hidden, encode_hidden};
    use std::time::Instant;
    let dim = 2048;
    for rows in [1usize, 64] {
        let values: Vec<f32> = (0..rows * dim).map(|i| i as f32).collect();
        let n = if rows == 1 { 20_000 } else { 2_000 };
        for direct in [false, true] {
            let (mut a, mut b) = pair();
            let reader = thread::spawn(move || {
                let (mut buf, mut v, mut st) = (Vec::new(), Vec::new(), Vec::new());
                for _ in 0..n {
                    if direct {
                        let h = b.recv_header().unwrap();
                        b.recv_hidden(&h, dim, &mut v, &mut st).unwrap();
                    } else {
                        b.recv_frame(&mut buf).unwrap();
                        decode_hidden(&buf, dim, &mut v, &mut st).unwrap();
                    }
                }
            });
            let mut enc = Vec::new();
            let t0 = Instant::now();
            for i in 0..n {
                if direct {
                    a.send_hidden(0, i, 0, &[1, 2], &values).unwrap();
                } else {
                    encode_hidden(&[1, 2], &values, &mut enc);
                    a.send(MsgType::Hidden, i, 0, &enc).unwrap();
                }
            }
            reader.join().unwrap();
            let per = t0.elapsed().as_secs_f64() * 1e6 / n as f64;
            println!("{rows:>2} rows ({:>3} KiB), {}: {per:.1} µs per frame", rows * dim * 4 / 1024, if direct { "direct " } else { "encoded" });
        }
    }
}
