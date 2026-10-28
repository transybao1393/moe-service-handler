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
fn direct_experts_and_partial_frames_match_encoded_ones() {
    use msh_net::{decode_experts, decode_partial, encode_experts, encode_partial, Pick};
    let (mut a, mut b) = pair();
    let dim = 4;
    let rows: Vec<f32> = (0..2 * dim).map(|i| i as f32 * 0.37 - 1.0).collect();
    let picks = [Pick { row: 1, expert: 9, gate: 0.75 }, Pick { row: 0, expert: 2, gate: 0.125 }];

    a.send_experts(7, 40, 3, &picks, &rows).unwrap();
    let mut buf = Vec::new();
    let h = b.recv_frame(&mut buf).unwrap();
    assert_eq!((h.msg_type, h.request_id, h.position), (MsgType::Experts, 7, 40));
    let mut encoded = Vec::new();
    encode_experts(3, &picks, &rows, &mut encoded);
    assert_eq!(buf, encoded);
    a.send(MsgType::Experts, 8, 41, &encoded).unwrap();
    let (mut got_picks, mut got_rows) = (Vec::new(), vec![5.0; 100]);
    let h = b.recv_header().unwrap();
    assert_eq!(b.recv_experts(&h, dim, &mut got_picks, &mut got_rows).unwrap(), 3);
    assert_eq!((got_picks.as_slice(), got_rows.as_slice()), (&picks[..], rows.as_slice()));
    let (mut old_picks, mut old_rows) = (Vec::new(), Vec::new());
    decode_experts(&encoded, dim, &mut old_picks, &mut old_rows).unwrap();
    assert_eq!((old_picks, old_rows), (got_picks, got_rows));

    // One output row per pick.
    b.send_partial(7, 40, 3, 55, &rows).unwrap();
    let h = a.recv_frame(&mut buf).unwrap();
    assert_eq!((h.msg_type, h.request_id, h.position), (MsgType::Partial, 7, 40));
    encode_partial(3, 55, &rows, &mut encoded);
    assert_eq!(buf, encoded);
    b.send(MsgType::Partial, 8, 41, &encoded).unwrap();
    let mut got = Vec::new();
    let h = a.recv_header().unwrap();
    assert_eq!(a.recv_partial(&h, dim, &mut got).unwrap(), (3, 55));
    assert_eq!(got, rows);
    let mut old = Vec::new();
    assert_eq!(decode_partial(&encoded, dim, &mut old).unwrap(), (3, 55));
    assert_eq!(old, got);
}

#[test]
fn malformed_experts_and_partial_payloads_are_skipped() {
    use msh_net::{encode_experts, Pick};
    let (mut a, mut b) = pair();
    let mut bad = Vec::new();
    // A pick for row 2 of the step's 1 row.
    encode_experts(0, &[Pick { row: 2, expert: 0, gate: 1.0 }], &[0.0; 4], &mut bad);
    a.send(MsgType::Experts, 1, 0, &bad).unwrap();
    // Not whole rows of 4 floats.
    a.send(MsgType::Partial, 2, 0, &[0, 0, 0, 0, 0, 0, 1, 2, 3]).unwrap();
    a.send(MsgType::Ping, 3, 0, b"next").unwrap();
    let h = b.recv_header().unwrap();
    assert!(b.recv_experts(&h, 4, &mut Vec::new(), &mut Vec::new()).is_err());
    let h = b.recv_header().unwrap();
    assert!(b.recv_partial(&h, 4, &mut Vec::new()).is_err());
    let mut buf = Vec::new();
    let h = b.recv_frame(&mut buf).unwrap();
    assert_eq!((h.msg_type, h.request_id, buf.as_slice()), (MsgType::Ping, 3, &b"next"[..]));
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

/// Two connected ends that send in chunks of at most `max` payload bytes.
fn chunked_pair(max: usize) -> (Connection, Connection) {
    let (a, b) = pair();
    (a.with_max_chunk(max), b.with_max_chunk(max))
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn chunked_rows_match_unchunked_ones_byte_for_byte() {
    use msh_net::{encode_experts, encode_hidden, encode_partial, Pick, FLAG_EVERY_TOKEN};
    let dim = 256;
    let values: Vec<f32> = (0..64 * dim).map(|i| (i as f32 * 0.37 - 2.0).sin()).collect();
    // 1000 picks: the prefix alone spans two chunks.
    let picks: Vec<Pick> = (0..1000).map(|i| Pick { row: (i % 64) as u16, expert: i as u16, gate: i as f32 / 7.0 }).collect();
    for max in [4096, 1 << 20] {
        let (mut a, mut b) = chunked_pair(max);
        a.send_hidden(FLAG_EVERY_TOKEN, 7, 40, &[11, 22], &values).unwrap();
        let h = b.recv_header().unwrap();
        assert_eq!((h.msg_type, h.flags, h.request_id, h.position), (MsgType::Hidden, FLAG_EVERY_TOKEN, 7, 40));
        let (mut got, mut stages) = (vec![1.0; 3], Vec::new());
        assert_eq!(b.recv_hidden(&h, dim, &mut got, &mut stages).unwrap(), 64);
        assert_eq!(bits(&got), bits(&values));
        assert_eq!(stages, [11, 22]);
        // Read whole, it is the encoded payload.
        a.send_hidden(0, 8, 0, &[5], &values).unwrap();
        let mut buf = Vec::new();
        b.recv_frame(&mut buf).unwrap();
        let mut enc = Vec::new();
        encode_hidden(&[5], &values, &mut enc);
        assert_eq!(buf, enc, "max {max}");

        a.send_experts(9, 3, 2, &picks, &values).unwrap();
        let h = b.recv_header().unwrap();
        let (mut p, mut rows) = (Vec::new(), Vec::new());
        assert_eq!(b.recv_experts(&h, dim, &mut p, &mut rows).unwrap(), 2);
        assert_eq!((p, bits(&rows)), (picks.clone(), bits(&values)));
        a.send_experts(9, 3, 2, &picks, &values).unwrap();
        b.recv_frame(&mut buf).unwrap();
        encode_experts(2, &picks, &values, &mut enc);
        assert_eq!(buf, enc);

        b.send_partial(9, 3, 2, 77, &values).unwrap();
        let h = a.recv_header().unwrap();
        let mut out = Vec::new();
        assert_eq!(a.recv_partial(&h, dim, &mut out).unwrap(), (2, 77));
        assert_eq!(bits(&out), bits(&values));
        b.send_partial(9, 3, 2, 77, &values).unwrap();
        a.recv_frame(&mut buf).unwrap();
        encode_partial(2, 77, &values, &mut enc);
        assert_eq!(buf, enc);
    }
}

#[test]
fn the_writer_sends_the_same_bytes_in_chunks() {
    use msh_net::{encode_experts, encode_hidden, encode_partial, Pick};
    let dim = 128;
    let values: Vec<f32> = (0..64 * dim).map(|i| (i as f32).cos()).collect();
    let picks = [Pick { row: 3, expert: 9, gate: 0.5 }];
    let (a, mut b) = chunked_pair(4096);
    let (_keep, w) = a.split().unwrap();
    let mut v = values.clone();
    w.send_hidden(0, 1, 0, &[4], &mut v).unwrap();
    w.send_experts(2, 0, 7, &picks, &std::sync::Arc::new(values.clone())).unwrap();
    let mut v = values.clone();
    w.send_partial(3, 0, 7, 9, &mut v).unwrap();
    let (mut buf, mut enc) = (Vec::new(), Vec::new());
    b.recv_frame(&mut buf).unwrap();
    encode_hidden(&[4], &values, &mut enc);
    assert_eq!(buf, enc);
    b.recv_frame(&mut buf).unwrap();
    encode_experts(7, &picks, &values, &mut enc);
    assert_eq!(buf, enc);
    b.recv_frame(&mut buf).unwrap();
    encode_partial(7, 9, &values, &mut enc);
    assert_eq!(buf, enc);
}

/// `payload` as hand-made chunks of `size` bytes: each frame's flags and payload, the first
/// carrying the total.
fn chunks(payload: &[u8], size: usize) -> Vec<(u8, Vec<u8>)> {
    use msh_net::FLAG_MORE;
    let parts: Vec<&[u8]> = payload.chunks(size).collect();
    let mut out = Vec::new();
    for (i, p) in parts.iter().enumerate() {
        let mut body = if i == 0 { (payload.len() as u32).to_le_bytes().to_vec() } else { Vec::new() };
        body.extend_from_slice(p);
        out.push((if i + 1 < parts.len() { FLAG_MORE } else { 0 }, body));
    }
    out
}

#[test]
fn chunks_of_two_requests_interleave() {
    use msh_net::{decode_hidden, encode_hidden};
    let (mut a, mut b) = pair();
    let dim = 16;
    let x: Vec<f32> = (0..10 * dim).map(|i| i as f32).collect();
    let y: Vec<f32> = (0..7 * dim).map(|i| -(i as f32)).collect();
    let (mut ex, mut ey) = (Vec::new(), Vec::new());
    encode_hidden(&[1], &x, &mut ex);
    encode_hidden(&[2, 3], &y, &mut ey);
    let (cx, cy) = (chunks(&ex, 100), chunks(&ey, 100));
    // Alternate chunk by chunk, with a ping of the same id in the middle.
    for i in 0..cx.len().max(cy.len()) {
        if let Some((f, p)) = cx.get(i) {
            a.send_flagged(MsgType::Hidden, *f, 1, 0, p).unwrap();
        }
        if i == 2 {
            a.send(MsgType::Ping, 1, 0, b"ping").unwrap();
        }
        if let Some((f, p)) = cy.get(i) {
            a.send_flagged(MsgType::Hidden, *f, 2, 5, p).unwrap();
        }
    }
    let mut buf = Vec::new();
    // The ping comes first; `y` is shorter, so it is complete before `x`.
    let h = b.recv_frame(&mut buf).unwrap();
    assert_eq!((h.msg_type, h.request_id, buf.as_slice()), (MsgType::Ping, 1, &b"ping"[..]));
    let h = b.recv_header().unwrap();
    assert_eq!((h.request_id, h.position, h.payload_len as usize, h.flags), (2, 5, ey.len(), 0));
    let (mut got, mut st) = (Vec::new(), Vec::new());
    b.recv_hidden(&h, dim, &mut got, &mut st).unwrap();
    assert_eq!((got.clone(), st.clone()), (y, vec![2, 3]));
    let h = b.recv_frame(&mut buf).unwrap();
    assert_eq!(h.request_id, 1);
    decode_hidden(&buf, dim, &mut got, &mut st).unwrap();
    assert_eq!(got, x);
}

/// The `Error` message `b` makes of a refused chunked message: its request and text.
fn refused(b: &mut Connection) -> (u64, String) {
    let mut buf = Vec::new();
    let h = b.recv_frame(&mut buf).unwrap();
    assert_eq!(h.msg_type, MsgType::Error, "{:?}", String::from_utf8_lossy(&buf));
    (h.request_id, String::from_utf8_lossy(&buf).into_owned())
}

#[test]
fn malformed_chunk_sequences_fail_only_their_request() {
    use msh_net::{encode_hidden, FLAG_MORE, MAX_PAYLOAD};
    let (mut a, mut b) = pair();
    let mut enc = Vec::new();
    encode_hidden(&[1], &[0.5; 64], &mut enc);
    let c = chunks(&enc, 100);
    // A token of request 3 while its hidden state is half sent.
    a.send_flagged(MsgType::Hidden, c[0].0, 3, 0, &c[0].1).unwrap();
    a.send(MsgType::Token, 3, 0, b"tok").unwrap();
    // More than the total it announced.
    a.send_flagged(MsgType::Hidden, FLAG_MORE, 4, 0, &[8, 0, 0, 0, 1, 2]).unwrap();
    a.send_flagged(MsgType::Hidden, 0, 4, 0, &[3, 4, 5, 6, 7, 8, 9]).unwrap();
    // Less.
    a.send_flagged(MsgType::Hidden, FLAG_MORE, 5, 0, &[8, 0, 0, 0, 1, 2]).unwrap();
    a.send_flagged(MsgType::Hidden, 0, 5, 0, &[3]).unwrap();
    // Over the limit: its chunks are skipped.
    let over = (MAX_PAYLOAD as u32 + 1).to_le_bytes();
    a.send_flagged(MsgType::Move, FLAG_MORE, 6, 0, &[&over[..], &[0; 50]].concat()).unwrap();
    a.send_flagged(MsgType::Move, 0, 6, 0, &[0; 50]).unwrap();
    // No total.
    a.send_flagged(MsgType::Move, FLAG_MORE, 7, 0, &[1, 2]).unwrap();
    a.send_flagged(MsgType::Move, 0, 7, 0, &[3]).unwrap();
    a.send(MsgType::Ping, 8, 0, b"after").unwrap();

    let (id, why) = refused(&mut b);
    assert!(id == 3 && why.contains("interrupted by a Token"), "{why}");
    let mut buf = Vec::new();
    let h = b.recv_frame(&mut buf).unwrap();
    assert_eq!((h.msg_type, h.request_id, buf.as_slice()), (MsgType::Token, 3, &b"tok"[..]));
    let (id, why) = refused(&mut b);
    assert!(id == 4 && why.contains("more than the 8 bytes"), "{why}");
    let (id, why) = refused(&mut b);
    assert!(id == 5 && why.contains("ended at 3 of 8"), "{why}");
    let (id, why) = refused(&mut b);
    assert!(id == 6 && why.contains("exceeds the limit"), "{why}");
    let (id, why) = refused(&mut b);
    assert!(id == 7 && why.contains("without the payload length"), "{why}");
    let h = b.recv_frame(&mut buf).unwrap();
    assert_eq!((h.msg_type, h.request_id), (MsgType::Ping, 8));

    // The connection ends inside a chunked message.
    a.send_flagged(MsgType::Hidden, c[0].0, 9, 0, &c[0].1).unwrap();
    a.close_send();
    match b.recv_frame(&mut buf) {
        Err(FrameError::Incomplete { msg_type: MsgType::Hidden, request_id: 9 }) => {}
        other => panic!("expected an incomplete message, got {other:?}"),
    }
}

/// The key property: with a big message queued, a small frame sent after it arrives first.
#[test]
fn a_small_frame_overtakes_a_big_one_queued_before_it() {
    use msh_net::FLAG_EVERY_TOKEN;
    let dim = 2048;
    for (max, chunked) in [(64 << 10, true), (32 << 20, false)] {
        let (a, mut b) = chunked_pair(max);
        let (_keep, w) = a.split().unwrap();
        // 8 MiB of request 1, then a ping, request 2's token and request 1's token.
        let mut big: Vec<f32> = (0..1024 * dim).map(|i| i as f32).collect();
        let want = big.clone();
        w.send_hidden(FLAG_EVERY_TOKEN, 1, 0, &[1], &mut big).unwrap();
        w.send(MsgType::Ping, 77, 0, b"ping").unwrap();
        w.send(MsgType::Token, 2, 0, b"t2").unwrap();
        w.send(MsgType::Token, 1, 1024, b"t1").unwrap();
        let (mut order, mut buf) = (Vec::new(), Vec::new());
        for _ in 0..4 {
            let h = b.recv_header().unwrap();
            if h.msg_type == MsgType::Hidden {
                let (mut v, mut st) = (Vec::new(), Vec::new());
                b.recv_hidden(&h, dim, &mut v, &mut st).unwrap();
                assert_eq!(v, want);
            } else {
                b.recv_payload(&h, &mut buf).unwrap();
            }
            order.push((h.msg_type, h.request_id));
        }
        let (ping, t2, hidden, t1) = ((MsgType::Ping, 77), (MsgType::Token, 2), (MsgType::Hidden, 1), (MsgType::Token, 1));
        if chunked {
            // The ping and the other request's token go ahead; request 1's own token keeps its
            // place behind request 1's hidden state.
            assert_eq!(order, [ping, t2, hidden, t1]);
        } else {
            assert_eq!(order, [hidden, ping, t2, t1]);
        }
    }
}

#[test]
fn bulk_messages_of_two_requests_take_turns_and_release_waits_for_both() {
    use msh_net::{encode_move, Pick};
    use std::sync::Arc;
    let dim = 1024;
    let (a, mut b) = chunked_pair(4096);
    let (_keep, w) = a.split().unwrap();
    let pick = [Pick { row: 0, expert: 1, gate: 1.0 }];
    w.send_experts(1, 0, 0, &pick, &Arc::new(vec![0.25f32; 64 * dim])).unwrap();
    w.send_experts(2, 0, 0, &pick, &Arc::new(vec![0.5f32; 4 * dim])).unwrap();
    let mut out = Vec::new();
    encode_move(0, 1, &[], &mut out);
    w.send(MsgType::Release, 0, 0, &out).unwrap();
    w.send(MsgType::Ping, 5, 0, &[]).unwrap();
    let (mut order, mut buf) = (Vec::new(), Vec::new());
    for _ in 0..4 {
        let h = b.recv_frame(&mut buf).unwrap();
        order.push((h.msg_type, h.request_id));
    }
    // The ping first; request 2's smaller message is done before request 1's, sent in turns;
    // `Release` waits for both.
    let want = [(MsgType::Ping, 5), (MsgType::Experts, 2), (MsgType::Experts, 1), (MsgType::Release, 0)];
    assert_eq!(order, want);
}

#[test]
fn closing_a_writer_sends_what_is_queued_then_end_of_stream() {
    let (a, mut b) = chunked_pair(4096);
    let (_keep, w) = a.split().unwrap();
    let mut big = vec![1.5f32; 100_000];
    w.send_hidden(0, 1, 0, &[], &mut big).unwrap();
    w.send(MsgType::Drop, 1, 0, &[]).unwrap();
    w.close();
    assert!(w.send(MsgType::Ping, 1, 0, &[]).is_err(), "closed");
    let (mut v, mut st, mut buf) = (Vec::new(), Vec::new(), Vec::new());
    let h = b.recv_header().unwrap();
    assert_eq!(b.recv_hidden(&h, 4, &mut v, &mut st).unwrap(), 25_000);
    assert_eq!(b.recv_frame(&mut buf).unwrap().msg_type, MsgType::Drop);
    match b.recv_frame(&mut buf) {
        Err(FrameError::Io(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {}
        other => panic!("expected end of stream, got {other:?}"),
    }
}

#[test]
fn a_write_error_reaches_the_callback_and_later_sends() {
    let (a, b) = chunked_pair(4096);
    let (_keep, w) = a.split().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    w.on_error(move |e| tx.send(e).unwrap());
    drop(b);
    // Writes fail once the peer has reset the connection.
    let mut sent = 0;
    while rx.try_recv().is_err() {
        assert!(sent < 10_000, "no error reported");
        let _ = w.send(MsgType::Ping, sent, 0, &[0; 1000]);
        sent += 1;
        thread::sleep(Duration::from_millis(1));
    }
    assert!(w.send(MsgType::Ping, 0, 0, &[]).is_err());
}

/// `cargo test -p msh-net --release --test loopback -- --ignored --nocapture`
#[test]
#[ignore]
fn ping_behind_a_hidden_state_timing() {
    use std::time::Instant;
    let dim = 2048;
    for (label, max) in [("unchunked", 1usize << 20), ("64 KiB chunks", 64 << 10)] {
        let (a, mut b) = chunked_pair(max);
        let (_keep, w) = a.split().unwrap();
        let (mut alone, mut behind, mut buf) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..300 {
            // A ping on its own, then one queued right behind a 512 KiB prompt chunk.
            let t0 = Instant::now();
            w.send(MsgType::Ping, i, 0, &[]).unwrap();
            b.recv_frame(&mut buf).unwrap();
            alone.push(t0.elapsed().as_secs_f64() * 1e6);
            let mut v = vec![0.5f32; 64 * dim];
            let t0 = Instant::now();
            w.send_hidden(0, 1, 0, &[1], &mut v).unwrap();
            w.send(MsgType::Ping, i, 0, &[]).unwrap();
            for _ in 0..2 {
                if b.recv_frame(&mut buf).unwrap().msg_type == MsgType::Ping {
                    behind.push(t0.elapsed().as_secs_f64() * 1e6);
                }
            }
        }
        for v in [&mut alone, &mut behind] {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        }
        let p = |v: &[f64], q: f64| v[((v.len() - 1) as f64 * q) as usize];
        println!(
            "{label:>13}: ping alone p50 {:.0} µs; behind a 512 KiB hidden p50 {:.0} µs, p90 {:.0} µs",
            p(&alone, 0.5),
            p(&behind, 0.5),
            p(&behind, 0.9)
        );
    }
}
