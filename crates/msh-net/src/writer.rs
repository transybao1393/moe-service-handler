//! The sending side of a split link: one thread owns the socket's write half and takes
//! messages from a queue, small frames first.
//!
//! Small messages (a payload of at most `max_chunk`: `Ping`, `Token`, `Drop`, `Error`, a
//! decode step's hidden state, ...) go on the high queue and are always sent first. Bigger
//! ones are bulk, sent in chunks: after draining the high queue the writer sends one chunk of
//! the bulk message at the head, moves on to the next request's bulk message, round-robin, and
//! checks the high queue again. So a small frame waits at most one chunk.
//!
//! Every request's messages keep their order: a message of a request that has bulk queued
//! waits behind it in that request's lane. Pings are no request's and never wait. `Own` and
//! `Release` change which worker holds an expert, so they wait for every bulk message queued
//! before them (an `Experts` for the old owner must reach it first).
//!
//! Callers hand big buffers over instead of copying them: [`Writer::send_hidden`] swaps the
//! caller's rows with a spare buffer, and sent buffers go back to a small pool that the
//! receiving side takes from. The bulk queue is bounded: a producer waits while it is full,
//! and small frames never do. When the last handle is dropped or [`Writer::close`] is called
//! the thread sends what is queued and shuts the socket's write side, so the peer reads end of
//! stream. A write error ends the link: queued messages are dropped, every later send fails
//! with it, and the callback set with [`Writer::on_error`] is told.

use std::collections::VecDeque;
use std::io::{self, Write};
use std::net::{Shutdown, SocketAddr, TcpStream};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::conn::Message;
use crate::frame::{chunk_header, f32_bytes, payload_range, write_experts_prefix, write_parts, write_stages, Header, MsgType, Pick, PARTIAL_PREFIX};

/// Bulk bytes queued on one link before producers wait. One message bigger than this is still
/// taken while the queue is empty.
pub const BULK_QUEUE_BYTES: usize = 4 << 20;
/// Spare row buffers kept for reuse.
pub const POOL_BUFFERS: usize = 8;
/// How long [`Writer::close`] waits for the queue to go out before cutting the socket.
const CLOSE_TIMEOUT: Duration = Duration::from_secs(5);

static POOL: Mutex<Vec<Vec<f32>>> = Mutex::new(Vec::new());

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// A spare row buffer, of any length and contents, or a new one.
pub fn take_rows() -> Vec<f32> {
    lock(&POOL).pop().unwrap_or_default()
}

/// Returns a row buffer for reuse.
pub fn give_rows(v: Vec<f32>) {
    if v.capacity() > 0 {
        let mut p = lock(&POOL);
        if p.len() < POOL_BUFFERS {
            p.push(v);
        }
    }
}

/// A message waiting to be sent.
struct Out {
    /// The whole payload's length.
    h: Header,
    head: Vec<u8>,
    rows: Option<Arc<Vec<f32>>>,
    rows_len: usize,
    /// Payload bytes already sent.
    off: usize,
}

impl Out {
    fn new(h: Header, head: Vec<u8>, rows: Option<Arc<Vec<f32>>>, rows_len: usize) -> Out {
        Out { h, head, rows, rows_len, off: 0 }
    }

    fn len(&self) -> usize {
        self.h.payload_len as usize
    }

    fn rows(&self) -> &[u8] {
        self.rows.as_deref().map_or(&[], |r| &f32_bytes(r)[..self.rows_len])
    }

    fn recycle(self) {
        if let Some(Ok(v)) = self.rows.map(Arc::try_unwrap) {
            give_rows(v);
        }
    }
}

/// One request's messages behind its bulk message at the front.
struct Lane {
    /// The request id; `None` for pings.
    key: Option<u64>,
    /// Creation order, for `Own` and `Release` to wait on.
    seq: u64,
    items: VecDeque<Out>,
}

#[derive(Default)]
struct Queues {
    high: VecDeque<Out>,
    lanes: VecDeque<Lane>,
    /// `Own` and `Release`, each with the last lane it waits for.
    fenced: VecDeque<(u64, Out)>,
    lane_seq: u64,
    bulk_bytes: usize,
    closing: bool,
    finished: bool,
    error: Option<(io::ErrorKind, String)>,
}

impl Queues {
    /// Queues `o` behind its request's bulk messages, or as a high frame.
    fn place(&mut self, o: Out, max_chunk: usize) {
        let key = (!matches!(o.h.msg_type, MsgType::Ping | MsgType::Pong)).then_some(o.h.request_id);
        if let Some(lane) = self.lanes.iter_mut().find(|l| l.key == key) {
            lane.items.push_back(o);
        } else if o.len() <= max_chunk {
            self.high.push_back(o);
        } else {
            self.lane_seq += 1;
            self.lanes.push_back(Lane { key, seq: self.lane_seq, items: VecDeque::from([o]) });
        }
    }

    /// Queues the fenced messages whose lanes have all gone.
    fn release_fences(&mut self, max_chunk: usize) {
        let oldest = self.lanes.iter().map(|l| l.seq).min();
        while let Some(&(last, _)) = self.fenced.front() {
            if oldest.is_some_and(|s| s <= last) {
                return;
            }
            let (_, o) = self.fenced.pop_front().unwrap();
            self.place(o, max_chunk);
        }
    }

    fn failed(&self) -> Option<io::Error> {
        self.error.as_ref().map(|(k, m)| io::Error::new(*k, m.clone()))
    }
}

struct Shared {
    q: Mutex<Queues>,
    /// The writer waits for messages, producers for bulk room, `close` for the end.
    work: Condvar,
    room: Condvar,
    done: Condvar,
    /// To cut the socket when closing takes too long.
    stream: TcpStream,
    on_error: Mutex<Option<Box<dyn FnOnce(String) + Send>>>,
}

struct Inner {
    shared: Arc<Shared>,
    peer_name: String,
    peer_addr: SocketAddr,
    max_chunk: usize,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        lock(&self.shared.q).closing = true;
        self.shared.work.notify_all();
    }
}

/// The sending side of a link; clones share it.
#[derive(Clone)]
pub struct Writer {
    inner: Arc<Inner>,
}

impl Writer {
    pub(crate) fn start(stream: TcpStream, peer_name: String, peer_addr: SocketAddr, max_chunk: usize) -> io::Result<Writer> {
        let shared = Arc::new(Shared {
            q: Mutex::new(Queues::default()),
            work: Condvar::new(),
            room: Condvar::new(),
            done: Condvar::new(),
            stream: stream.try_clone()?,
            on_error: Mutex::new(None),
        });
        let s = shared.clone();
        let thread = thread::Builder::new().name(format!("writer {peer_name}")).spawn(move || run(s, stream, max_chunk))?;
        Ok(Writer { inner: Arc::new(Inner { shared, peer_name, peer_addr, max_chunk, thread: Mutex::new(Some(thread)) }) })
    }

    pub fn peer_name(&self) -> &str {
        &self.inner.peer_name
    }

    pub fn peer_addr(&self) -> SocketAddr {
        self.inner.peer_addr
    }

    pub fn max_chunk(&self) -> usize {
        self.inner.max_chunk
    }

    /// Calls `f` with the error once a write fails; at once if one already has.
    pub fn on_error(&self, f: impl FnOnce(String) + Send + 'static) {
        let q = lock(&self.inner.shared.q);
        match q.failed() {
            Some(e) => {
                drop(q);
                f(e.to_string())
            }
            None => *lock(&self.inner.shared.on_error) = Some(Box::new(f)),
        }
    }

    /// Whether the link has failed or closed.
    pub fn is_closed(&self) -> bool {
        let q = lock(&self.inner.shared.q);
        q.closing || q.finished
    }

    fn push(&self, o: Out) -> io::Result<()> {
        let (s, max) = (&self.inner.shared, self.inner.max_chunk);
        let len = o.len();
        let bulk = len > max;
        let mut q = lock(&s.q);
        while bulk && q.bulk_bytes > 0 && q.bulk_bytes + len > BULK_QUEUE_BYTES && !q.finished && !q.closing {
            q = s.room.wait(q).unwrap_or_else(|e| e.into_inner());
        }
        if let Some(e) = q.failed() {
            return Err(e);
        }
        if q.closing || q.finished {
            return Err(io::Error::new(io::ErrorKind::BrokenPipe, "link closed"));
        }
        if bulk {
            q.bulk_bytes += len;
        }
        let fence = matches!(o.h.msg_type, MsgType::Own | MsgType::Release);
        if fence && !(q.lanes.is_empty() && q.fenced.is_empty()) {
            let last = q.lane_seq;
            q.fenced.push_back((last, o));
        } else {
            q.place(o, max);
        }
        drop(q);
        s.work.notify_one();
        Ok(())
    }

    /// Queues one message; the payload is copied.
    pub fn send(&self, msg_type: MsgType, request_id: u64, position: u32, payload: &[u8]) -> io::Result<()> {
        self.send_flagged(msg_type, 0, request_id, position, payload)
    }

    pub fn send_flagged(&self, msg_type: MsgType, flags: u8, request_id: u64, position: u32, payload: &[u8]) -> io::Result<()> {
        self.send_owned(msg_type, flags, request_id, position, payload.to_vec())
    }

    /// Queues one message, taking its payload.
    pub fn send_owned(&self, msg_type: MsgType, flags: u8, request_id: u64, position: u32, payload: Vec<u8>) -> io::Result<()> {
        let h = Header::new(msg_type, request_id, position, payload.len()).with_flags(flags);
        self.push(Out::new(h, payload, None, 0))
    }

    /// Queues a received message to pass on as it is.
    pub fn send_message(&self, mut m: Message) -> io::Result<()> {
        let rows_len = m.rows_len();
        let rows = (rows_len > 0).then(|| Arc::new(std::mem::take(&mut m.rows)));
        let h = Header::new(m.header.msg_type, m.header.request_id, m.header.position, m.head.len() + rows_len)
            .with_flags(m.header.flags);
        self.push(Out::new(h, std::mem::take(&mut m.head), rows, rows_len))
    }

    /// Queues a `Hidden` frame (the [`crate::encode_hidden`] payload). Takes `values` without
    /// a copy (NET-6) and leaves a spare buffer, of any length and contents, in its place.
    pub fn send_hidden(&self, flags: u8, request_id: u64, position: u32, stage_us: &[u32], values: &mut Vec<f32>) -> io::Result<()> {
        let mut head = Vec::with_capacity(2 + 4 * stage_us.len());
        write_stages(stage_us, &mut head);
        let rows = std::mem::replace(values, take_rows());
        let n = rows.len() * 4;
        let h = Header::new(MsgType::Hidden, request_id, position, head.len() + n).with_flags(flags);
        self.push(Out::new(h, head, Some(Arc::new(rows)), n))
    }

    /// Queues an `Experts` frame (the [`crate::encode_experts`] payload); `rows` is shared,
    /// not copied, so one step's rows can go to every worker.
    pub fn send_experts(&self, request_id: u64, position: u32, layer: u16, picks: &[Pick], rows: &Arc<Vec<f32>>) -> io::Result<()> {
        let mut head = Vec::new();
        write_experts_prefix(layer, picks, &mut head);
        let n = rows.len() * 4;
        let h = Header::new(MsgType::Experts, request_id, position, head.len() + n);
        self.push(Out::new(h, head, Some(rows.clone()), n))
    }

    /// Queues a `Partial` frame (the [`crate::encode_partial`] payload), taking `values` as
    /// [`Writer::send_hidden`] does.
    pub fn send_partial(&self, request_id: u64, position: u32, layer: u16, compute_us: u32, values: &mut Vec<f32>) -> io::Result<()> {
        let mut head = Vec::with_capacity(PARTIAL_PREFIX);
        head.extend_from_slice(&layer.to_le_bytes());
        head.extend_from_slice(&compute_us.to_le_bytes());
        let rows = std::mem::replace(values, take_rows());
        let n = rows.len() * 4;
        let h = Header::new(MsgType::Partial, request_id, position, PARTIAL_PREFIX + n);
        self.push(Out::new(h, head, Some(Arc::new(rows)), n))
    }

    /// Cuts the link at once: what is queued is dropped, and the peer reads end of stream.
    pub fn abort(&self) {
        let s = &self.inner.shared;
        lock(&s.q).closing = true;
        let _ = s.stream.shutdown(Shutdown::Both);
        s.work.notify_all();
        s.room.notify_all();
    }

    /// Sends what is queued, then shuts the socket's write side, so the peer reads end of
    /// stream; waits for that, up to a few seconds, then cuts the socket. Later sends fail.
    pub fn close(&self) {
        let s = &self.inner.shared;
        let mut q = lock(&s.q);
        q.closing = true;
        s.work.notify_all();
        s.room.notify_all();
        let deadline = Instant::now() + CLOSE_TIMEOUT;
        while !q.finished {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                let _ = s.stream.shutdown(Shutdown::Both);
                return;
            }
            q = s.done.wait_timeout(q, left).unwrap_or_else(|e| e.into_inner()).0;
        }
        drop(q);
        if let Some(t) = lock(&self.inner.thread).take() {
            let _ = t.join();
        }
    }
}

/// The writer thread.
fn run(s: Arc<Shared>, mut stream: TcpStream, max_chunk: usize) {
    let mut batch = Vec::new();
    let result = serve(&s, &mut stream, max_chunk, &mut batch);
    let mut q = lock(&s.q);
    q.finished = true;
    q.high.clear();
    q.lanes.clear();
    q.fenced.clear();
    let failed = result.err();
    let mut tell = None;
    if let Some(e) = &failed {
        q.error = Some((e.kind(), e.to_string()));
        let _ = stream.shutdown(Shutdown::Both);
        // Under the queue lock, so `on_error` either sees the error or leaves its callback.
        tell = lock(&s.on_error).take();
    }
    drop(q);
    s.done.notify_all();
    s.room.notify_all();
    if let (Some(e), Some(f)) = (failed, tell) {
        f(e.to_string());
    }
}

fn serve(s: &Shared, stream: &mut TcpStream, max_chunk: usize, batch: &mut Vec<u8>) -> io::Result<()> {
    loop {
        let mut q = lock(&s.q);
        while q.high.is_empty() && q.lanes.is_empty() && !q.closing {
            q = s.work.wait(q).unwrap_or_else(|e| e.into_inner());
        }
        let high = std::mem::take(&mut q.high);
        let job = q.lanes.front_mut().and_then(|l| l.items.pop_front());
        drop(q);
        // Every small frame in one write, with the next chunk if there is one.
        for o in high {
            chunk_header(&o.h, 0, usize::MAX, batch);
            batch.extend_from_slice(&o.head);
            batch.extend_from_slice(o.rows());
            o.recycle();
        }
        let Some(mut o) = job else {
            if !batch.is_empty() {
                write_parts(stream, &mut [batch])?;
                batch.clear();
            }
            let q = lock(&s.q);
            if q.closing && q.high.is_empty() && q.lanes.is_empty() {
                drop(q);
                stream.flush()?;
                // Ignored: the peer may have closed already.
                let _ = stream.shutdown(Shutdown::Write);
                return Ok(());
            }
            continue;
        };
        let n = chunk_header(&o.h, o.off, max_chunk, batch);
        let (a, b) = payload_range(&o.head, o.rows(), o.off, n);
        write_parts(stream, &mut [batch, a, b])?;
        batch.clear();
        o.off += n;
        let mut q = lock(&s.q);
        let mut lane = q.lanes.pop_front().expect("only the writer takes lanes");
        if o.off < o.len() {
            lane.items.push_front(o);
        } else {
            if o.len() > max_chunk {
                q.bulk_bytes -= o.len();
                s.room.notify_all();
            }
            o.recycle();
            // What waited behind it and is small goes ahead now.
            while lane.items.front().is_some_and(|i| i.len() <= max_chunk) {
                let next = lane.items.pop_front().unwrap();
                q.high.push_back(next);
            }
        }
        if lane.items.is_empty() {
            q.release_fences(max_chunk);
        } else {
            // Round-robin: the next request's bulk message goes next.
            q.lanes.push_back(lane);
        }
    }
}
