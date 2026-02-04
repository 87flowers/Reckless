use crate::{
    board::Board,
    search::{self, Report},
    thread::{SharedContext, Status, ThreadData},
    time::{Limits, TimeManager},
};
use std::sync::Arc;

mod channel;
mod sync_unsafe_cell;

pub use sync_unsafe_cell::SyncUnsafeCell;

type ThreadDataVec = Arc<Vec<SyncUnsafeCell<Option<ThreadData>>>>;

#[derive(Clone)]
#[allow(clippy::large_enum_variant)]
enum Msg {
    Ping,
    Go(Board, TimeManager, Report, usize),
    Clear,
    Quit,
}

pub struct ThreadPool {
    pub workers: Vec<WorkerThread>,
    board: Board,
    tds: ThreadDataVec,
    shared: Arc<SharedContext>,
    channel: channel::Sender<Msg>,
}

impl ThreadPool {
    pub fn available_threads() -> usize {
        const MINIMUM_THREADS: usize = 512;

        match std::thread::available_parallelism() {
            Ok(threads) => (4 * threads.get()).max(MINIMUM_THREADS),
            Err(_) => MINIMUM_THREADS,
        }
    }

    pub fn new(shared: Arc<SharedContext>) -> Self {
        Self::construct(shared, 1)
    }

    pub fn set_count(&mut self, threads: usize) {
        self.channel.send(&Msg::Quit);
        self.workers.drain(..).for_each(WorkerThread::join);

        *self = Self::construct(self.shared.clone(), threads);
    }

    fn construct(shared: Arc<SharedContext>, threads: usize) -> Self {
        let tds = Arc::new({
            let mut v = Vec::with_capacity(threads);
            (0..threads).for_each(|_| v.push(SyncUnsafeCell::new(None)));
            v
        });

        let (mut tx, rxs) = channel::channel(threads);
        let workers = make_worker_threads(shared.clone(), &tds, threads, rxs);
        assert!(workers.len() == threads);

        // SAFETY: Ensure all of tds have been initialized.
        tx.send(&Msg::Ping);

        Self {
            workers,
            board: Board::starting_position(),
            tds,
            shared,
            channel: tx,
        }
    }

    pub fn main_thread(&mut self) -> &SyncUnsafeCell<Option<ThreadData>> {
        &self.tds[0]
    }

    pub fn len(&self) -> usize {
        self.workers.len()
    }

    pub fn clear(&mut self) {
        self.board = Board::starting_position();
        self.channel.send(&Msg::Clear);
        self.channel.send(&Msg::Ping);
    }

    pub fn wait(&mut self) {
        self.channel.send(&Msg::Ping);
    }

    pub fn set_board(&mut self, board: Board) {
        self.board = board;
    }

    pub fn board(&self) -> &Board {
        &self.board
    }

    pub fn execute_searches(
        &mut self, time_manager: TimeManager, report: Report, multi_pv: usize, shared: &Arc<SharedContext>,
    ) {
        shared.tt.increment_age();

        shared.nodes.reset();
        shared.tb_hits.reset();
        shared.status.set(Status::RUNNING);

        self.channel.send(&Msg::Go(self.board.clone(), time_manager, report, multi_pv));
    }
}

pub struct WorkerThread {
    handle: std::thread::JoinHandle<()>,
}

impl WorkerThread {
    pub fn join(self) {
        self.handle.join().expect("Worker thread panicked");
    }
}

fn make_worker_thread(
    shared: Arc<SharedContext>, tds: ThreadDataVec, id: usize, bind: bool, mut channel: channel::Receiver<Msg>,
) -> WorkerThread {
    let handle = std::thread::spawn(move || {
        #[cfg(feature = "numa")]
        if bind {
            crate::numa::bind_thread(id);
        }
        #[cfg(not(feature = "numa"))]
        let _ = bind;

        let td = &tds[id];
        unsafe { *td.get() = Some(ThreadData::new(shared.clone())) };

        loop {
            match channel.recv(|m| m.clone()) {
                Msg::Ping => {}
                Msg::Quit => break,
                Msg::Clear => unsafe { *td.get() = Some(ThreadData::new(shared.clone())) },
                Msg::Go(board, time_manager, report, multi_pv) => {
                    {
                        let td = unsafe { td.get().as_mut().unwrap().as_mut().unwrap() };

                        td.board = board;

                        if id == 0 {
                            td.time_manager = time_manager;
                            td.multi_pv = multi_pv;
                            search::start(td, report);
                            td.shared.status.set(Status::STOPPED);
                        } else {
                            td.time_manager = TimeManager::new(Limits::Infinite, 0, 0);
                            search::start(td, Report::None);
                        }
                    }

                    if id == 0 && report != Report::None {
                        let tds: Vec<_> =
                            tds.iter().map(|td| unsafe { td.get().as_mut().unwrap().as_ref().unwrap() }).collect();
                        search::end(&tds);
                    }
                }
            }
        }
    });

    WorkerThread { handle }
}

fn make_worker_threads(
    shared: Arc<SharedContext>, tds: &ThreadDataVec, num_threads: usize,
    channels: impl Iterator<Item = channel::Receiver<Msg>>,
) -> Vec<WorkerThread> {
    let concurrency = std::thread::available_parallelism().map_or(1, |n| n.get());
    let bind = num_threads >= concurrency / 2;

    channels.enumerate().map(|(id, ch)| make_worker_thread(shared.clone(), tds.clone(), id, bind, ch)).collect()
}
