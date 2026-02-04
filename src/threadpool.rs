use std::sync::{Arc, RwLock};

use crate::{
    board::Board,
    search::{self, Report},
    thread::{SharedContext, Status, ThreadData},
    time::{Limits, TimeManager},
};

mod channel;

type ThreadDataVec = Vec<Arc<RwLock<Option<ThreadData>>>>;

#[derive(Clone)]
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
        let tds = vec![Arc::new(RwLock::new(None))];

        let (mut tx, rxs) = channel::channel(1);
        let workers = make_worker_threads(shared.clone(), &tds, 1, rxs);
        assert!(workers.len() == 1);

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

    pub fn set_count(&mut self, threads: usize) {
        self.channel.send(&Msg::Quit);
        self.workers.drain(..).for_each(WorkerThread::join);

        self.tds = vec![Arc::new(RwLock::new(None)); threads];

        let (tx, rxs) = channel::channel(threads);
        self.channel = tx;
        self.workers = make_worker_threads(self.shared.clone(), &self.tds, threads, rxs);
        assert!(self.workers.len() == threads);

        // SAFETY: Ensure all of tds have been initialized.
        self.channel.send(&Msg::Ping);
    }

    pub fn main_thread(&mut self) -> &RwLock<Option<ThreadData>> {
        &*self.tds[0]
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
    shared: Arc<SharedContext>, tds: &ThreadDataVec, id: usize, bind: bool, mut channel: channel::Receiver<Msg>,
) -> WorkerThread {
    let tds = tds.clone();

    let handle = std::thread::spawn(move || {
        #[cfg(feature = "numa")]
        if bind {
            crate::numa::bind_thread(id);
        }
        #[cfg(not(feature = "numa"))]
        let _ = bind;

        let td = tds[id].clone();
        {
            let mut td = td.write().unwrap();
            *td = Some(ThreadData::new(shared));
        }

        loop {
            match channel.recv(|m| m.clone()) {
                Msg::Ping => {}
                Msg::Quit => break,
                Msg::Clear => {
                    let mut td = td.write().unwrap();
                    let shared = td.as_ref().unwrap().shared.clone();
                    *td = Some(ThreadData::new(shared));
                }
                Msg::Go(board, time_manager, report, multi_pv) => {
                    {
                        let mut td = td.write().unwrap();
                        let td = td.as_mut().unwrap();

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
                        let tds = tds.iter().map(|td| td.read().unwrap()).collect::<Vec<_>>();
                        let tds = tds.iter().map(|td| td.as_ref().unwrap()).collect::<Vec<_>>();
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

    channels.enumerate().map(|(id, ch)| make_worker_thread(shared.clone(), tds, id, bind, ch)).collect()
}
