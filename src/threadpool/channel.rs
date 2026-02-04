//! Spmc broadcast channel with capacity of 1.
//! Implementation very heavily influenced by @Sp00ph, discussions with him, and his implementation.

use std::sync::{
    Arc,
    atomic::{AtomicPtr, AtomicU32, Ordering},
};

pub struct Futex {
    threads: u32,
    generation: bool,
}

impl Futex {
    const THREADS_MASK: u32 = u32::MAX >> 1;

    fn pack(self) -> u32 {
        debug_assert!(self.threads <= Futex::THREADS_MASK);
        self.threads | (self.generation as u32) << 31
    }

    fn unpack(raw: u32) -> Futex {
        Futex {
            threads: raw & Futex::THREADS_MASK,
            generation: (raw >> 31) != 0,
        }
    }
}

struct SharedData<T: Sync> {
    msg: AtomicPtr<T>,
    futex: AtomicU32,
    receiver_count: u32,
}

impl<T: Sync> SharedData<T> {
    fn read_futex(&self, ordering: Ordering) -> Futex {
        Futex::unpack(self.futex.load(ordering))
    }

    fn write_futex(&self, f: Futex, ordering: Ordering) {
        self.futex.store(f.pack(), ordering);
    }

    fn futex_wake_all(&self) {
        atomic_wait::wake_all(&self.futex);
    }

    fn futex_wait(&self, f: Futex) {
        atomic_wait::wait(&self.futex, f.pack());
    }

    fn decrement_futex(&self, ordering: Ordering) -> Futex {
        Futex::unpack(self.futex.fetch_sub(1, ordering))
    }
}

pub struct Sender<T: Sync> {
    shared: Arc<SharedData<T>>,
}

pub struct Receiver<T: Sync> {
    shared: Arc<SharedData<T>>,
    generation: bool,
}

/// Creates a channel with `receiver_count` receivers. It is not possible to change the receiver count.
/// All receivers must handle the message. A deadlock will occur if a receiver is dropped or fails to
/// handle a message, as the sender blocks until the message has been received.
/// `receiver_count` must be at least 1 and is limited to 1 << 31 - 1 receivers.
pub fn channel<T: Sync>(receiver_count: usize) -> (Sender<T>, impl Iterator<Item = Receiver<T>>) {
    assert!((1..=Futex::THREADS_MASK as usize).contains(&receiver_count));

    let shared = Arc::new(SharedData {
        msg: AtomicPtr::new(std::ptr::null_mut()),
        futex: AtomicU32::new(0),
        receiver_count: receiver_count as u32,
    });

    let tx = Sender { shared: shared.clone() };
    let rxs = std::iter::repeat_n(shared, receiver_count).map(|shared| Receiver { shared, generation: false });

    (tx, rxs)
}

impl<T: Sync> Sender<T> {
    /// Synchronously broadcasts a message to all receivers. Blocks until read by all receivers.
    pub fn send(&mut self, msg: &T) {
        let generation = {
            let f = self.shared.read_futex(Ordering::Relaxed);

            // Since send is a blocking operation, there should never be any receivers outstanding.
            debug_assert!(f.threads == 0);

            !f.generation
        };

        // SAFETY: send waits until all receivers have handled the message, therefore this pointer
        // is always valid when dereferenced by the receivers.
        self.shared.msg.store(std::ptr::from_ref(msg).cast_mut(), Ordering::Relaxed);

        self.shared.write_futex(Futex { threads: self.shared.receiver_count, generation }, Ordering::Release);
        self.shared.futex_wake_all();

        loop {
            let f = self.shared.read_futex(Ordering::Acquire);
            // We are the only thread that updates generation, this should never happen.
            debug_assert!(f.generation == generation);
            if f.threads == 0 {
                // No more outstanding receivers.
                break;
            }
            self.shared.futex_wait(f);
        }

        // Ensures misbehaving readers trap.
        self.shared.msg.store(std::ptr::null_mut(), Ordering::Relaxed);

        // Sanity check: Ensures msg is valid for the entirely of this function.
        let _ = msg;
    }
}

impl<T: Sync> Receiver<T> {
    /// Synchronously received a broadcasted message, and calls handler on it, returning its result.
    pub fn recv<R, F: FnOnce(&T) -> R>(&mut self, handler: F) -> R {
        // Wait until next generation
        self.generation = loop {
            let f = self.shared.read_futex(Ordering::Acquire);
            if f.generation != self.generation {
                // This should never happen as there should be at least one receiver (us!).
                debug_assert!(f.threads > 0);
                break f.generation;
            }
            self.shared.futex_wait(f);
        };

        // SAFETY: Here, msg is valid because:
        // - send has updated the futex, which implies that it has written a valid pointer.
        // - send blocks until all receivers have read msg, ensuring the reference remains live.
        let msg = unsafe { self.shared.msg.load(Ordering::Relaxed).as_ref().unwrap() };
        let ret = handler(msg);

        if self.shared.decrement_futex(Ordering::Release).threads == 1 {
            // We are the last receiver to handle the message. Wake the sender.
            self.shared.futex_wake_all();
        }

        ret
    }
}
