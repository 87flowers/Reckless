use crate::nnue::INPUT_BUCKETS;

pub mod psq;
pub mod threats;

pub use psq::PstAccumulator;
pub use threats::ThreatAccumulator;

#[derive(Clone, Default)]
pub struct AccumulatorCache {
    entries: Box<[[[psq::CacheEntry; INPUT_BUCKETS]; 2]; 2]>,
}
