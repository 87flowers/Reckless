/// Represents the sets of random numbers used to produce an *almost* unique hash value
/// for a position using [Zobrist Hashing](https://en.wikipedia.org/wiki/Zobrist_hashing)
/// generated using the SplitMix64 pseudorandom number generator.
#[allow(unused)]
#[derive(Debug)]
pub struct Zobrist {
    pub pieces: [[u64; 64]; 12],
    pub en_passant: [u64; 64],
    pub castling: [u64; 16],
    pub side: u64,
    pub halfmove_clock: [u64; 16],
}
