use crate::{attacks::*, magics::*};

const ROOK_DIRECTIONS: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const BISHOP_DIRECTIONS: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

pub fn generate_king_map() -> [u64; 64] {
    generate_map(king_attacks)
}

pub fn generate_knight_map() -> [u64; 64] {
    generate_map(knight_attacks)
}

fn generate_map<F: Fn(u8) -> u64>(f: F) -> [u64; 64] {
    let mut map = [0; 64];
    for square in 0..64 {
        map[square as usize] = f(square as u8);
    }

    map
}

pub fn generate_pawn_map() -> [[u64; 64]; 2] {
    [
        generate_map(|square| pawn_attacks(square, Color::White)),
        generate_map(|square| pawn_attacks(square, Color::Black)),
    ]
}

pub fn generate_rook_map() -> Vec<u64> {
    generate_sliding_map(ROOK_MAP_SIZE, &ROOK_MAGICS, &ROOK_DIRECTIONS)
}

pub fn generate_bishop_map() -> Vec<u64> {
    generate_sliding_map(BISHOP_MAP_SIZE, &BISHOP_MAGICS, &BISHOP_DIRECTIONS)
}

pub fn generate_diagonal_table() -> Vec<u64> {
    let mut map = vec![0; 64];
    for square in 0..64 {
        map[square as usize] = sliding_attacks(square, 0, &[(1, 1), (-1, -1)]);
    }
    map
}

pub fn generate_anti_diagonal_table() -> Vec<u64> {
    let mut map = vec![0; 64];
    for square in 0..64 {
        map[square as usize] = sliding_attacks(square, 0, &[(1, -1), (-1, 1)]);
    }
    map
}

pub fn generate_between_map() -> [[u64; 64]; 64] {
    let mut map = [[0; 64]; 64];
    for a in 0..64 {
        for b in 0..64 {
            let a_bb = 1 << a;
            let b_bb = 1 << b;

            map[a as usize][b as usize] = if sliding_attacks(a, 0, &ROOK_DIRECTIONS) & b_bb != 0 {
                sliding_attacks(a, b_bb, &ROOK_DIRECTIONS) & sliding_attacks(b, a_bb, &ROOK_DIRECTIONS)
            } else if sliding_attacks(a, 0, &BISHOP_DIRECTIONS) & b_bb != 0 {
                sliding_attacks(a, b_bb, &BISHOP_DIRECTIONS) & sliding_attacks(b, a_bb, &BISHOP_DIRECTIONS)
            } else {
                0
            };
        }
    }
    map
}

pub fn generate_ray_pass_map() -> [[u64; 64]; 64] {
    let mut map = [[0; 64]; 64];
    for a in 0..64 {
        for b in 0..64 {
            let a_bb = 1 << a;
            let b_bb = 1 << b;

            map[a as usize][b as usize] = if sliding_attacks(a, 0, &ROOK_DIRECTIONS) & b_bb != 0 {
                b_bb | (sliding_attacks(a, 0, &ROOK_DIRECTIONS) & sliding_attacks(b, a_bb, &ROOK_DIRECTIONS))
            } else if sliding_attacks(a, 0, &BISHOP_DIRECTIONS) & b_bb != 0 {
                b_bb | (sliding_attacks(a, 0, &BISHOP_DIRECTIONS) & sliding_attacks(b, a_bb, &BISHOP_DIRECTIONS))
            } else {
                0
            };
        }
    }
    map
}

fn generate_sliding_map(size: usize, magics: &[MagicEntry], directions: &[(i8, i8)]) -> Vec<u64> {
    let mut map = vec![0; size];

    for square in 0..64 {
        let entry = &magics[square as usize];

        let mut occupancies = 0u64;
        for _ in 0..get_permutation_count(entry.mask) {
            let hash = magic_index(occupancies, entry);
            map[hash] = sliding_attacks(square, occupancies, directions);

            occupancies = occupancies.wrapping_sub(entry.mask) & entry.mask;
        }
    }

    map
}

const fn get_permutation_count(mask: u64) -> u64 {
    1 << mask.count_ones()
}

const fn magic_index(occupancies: u64, entry: &MagicEntry) -> usize {
    let mut hash = occupancies & entry.mask;
    hash = hash.wrapping_mul(entry.magic) >> entry.shift;
    hash as usize + entry.offset
}
