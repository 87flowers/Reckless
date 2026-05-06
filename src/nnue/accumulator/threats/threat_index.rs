use crate::{
    lookup::attacks,
    types::{Bitboard, Color, Piece, PieceType, Square},
};

#[derive(Copy, Clone)]
struct PiecePair {
    // Bit layout:
    // - bits 0..23: base index contribution for this piece-pair
    // - bits 30..31 : exclusion flags (semi/excluded)
    inner: u32,
}

impl PiecePair {
    const fn new(excluded: bool, semi_excluded: bool, base: i32) -> Self {
        Self {
            inner: (((semi_excluded && !excluded) as u32) << 30)
                | ((excluded as u32) << 31)
                | ((base & 0x3FFFFFFF) as u32),
        }
    }

    const fn base(self, attacking: Square, attacked: Square) -> isize {
        let below = ((attacking as u8) < (attacked as u8)) as u32;
        ((self.inner.wrapping_add(below << 30)) & 0x80FFFFFF) as i32 as isize
    }
}

static mut PIECE_PAIR_LOOKUP: [[PiecePair; 12]; 12] = [[PiecePair { inner: 0 }; 12]; 12];
static mut PIECE_OFFSET_LOOKUP: [[i32; 64]; 12] = [[0; 64]; 12];
static mut ATTACK_INDEX_LOOKUP: [[[u8; 64]; 64]; 12] = [[[0; 64]; 64]; 12];
static mut PIECE_LACK_LOOKUP: [i32; 12] = [0; 12];
const LACK_RAY_LOOKUP: [[i32; 8]; 6] = [
    [-1; 8],
    [-1; 8],
    [-1, 0, -1, 64, -1, 128, -1, 192],
    [0, -1, 64, -1, 128, -1, 192, -1],
    [0, 64, 128, 192, 256, 320, 384, 448],
    [-1; 8],
];

pub fn initialize() {
    #[rustfmt::skip]
    const PIECE_INTERACTION_MAP: [[i32; 6]; 6] = [
        [0,  1, -1,  2, -1, -1],
        [0,  1,  2,  3,  4, -1],
        [0,  1,  2,  3, -1, -1],
        [0,  1,  2,  3, -1, -1],
        [0,  1,  2,  3,  4, -1],
        [0,  1,  2,  3, -1, -1],
    ];

    const PIECE_TARGET_COUNT: [i32; 6] = [6, 10, 8, 8, 10, 8];

    let mut offset = 0;
    let mut piece_offset = [0; Piece::NUM];
    let mut offset_table = [0; Piece::NUM];

    for piece_color in [Color::White, Color::Black] {
        for piece_type in 0..PieceType::NUM {
            let piece_type = PieceType::new(piece_type);
            let piece = Piece::new(piece_color, piece_type);

            let mut count = 0;

            for (square, entry) in unsafe { PIECE_OFFSET_LOOKUP[piece].iter_mut().enumerate() } {
                *entry = count;

                if piece_type != PieceType::Pawn || (8..56).contains(&square) {
                    count += attacks(piece, Square::new(square as u8), Bitboard(0)).popcount() as i32;
                }
            }

            piece_offset[piece] = count;
            offset_table[piece] = offset;

            offset += PIECE_TARGET_COUNT[piece_type] * count;

            unsafe { PIECE_LACK_LOOKUP[piece] = offset };

            offset += match piece_type {
                PieceType::None | PieceType::Pawn | PieceType::Knight | PieceType::King => 0,
                PieceType::Bishop | PieceType::Rook => 4 * 64,
                PieceType::Queen => 8 * 64,
            };
        }
    }

    for attacking in Piece::ALL {
        for attacked in Piece::ALL {
            let attacking_piece = attacking.piece_type();
            let attacking_color = attacking.color();

            let attacked_piece = attacked.piece_type();
            let attacked_color = attacked.color();

            let map = PIECE_INTERACTION_MAP[attacking_piece][attacked_piece];
            let base = offset_table[attacking]
                + ((attacked_color as i32) * (PIECE_TARGET_COUNT[attacking_piece] / 2) + map) * piece_offset[attacking];

            let enemy = attacking_color != attacked_color;
            let semi_excluded = attacking_piece == attacked_piece && (enemy || attacking_piece != PieceType::Pawn);
            let excluded = map < 0;

            unsafe { PIECE_PAIR_LOOKUP[attacking][attacked] = PiecePair::new(excluded, semi_excluded, base) };
        }
    }

    for piece in Piece::ALL {
        for (from, row) in unsafe { ATTACK_INDEX_LOOKUP[piece].iter_mut().enumerate() } {
            let attacks = attacks(piece, Square::new(from as u8), Bitboard(0));

            for (to, entry) in row.iter_mut().enumerate() {
                *entry = (Bitboard((1u64 << to) - 1) & attacks).popcount() as u8;
            }
        }
    }
}

pub fn threat_index(piece: Piece, from: Square, attacked: Piece, to: Square, mirrored: bool, pov: Color) -> isize {
    let from = from.relative_to(pov) ^ (7 * (mirrored as u8));
    let to = to.relative_to(pov) ^ (7 * (mirrored as u8));

    let attacking = (piece as usize) ^ (pov as usize);
    let attacked = (attacked as usize) ^ (pov as usize);

    unsafe {
        let pair = PIECE_PAIR_LOOKUP[attacking][attacked];

        pair.base(from, to)
            + PIECE_OFFSET_LOOKUP[attacking][from] as isize
            + ATTACK_INDEX_LOOKUP[attacking][from][to] as isize
    }
}

pub fn lack_index(piece: Piece, from: Square, ray: u8, mirrored: bool, pov: Color) -> isize {
    let ray = if pov == Color::Black { 8u8.wrapping_sub(ray) } else { ray };
    let ray = if mirrored { 4u8.wrapping_sub(ray) } else { ray };
    let ray = ray % 8;

    unsafe {
        PIECE_LACK_LOOKUP[piece] as isize + LACK_RAY_LOOKUP[piece.piece_type()][ray as usize] as isize + from as isize
    }
}
