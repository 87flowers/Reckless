use crate::types::{Bitboard, Color, Piece, PieceType, Square};

include!(concat!(env!("OUT_DIR"), "/lookup.rs"));

pub fn initialize() {}

pub fn relative_diagonal(color: Color, sq: Square) -> Bitboard {
    unsafe { Bitboard(*DIAGONALS[color as usize].get_unchecked(sq as usize)) }
}

pub fn attacks(piece: Piece, square: Square, occupancies: Bitboard) -> Bitboard {
    match piece.piece_type() {
        PieceType::Pawn => pawn_attacks(square, piece.color()),
        PieceType::Knight => knight_attacks(square),
        PieceType::Bishop => bishop_attacks(square, occupancies),
        PieceType::Rook => rook_attacks(square, occupancies),
        PieceType::Queen => queen_attacks(square, occupancies),
        PieceType::King => king_attacks(square),
        PieceType::None => Bitboard(0),
    }
}

pub fn pawn_attacks(square: Square, color: Color) -> Bitboard {
    unsafe { Bitboard(*PAWN_MAP.get_unchecked(color as usize).get_unchecked(square as usize)) }
}

pub fn king_attacks(square: Square) -> Bitboard {
    unsafe { Bitboard(*KING_MAP.get_unchecked(square as usize)) }
}

pub fn knight_attacks(square: Square) -> Bitboard {
    unsafe { Bitboard(*KNIGHT_MAP.get_unchecked(square as usize)) }
}

pub fn rook_attacks(square: Square, occupancies: Bitboard) -> Bitboard {
    unsafe {
        let entry = ROOK_MAGICS.get_unchecked(square as usize);
        let index = magic_index(occupancies, entry);

        Bitboard(*ROOK_MAP.get_unchecked(index as usize))
    }
}

pub fn ray_pass(square1: Square, square2: Square) -> Bitboard {
    unsafe { Bitboard(*RAYPASS[square1 as usize].get_unchecked(square2 as usize)) }
}

pub fn between(square1: Square, square2: Square) -> Bitboard {
    unsafe { Bitboard(*BETWEEN[square1 as usize].get_unchecked(square2 as usize)) }
}

pub fn bishop_attacks(square: Square, occupancies: Bitboard) -> Bitboard {
    unsafe {
        let entry = BISHOP_MAGICS.get_unchecked(square as usize);
        let index = magic_index(occupancies, entry);

        Bitboard(*BISHOP_MAP.get_unchecked(index as usize))
    }
}

pub fn queen_attacks(square: Square, occupancies: Bitboard) -> Bitboard {
    rook_attacks(square, occupancies) | bishop_attacks(square, occupancies)
}

const fn magic_index(occupancies: Bitboard, entry: &MagicEntry) -> u32 {
    let mut hash = occupancies.0 & entry.mask;
    hash = hash.wrapping_mul(entry.magic) >> entry.shift;
    hash as u32 + entry.offset
}
