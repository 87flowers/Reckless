use std::arch::x86_64::*;

use crate::types::{Piece, Square};

pub fn ray_permuation(focus: Square) -> (__m512i, u64) {
    unsafe {
        // We use the 0x88 board representation here for intermediate calculations.
        // We convert to and from this representation to avoid a 4KiB LUT.
        let offsets: [u8; 64] = [
            0x1F, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, // N
            0x21, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, // NE
            0x12, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, // E
            0xF2, 0xF1, 0xE2, 0xD3, 0xC4, 0xB5, 0xA6, 0x97, // SE
            0xE1, 0xF0, 0xE0, 0xD0, 0xC0, 0xB0, 0xA0, 0x90, // S
            0xDF, 0xEF, 0xDE, 0xCD, 0xBC, 0xAB, 0x9A, 0x89, // SW
            0xEE, 0xFF, 0xFE, 0xFD, 0xFC, 0xFB, 0xFA, 0xF9, // W
            0x0E, 0x0F, 0x1E, 0x2D, 0x3C, 0x4B, 0x5A, 0x69, // NW
        ];
        let offsets = _mm512_loadu_si512(offsets.as_ptr().cast());

        let focus = focus as u8;
        let focus = focus + (focus & 0x38);

        let coords = _mm512_add_epi8(offsets, _mm512_set1_epi8(focus as i8));

        let perm = _mm512_gf2p8affine_epi64_epi8(coords, _mm512_set1_epi64(0x0102041020400000), 0);
        let mask = _mm512_testn_epi8_mask(coords, _mm512_set1_epi8(0x88u8 as i8));

        (perm, mask)
    }
}

pub fn closest_on_rays(occupied: u64) -> u64 {
    let o = occupied | 0x8181818181818181;
    let x = o ^ (o - 0x0303030303030303);
    x & occupied
}

pub fn ray_fill(x: u64) -> u64 {
    let x = (x + 0x7E7E7E7E7E7E7E7E) & 0x8080808080808080;
    x - (x >> 7)
}

pub fn board_to_rays(perm: __m512i, valid: u64, board: __m512i) -> (__m512i, __m512i) {
    unsafe {
        let lut: [u8; 16] = [
            //   White,      Black,
            0b00000001, 0b00000010, // Pawn
            0b00000100, 0b00000100, // Knight
            0b00001000, 0b00001000, // Bishop
            0b00010000, 0b00010000, // Rook
            0b00100000, 0b00100000, // Queen
            0b01000000, 0b01000000, // King
            0, 0, 0, 0,
        ];
        let lut = _mm_loadu_si128(lut.as_ptr().cast());

        let pboard = _mm512_permutexvar_epi8(perm, board);
        let rays = _mm512_maskz_shuffle_epi8(valid, _mm512_broadcast_i32x4(lut), pboard);
        (pboard, rays)
    }
}

pub fn attackers_along_rays(victim: Piece, rays: __m512i) -> u64 {
    const HORSE: u8 = 0b00000100; // knight
    const ORTH: u8 = 0b00110000; // rook and queen
    const DIAG: u8 = 0b00101000; // bishop and queen
    const ORTHO_NEAR: u8 = 0b01110000; // king, rook and queen
    const WPAWN_NEAR: u8 = 0b01101001; // wp, king, bishop, queen
    const BPAWN_NEAR: u8 = 0b01101010; // bp, king, bishop, queen

    const MASK: [u8; 64] = [
        HORSE, ORTHO_NEAR, ORTH, ORTH, ORTH, ORTH, ORTH, ORTH, // N
        HORSE, BPAWN_NEAR, DIAG, DIAG, DIAG, DIAG, DIAG, DIAG, // NE
        HORSE, ORTHO_NEAR, ORTH, ORTH, ORTH, ORTH, ORTH, ORTH, // E
        HORSE, WPAWN_NEAR, DIAG, DIAG, DIAG, DIAG, DIAG, DIAG, // SE
        HORSE, ORTHO_NEAR, ORTH, ORTH, ORTH, ORTH, ORTH, ORTH, // S
        HORSE, WPAWN_NEAR, DIAG, DIAG, DIAG, DIAG, DIAG, DIAG, // SW
        HORSE, ORTHO_NEAR, ORTH, ORTH, ORTH, ORTH, ORTH, ORTH, // W
        HORSE, BPAWN_NEAR, DIAG, DIAG, DIAG, DIAG, DIAG, DIAG, // NW
    ];

    const INCLUDE: [i8; 6] = [
        0b01111111, // Pawn
        0b01111111, // Knight
        0b01111100, // Bishop
        0b01111111, // Rook
        0b00100100, // Queen
        0b00111100, // King
    ];

    unsafe {
        let mask = _mm512_and_si512(
            _mm512_loadu_si512(MASK.as_ptr().cast()),
            _mm512_set1_epi8(INCLUDE[victim.piece_type() as usize]),
        );

        _mm512_test_epi8_mask(rays, mask)
    }
}

pub fn attacking_along_rays(attacker: Piece, rays: __m512i) -> u64 {
    const LUT: [(u64, i8); 12] = [
        (0x02_00_00_00_00_00_02_00, 0b00010111), // WhitePawn
        (0x00_00_02_00_02_00_00_00, 0b00010111), // BlackPawn
        (0x01_01_01_01_01_01_01_01, 0b01111111), // WhiteKnight
        (0x01_01_01_01_01_01_01_01, 0b01111111), // BlackKnight
        (0xFE_00_FE_00_FE_00_FE_00, 0b01011111), // WhiteBishop
        (0xFE_00_FE_00_FE_00_FE_00, 0b01011111), // BlackBishop
        (0x00_FE_00_FE_00_FE_00_FE, 0b01011111), // WhiteRook
        (0x00_FE_00_FE_00_FE_00_FE, 0b01011111), // BlackRook
        (0xFE_FE_FE_FE_FE_FE_FE_FE, 0b01111111), // WhiteQueen
        (0xFE_FE_FE_FE_FE_FE_FE_FE, 0b01111111), // BlackQueen
        (0x02_02_02_02_02_02_02_02, 0b00011111), // WhiteKing
        (0x02_02_02_02_02_02_02_02, 0b00011111), // BlackKing
    ];

    unsafe {
        let (mask, include) = LUT[attacker as usize];
        let mask = _mm512_maskz_set1_epi8(mask, include);
        _mm512_test_epi8_mask(rays, mask)
    }
}

pub fn sliders_along_rays(rays: __m512i) -> u64 {
    unsafe {
        let orth = 0b00110000; // rook and queen
        let diag = 0b00101000; // bishop and queen

        let mask: [u8; 64] = [
            0x80, orth, orth, orth, orth, orth, orth, orth, // N
            0x80, diag, diag, diag, diag, diag, diag, diag, // NE
            0x80, orth, orth, orth, orth, orth, orth, orth, // E
            0x80, diag, diag, diag, diag, diag, diag, diag, // SE
            0x80, orth, orth, orth, orth, orth, orth, orth, // S
            0x80, diag, diag, diag, diag, diag, diag, diag, // SW
            0x80, orth, orth, orth, orth, orth, orth, orth, // W
            0x80, diag, diag, diag, diag, diag, diag, diag, // NW
        ];
        let mask = _mm512_loadu_si512(mask.as_ptr().cast());

        _mm512_test_epi8_mask(rays, mask) & 0xFEFEFEFEFEFEFEFE
    }
}

pub fn flip_rays(x: __m512i) -> __m512i {
    unsafe { _mm512_shuffle_i64x2(x, x, 0b01001110) }
}
