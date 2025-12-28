use crate::types::Bitboard;
use core::arch::x86_64::__m512i;
use std::arch::x86_64::{
    _mm512_add_epi8, _mm512_cmpneq_epi8_mask, _mm512_maskz_set1_epi8, _mm512_setzero_si512, _mm512_sub_epi8,
};

#[derive(Copy, Clone)]
pub struct BitboardCounter {
    bb: __m512i,
}

impl Default for BitboardCounter {
    fn default() -> Self {
        BitboardCounter { bb: unsafe { _mm512_setzero_si512() } }
    }
}

impl BitboardCounter {
    pub fn update(&mut self, delta: [Bitboard; 2]) {
        unsafe {
            self.bb = _mm512_sub_epi8(self.bb, _mm512_maskz_set1_epi8(delta[0].0, 1));
            self.bb = _mm512_add_epi8(self.bb, _mm512_maskz_set1_epi8(delta[1].0, 1));
        }
    }

    pub fn add(&mut self, delta: Bitboard) {
        unsafe {
            self.bb = _mm512_add_epi8(self.bb, _mm512_maskz_set1_epi8(delta.0, 1));
        }
    }

    pub fn reduce(&self) -> Bitboard {
        Bitboard(unsafe { _mm512_cmpneq_epi8_mask(self.bb, _mm512_setzero_si512()) })
    }
}
