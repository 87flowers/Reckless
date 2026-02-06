use crate::{
    nnue::{
        Aligned, DEQUANT_MULTIPLIER, FT_QUANT, FT_SHIFT, L1_SIZE, L2_SIZE, L3_SIZE, PARAMETERS, SparseEntry,
        accumulator::{PstAccumulator, ThreatAccumulator},
        simd,
    },
    types::Color,
};

pub unsafe fn activate_ft(pst: &PstAccumulator, threat: &ThreatAccumulator, stm: Color) -> Aligned<[u8; L1_SIZE]> {
    let mut output = Aligned::new([0; L1_SIZE]);

    let zero = simd::splat_i16(0);
    let one = simd::splat_i16(FT_QUANT as i16);

    for flip in [0, 1] {
        let pst_input = &pst.values[stm as usize ^ flip];
        let threat_input = &threat.values[stm as usize ^ flip];

        for i in (0..L1_SIZE / 2).step_by(2 * simd::I16_LANES) {
            let lhs1 = *pst_input.as_ptr().add(i).cast();
            let lhs2 = *pst_input.as_ptr().add(i + simd::I16_LANES).cast();

            let rhs1 = *pst_input.as_ptr().add(i + L1_SIZE / 2).cast();
            let rhs2 = *pst_input.as_ptr().add(i + L1_SIZE / 2 + simd::I16_LANES).cast();

            let threat_lhs1 = *threat_input.as_ptr().add(i).cast();
            let threat_lhs2 = *threat_input.as_ptr().add(i + simd::I16_LANES).cast();

            let threat_rhs1 = *threat_input.as_ptr().add(i + L1_SIZE / 2).cast();
            let threat_rhs2 = *threat_input.as_ptr().add(i + L1_SIZE / 2 + simd::I16_LANES).cast();

            let lhs1_clipped = simd::clamp_i16(simd::add_i16(lhs1, threat_lhs1), zero, one);
            let lhs2_clipped = simd::clamp_i16(simd::add_i16(lhs2, threat_lhs2), zero, one);

            let rhs1_clipped = simd::min_i16(simd::add_i16(rhs1, threat_rhs1), one);
            let rhs2_clipped = simd::min_i16(simd::add_i16(rhs2, threat_rhs2), one);

            let shifted1 = simd::shift_left_i16::<{ 16 - FT_SHIFT }>(lhs1_clipped);
            let shifted2 = simd::shift_left_i16::<{ 16 - FT_SHIFT }>(lhs2_clipped);

            let product1 = simd::mul_high_i16(shifted1, rhs1_clipped);
            let product2 = simd::mul_high_i16(shifted2, rhs2_clipped);

            let packed = simd::packus(product1, product2);
            let unpacked = simd::permute(packed);

            *output.as_mut_ptr().add(i + flip * L1_SIZE / 2).cast() = unpacked;
        }
    }

    output
}

pub unsafe fn propagate_l1(ft_out: Aligned<[u8; L1_SIZE]>, nnz: &[u8], bucket: usize) -> Aligned<[f32; L2_SIZE]> {
    const CHUNKS: usize = 4;

    let mut pre_activations = Aligned::new([simd::zeroed(); L2_SIZE / simd::F32_LANES]);

    let packed = std::slice::from_raw_parts(ft_out.as_ptr().cast::<i32>(), L1_SIZE / CHUNKS);

    let mut pairs = nnz.chunks_exact(2);

    for pair in &mut pairs {
        let index1 = *pair.get_unchecked(0) as usize;
        let index2 = *pair.get_unchecked(1) as usize;

        let input1 = simd::splat_i32(*packed.get_unchecked(index1));
        let input2 = simd::splat_i32(*packed.get_unchecked(index2));

        let weights1 = PARAMETERS.l1_weights[bucket].as_ptr().add(index1 * L2_SIZE * CHUNKS);
        let weights2 = PARAMETERS.l1_weights[bucket].as_ptr().add(index2 * L2_SIZE * CHUNKS);

        for j in (0..L2_SIZE).step_by(simd::F32_LANES) {
            let weights1 = *weights1.add(j * CHUNKS).cast();
            let weights2 = *weights2.add(j * CHUNKS).cast();

            let vector = &mut pre_activations[j / simd::F32_LANES];
            *vector = simd::double_dpbusd(*vector, input1, weights1, input2, weights2);
        }
    }

    if let Some(last) = pairs.remainder().first() {
        let index = *last as usize;
        let input = simd::splat_i32(*packed.get_unchecked(index));
        let weights = PARAMETERS.l1_weights[bucket].as_ptr().add(index * L2_SIZE * CHUNKS);

        for j in (0..L2_SIZE).step_by(simd::F32_LANES) {
            let weights = *weights.add(j * CHUNKS).cast();
            let vector = &mut pre_activations[j / simd::F32_LANES];
            *vector = simd::dpbusd(*vector, input, weights);
        }
    }

    let mut output = Aligned::new([0.0; L2_SIZE]);

    let zero = simd::zero_f32();
    let one = simd::splat_f32(1.0);
    let dequant = simd::splat_f32(DEQUANT_MULTIPLIER);

    for i in (0..L2_SIZE).step_by(simd::F32_LANES) {
        let biases = *PARAMETERS.l1_biases[bucket].as_ptr().add(i).cast();
        let vector = simd::mul_add_f32(simd::convert_to_f32(pre_activations[i / simd::F32_LANES]), dequant, biases);
        *output.as_mut_ptr().add(i).cast() = simd::clamp_f32(vector, zero, one);
    }

    output
}

pub unsafe fn propagate_l2(l1_out: Aligned<[f32; L2_SIZE]>, bucket: usize) -> Aligned<[f32; L3_SIZE]> {
    let mut output = Aligned::new(PARAMETERS.l2_biases[bucket]);

    for i in 0..L2_SIZE {
        let input = simd::splat_f32(l1_out[i]);
        let weights = PARAMETERS.l2_weights[bucket][i].as_ptr();

        for j in (0..L3_SIZE).step_by(simd::F32_LANES) {
            let weights = *weights.add(j).cast();
            let vector = output.as_mut_ptr().add(j).cast();
            *vector = simd::mul_add_f32(weights, input, *vector);
        }
    }

    let zero = simd::zero_f32();
    let one = simd::splat_f32(1.0);

    for i in (0..L3_SIZE).step_by(simd::F32_LANES) {
        let vector = output.as_mut_ptr().add(i).cast();
        *vector = simd::clamp_f32(*vector, zero, one);
    }

    output
}

pub unsafe fn propagate_l3(l2_out: Aligned<[f32; L3_SIZE]>, bucket: usize) -> f32 {
    const LANES: usize = 16 / simd::F32_LANES;

    let input = l2_out.as_ptr();
    let weights = PARAMETERS.l3_weights[bucket].as_ptr();

    let mut output = [simd::zero_f32(); LANES];

    for (lane, result) in output.iter_mut().enumerate() {
        for i in (0..L3_SIZE).step_by(LANES * simd::F32_LANES) {
            let a = *weights.add(i + lane * simd::F32_LANES).cast();
            let b = *input.add(i + lane * simd::F32_LANES).cast();

            *result = simd::mul_add_f32(a, b, *result);
        }
    }

    simd::horizontal_sum(output) + PARAMETERS.l3_biases[bucket]
}

#[cfg(all(
    not(all(target_feature = "bmi2", target_feature = "avx2")),
    not(all(target_feature = "avx512vl", target_feature = "avx512vbmi"))
))]
pub unsafe fn find_nnz(
    ft_out: &Aligned<[u8; L1_SIZE]>, nnz_table: &[SparseEntry],
) -> (Aligned<[u8; L1_SIZE / 4]>, usize) {
    let mut indexes = Aligned::new([0; L1_SIZE / 4]);
    let mut count = 0;

    let increment = 0x0808080808080808;
    let mut base: u64 = 0;

    for i in (0..L1_SIZE).step_by(2 * simd::I16_LANES) {
        let mask = simd::nnz_bitmask(*ft_out.as_ptr().add(i).cast());

        for offset in (0..simd::I32_LANES).step_by(8) {
            let slice = (mask >> offset) & 0xFF;
            let entry = nnz_table.get_unchecked(slice as usize);

            let store = indexes.as_mut_ptr().add(count).cast();
            std::ptr::write_unaligned(store, base + entry.indexes);

            count += entry.count;
            base += increment;
        }
    }

    (indexes, count)
}

#[cfg(all(
    all(target_feature = "bmi2", target_feature = "avx2"),
    not(all(target_feature = "avx512vl", target_feature = "avx512vbmi"))
))]
pub unsafe fn find_nnz(ft_out: &Aligned<[u8; L1_SIZE]>, _: &[SparseEntry]) -> (Aligned<[u8; L1_SIZE / 4]>, usize) {
    use std::arch::x86_64::*;

    let mut indexes = Aligned::new([0; L1_SIZE / 4]);
    let mut count = 0;

    let increment = 0x2020202020202020;
    let mut base0 = 0x0b0a090803020100;
    let mut base1 = 0x1b1a191813121110;
    let mut base2 = 0x0f0e0d0c07060504;
    let mut base3 = 0x1f1e1d1c17161514;

    for i in (0..L1_SIZE).step_by(8 * simd::I16_LANES) {
        let vector0 = *ft_out.as_ptr().add(i).cast();
        let vector1 = *ft_out.as_ptr().add(i + 2 * simd::I16_LANES).cast();
        let vector2 = *ft_out.as_ptr().add(i + 4 * simd::I16_LANES).cast();
        let vector3 = *ft_out.as_ptr().add(i + 6 * simd::I16_LANES).cast();
        let mask01 = _mm256_packs_epi32(vector0, vector1);
        let mask23 = _mm256_packs_epi32(vector2, vector3);
        let mask = _mm256_packs_epi16(mask01, mask23);
        let mask = _mm256_cmpgt_epi8(mask, _mm256_setzero_si256());
        let mask: [u64; 4] = std::mem::transmute(mask);

        let compressed0 = _pext_u64(base0, mask[0]);
        let compressed1 = _pext_u64(base1, mask[1]);
        let compressed2 = _pext_u64(base2, mask[2]);
        let compressed3 = _pext_u64(base3, mask[3]);

        let store = indexes.as_mut_ptr().add(count).cast();
        std::ptr::write_unaligned(store, compressed0);
        count += (mask[0].count_ones() / 8) as usize;

        let store = indexes.as_mut_ptr().add(count).cast();
        std::ptr::write_unaligned(store, compressed1);
        count += (mask[1].count_ones() / 8) as usize;

        let store = indexes.as_mut_ptr().add(count).cast();
        std::ptr::write_unaligned(store, compressed2);
        count += (mask[2].count_ones() / 8) as usize;

        let store = indexes.as_mut_ptr().add(count).cast();
        std::ptr::write_unaligned(store, compressed3);
        count += (mask[3].count_ones() / 8) as usize;

        base0 += increment;
        base1 += increment;
        base2 += increment;
        base3 += increment;
    }

    (indexes, count)
}

#[cfg(all(target_feature = "avx512vl", target_feature = "avx512vbmi"))]
pub unsafe fn find_nnz(ft_out: &Aligned<[u8; L1_SIZE]>, _: &[SparseEntry]) -> (Aligned<[u8; L1_SIZE / 4]>, usize) {
    use std::arch::x86_64::*;

    let mut indexes = Aligned::new([0; L1_SIZE / 4]);
    let mut count = 0;

    let increment = _mm512_set1_epi8(64);
    let mut base = _mm512_set_epi8(
        63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36,
        35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8,
        7, 6, 5, 4, 3, 2, 1, 0,
    );

    for i in (0..L1_SIZE).step_by(8 * simd::I16_LANES) {
        let mask0 = simd::nnz_bitmask(*ft_out.as_ptr().add(i).cast());
        let mask1 = simd::nnz_bitmask(*ft_out.as_ptr().add(i + 2 * simd::I16_LANES).cast());
        let mask2 = simd::nnz_bitmask(*ft_out.as_ptr().add(i + 4 * simd::I16_LANES).cast());
        let mask3 = simd::nnz_bitmask(*ft_out.as_ptr().add(i + 6 * simd::I16_LANES).cast());

        let mask01 = _mm512_kunpackw(mask1 as u32, mask0 as u32);
        let mask23 = _mm512_kunpackw(mask3 as u32, mask2 as u32);
        let mask = _mm512_kunpackd(mask23 as u64, mask01 as u64);

        let compressed = _mm512_maskz_compress_epi8(mask, base);

        let store = indexes.as_mut_ptr().add(count).cast();
        _mm512_storeu_si512(store, compressed);

        count += mask.count_ones() as usize;
        base = _mm512_add_epi8(base, increment);
    }

    (indexes, count)
}
