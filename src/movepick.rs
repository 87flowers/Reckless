use crate::{
    search::NodeType,
    thread::ThreadData,
    types::{ArrayVec, MAX_MOVES, Move, MoveList, PieceType},
};

#[derive(Copy, Clone, Eq, PartialEq, PartialOrd)]
pub enum Stage {
    HashMove,
    GenerateNoisy,
    GoodNoisy,
    GenerateQuiet,
    Quiet,
    BadNoisy,
}

pub struct MovePicker {
    list: MoveList,
    tt_move: Move,
    threshold: Option<i32>,
    stage: Stage,
    bad_noisy: ArrayVec<Move, MAX_MOVES>,
    bad_noisy_idx: usize,
}

impl MovePicker {
    pub const fn new(tt_move: Move) -> Self {
        Self {
            list: MoveList::new(),
            tt_move,
            threshold: None,
            stage: if tt_move.is_some() { Stage::HashMove } else { Stage::GenerateNoisy },
            bad_noisy: ArrayVec::new(),
            bad_noisy_idx: 0,
        }
    }

    pub const fn new_probcut(threshold: i32) -> Self {
        Self {
            list: MoveList::new(),
            tt_move: Move::NULL,
            threshold: Some(threshold),
            stage: Stage::GenerateNoisy,
            bad_noisy: ArrayVec::new(),
            bad_noisy_idx: 0,
        }
    }

    pub const fn new_qsearch() -> Self {
        Self {
            list: MoveList::new(),
            tt_move: Move::NULL,
            threshold: None,
            stage: Stage::GenerateNoisy,
            bad_noisy: ArrayVec::new(),
            bad_noisy_idx: 0,
        }
    }

    pub const fn stage(&self) -> Stage {
        self.stage
    }

    pub fn next<NODE: NodeType>(&mut self, td: &ThreadData, skip_quiets: bool, ply: isize) -> Option<Move> {
        if self.stage == Stage::HashMove {
            self.stage = Stage::GenerateNoisy;

            if td.board.is_pseudo_legal(self.tt_move) {
                return Some(self.tt_move);
            }
        }

        if self.stage == Stage::GenerateNoisy {
            self.stage = Stage::GoodNoisy;
            td.board.append_noisy_moves(&mut self.list);
            self.score_noisy(td);
        }

        if self.stage == Stage::GoodNoisy {
            while !self.list.is_empty() {
                let index = self.find_best_score_index();
                let entry = &self.list.remove(index);
                if entry.mv == self.tt_move {
                    continue;
                }

                let threshold = self.threshold.unwrap_or_else(|| -entry.score / 46 + 109);
                if !td.board.see(entry.mv, threshold) {
                    self.bad_noisy.push(entry.mv);
                    continue;
                }

                if NODE::ROOT {
                    self.score_noisy(td);
                }

                return Some(entry.mv);
            }

            self.stage = Stage::GenerateQuiet;
        }

        if self.stage == Stage::GenerateQuiet {
            if !skip_quiets {
                self.stage = Stage::Quiet;
                td.board.append_quiet_moves(&mut self.list);
                self.score_quiet(td, ply);
            } else {
                self.stage = Stage::BadNoisy;
            }
        }

        if self.stage == Stage::Quiet {
            if !skip_quiets {
                while !self.list.is_empty() {
                    let index = self.find_best_score_index();
                    let entry = &self.list.remove(index);
                    if entry.mv == self.tt_move {
                        continue;
                    }

                    if NODE::ROOT {
                        self.score_quiet(td, ply);
                    }

                    return Some(entry.mv);
                }
            }

            self.stage = Stage::BadNoisy;
        }

        // Stage::BadNoisy
        while self.bad_noisy_idx < self.bad_noisy.len() {
            let mv = self.bad_noisy[self.bad_noisy_idx];
            self.bad_noisy_idx += 1;

            if mv == self.tt_move {
                continue;
            }

            return Some(mv);
        }

        None
    }

    #[cfg(not(target_feature = "avx2"))]
    fn find_best_score_index(&self) -> usize {
        let mut best_index = 0;
        let mut best_score = i32::MIN;

        for (index, entry) in self.list.iter().enumerate() {
            if entry.score >= best_score {
                best_index = index;
                best_score = entry.score;
            }
        }

        best_index
    }

    #[cfg(target_feature = "avx2")]
    fn find_best_score_index(&self) -> usize {
        use std::arch::x86_64::*;
        unsafe {
            let invalid = _mm256_set1_epi64x(i64::MIN);
            let step = _mm256_set1_epi64x(8);
            let last_index = _mm256_set1_epi64x(self.list.len() as i64 - 1);
            let mask = _mm256_set1_epi64x(0xFFFFFFFF00000000u64 as i64);

            let mut index0 = _mm256_set_epi64x(3, 2, 1, 0);
            let mut index1 = _mm256_set_epi64x(7, 6, 5, 4);
            let mut best0 = invalid;
            let mut best1 = invalid;

            for i in (0..self.list.len()).step_by(8) {
                // SAFETY: This will never read beyond the end of the list, because MAX_MOVES is a multiple of 8.
                let curr0 = _mm256_loadu_si256(self.list.as_ptr().add(i).cast());
                let curr1 = _mm256_loadu_si256(self.list.as_ptr().add(i + 4).cast());
                let curr0 = _mm256_or_si256(_mm256_and_si256(curr0, mask), index0);
                let curr1 = _mm256_or_si256(_mm256_and_si256(curr1, mask), index1);
                let curr0 = _mm256_blendv_epi8(curr0, invalid, _mm256_cmpgt_epi64(index0, last_index));
                let curr1 = _mm256_blendv_epi8(curr1, invalid, _mm256_cmpgt_epi64(index1, last_index));

                best0 = _mm256_blendv_epi8(best0, curr0, _mm256_cmpgt_epi64(curr0, best0));
                best1 = _mm256_blendv_epi8(best1, curr1, _mm256_cmpgt_epi64(curr1, best1));

                index0 = _mm256_add_epi64(index0, step);
                index1 = _mm256_add_epi64(index1, step);
            }

            let best = _mm256_blendv_epi8(best0, best1, _mm256_cmpgt_epi64(best1, best0));
            let best = std::mem::transmute::<__m256i, [i64; 4]>(best);
            let best = best.iter().max().unwrap();
            (best & 0xFF) as usize
        }
    }

    fn score_noisy(&mut self, td: &ThreadData) {
        let threats = td.board.threats();

        for entry in self.list.iter_mut() {
            let mv = entry.mv;

            if mv == self.tt_move {
                entry.score = i32::MIN;
                continue;
            }

            let captured =
                if entry.mv.is_en_passant() { PieceType::Pawn } else { td.board.piece_on(mv.to()).piece_type() };

            entry.score =
                16 * captured.value() + td.noisy_history.get(threats, td.board.moved_piece(mv), mv.to(), captured);
        }
    }

    fn score_quiet(&mut self, td: &ThreadData, ply: isize) {
        let threats = td.board.threats();
        let side = td.board.side_to_move();

        for entry in self.list.iter_mut() {
            let mv = entry.mv;

            if mv == self.tt_move {
                entry.score = i32::MIN;
                continue;
            }

            entry.score = (994 * td.quiet_history.get(threats, side, mv)
                + 1049 * td.conthist(ply, 1, mv)
                + 990 * td.conthist(ply, 2, mv)
                + 969 * td.conthist(ply, 4, mv)
                + 1088 * td.conthist(ply, 6, mv))
                / 1024;
        }
    }
}
