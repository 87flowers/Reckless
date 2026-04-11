use std::ops::{Index, IndexMut};

use crate::{
    history::ContinuationHistorySubtable,
    types::{MAX_PLY, Move, Piece, Score},
};

pub struct Stack {
    data: [StackEntry; MAX_PLY + 16],
    conthist_sentinel: ContinuationHistorySubtable,
    contcorrhist_sentinel: [[i16; 64]; 13],
}

impl Stack {
    pub fn sentinel(&mut self) -> &mut StackEntry {
        unsafe { self.data.get_unchecked_mut(0) }
    }

    pub fn new() -> Box<Self> {
        let mut stack = Box::new(Self::default());
        let conthist_ptr = &raw mut stack.conthist_sentinel;
        let contcorrhist_ptr = &raw mut stack.contcorrhist_sentinel;
        for entry in &mut stack.data {
            entry.conthiste = conthist_ptr;
            entry.conthisto = [conthist_ptr; 2];
            entry.contcorrhist = contcorrhist_ptr;
        }
        stack
    }
}

impl Default for Stack {
    fn default() -> Self {
        Self {
            data: [StackEntry::default(); MAX_PLY + 16],
            conthist_sentinel: [[0; 64]; 13],
            contcorrhist_sentinel: [[0; 64]; 13],
        }
    }
}

#[derive(Copy, Clone)]
pub struct StackEntry {
    pub mv: Move,
    pub piece: Piece,
    pub eval: i32,
    pub excluded: Move,
    pub tt_move: Move,
    pub tt_pv: bool,
    pub cutoff_count: i32,
    pub move_count: i32,
    pub reduction: i32,
    pub conthiste: *mut ContinuationHistorySubtable,
    pub conthisto: [*mut ContinuationHistorySubtable; 2],
    pub contcorrhist: *mut [[i16; 64]; 13],
}

unsafe impl Send for StackEntry {}

impl Default for StackEntry {
    fn default() -> Self {
        Self {
            mv: Move::NULL,
            piece: Piece::None,
            eval: Score::NONE,
            excluded: Move::NULL,
            tt_move: Move::NULL,
            tt_pv: false,
            cutoff_count: 0,
            move_count: 0,
            reduction: 0,
            conthiste: std::ptr::null_mut(),
            conthisto: [std::ptr::null_mut(); 2],
            contcorrhist: std::ptr::null_mut(),
        }
    }
}

impl Index<isize> for Stack {
    type Output = StackEntry;

    fn index(&self, index: isize) -> &Self::Output {
        debug_assert!(index + 8 >= 0 && index < MAX_PLY as isize + 16);
        unsafe { self.data.get_unchecked((index + 8) as usize) }
    }
}

impl IndexMut<isize> for Stack {
    fn index_mut(&mut self, index: isize) -> &mut Self::Output {
        debug_assert!(index + 8 >= 0 && index < MAX_PLY as isize + 16);
        unsafe { self.data.get_unchecked_mut((index + 8) as usize) }
    }
}
