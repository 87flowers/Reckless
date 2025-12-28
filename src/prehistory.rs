use crate::{
    board::Board,
    types::{Move, Piece},
};

#[derive(Clone)]
pub struct Prehistory {
    pub mv: Move,
    pub in_check: bool,
    pub moved_piece: Piece,
}

impl Prehistory {
    pub fn new(board: &Board, mv: Move) -> Prehistory {
        Prehistory {
            mv: mv,
            in_check: board.in_check(),
            moved_piece: board.moved_piece(mv),
        }
    }
}
