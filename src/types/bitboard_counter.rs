use crate::types::Bitboard;

#[derive(Copy, Clone, Default, Eq, PartialEq)]
pub struct BitboardCounter {
    bb: [Bitboard; 4],
}

fn adder(a: Bitboard, b: Bitboard, c: Bitboard) -> (Bitboard, Bitboard) {
    (a ^ b ^ c, (a & b) | (c & (a ^ b)))
}

impl BitboardCounter {
    pub fn update(&mut self, delta: [Bitboard; 2]) {
        let [sub, add] = delta;
        let mut carry: Bitboard;
        (self.bb[0], carry) = adder(self.bb[0], sub, add);
        (self.bb[1], carry) = adder(self.bb[1], sub, carry);
        (self.bb[2], carry) = adder(self.bb[2], sub, carry);
        (self.bb[3], _) = adder(self.bb[3], sub, carry);
    }

    pub fn add(&mut self, delta: Bitboard) {
        self.update([Bitboard::default(), delta]);
    }

    pub fn reduce(&self) -> Bitboard {
        self.bb[0] | self.bb[1] | self.bb[2] | self.bb[3]
    }
}
