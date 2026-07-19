use crate::{thread::ThreadData, types::Score};

pub fn correct_eval(td: &ThreadData, raw_eval: i32, correction_value: i32, multiplicative_correction: i32) -> i32 {
    let mut eval = (raw_eval * (21032 + td.board.material())
        + td.optimism[td.board.side_to_move()] * (1548 + td.board.material()))
        / 27015;

    eval = eval * (200 - td.board.fiftymove_clock() as i32) / 200;

    eval += correction_value;

    (eval as i64 * multiplicative_correction as i64 / 2048)
        .clamp((-Score::TB_WIN_IN_MAX + 1) as i64, (Score::TB_WIN_IN_MAX - 1) as i64) as i32
}
