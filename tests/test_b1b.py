import importlib.util
import json
from pathlib import Path

import pytest

from mastermind.engine import get_game

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("b1b", ROOT / "experiments/study_b/fit_llm_human_models.py")
b1b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b1b)


def _turn(guess, black=0, white=0):
    return {"guess": guess, "black": black, "white": white}


def _write(tmp_path, recs):
    (tmp_path / "m_free_T0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))


def test_llm_games_drops_invalid_turns_and_keeps_order(tmp_path, monkeypatch):
    monkeypatch.setattr(b1b, "B1", tmp_path)
    mm = get_game(4, 4)
    _write(tmp_path, [
        {"game_id": "free:10", "solved": False, "turns": [_turn("1122", 1, 0), _turn(None)]},
        {"game_id": "free:2", "solved": True, "turns": [_turn(None), _turn("1234", 4, 0)]},
        {"game_id": "free:3", "solved": False, "turns": [_turn(None)]},
    ])
    games, n = b1b.llm_games(mm, "m", "T0")
    assert n == 3
    assert games == [(2, [(mm.index("1234"), (4, 0))])]
    games, _ = b1b.llm_games(mm, "m", "T0", solved_only=False)
    assert [i for i, _ in games] == [2, 10]  # numeric order; the all-invalid game has no decisions
    assert games[1][1] == [(mm.index("1122"), (1, 0))]


def test_llm_games_rejects_duplicate_games(tmp_path, monkeypatch):
    monkeypatch.setattr(b1b, "B1", tmp_path)
    rec = {"game_id": "free:0", "solved": True, "turns": [_turn("1234", 4, 0)]}
    _write(tmp_path, [rec, rec])
    with pytest.raises(AssertionError):
        b1b.llm_games(get_game(4, 4), "m", "T0")


def test_percentile_counts_ties_half():
    assert b1b.percentile(2, [1, 2, 3, 4]) == 0.375
    assert b1b.percentile(0, [1, 2]) == 0.0
