import ast
import csv
from pathlib import Path

import numpy as np
import pytest

from mastermind.engine import get_game

ROOT = Path(__file__).resolve().parents[1]
E3 = ROOT / "data/human/cheyette2025_osf_vjrqu/Experiments/E3/mastermind_E3.csv"


@pytest.fixture(scope="module")
def mm():
    return get_game(4, 4)


def test_code_space(mm):
    assert mm.N == 256
    assert mm.to_str(0) == "1111" and mm.to_str(255) == "4444"


def test_feedback_examples(mm):
    # Example from Cheyette et al.: guess 1233 vs code 1344 -> 1 black, 1 white
    assert mm.feedback([1, 2, 3, 3], [1, 3, 4, 4]) == (1, 1)
    assert mm.feedback([1, 1, 2, 2], [2, 2, 1, 1]) == (0, 4)
    assert mm.feedback([1, 2, 3, 4], [1, 2, 3, 4]) == (4, 0)


def test_table_matches_scalar_feedback(mm):
    rng = np.random.default_rng(0)
    for _ in range(500):
        q, s = rng.integers(mm.N, size=2)
        assert mm.decode(mm.table[q, s]) == mm.feedback(mm.codes[q], mm.codes[s])


def test_eig_is_outcome_entropy(mm):
    full = np.ones(mm.N, bool)
    e = mm.eig(full)
    q = mm.index("1111")
    p = mm.outcome_probs(q, full)
    p = p[p > 0]
    assert e[q] == pytest.approx(-(p * np.log2(p)).sum())
    # A guess with more distinct digits is more informative at the start
    assert e[mm.index("1234")] > e[mm.index("1111")]


def test_greedy_eig_solves_quickly(mm):
    lengths = [len(mm.play_greedy_eig(s)) for s in range(mm.N)]
    assert max(lengths) <= 6
    # Cheyette et al. (2025) report ~3.6 guesses for their information-maximizing model
    assert 3.0 < np.mean(lengths) < 4.5


@pytest.mark.skipif(not E3.exists(), reason="human data not downloaded")
def test_matches_human_data_codes_remaining(mm):
    """Reproduce the authors' n_remaining_prev / n_remaining_post for every human guess."""
    games = {}
    with open(E3) as f:
        for r in csv.DictReader(f):
            games.setdefault((r["pid"], r["trial_id"]), []).append(r)
    checked = 0
    for rows in games.values():
        rows.sort(key=lambda r: int(r["guess_number"]))
        mask = np.ones(mm.N, bool)
        for r in rows:
            if r["n_remaining_post"] in ("", "NA"):
                continue
            q = mm.index(ast.literal_eval(r["guess"]))
            assert mask.sum() == int(float(r["n_remaining_prev"]))
            mask = mm.consistent([(q, (int(r["n_green"]), int(r["n_yellow"])))], mask)
            assert mask.sum() == int(float(r["n_remaining_post"]))
            checked += 1
    assert checked > 3000
