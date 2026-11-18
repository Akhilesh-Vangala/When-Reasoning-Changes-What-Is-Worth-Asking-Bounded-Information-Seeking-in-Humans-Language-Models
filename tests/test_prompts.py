import numpy as np

from mastermind import prompts
from mastermind.engine import get_game

mm = get_game(4, 4)


def _state(seed=1):
    rng = np.random.default_rng(seed)
    secret = int(rng.integers(mm.N))
    hist = [(q, mm.decode(mm.table[q, secret])) for q in rng.integers(mm.N, size=2)]
    return secret, hist


def test_rules_example_is_correct():
    assert mm.feedback([1, 2, 1, 3], [1, 1, 2, 2]) == (1, 2)


def test_no_leakage_of_secret_or_survivors():
    secret, hist = _state()
    mask = mm.consistent(hist)
    for task in ("guess", "identify"):
        for cond in prompts.GUIDANCE:
            p, _ = prompts.build(mm, task, cond, hist)
            assert mm.to_str(secret) not in p.replace(" ", "") or any(mm.to_str(q) == mm.to_str(secret) for q, _ in hist)
            survivors = [mm.to_str(i) for i in np.flatnonzero(mask) if all(mm.to_str(i) != mm.to_str(q) for q, _ in hist)]
            assert not any(s in p for s in survivors if s != "1234")
            for word in ("EIG", "information gain", "remaining", "survivor"):
                assert word not in p


def test_length_control_matches_structured_aid():
    n1, n3 = len(prompts.GUIDANCE["T1"].split()), len(prompts.GUIDANCE["T3"].split())
    assert abs(n1 - n3) / n3 < 0.15


def test_parsing():
    assert prompts.parse(mm, "guess", 'thinking...\n{"guess": "1234"}') == mm.index("1234")
    assert prompts.parse(mm, "guess", '{"guess": "1235"}') is None
    assert prompts.parse(mm, "identify", 'x {"code": "4411"}') == mm.index("4411")
    assert prompts.parse(mm, "choice", '{"choice": "b"}') == "B"
    codes = [mm.index("1111"), mm.index("2222")]
    got = prompts.parse(mm, "classify", '{"1111": "possible", "2222": "Impossible", "3333": "possible"}', codes)
    assert got == {codes[0]: True, codes[1]: False}
    assert prompts.parse(mm, "guess", "no json here") is None


def test_template_hash_is_stable_and_distinct():
    hashes = {prompts.template_hash(t, c) for t in prompts.TASKS for c in prompts.GUIDANCE}
    assert len(hashes) == len(prompts.TASKS) * len(prompts.GUIDANCE) - len(prompts.TASKS)  # T0 and T4 share text


def test_classify_example_does_not_label_listed_codes():
    secret, hist = _state(3)
    codes = [5, 17, 200]
    p, _ = prompts.build(mm, "classify", "T0", hist, codes=codes)
    tail = p.split("Codes:")[1].split("\n", 1)[1]
    assert not any(mm.to_str(c) in tail for c in codes)
