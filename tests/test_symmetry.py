import numpy as np

from mastermind.engine import get_game
from mastermind.symmetry import family_key, transforms

mm = get_game(4, 4)


def test_transforms_preserve_feedback():
    T = transforms()
    assert T.shape == (576, 256)
    rng = np.random.default_rng(0)
    for t in T[rng.choice(576, 20, replace=False)]:
        q, s = rng.integers(256, size=2)
        assert mm.table[q, s] == mm.table[t[q], t[s]]


def test_eig_and_vstar_invariant():
    rng = np.random.default_rng(1)
    s = int(rng.integers(256))
    hist = [(q, mm.decode(mm.table[q, s])) for q in (3, 200)]
    t = transforms()[123]
    hist2 = [(int(t[q]), fb) for q, fb in hist]
    m1, m2 = mm.consistent(hist), mm.consistent(hist2)
    assert m1.sum() == m2.sum()
    assert np.isclose(np.sort(mm.eig(m1)), np.sort(mm.eig(m2))).all()


def test_family_key_ignores_order_and_relabeling():
    rng = np.random.default_rng(2)
    s = int(rng.integers(256))
    hist = [(q, mm.decode(mm.table[q, s])) for q in (17, 99, 250)]
    t = transforms()[300]
    relabeled = [(int(t[q]), fb) for q, fb in hist][::-1]
    assert family_key(hist) == family_key(relabeled)
    other = [(q, mm.decode(mm.table[q, (s + 1) % 256])) for q in (17, 99, 250)]
    assert family_key(other) != family_key(hist) or mm.consistent(other).sum() == mm.consistent(hist).sum()
