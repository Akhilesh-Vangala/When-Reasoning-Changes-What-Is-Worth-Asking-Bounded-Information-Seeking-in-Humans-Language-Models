import json

import numpy as np

from mastermind import process_models as pm
from mastermind.engine import get_game

mm = get_game(4, 4)
fams = [f for f in json.load(open("data/splits/families_v1.json")) if f["split"] == "development"][:12]


def _items(answer_fn):
    items = []
    for f in fams:
        h = [(mm.index(g), (b, w)) for g, b, w in f["history"]]
        mask = mm.consistent(h)
        xs = list(np.flatnonzero(mask)[:4]) + list(np.flatnonzero(~mask)[:4])
        items.append(pm.Item(mm, h, "classify", answer_fn(xs, mask)))
        items.append(pm.Item(mm, h, "identify", int(np.flatnonzero(mask)[0])))
    return items


def test_exact_answers_favor_exact_model_with_low_noise():
    items = _items(lambda xs, mask: [(int(x), bool(mask[x])) for x in xs])
    f0 = pm.fit("M0", items)
    f1 = pm.fit("M1", items)
    assert f0["params"]["zeta"] < 0.05
    assert f1["loglik"] >= f0["loglik"] - 1e-6


def test_m3_full_retention_equals_m0():
    items = _items(lambda xs, mask: [(int(x), bool(mask[x])) for x in xs])
    p0 = {"zeta": 0.1}
    p3 = {"k0": 30.0, "k1": 0.0, "zeta": 0.1}
    assert np.isclose(pm.loglik("M0", p0, items), pm.loglik("M3", p3, items), atol=1e-6)


def test_noise_recovery_m1():
    rng = np.random.default_rng(0)
    items = _items(lambda xs, mask: [(int(x), bool(mask[x])) for x in xs]) * 6
    sim = pm.simulate("M1", {"nu": 0.3, "zeta": 0.05}, items, None, rng)
    f = pm.fit("M1", sim)
    assert abs(f["params"]["nu"] - 0.3) < 0.15


def test_predicted_usefulness_exact_model_equals_vstar():
    from mastermind import usefulness as us
    f = fams[0]
    h = [(mm.index(g), (b, w)) for g, b, w in f["history"]]
    mask = mm.consistent(h)
    q = int(np.flatnonzero(~mask)[0])
    u = us.predicted_usefulness(mm, "M0", {"zeta": 0.0}, h, q, None)
    vstar = len(set(mm.table[q, mask].tolist())) / mask.sum()
    assert np.isclose(u, vstar)


def test_full_support_recovers_exact_usefulness():
    from mastermind import usefulness as us
    f = fams[1]
    h = [(mm.index(g), (b, w)) for g, b, w in f["history"]]
    mask = mm.consistent(h)
    q = int(np.flatnonzero(~mask)[3])
    u_noisy = us.predicted_usefulness(mm, "M1", {"nu": 0.4, "zeta": 0.0}, h, q, None)
    u_full = us.predicted_usefulness(mm, "M1", {"nu": 0.4, "zeta": 0.0}, h, q, None, delta=1.0)
    u_exact = us.predicted_usefulness(mm, "M0", {"zeta": 0.0}, h, q, None)
    assert u_noisy < u_full and np.isclose(u_full, u_exact)
