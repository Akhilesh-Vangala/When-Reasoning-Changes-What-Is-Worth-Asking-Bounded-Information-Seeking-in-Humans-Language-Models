import numpy as np

from mastermind import human_models as hm

task = hm.Task()
mm = task.mm


def test_information_gain_equals_eig_for_exact_beliefs():
    # With exact updating, lam = 0 and a very low temperature, the utility is the EIG (bits).
    rng = np.random.default_rng(0)
    secret = int(rng.integers(mm.N))
    hist = [(q, mm.decode(mm.table[q, secret])) for q in (5, 77)]
    mask = mm.consistent(hist)
    belief = mask / mask.sum()
    p_out = task.onehot @ belief
    post = task.onehot * belief[None, None, :]
    z = post.sum(-1, keepdims=True)
    post = np.divide(post, z, out=np.zeros_like(post), where=z > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = np.where(post > 0, post * np.log2(post / belief), 0).sum(-1)
    ig = (p_out * kl).sum(-1)
    assert np.allclose(ig, mm.eig(mask))


def test_unbounded_low_temperature_prefers_max_eig():
    game = mm.play_greedy_eig(10)
    ll_sharp = hm.loglik(task, "unbounded", {"lam": 1e-3, "tau": 1e-3}, [game])
    ll_flat = hm.loglik(task, "unbounded", {"lam": 1e-3, "tau": 100.0}, [game])
    assert ll_sharp > ll_flat
    assert np.isclose(ll_flat, -len(game) * np.log(mm.N), atol=1.0)


def test_leaky_with_zero_noise_matches_unbounded():
    game = mm.play_greedy_eig(42)
    p = {"lam": 2.0, "tau": 0.3}
    assert np.isclose(hm.loglik(task, "unbounded", p, [game]), hm.loglik(task, "leaky", {**p, "eps": 0.0}, [game]))
