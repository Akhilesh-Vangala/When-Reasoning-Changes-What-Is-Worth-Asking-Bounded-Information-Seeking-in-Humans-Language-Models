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


def _simulated(model, p, n_games, seed):
    rng = np.random.default_rng(seed)
    tables = hm._tables(task, model, p, None)
    return [hm.simulate_game(task, model, p, int(rng.integers(mm.N)), rng, None, tables) for _ in range(n_games)]


def test_converged_fit_is_at_least_greedy_and_nested():
    games = _simulated("leaky", {"eps": 0.05, "lam": 1.0, "tau": 0.3}, 6, 1)
    paper = {m: hm.fit(task, m, games, seed=0, max_evals=150) for m in ("unbounded", "leaky")}
    out = hm.fit_converged_models(task, ["unbounded", "leaky"], games, paper=paper)
    for m in out:
        assert out[m]["loglik"] >= paper[m]["loglik"] - 1e-9
        assert out[m]["procedure"] == "converged"
    assert out["leaky"]["loglik"] >= out["unbounded"]["loglik"] - hm.NEST_TOL


def test_converged_fit_reaches_a_known_interior_optimum():
    games = _simulated("unbounded", {"lam": 2.0, "tau": 0.3}, 8, 2)
    a = hm.fit_converged(task, "unbounded", games, paper={"params": {"lam": 50.0, "tau": 5.0}, "loglik": -np.inf})
    b = hm.fit_converged(task, "unbounded", games, paper={"params": {"lam": 0.01, "tau": 0.05}, "loglik": -np.inf})
    assert abs(a["loglik"] - b["loglik"]) < 0.05  # same maximum from far-apart starts
    assert a["at_limit"] == []


def test_limits_are_respected_and_flagged():
    lims = {**hm.BOUNDS, "tau": (0.5, 100.0)}  # optimum below the lower limit
    games = _simulated("unbounded", {"lam": 2.0, "tau": 0.1}, 6, 3)
    f = hm.fit_converged(task, "unbounded", games, bounds=lims, paper={"params": {"lam": 1.0, "tau": 1.0}, "loglik": -np.inf})
    assert f["params"]["tau"] >= 0.5 - 1e-12
    assert "tau" in f["at_limit"]
