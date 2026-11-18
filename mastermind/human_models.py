"""The three models of Cheyette et al. (2025, Methods and Eqs. 4, 6-8), fitted to free-play guesses.

  unbounded  exact belief updating; noise only in the choice           params: lam, tau
  leaky      belief updating with a Bernoulli likelihood, eps noise    params: eps, lam, tau
  bounded    leaky model whose likelihood for each piece of feedback is
             the alpha-weighted mixture of cached approximations       params: eps, alpha, lam, tau
             (beta = 1, not fitted)

Utility of query q given belief P over codes:
  U(q) = lam * P(q) + sum_o P(o | q) * KL[ P*(H | o) || P(H) ]      (bits)
where P(o | q) uses the true feedback function and P* is the model's updated belief.
Choice: P(choose q) proportional to exp(U(q) / tau), over all 256 codes.
The likelihood of feedback under a representation pi(h) in [0, 1] is (1 - eps) * pi(h) + eps * (1 - pi(h)),
so the bounded model reduces to the leaky model when every approximation is exact.
"""
import numpy as np
from scipy.special import logsumexp, softmax

from mastermind.engine import get_game

N_OUT = 25  # outcome codes black * 5 + white for 4 positions

MODELS = {
    "unbounded": ["lam", "tau"],
    "leaky": ["eps", "lam", "tau"],
    "bounded": ["eps", "alpha", "lam", "tau"],
}


class Task:
    """Feedback tables shared by all models."""

    def __init__(self):
        self.mm = get_game(4, 4)
        F = self.mm.table.astype(np.int64)  # F[q, h]
        self.onehot = np.zeros((self.mm.N, N_OUT, self.mm.N), bool)
        self.onehot[np.arange(self.mm.N)[:, None], F, np.arange(self.mm.N)[None, :]] = True
        self.onehot_flat = self.onehot.reshape(self.mm.N * N_OUT, self.mm.N).astype(np.float64)


class BoundedCache:
    """Rebuilt approximation cache (experiments/study_a/build_cache.py)."""

    def __init__(self, path):
        z = np.load(path)
        self.mm = get_game(4, 4)
        self.off = z["case_offsets"]
        self.ext = z["ext"].astype(np.float64)
        self.logp, self.dist = z["logp"], z["dist"].astype(np.float64)
        self.case_pattern = z["case_pattern"]
        self.case_feedback = z["case_feedback"]
        self.query_pattern, self.query_perm = z["query_pattern"], z["query_perm"]
        self.patterns = ["1111", "1112", "1122", "1123", "1234"]
        self._memo = {}

    def representation(self, alpha):
        """pi[q, o, h]: probability that code h is kept after query q gets outcome o, at cost alpha."""
        key = round(float(alpha), 10)
        if key in self._memo:
            return self._memo[key]
        mm = self.mm
        per_case = np.zeros((len(self.case_pattern), mm.N))
        for c in range(len(self.case_pattern)):
            a, b = self.off[c], self.off[c + 1]
            w = softmax(alpha * self.logp[a:b] - self.dist[a:b])
            per_case[c] = w @ self.ext[a:b]
        index = {(str(p), int(fb[0]), int(fb[1])): c for c, (p, fb) in enumerate(zip(self.case_pattern, self.case_feedback))}
        pi = np.zeros((mm.N, N_OUT, mm.N))
        for q in range(mm.N):
            pat = self.patterns[self.query_pattern[q]]
            T = self.query_perm[q]
            for o in set(mm.table[q].tolist()):
                b, w = mm.decode(o)
                pi[q, o] = per_case[index[(pat, b, w)]][T]
        if len(self._memo) > 64:
            self._memo.clear()
        self._memo[key] = pi
        return pi


def representation(task, model, p, cache=None):
    if model == "bounded":
        return cache.representation(p["alpha"])
    return task.onehot.astype(np.float64)


def _tables(task, model, p, cache):
    """Likelihood L[q, o, h] and L log2 L, flattened to (q*o, h) for matrix-vector products."""
    rep = representation(task, model, p, cache)
    eps = p.get("eps", 0.0)
    lik = (1 - eps) * rep + eps * (1 - rep)
    with np.errstate(divide="ignore", invalid="ignore"):
        liklog = np.where(lik > 0, lik * np.log2(lik), 0.0)
    n = task.mm.N
    return lik.reshape(n * N_OUT, n), liklog.reshape(n * N_OUT, n)


def game_loglik(task, model, p, game, cache=None, tables=None):
    """Sum of log P(choose q_t) over a game [(q, (black, white)), ...].

    With posterior P*(h | o) = P(h) L(o | h) / z(o) and z(o) = sum_h P(h) L(o | h),
      KL[P* || P] = (sum_h P(h) L log2 L) / z  -  log2 z,
    so each decision needs only three matrix-vector products."""
    mm = task.mm
    n = mm.N
    lik, liklog = tables if tables is not None else _tables(task, model, p, cache)
    onehot = task.onehot_flat
    belief = np.full(n, 1.0 / n)
    total = 0.0
    for q_t, (b, w) in game:
        o_t = mm.encode(b, w)
        p_out = (onehot @ belief).reshape(n, N_OUT)
        z = (lik @ belief).reshape(n, N_OUT)
        a = (liklog @ belief).reshape(n, N_OUT)
        with np.errstate(divide="ignore", invalid="ignore"):
            kl = np.where(z > 0, a / z - np.log2(z), 0.0)
        utility = p["lam"] * belief + (p_out * kl).sum(-1)
        logits = utility / p["tau"]
        total += logits[q_t] - logsumexp(logits)
        belief = belief * lik[q_t * N_OUT + o_t]
        belief /= belief.sum()
    return float(total)


def loglik(task, model, p, games, cache=None):
    tables = _tables(task, model, p, cache)
    return sum(game_loglik(task, model, p, g, cache, tables) for g in games)


def simulate_game(task, model, p, secret, rng, cache=None, tables=None, max_guesses=15):
    """Play one game by sampling each guess from the model's choice rule (for recovery checks)."""
    mm = task.mm
    n = mm.N
    lik, liklog = tables if tables is not None else _tables(task, model, p, cache)
    onehot = task.onehot_flat
    belief = np.full(n, 1.0 / n)
    game = []
    for _ in range(max_guesses):
        p_out = (onehot @ belief).reshape(n, N_OUT)
        z = (lik @ belief).reshape(n, N_OUT)
        a = (liklog @ belief).reshape(n, N_OUT)
        with np.errstate(divide="ignore", invalid="ignore"):
            kl = np.where(z > 0, a / z - np.log2(z), 0.0)
        logits = (p["lam"] * belief + (p_out * kl).sum(-1)) / p["tau"]
        q = int(rng.choice(n, p=softmax(logits)))
        b, w = mm.decode(mm.table[q, secret])
        game.append((q, (b, w)))
        if b == mm.P:
            break
        belief = belief * lik[q * N_OUT + mm.encode(b, w)]
        belief /= belief.sum()
    return game


# ---- fitting (paper: greedy stochastic search) ------------------------------
BOUNDS = {"lam": (1e-3, 1e3), "tau": (1e-3, 1e2), "alpha": (1e-4, 100.0), "eps": (1e-6, 0.5)}


def _propose(name, value, rng, scale=0.1):
    if name == "eps":
        x = np.log(value / (1 - value)) + rng.normal(0, scale)
        v = 1 / (1 + np.exp(-x))
    else:
        v = float(np.exp(np.log(value) + rng.normal(0, scale)))
    lo, hi = BOUNDS[name]
    return float(np.clip(v, lo, hi))


def fit(task, model, games, cache=None, starts=None, seed=0, patience=100, min_gain=1.0, max_evals=3000):
    """Maximum-likelihood parameters by the paper's greedy stochastic search, from several starts.
    Stops a run when the log-likelihood improves by less than `min_gain` over `patience` proposals."""
    rng = np.random.default_rng(seed)
    names = MODELS[model]
    starts = starts or [dict(zip(names, s)) for s in _default_starts(model)]
    best_p, best_ll, evals = None, -np.inf, 0
    for start in starts:
        p = {k: start[k] for k in names}
        ll = loglik(task, model, p, games, cache)
        evals += 1
        anchor = ll
        since = 0
        while evals < max_evals:
            name = names[rng.integers(len(names))]
            cand = dict(p)
            cand[name] = _propose(name, p[name], rng)
            cl = loglik(task, model, cand, games, cache)
            evals += 1
            since += 1
            if cl > ll:
                p, ll = cand, cl
            if since >= patience:
                if ll - anchor < min_gain:
                    break
                anchor, since = ll, 0
        if ll > best_ll:
            best_p, best_ll = p, ll
    k = len(names)
    return {"model": model, "params": best_p, "loglik": best_ll, "k": k, "aic": 2 * k - 2 * best_ll, "evals": evals}


def _default_starts(model):
    if model == "unbounded":
        return [(1.0, 0.5), (5.0, 0.2), (0.3, 1.0)]
    if model == "leaky":
        return [(0.01, 1.0, 0.5), (0.1, 5.0, 0.2), (0.001, 0.3, 1.0)]
    return [(0.01, a, 1.0, 0.5) for a in (0.01, 0.3, 3.0)] + [(0.1, 1.0, 5.0, 0.2)]
