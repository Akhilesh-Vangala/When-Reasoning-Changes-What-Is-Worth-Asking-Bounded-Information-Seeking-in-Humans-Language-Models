"""Process models of how a reasoner uses Mastermind feedback, fitted to the B2 battery.

Each model gives a clue weight l_i(s) for every clue i of a history; compatibility g(s) = prod_i l_i(s)
and belief b(s) proportional to g(s) (uniform prior). One shared observation model:
  classify ("is x still possible?"):  P(yes) = (1 - zeta) g(x) + zeta / 2
  identify (name one code):           P(d)   = (1 - zeta) b(d) + zeta / 256
Unparsed answers are format failures, counted separately and left out of the likelihood.
Decision temperature is fixed at 1 (not identifiable from final codes under a uniform prior).

  M0   exact inference + reporting noise    l = c                                   params: zeta
  M1   noisy evidence use                   l = (1 - nu) c + nu / 2                 params: nu, zeta
  M2c  conservative approximation (human)   l = (1 - nu) pi_c(alpha) + nu / 2       params: alpha, nu, zeta
  M2s  symmetric approximation              l = (1 - nu) pi_s(alpha) + nu / 2       params: alpha, nu, zeta
  M3   evidence retention                   clue i kept with r_i = sigmoid(k0 + k1 * age_i),
                                            age 0 = newest; exact over the 2^k kept subsets   params: k0, k1, zeta
"""
from itertools import product

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from mastermind.engine import get_game

MODELS = {"M0": ["zeta"], "M1": ["nu", "zeta"], "M2c": ["alpha", "nu", "zeta"],
          "M2s": ["alpha", "nu", "zeta"], "M3": ["k0", "k1", "zeta"]}
TRANSFORM = {"zeta": "unit", "nu": "unit", "alpha": "log", "k0": "real", "k1": "real"}
START = {"zeta": 0.1, "nu": 0.1, "alpha": 0.3, "k0": 2.0, "k1": -0.5}


class Item:
    """One battery response, pre-processed: clue indicator rows c[i, s] and the observed answer."""

    def __init__(self, mm, history, task, answer):
        self.task = task
        self.clues = [(q, mm.encode(b, w)) for q, (b, w) in history]
        self.c = np.stack([mm.table[q] == o for q, o in self.clues]).astype(np.float64)  # [k, 256]
        self.answer = answer  # classify: list of (code_idx, said_possible); identify: code_idx


def clue_weights(model, p, item, caches):
    """l[i, s] for every clue of the item (M3 handled separately)."""
    if model == "M0":
        return item.c
    if model == "M1":
        return (1 - p["nu"]) * item.c + p["nu"] / 2
    rep = caches["M2c" if model == "M2c" else "M2s"].representation(p["alpha"])
    pi = np.stack([rep[q, o] for q, o in item.clues])
    return (1 - p["nu"]) * pi + p["nu"] / 2


def item_loglik(model, p, item, caches):
    zeta = p["zeta"]
    if model == "M3":
        k = len(item.clues)
        ages = np.arange(k)[::-1]  # newest clue has age 0
        r = expit(p["k0"] + p["k1"] * ages)
        if item.task == "classify":
            g = np.prod(r[:, None] * item.c + (1 - r[:, None]), axis=0)   # independence gives the closed form
        else:
            b = np.zeros(item.c.shape[1])
            for kept in product([0, 1], repeat=k):
                kept = np.array(kept, bool)
                w = np.prod(np.where(kept, r, 1 - r))
                g_s = np.prod(item.c[kept], axis=0) if kept.any() else np.ones(item.c.shape[1])
                b += w * g_s / g_s.sum()
    else:
        l_ = clue_weights(model, p, item, caches)
        g = np.prod(l_, axis=0)
        if item.task == "identify":
            b = g / g.sum()
    if item.task == "classify":
        ll = 0.0
        for x, said in item.answer:
            py = (1 - zeta) * g[x] + zeta / 2
            ll += np.log(py if said else 1 - py)
        return ll
    return float(np.log((1 - zeta) * b[item.answer] + zeta / item.c.shape[1]))


def loglik(model, p, items, caches=None):
    return sum(item_loglik(model, p, it, caches) for it in items)


def _to_vec(model, p):
    out = []
    for k in MODELS[model]:
        t = TRANSFORM[k]
        out.append(logit(p[k]) if t == "unit" else np.log(p[k]) if t == "log" else p[k])
    return np.array(out)


def _from_vec(model, v):
    p = {}
    for k, x in zip(MODELS[model], v):
        t = TRANSFORM[k]
        p[k] = float(expit(x)) if t == "unit" else float(np.exp(np.clip(x, -9.2, 4.6))) if t == "log" else float(x)
    return p


def fit(model, items, caches=None, starts=None):
    """Maximum likelihood with Nelder-Mead on transformed parameters, several starts."""
    starts = starts or [START] + [{**START, "alpha": a, "zeta": z} for a in (0.03, 3.0) for z in (0.02, 0.3)]
    best = None
    for s in starts:
        v0 = _to_vec(model, {k: s[k] for k in MODELS[model]})
        res = minimize(lambda v: -loglik(model, _from_vec(model, v), items, caches), v0, method="Nelder-Mead",
                       options={"xatol": 1e-3, "fatol": 1e-3, "maxiter": 400})
        if best is None or res.fun < best.fun:
            best = res
        if len(MODELS[model]) == 1:
            break
    k = len(MODELS[model])
    ll = -float(best.fun)
    return {"model": model, "params": _from_vec(model, best.x), "loglik": ll, "k": k, "aic": 2 * k - 2 * ll}


def simulate(model, p, items, caches, rng):
    """Synthetic answers from a model on the same items (for parameter and model recovery)."""
    mm = get_game(4, 4)
    out = []
    for it in items:
        zeta = p["zeta"]
        if model == "M3":
            k = len(it.clues)
            r = expit(p["k0"] + p["k1"] * np.arange(k)[::-1])
            kept = rng.random(k) < r
            g = np.prod(it.c[kept], axis=0) if kept.any() else np.ones(mm.N)
        else:
            g = np.prod(clue_weights(model, p, it, caches), axis=0)
        new = Item.__new__(Item)
        new.task, new.clues, new.c = it.task, it.clues, it.c
        if it.task == "classify":
            new.answer = [(x, bool(rng.random() < (1 - zeta) * g[x] + zeta / 2)) for x, _ in it.answer]
        else:
            probs = (1 - zeta) * g / g.sum() + zeta / mm.N
            new.answer = int(rng.choice(mm.N, p=probs / probs.sum()))
        out.append(new)
    return out
