"""C-c: the prospective prediction test.

  calibrate  base process model from B3 (baseline T0 fit) + support map delta for T3 fitted on the
             support-calibration usefulness runs; rival predictors fitted on the same calibration data
  select     choose confirmation query pairs WITHOUT outcome data, in three strata
             (a) matched V*, largest predicted |Gamma|; (b) similar EIG, different response burden
             (distinct digits); (c) predicted Gamma near zero
  lock       write prereg/prediction_lock.json: predicted Gamma for every predictor and pair, plus the
             hashes of the splits, frozen prompts, calibration and pair manifest. Commit it to git
             BEFORE the confirmation runs; the commit is the lock.
  score      observed Gamma = [U(q1,T3) - U(q2,T3)] - [U(q1,T0) - U(q2,T0)] from the confirmation runs;
             family-level MSPE per predictor; paired differences vs each rival with a cluster bootstrap

Predictors: support_map (the model), reporting_only (only response noise changes), uniform_gain
(Gamma = 0), difficulty_regression (ridge on V*, EIG, K, clues, digits, predicted baseline U),
ideal_noise (exact reasoner whose response noise changes).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mastermind import human_models as hm  # noqa: E402
from mastermind import process_models as pm  # noqa: E402
from mastermind import usefulness as us  # noqa: E402
from mastermind.engine import get_game  # noqa: E402
from mastermind.states import n_distinct  # noqa: E402

FAMILIES = ROOT / "data/splits/families_v1.json"
UC = ROOT / "results/study_c/usefulness_choice"
FITS = ROOT / "results/study_b/process_fits"
CAL = ROOT / "results/study_c/cc_calibration.json"
PAIRS = ROOT / "data/splits/confirmation_pairs_v1.json"
LOCK = ROOT / "prereg/prediction_lock.json"
CACHES = {"M2c": ROOT / "data/model_cache/bounded_cache_v1.npz", "M2s": ROOT / "data/model_cache/symmetric_cache_v1.npz"}
PREDICTORS = ["support_map", "reporting_only", "uniform_gain", "difficulty_regression", "ideal_noise"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def families(mm):
    return {f["family_id"]: [(mm.index(g), (b, w)) for g, b, w in f["history"]] for f in json.loads(FAMILIES.read_text())}


def branch_records(model, condition, split):
    path = UC / f"{model}_{condition}_{split}.jsonl"
    pairs = {p["pair_id"]: p for p in json.loads((UC / f"{model}_{condition}_{split}_pairs.json").read_text())}
    rows = [json.loads(line) for line in open(path)]
    return [r for r in rows if r["kind"] == "branch"], pairs


def observed_u(rows, pairs):
    """{(family, query code): U} from branch records."""
    groups = {}
    for r in rows:
        p = pairs[r["pair_id"]]
        groups.setdefault((p["family_id"], p[r["query"]]), []).append(r)
    return {k: us.observed_usefulness(v) for k, v in groups.items()}


class Branches:
    """Identification answers in branches, pre-processed for fast likelihoods over delta / zeta."""

    def __init__(self, mm, fams, rows, pairs, model, theta, caches):
        self.items = []
        for r in rows:
            if r["answer"] is None:
                continue
            p = pairs[r["pair_id"]]
            hist = fams[p["family_id"]] + [(mm.index(p[r["query"]]), tuple(r["outcome"]))]
            clues = [(h, mm.encode(b, w)) for h, (b, w) in hist]
            c = np.stack([mm.table[h] == o for h, o in clues]).astype(np.float64)
            item = pm.Item.__new__(pm.Item)
            item.clues, item.c = clues, c
            base = c if model == "M0" else pm.clue_weights(model, theta, item, caches)
            self.items.append((base, c, mm.index(r["answer"])))

    def loglik(self, delta, zeta):
        ll = 0.0
        for base, c, d in self.items:
            g = np.prod((1 - delta) * base + delta * c, axis=0)
            ll += np.log((1 - zeta) * g[d] / g.sum() + zeta / g.size)
        return ll


def features(mm, hist, q, u0_pred):
    mask = mm.consistent(hist)
    return [len(set(mm.table[q, mask].tolist())) / mask.sum(), float(mm.eig(mask, [q])[0]),
            float(mask.sum()), float(len(hist)), float(n_distinct(mm, q)), u0_pred]


def calibrate(model):
    mm = get_game(4, 4)
    caches = {k: hm.BoundedCache(v) for k, v in CACHES.items()}
    b3 = json.loads((FITS / f"{model}_T0.json").read_text())
    proc = b3["best_heldout"]
    theta = b3["fits"][proc]["params"]
    fams = families(mm)
    rows3, pairs3 = branch_records(model, "T3", "support_calibration")
    rows0, pairs0 = branch_records(model, "T0", "support_calibration")
    br3 = Branches(mm, fams, rows3, pairs3, proc, theta, caches)
    br0_exact = Branches(mm, fams, rows0, pairs0, "M0", {}, caches)
    br3_exact = Branches(mm, fams, rows3, pairs3, "M0", {}, caches)

    delta = minimize_scalar(lambda d: -br3.loglik(d, theta["zeta"]), bounds=(0, 1), method="bounded").x
    zeta_rep = minimize_scalar(lambda z: -br3.loglik(0.0, z), bounds=(1e-4, 0.999), method="bounded").x
    zeta_ideal = {"T0": minimize_scalar(lambda z: -br0_exact.loglik(0.0, z), bounds=(1e-4, 0.999), method="bounded").x,
                  "T3": minimize_scalar(lambda z: -br3_exact.loglik(0.0, z), bounds=(1e-4, 0.999), method="bounded").x}

    # Difficulty regression: observed per-query change in U (T3 - T0) on both calibration splits.
    X, y = [], []
    for split in ("support_calibration", "choice_calibration"):
        try:
            r3, p3 = branch_records(model, "T3", split)
            r0, p0 = branch_records(model, "T0", split)
        except FileNotFoundError:
            continue
        u3, u0 = observed_u(r3, p3), observed_u(r0, p0)
        for (fam, qs) in set(u3) & set(u0):
            q = mm.index(qs)
            X.append(features(mm, fams[fam], q, us.predicted_usefulness(mm, proc, theta, fams[fam], q, caches)))
            y.append(u3[(fam, qs)] - u0[(fam, qs)])
    X, y = np.array(X), np.array(y)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    lam = 1.0
    beta = np.linalg.solve(Z.T @ Z + lam * np.diag([0] + [1] * (Z.shape[1] - 1)), Z.T @ y)

    cal = {"model": model, "process_model": proc, "theta_T0": theta, "delta_T3": float(delta),
           "reporting_only": {"zeta_T0": theta["zeta"], "zeta_T3": float(zeta_rep)},
           "ideal_noise": {k: float(v) for k, v in zeta_ideal.items()},
           "difficulty_regression": {"features": ["vstar", "eig", "K", "n_clues", "distinct_digits", "pred_U_T0"],
                                     "mean": mu.tolist(), "sd": sd.tolist(), "beta": beta.tolist(), "ridge_lambda": lam, "n": int(len(y))},
           "data": {"support_cal_T3_answers": len(br3.items), "support_cal_T0_answers": len(br0_exact.items)}}
    CAL.parent.mkdir(parents=True, exist_ok=True)
    CAL.write_text(json.dumps(cal, indent=2))
    print(json.dumps(cal, indent=2))


def predict_gamma(mm, cal, hist, q1, q2, caches):
    """Predicted Gamma for each predictor."""
    proc, th, d = cal["process_model"], cal["theta_T0"], cal["delta_T3"]
    U = lambda q, model, p, delta=0.0: us.predicted_usefulness(mm, model, p, hist, q, caches, delta)
    base = {q: U(q, proc, th) for q in (q1, q2)}
    out = {}
    sup = {q: U(q, proc, th, d) for q in (q1, q2)}
    out["support_map"] = (sup[q1] - sup[q2]) - (base[q1] - base[q2])
    rep = {q: U(q, proc, {**th, "zeta": cal["reporting_only"]["zeta_T3"]}) for q in (q1, q2)}
    out["reporting_only"] = (rep[q1] - rep[q2]) - (base[q1] - base[q2])
    out["uniform_gain"] = 0.0
    dr = cal["difficulty_regression"]
    def dU(q):
        x = (np.array(features(mm, hist, q, base[q])) - np.array(dr["mean"])) / np.array(dr["sd"])
        return float(np.r_[1.0, x] @ np.array(dr["beta"]))
    out["difficulty_regression"] = dU(q1) - dU(q2)
    i0 = {q: U(q, "M0", {"zeta": cal["ideal_noise"]["T0"]}) for q in (q1, q2)}
    i3 = {q: U(q, "M0", {"zeta": cal["ideal_noise"]["T3"]}) for q in (q1, q2)}
    out["ideal_noise"] = (i3[q1] - i3[q2]) - (i0[q1] - i0[q2])
    return out, base


def select(pairs_per_stratum, seed=11):
    mm = get_game(4, 4)
    caches = {k: hm.BoundedCache(v) for k, v in CACHES.items()}
    cal = json.loads(CAL.read_text())
    fams = {f["family_id"]: f for f in json.loads(FAMILIES.read_text()) if f["split"] == "confirmation"}
    rng = np.random.default_rng(seed)
    rows = []
    for fid, f in fams.items():
        hist = [(mm.index(g), (b, w)) for g, b, w in f["history"]]
        mask = mm.consistent(hist)
        eig = mm.eig(mask)
        probes = [int(q) for q in np.flatnonzero(~mask & (eig > 0))]
        cands = [tuple(sorted(rng.choice(probes, 2, replace=False).tolist())) for _ in range(60)]
        scored = []
        for q1, q2 in set(cands):
            g = predict_gamma(mm, cal, hist, q1, q2, caches)[0]["support_map"]
            m1, m2 = (len(set(mm.table[q, mask].tolist())) for q in (q1, q2))
            scored.append({"q1": q1, "q2": q2, "gamma_hat": g, "vstar_equal": m1 == m2,
                           "eig_gap": abs(eig[q1] - eig[q2]), "digit_gap": abs(n_distinct(mm, q1) - n_distinct(mm, q2))})
        strata = {
            "a_matched_vstar": sorted([s for s in scored if s["vstar_equal"]], key=lambda s: -abs(s["gamma_hat"])),
            "b_similar_eig_diff_burden": sorted([s for s in scored if s["eig_gap"] < 0.2 and s["digit_gap"] >= 1], key=lambda s: -abs(s["gamma_hat"])),
            "c_predicted_zero": sorted(scored, key=lambda s: abs(s["gamma_hat"])),
        }
        used = set()
        for name, lst in strata.items():
            k = 0
            for s in lst:
                if k == pairs_per_stratum:
                    break
                if (s["q1"], s["q2"]) in used:
                    continue
                used.add((s["q1"], s["q2"]))
                q1, q2 = (s["q1"], s["q2"]) if s["gamma_hat"] >= 0 else (s["q2"], s["q1"])
                rows.append({"pair_id": f"{fid}:{name[0]}{k}", "family_id": fid, "stratum": name,
                             "q_simple": mm.to_str(q1), "q_complex": mm.to_str(q2),
                             "note": "q_simple = q1 and q_complex = q2 in the Gamma definition (field names shared with the runner)"})
                k += 1
    PAIRS.write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} pairs over {len(fams)} families -> {PAIRS}")


def lock():
    mm = get_game(4, 4)
    caches = {k: hm.BoundedCache(v) for k, v in CACHES.items()}
    cal = json.loads(CAL.read_text())
    pairs = json.loads(PAIRS.read_text())
    fams = families(mm)
    preds = []
    for p in pairs:
        g, base = predict_gamma(mm, cal, fams[p["family_id"]], mm.index(p["q_simple"]), mm.index(p["q_complex"]), caches)
        preds.append({"pair_id": p["pair_id"], "family_id": p["family_id"], "stratum": p["stratum"],
                      "gamma_hat": {k: round(float(v), 6) for k, v in g.items()}})
    lockdoc = {"locked_at_utc": __import__("datetime").datetime.utcnow().isoformat(timespec="seconds"),
               "primary_endpoint": "family-level MSPE of Gamma; support_map vs each rival; cluster bootstrap over families",
               "hashes": {"families_v1": sha(FAMILIES), "pairs_v1": sha(PAIRS), "calibration": sha(CAL),
                          "prompt_freeze_v1": sha(ROOT / "configs/prompt_freeze_v1.json"),
                          "model_registry": sha(ROOT / "configs/model_registry.json")},
               "calibration": cal, "predictions": preds}
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(json.dumps(lockdoc, indent=1))
    print(f"wrote {LOCK} with {len(preds)} predictions. Commit it now:  git add prereg data/splits && git commit")


def score(model, n_boot=5000, seed=0):
    lockdoc = json.loads(LOCK.read_text())
    r3, p3 = branch_records(model, "T3", "confirmation")
    r0, p0 = branch_records(model, "T0", "confirmation")
    u3, u0 = observed_u(r3, p3), observed_u(r0, p0)
    pairs = {p["pair_id"]: p for p in json.loads(PAIRS.read_text())}
    rows = []
    for pr in lockdoc["predictions"]:
        p = pairs[pr["pair_id"]]
        k1, k2 = (p["family_id"], p["q_simple"]), (p["family_id"], p["q_complex"])
        if not all(k in u3 and k in u0 for k in (k1, k2)):
            continue
        obs = (u3[k1] - u3[k2]) - (u0[k1] - u0[k2])
        rows.append({"family": p["family_id"], "stratum": p["stratum"], "obs": obs, **{k: pr["gamma_hat"][k] for k in PREDICTORS}})
    fam_ids = sorted({r["family"] for r in rows})
    def fam_mspe(rs, k):
        per = {}
        for r in rs:
            per.setdefault(r["family"], []).append((r[k] - r["obs"]) ** 2)
        return {f: float(np.mean(v)) for f, v in per.items()}
    per_fam = {k: fam_mspe(rows, k) for k in PREDICTORS}
    mspe = {k: float(np.mean(list(per_fam[k].values()))) for k in PREDICTORS}
    rng = np.random.default_rng(seed)
    diffs = {}
    for k in PREDICTORS[1:]:
        d = np.array([per_fam[k][f] - per_fam["support_map"][f] for f in fam_ids])
        boots = [rng.choice(d, len(d)).mean() for _ in range(n_boot)]
        diffs[k] = {"mspe_rival_minus_model": round(float(d.mean()), 6),
                    "ci95": [round(float(np.percentile(boots, 2.5)), 6), round(float(np.percentile(boots, 97.5)), 6)],
                    "p_one_sided": float(np.mean(np.array(boots) <= 0))}
    obs = np.array([r["obs"] for r in rows])
    pred = np.array([r["support_map"] for r in rows])
    big = np.abs(obs) >= np.percentile(np.abs(obs), 75)
    res = {"pairs_scored": len(rows), "families": len(fam_ids), "mspe": mspe, "model_vs_rivals": diffs,
           "calibration_slope": float(np.polyfit(pred, obs, 1)[0]) if len(rows) > 2 else None,
           "bias": float(np.mean(pred - obs)), "sign_accuracy_large_effects": float(np.mean(np.sign(pred[big]) == np.sign(obs[big]))),
           "mean_observed_gamma_by_stratum": {s: float(np.mean([r["obs"] for r in rows if r["stratum"] == s])) for s in {r["stratum"] for r in rows}}}
    (ROOT / "results/study_c/cc_score.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate"); c.add_argument("--model", default="qwen3-32b")
    s = sub.add_parser("select"); s.add_argument("--pairs-per-stratum", type=int, default=1)
    sub.add_parser("lock")
    sc = sub.add_parser("score"); sc.add_argument("--model", default="qwen3-32b")
    args = ap.parse_args()
    {"calibrate": lambda: calibrate(args.model), "select": lambda: select(args.pairs_per_stratum),
     "lock": lock, "score": lambda: score(args.model)}[args.cmd]()


if __name__ == "__main__":
    main()
