"""Study C, post hoc: how informative the preregistered nulls are (H3 and H4).

Not part of the preregistration. The confirmatory results stay those of `prediction_test.py score`
(`cc_score.json`) and `choice_shift.py --split c_confirmation`. This script reads the same locked predictions and
the same confirmation records, through the same functions, and asks three questions the plan left open:

  1. Equivalence. Can the smallest effect of interest be ruled out? The plan sized the study to detect a predictor
     that explains 20% of the true variance of Gamma, so the SESOI for an MSPE gain is 0.2 * tau^2. tau^2 is taken
     two ways: the planned value (from calibration) and the value measured on the confirmation data. The test is the
     one-sided upper 95% bootstrap bound of the gain against the SESOI (equivalent to the upper half of a TOST).
  2. Bayes factors. BF01 for "gain = 0" against "gain > 0" with a half-normal prior whose scale is the SESOI (also
     half and twice it), using the bootstrap standard error (normal approximation). For H4, BF01 for the
     correlation between measured Gamma and the choice shift, Fisher z, half-normal prior on z.
  3. Design diagnosis. What could the locked primary predictor have gained over Gamma = 0 if its predictions had
     been exactly right? If Gamma = Gamma_hat + independent rest, the expected gain is mean(Gamma_hat^2). That is
     compared with the SESOI, and the power of the locked test for a correct predictor is simulated.

Noise variance of each pair's Gamma is computed analytically (unbiased, within-branch variance with ddof = 1),
the "corrected" estimator of `noise_check.py`. Bootstraps resample families (10,000 resamples, seed 0, as in
the preregistered score; the same indices are reused everywhere).

  python experiments/study_c/informative_nulls.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/study_c"))
import prediction_test as P  # noqa: E402
from choice_shift import logit  # noqa: E402

MODEL = "qwen3-32b"
N_BOOT = 10000
ALPHA_PER_TEST = 0.0125  # the plan's one-sided alpha per rival (0.05 / 4)
OUT = ROOT / "results/study_c/cc_informative_nulls.json"
KEYS = [("T0", "q_simple"), ("T0", "q_complex"), ("T3", "q_simple"), ("T3", "q_complex")]


def u_var(rows):
    """Unbiased sampling variance of U for one (condition, pair, query): sum_a (K_a/K)^2 var(Y_a) / n_a."""
    by, ka = {}, {}
    for r in rows:
        o = tuple(r["outcome"])
        ka[o] = r["K_a"]
        by.setdefault(o, []).append(r["Y"])
    K = sum(ka.values())
    return sum((ka[o] / K) ** 2 * np.var(v, ddof=1) / len(v) for o, v in by.items() if len(v) > 1)


def half_normal_bf01(est, se, scale):
    """BF01 for theta = 0 vs theta ~ HalfNormal(scale), likelihood est ~ N(theta, se^2)."""
    g = np.linspace(0, 12 * scale, 40001)
    prior = 2 * stats.norm.pdf(g, 0, scale)
    m1 = np.trapezoid(stats.norm.pdf(est, g, se) * prior, g)
    m0 = stats.norm.pdf(est, 0, se)
    return float(m0 / m1)


def r3(x):
    return float(f"{x:.3g}")


def main():
    lock = json.loads(P.LOCK.read_text())
    b0, c0, pairs = P.records(MODEL, "T0", P.CONF_SPLIT, P.BASE_CAP)
    b3, c3, _ = P.records(MODEL, "T3", P.CONF_SPLIT, P.BASE_CAP)
    groups = P.group({"T0": b0, "T3": b3})
    gam = P.observed_gamma(groups, pairs)

    preds = [p for p in lock["predictions"] if p["pair_id"] in gam]
    pids = [p["pair_id"] for p in preds]
    obs = np.array([gam[p][0] for p in pids])
    noise = np.array([sum(u_var(groups[(c, pid, q)]) for c, q in KEYS) for pid in pids])
    fam_ids = sorted({p["family_id"] for p in preds})
    fam_of = np.array([fam_ids.index(p["family_id"]) for p in preds])
    members = [np.flatnonzero(fam_of == i) for i in range(len(fam_ids))]
    idx = np.random.default_rng(0).integers(len(fam_ids), size=(N_BOOT, len(fam_ids)))  # = score's resamples

    def per_family(x):
        return np.array([x[m].mean() for m in members])

    def boot_pairs(fn, n=N_BOOT):
        return np.array([fn(np.concatenate([members[i] for i in b])) for b in idx[:n]])

    # ---- true variance of Gamma (signal) ----
    tau2 = float(obs.var() - noise.mean())
    tau2_boot = boot_pairs(lambda s: obs[s].var() - noise[s].mean(), 4000)
    reliability = tau2 / float(obs.var())
    tau2_planned = float(json.loads((ROOT / "results/study_c/cc_power.json").read_text())["tau_sd"]) ** 2
    signal = {"observed_sd": r3(obs.std()), "noise_sd": r3(np.sqrt(noise.mean())), "true_sd": r3(np.sqrt(max(tau2, 0))),
              "true_variance": r3(tau2), "true_variance_ci95": [r3(np.percentile(tau2_boot, 2.5)), r3(np.percentile(tau2_boot, 97.5))],
              "p_true_variance_le_0": r3(np.mean(tau2_boot <= 0)), "reliability_of_gamma": r3(reliability),
              "planned_true_sd": r3(np.sqrt(tau2_planned)),
              "note": "the aid's effect on usefulness varies across pairs beyond sampling noise, but less than planned"}

    # ---- H3: equivalence and Bayes factors ----
    G = {k: np.array([p["gamma_hat"][k] for p in preds]) for k in P.PREDICTORS}
    sq = {k: per_family((G[k] - obs) ** 2) for k in P.PREDICTORS}
    sesoi = {"planned": 0.2 * tau2_planned, "measured": 0.2 * tau2}
    h3 = {"sesoi_mspe_gain": {k: r3(v) for k, v in sesoi.items()},
          "sesoi_definition": "0.2 * tau^2: the gain of a predictor explaining 20% of the true variance of Gamma (the plan's power target)",
          "primary_vs_rivals": {}}
    for rival in P.RIVALS:
        d = sq[rival] - sq[P.PRIMARY]
        b = d[idx].mean(1)
        se = float(b.std(ddof=1))
        upper95 = float(np.percentile(b, 95))
        h3["primary_vs_rivals"][rival] = {
            "mspe_gain": r3(d.mean()), "ci90": [r3(np.percentile(b, 5)), r3(upper95)], "bootstrap_se": r3(se),
            "equivalence": {k: {"upper95_below_sesoi": bool(upper95 < v), "p_equivalence": r3(max(np.mean(b >= v), 1 / (N_BOOT + 1)))}
                            for k, v in sesoi.items()},
            "bf01": {f"{k}_sesoi_x{m}": r3(half_normal_bf01(float(d.mean()), se, m * v)) for k, v in sesoi.items() for m in (0.5, 1, 2)},
        }
    # scale-free: correlation of each predictor with observed Gamma, vs the correlation a SESOI predictor would show
    corr = {}
    for k in P.PREDICTORS:
        if G[k].std() == 0:
            corr[k] = {"pearson_r": None, "note": "constant prediction"}
            continue
        rb = boot_pairs(lambda s, k=k: np.corrcoef(G[k][s], obs[s])[0, 1], 4000)
        corr[k] = {"pearson_r": r3(np.corrcoef(G[k], obs)[0, 1]), "ci95": [r3(np.percentile(rb, 2.5)), r3(np.percentile(rb, 97.5))],
                   "prediction_sd": r3(G[k].std())}
    h3["correlation_with_observed_gamma"] = corr
    h3["correlation_expected_for_sesoi_predictor"] = r3(np.sqrt(0.2 * reliability))

    # ---- design diagnosis: could the locked primary have passed if it were exactly right? ----
    gh = G[P.PRIMARY]
    max_gain = float(np.mean(gh ** 2))
    rest_var = max(tau2 - float(gh.var()), 0.0)
    rng = np.random.default_rng(1)
    n_sim, hits = 2000, 0
    for _ in range(n_sim):  # Gamma = Gamma_hat + independent rest; observed = Gamma + sampling noise
        sim = gh + rng.normal(0, np.sqrt(rest_var), len(gh)) + rng.normal(0, np.sqrt(noise))
        d = per_family(sim ** 2 - (gh - sim) ** 2)  # MSPE(Gamma = 0) - MSPE(Gamma_hat), per family
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
        hits += t > stats.norm.ppf(1 - ALPHA_PER_TEST)
    design = {"locked_primary_prediction_sd": r3(gh.std()), "true_sd_of_gamma": r3(np.sqrt(max(tau2, 0))),
              "ratio_true_to_predicted_sd": r3(np.sqrt(max(tau2, 0)) / gh.std()),
              "max_expected_gain_if_exactly_right": r3(max_gain), "as_share_of_true_variance": r3(max_gain / tau2),
              "sesoi_measured": r3(sesoi["measured"]),
              "power_if_exactly_right_vs_uniform_gain": r3(hits / n_sim),
              "power_method": f"{n_sim} simulations, family-level one-sided z test at alpha {ALPHA_PER_TEST} (normal approximation to the locked bootstrap test)",
              "planned_power": 0.864,
              "note": "the planned power is for a predictor whose spread matches a 20% share of tau^2; the locked predictor's "
                      "spread is far smaller, so even a perfectly correct support_map could not have beaten Gamma = 0"}

    # ---- H4: choice shift on measured Gamma ----
    k = {c: {} for c in ("T0", "T3")}
    for c, rows in (("T0", c0), ("T3", c3)):
        for r in rows:
            if r["parsed"]:
                a = k[c].setdefault(r["pair_id"], [0, 0])
                a[0] += r["chose"] == "q_simple"
                a[1] += 1
    keep = [i for i, p in enumerate(pids) if p in k["T0"] and p in k["T3"]]
    x, y = obs[keep], np.array([logit(*k["T3"][pids[i]]) - logit(*k["T0"][pids[i]]) for i in keep])
    fam_h4 = fam_of[keep]
    mem_h4 = [np.flatnonzero(fam_h4 == i) for i in range(len(fam_ids))]

    def boot_h4(fn, n=4000):
        out = []
        for b in idx[:n]:
            s = np.concatenate([mem_h4[i] for i in b])
            out.append(fn(s))
        return np.array(out)

    r = float(np.corrcoef(x, y)[0, 1])
    rb = boot_h4(lambda s: np.corrcoef(x[s], y[s])[0, 1])
    # split-half reliability of the choice shift (samples 0-3 vs 4-7)
    kk = {h: {c: {} for c in ("T0", "T3")} for h in (0, 1)}
    for c, rows in (("T0", c0), ("T3", c3)):
        for rr in rows:
            if rr["parsed"]:
                a = kk[int(rr["sample"] >= 4)][c].setdefault(rr["pair_id"], [0, 0])
                a[0] += rr["chose"] == "q_simple"
                a[1] += 1
    ok = [p for p in pids if all(p in kk[h][c] for h in (0, 1) for c in ("T0", "T3"))]
    ys = [np.array([logit(*kk[h]["T3"][p]) - logit(*kk[h]["T0"][p]) for p in ok]) for h in (0, 1)]
    rh = float(np.corrcoef(*ys)[0, 1])
    rel_y = 2 * rh / (1 + rh)
    z, se_z = float(np.arctanh(r)), 1 / np.sqrt(len(x) - 3)
    h4 = {"pairs": len(x), "pearson_r_shift_vs_measured_gamma": r3(r), "ci95": [r3(np.percentile(rb, 2.5)), r3(np.percentile(rb, 97.5))],
          "reliability": {"measured_gamma": r3(reliability), "choice_shift_split_half_spearman_brown": r3(rel_y)},
          "max_observable_r_if_true_r_is_1": r3(np.sqrt(max(reliability, 0) * max(rel_y, 0))),
          "disattenuated_upper95": r3(np.percentile(rb, 97.5) / np.sqrt(max(reliability, 1e-9) * max(rel_y, 1e-9))),
          "bf01_positive_correlation": {f"prior_scale_z_{s}": r3(half_normal_bf01(z, se_z, s)) for s in (0.1, 0.2, 0.3)},
          "note": "observed-scale correlation; the disattenuated bound is how large the true correlation could be "
                  "given the low reliability of both measures"}

    out = {"status": "post hoc, not preregistered; the confirmatory results are cc_score.json and cb_choice_shift_c_confirmation.json",
           "pairs": len(pids), "families": len(fam_ids), "signal": signal, "h3": h3, "design_diagnosis": design, "h4": h4}
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
