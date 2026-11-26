"""B1 re-analysis (post hoc): the signature effect for every model, against the ideal player, with the
paper's control for initial uncertainty.

Why: `analyze_b1.py` tests the raw Max-EIG minus Simple difference against zero, and only for models that pass
the solve-rate rule (set after the solve rates were seen). Two problems follow:
  1. The ideal player is itself faster after Max-EIG histories (they leave fewer codes), so a raw difference of
     zero is not "no human signature": it is already a shift away from the ideal player toward people.
  2. The rule keeps only the most capable models, and capability is what moves play toward the ideal player.
This script reports, for people, every model and the ideal player, on the same 113 Simple and Max-EIG
stimuli (unsolved games = 16, as in `analyze_b1.py`):

  raw            mean guesses, Max-EIG minus Simple (the existing contrast; reproduces b1_summary.json)
  controlled     OLS: guesses ~ Max-EIG + ln(codes remaining) + supplied guesses; SEs clustered by stimulus
                 (the paper's Experiment 1 control for initial uncertainty)
  vs_ideal       the same two contrasts on excess guesses (guesses minus the ideal player's guesses on that
                 stimulus): the part of the effect the ideal player does not show. Positive = the human direction.
                 This is the primary quantity of the re-analysis.
  hazard         discrete-time logistic model of solving per turn, + ln(codes remaining), censoring-aware,
                 SEs clustered by stimulus (negative log-odds = slower after Max-EIG)
  vs_people      model minus people on vs_ideal, bootstrap paired by stimulus (people and models played the
                 same stimuli, so stimuli are resampled jointly), and the model's share of the human effect
  holm           Holm across all six models (two-sided, vs_ideal bootstrap), and across the three that pass
                 the solve-rate rule, so the rule is a sensitivity check rather than a filter

Bootstraps resample stimuli within condition, 20,000 draws, seeds fixed.

  python experiments/study_b/b1_reanalysis.py
"""
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/study_b"))
from analyze_b1 import B1, MIN_FREE_SOLVE, UNSOLVED, human_e1, model_e1  # noqa: E402
from mastermind.engine import get_game  # noqa: E402

MODELS = ["qwen3-32b", "phi-4", "olmo2-32b", "gpt-oss-120b", "claude-haiku-5-5", "claude-opus-5-5"]
CONDS = ("Simple", "Max-EIG")
N_BOOT = 20000
OUT = ROOT / "results/study_b/b1_reanalysis.json"


def prep(df, stimuli):
    by_id = {s["stimulus_id"]: s for s in stimuli}
    d = df[df.condition.isin(CONDS)].copy()
    d["maxeig"] = (d.condition == "Max-EIG").astype(float)
    d["ln_remaining"] = np.log(d.codes_remaining.astype(float))
    d["n_supplied"] = [len(by_id[s]["supplied"]) for s in d.stimulus]
    d["excess"] = d.guesses - d.ideal
    return d


def ols(d, y, controlled):
    f = f"{y} ~ maxeig" + (" + ln_remaining + n_supplied" if controlled else "")
    r = smf.ols(f, d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.stimulus)[0]})
    lo, hi = r.conf_int().loc["maxeig"]
    return {"estimate": round(float(r.params["maxeig"]), 3), "ci95": [round(float(lo), 2), round(float(hi), 2)],
            "p": float(r.pvalues["maxeig"])}


def hazard(d):
    rows = []
    for gid, r in enumerate(d.itertuples()):
        for t in range(1, r.turns + 1):
            rows.append({"stimulus": r.stimulus, "turn": t, "maxeig": r.maxeig, "ln_remaining": r.ln_remaining,
                         "event": int(r.solved and t == r.turns)})
    pp = pd.DataFrame(rows)
    X = pd.get_dummies(pp.turn.clip(upper=10), prefix="t", drop_first=True, dtype=float)
    X = sm.add_constant(X.assign(maxeig=pp.maxeig, ln_remaining=pp.ln_remaining))
    r = sm.GLM(pp.event, X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(pp.stimulus)[0]})
    b, se = float(r.params["maxeig"]), float(r.bse["maxeig"])
    return {"log_odds_maxeig": round(b, 3), "ci95": [round(b - 1.96 * se, 2), round(b + 1.96 * se, 2)], "p": float(r.pvalues["maxeig"]),
            "note": "controls ln(codes remaining); negative = slower to solve after Max-EIG (human direction)"}


class StimBoot:
    """Stimulus-level sums and counts for each agent, aligned on one stimulus list per condition, with one set of
    resampling indices shared by all agents (paired bootstrap)."""

    def __init__(self, frames, seed=0):
        self.frames = frames
        rng = np.random.default_rng(seed)
        self.stims, self.idx = {}, {}
        for c in CONDS:
            ids = sorted(set.intersection(*[set(f[f.condition == c].stimulus) for f in frames.values()]))
            self.stims[c] = ids
            self.idx[c] = rng.integers(len(ids), size=(N_BOOT, len(ids)))

    def draws(self, agent, y):
        f = self.frames[agent]
        means = {}
        for c in CONDS:
            g = f[f.condition == c].groupby("stimulus")[y].agg(["sum", "size"]).reindex(self.stims[c])
            s, k = g["sum"].to_numpy(float), g["size"].to_numpy(float)
            means[c] = s[self.idx[c]].sum(1) / k[self.idx[c]].sum(1)
        return means["Max-EIG"] - means["Simple"]


def point(d, y):
    m = d.groupby("condition")[y].mean()
    return float(m["Max-EIG"] - m["Simple"])


def ci(x):
    return [round(float(np.percentile(x, 2.5)), 2), round(float(np.percentile(x, 97.5)), 2)]


def boot_p(x):
    """Two-sided bootstrap p for 'effect = 0', (count + 1) / (B + 1) so it is never exactly 0."""
    k = min((x <= 0).sum(), (x >= 0).sum())
    return float(min(1.0, 2 * (k + 1) / (len(x) + 1)))


def holm(ps):
    names = sorted(ps, key=ps.get)
    out, running = {}, 0.0
    for i, n in enumerate(names):
        running = max(running, min(1.0, (len(names) - i) * ps[n]))
        out[n] = running
    return out


def main():
    mm = get_game(4, 4)  # noqa: F841  (loads the engine tables used by human_e1)
    stimuli = json.loads((ROOT / "data/stimuli/e1_stimuli.json").read_text())
    summary = json.loads((ROOT / "results/study_b/b1_summary.json").read_text())

    frames = {"humans": prep(human_e1(get_game(4, 4), stimuli), stimuli)}
    for m in MODELS:
        frames[m] = prep(model_e1(m, stimuli), stimuli)
    # the ideal (greedy Max-EIG) player: one game per stimulus, its own guess count, solved
    ideal = frames["humans"].drop_duplicates("stimulus").copy()
    ideal = ideal.assign(guesses=ideal.ideal, excess=0.0, solved=True, turns=ideal.ideal)
    frames["ideal"] = ideal

    boot = StimBoot(frames, seed=zlib.crc32(b"b1_reanalysis"))
    h_ex = boot.draws("humans", "excess")
    h_pt = point(frames["humans"], "excess")

    out = {
        "status": "post hoc re-analysis of the saved B1 records; not preregistered",
        "measure": f"guesses after the supplied history (unsolved = {UNSOLVED}); excess = guesses minus the ideal player's",
        "sign": "Max-EIG minus Simple; positive = slower after informative but complex histories (the human direction). "
                "The paper reports Simple minus Max-EIG (B = -2.29), the same effect with the opposite sign.",
        "stimuli": {c: len(boot.stims[c]) for c in CONDS},
        "agents": {},
    }
    p_all = {}
    for a in ["humans", "ideal"] + MODELS:
        d = frames[a]
        raw_d = boot.draws(a, "guesses")
        r = {"games": int(len(d)), "solve_rate": round(float(d.solved.mean()), 3),
             "raw": {"estimate": round(point(d, "guesses"), 3), "ci95_boot": ci(raw_d)}}
        if a != "ideal":
            ex_d = boot.draws(a, "excess")
            r["raw"]["ols_controlled"] = ols(d, "guesses", True)
            r["vs_ideal"] = {"estimate": round(point(d, "excess"), 3), "ci95_boot": ci(ex_d), "p_boot": boot_p(ex_d),
                             "ols_controlled": ols(d, "excess", True)}
            r["hazard"] = hazard(d)
        if a in MODELS:
            fp = summary[a]["free_play"]
            r["free_play_solve_rate"] = fp["solve_rate"]
            r["free_play_eig_relative_to_max"] = fp["eig_relative_to_max_mean"]
            r["passes_solve_rate_rule"] = bool(fp["solve_rate"] >= MIN_FREE_SOLVE)
            diff = ex_d - h_ex
            share = ex_d / h_ex
            r["vs_people"] = {"model_minus_people": round(point(d, "excess") - h_pt, 3), "ci95_paired": ci(diff),
                              "share_of_human_effect": round(point(d, "excess") / h_pt, 2), "share_ci95_paired": ci(share),
                              "note": "on excess guesses (ideal-relative); bootstrap paired by stimulus"}
            # the existing raw estimate must reproduce b1_summary.json
            old = summary[a].get("signature_effect_maxeig_minus_simple", summary[a].get("descriptive", {}).get("maxeig_minus_simple"))
            r["raw"]["matches_b1_summary"] = bool(old is None or abs(round(point(d, "guesses"), 2) - old) < 0.006)
            p_all[a] = r["vs_ideal"]["p_boot"]
        out["agents"][a] = r

    eligible = [m for m in MODELS if out["agents"][m]["passes_solve_rate_rule"]]
    out["holm_vs_ideal"] = {
        "all_six_models": {m: round(p, 6) for m, p in holm(p_all).items()},
        "solve_rate_rule_models_only": {m: round(p, 6) for m, p in holm({m: p_all[m] for m in eligible}).items()},
        "note": "two-sided bootstrap p for the ideal-relative effect; alpha = 0.05",
    }
    cap = sorted(MODELS, key=lambda m: out["agents"][m]["free_play_eig_relative_to_max"])
    out["capability_gradient"] = {
        "order": "models by free-play query quality (mean EIG relative to the best query; ideal = 1)",
        "rows": [{"model": m, "eig_relative_to_max": out["agents"][m]["free_play_eig_relative_to_max"],
                  "vs_ideal": out["agents"][m]["vs_ideal"]["estimate"], "ci95": out["agents"][m]["vs_ideal"]["ci95_boot"],
                  "share_of_human_effect": out["agents"][m]["vs_people"]["share_of_human_effect"]} for m in cap],
        "spearman_capability_vs_effect": round(float(pd.Series([out["agents"][m]["free_play_eig_relative_to_max"] for m in cap])
                                                     .corr(pd.Series([out["agents"][m]["vs_ideal"]["estimate"] for m in cap]), method="spearman")), 3),
        "note": "six models: descriptive only",
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
