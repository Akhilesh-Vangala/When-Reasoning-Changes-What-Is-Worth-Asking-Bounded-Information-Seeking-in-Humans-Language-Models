"""B1 analysis, every model against people on the same 132 Experiment 1 positions.

Effect measure: guesses after the supplied history, unsolved games counted as 16
(one more than the 15-guess limit), so failures are not dropped.
  signature effect   Max-EIG minus Simple (positive = worse after informative but complex histories,
                     the human pattern); bootstrap 95% CI resampling stimuli within condition
  relative to ideal  the same effect minus the ideal player's effect (ideal is faster after Max-EIG)
  hazard check       discrete-time logistic model of solving at each turn, condition + turn,
                     cluster-robust SEs by stimulus (censoring-aware test of the contrast)
  one-code subset    Max-EIG histories that leave one code (ideal needs 1 guess)
  free play          guesses scored against EIG: share of guesses at the max-EIG value, mean EIG / max EIG
The human-model difference is reported as an estimate with a CI, not a test.

  python experiments/study_b/analyze_b1.py --models qwen3-32b gpt-oss-120b phi-4 olmo2-32b
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mastermind.engine import get_game  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments/study_a"))
from e1_reproduction import load_games  # noqa: E402

B1 = ROOT / "results/study_b/b1"
UNSOLVED = 16


def human_e1(mm, stimuli):
    key = {(s["condition"], tuple((g["guess"], g["black"], g["white"]) for g in s["supplied"]), s["secret"]): s for s in stimuli}
    rows = []
    for g in load_games(mm):
        k = (g["condition"], tuple((mm.to_str(q), b, w) for q, (b, w) in g["supplied"]), mm.to_str(g["secret"]))
        s = key[k]
        rows.append({"agent": "humans", "stimulus": s["stimulus_id"], "condition": g["condition"],
                     "guesses": len(g["human"]) if g["solved"] else UNSOLVED, "solved": bool(g["solved"]),
                     "ideal": s["ideal_guesses_to_solve"], "codes_remaining": s["codes_remaining"]})
    return pd.DataFrame(rows)


def model_e1(model, stimuli):
    by_id = {s["stimulus_id"]: s for s in stimuli}
    rows = []
    for line in open(B1 / f"{model}_e1_T0.jsonl"):
        g = json.loads(line)
        sid = g["game_id"].split(":")[0]
        s = by_id[sid]
        rows.append({"agent": model, "stimulus": sid, "condition": s["condition"],
                     "guesses": g["own_guesses"] if g["solved"] else UNSOLVED, "solved": g["solved"],
                     "ideal": s["ideal_guesses_to_solve"], "codes_remaining": s["codes_remaining"],
                     "invalid_turns": g["invalid_turns"]})
    return pd.DataFrame(rows)


def effect(df):
    m = df.groupby("condition")["guesses"].mean()
    return float(m.get("Max-EIG", np.nan) - m.get("Simple", np.nan))


def boot_ci(df, rng, n=2000):
    """Bootstrap over stimuli (within condition), keeping all games of a resampled stimulus."""
    groups = {c: [g for _, g in d.groupby("stimulus")] for c, d in df[df.condition.isin(["Simple", "Max-EIG"])].groupby("condition")}
    out = []
    for _ in range(n):
        parts = []
        for c, gs in groups.items():
            parts += [gs[i] for i in rng.integers(len(gs), size=len(gs))]
        out.append(effect(pd.concat(parts)))
    return [round(float(np.percentile(out, 2.5)), 2), round(float(np.percentile(out, 97.5)), 2)], np.array(out)


def hazard_test(df):
    """Person-period data: one row per turn at risk; solved on that turn or not."""
    rows = []
    for gid, r in enumerate(df[df.condition.isin(["Simple", "Max-EIG"])].itertuples()):
        last = r.guesses if r.solved else 15
        for t in range(1, last + 1):
            rows.append({"game": gid, "stimulus": r.stimulus, "turn": t, "maxeig": int(r.condition == "Max-EIG"),
                         "event": int(r.solved and t == last)})
    pp = pd.DataFrame(rows)
    X = sm.add_constant(pd.get_dummies(pp["turn"].clip(upper=10), prefix="t", drop_first=True, dtype=float).assign(maxeig=pp["maxeig"]))
    res = sm.GLM(pp["event"], X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(pp["stimulus"])[0]})
    b, se = res.params["maxeig"], res.bse["maxeig"]
    return {"log_odds_maxeig": round(float(b), 3), "se": round(float(se), 3), "z": round(float(b / se), 2),
            "p": float(res.pvalues["maxeig"]), "note": "negative = slower to solve after Max-EIG histories"}


def free_play(mm, model):
    path = B1 / f"{model}_free_T0.jsonl"
    if not path.exists():
        return None
    games = [json.loads(line) for line in open(path)]
    turns = [t for g in games for t in g["turns"] if t.get("guess") is not None and t["codes_remaining"] > 1]
    rel = [t["eig"] / t["max_eig"] for t in turns if t["max_eig"] > 0]
    solved = [g for g in games if g["solved"]]
    return {"games": len(games), "solve_rate": round(len(solved) / len(games), 3),
            "guesses_solved_mean": round(float(np.mean([g["own_guesses"] for g in solved])), 2) if solved else None,
            "eig_relative_to_max_mean": round(float(np.mean(rel)), 3),
            "share_at_max_eig": round(float(np.mean([r > 0.999 for r in rel])), 3),
            "share_consistent_guesses": round(float(np.mean([t["consistent"] for t in turns])), 3),
            "invalid_turn_rate": round(sum(g["invalid_turns"] for g in games) / max(1, sum(g["own_guesses"] for g in games)), 3)}


def summarize(df, rng, label):
    ci, draws = boot_ci(df, rng)
    one = df[(df.condition == "Max-EIG") & (df.codes_remaining == 1)]
    ideal_effect = float(df[df.condition == "Max-EIG"].ideal.mean() - df[df.condition == "Simple"].ideal.mean())
    return {
        "games": int(len(df)),
        "solve_rate_by_condition": df.groupby("condition").solved.mean().round(3).to_dict(),
        "mean_guesses_by_condition": df.groupby("condition").guesses.mean().round(2).to_dict(),
        "signature_effect_maxeig_minus_simple": round(effect(df), 2), "ci95": ci,
        "ideal_effect": round(ideal_effect, 2),
        "effect_relative_to_ideal": round(effect(df) - ideal_effect, 2),
        "one_code_subset": {"games": int(len(one)), "mean_guesses": round(float(one.guesses.mean()), 2) if len(one) else None,
                            "solve_rate": round(float(one.solved.mean()), 3) if len(one) else None},
        "hazard_check": hazard_test(df),
    }, draws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()
    mm = get_game(4, 4)
    stimuli = json.loads((ROOT / "data/stimuli/e1_stimuli.json").read_text())
    rng = np.random.default_rng(0)

    human = human_e1(mm, stimuli)
    out = {"measure": f"guesses after the supplied history; unsolved = {UNSOLVED}"}
    out["humans"], h_draws = summarize(human, rng, "humans")
    for m in args.models:
        if not (B1 / f"{m}_e1_T0.jsonl").exists():
            continue
        df = model_e1(m, stimuli)
        res, draws = summarize(df, rng, m)
        diff = draws - h_draws
        res["difference_from_humans"] = {"estimate": round(res["signature_effect_maxeig_minus_simple"] - out["humans"]["signature_effect_maxeig_minus_simple"], 2),
                                         "ci95": [round(float(np.percentile(diff, 2.5)), 2), round(float(np.percentile(diff, 97.5)), 2)]}
        res["invalid_turn_rate"] = round(float(df.invalid_turns.sum() / max(1, df.guesses.sum())), 3)
        res["free_play"] = free_play(mm, m)
        out[m] = res
    (ROOT / "results/study_b/b1_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
