"""Fit the unbounded, leaky and bounded models to each Experiment 3 participant
(solved games, as in the paper) and compare them by AIC.

  python experiments/study_a/fit_human_models.py fit --participant 0     # one participant, all 3 models
  python experiments/study_a/fit_human_models.py summarize               # all participants -> table
  add --include-unsolved to both for the sensitivity fit on all games
  python experiments/study_a/fit_human_models.py heldout                 # held-out participants + game-index trend
  python experiments/study_a/fit_human_models.py fit --participant 0 --models bounded --half early   (then late; then `halves`)
Targets (paper, Mastermind): bounded beats unbounded by mean dAIC 82.5 and leaky by 72.3; 37/39 favor bounded.
"""
import argparse
import ast
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mastermind import human_models as hm  # noqa: E402

E3 = ROOT / "data/human/cheyette2025_osf_vjrqu/Experiments/E3"
CACHE = ROOT / "data/model_cache/bounded_cache_v1.npz"
OUT = ROOT / "results/study_a/e3_fits"


def participant_games(mm, solved_only=True):
    keep = {r["uniqueid"] for r in csv.DictReader(open(E3 / "analysis/e3_human_preprocessed.csv")) if r["game"] == "Mastermind"}
    games = defaultdict(list)
    for r in csv.DictReader(open(E3 / "mastermind_E3.csv")):
        if r["uniqueid"] in keep:
            games[(r["uniqueid"], r["trial_id"])].append(r)
    by = defaultdict(list)
    for (uid, _), rows in sorted(games.items()):
        rows.sort(key=lambda r: int(r["guess_number"]))
        g = [(mm.index(ast.literal_eval(r["guess"])), (int(r["n_green"]), int(r["n_yellow"]))) for r in rows]
        if not solved_only or g[-1][1][0] == mm.P:
            by[uid].append(g)
    return dict(sorted(by.items()))


def out_dir(include_unsolved=False, half=None):
    return OUT.with_name(OUT.name + ("_with_unsolved" if include_unsolved else "") + (f"_{half}" if half else ""))


def fit_one(idx, models, seed, include_unsolved=False, half=None):
    task = hm.Task()
    cache = hm.BoundedCache(CACHE) if "bounded" in models else None
    uid, games = list(participant_games(task.mm, solved_only=not include_unsolved).items())[idx]
    if half:  # game-index sensitivity: session games 1-5 vs 6-10
        order = trial_order()
        games = [g for g in games if (order[(uid, tuple(q for q, _ in g))] <= 5) == (half == "early")]
    out = {"participant": idx, "uniqueid": uid, "games": len(games), "decisions": sum(len(g) for g in games), "fits": {}}
    for m in models:
        out["fits"][m] = hm.fit(task, m, games, cache=cache, seed=seed + idx)
        print(idx, m, round(out["fits"][m]["loglik"], 1), out["fits"][m]["params"], flush=True)
    d = out_dir(include_unsolved, half)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"participant_{idx:02d}.json").write_text(json.dumps(out, indent=1))


def summarize(include_unsolved=False):
    fits = [json.loads(p.read_text()) for p in sorted(out_dir(include_unsolved).glob("participant_*.json"))]
    aic = {m: np.array([f["fits"][m]["aic"] for f in fits]) for m in hm.MODELS}
    d_unb = aic["unbounded"] - aic["bounded"]
    d_leaky = aic["leaky"] - aic["bounded"]
    summary = {
        "participants": len(fits),
        "mean_dAIC_bounded_vs_unbounded": round(float(d_unb.mean()), 1),
        "mean_dAIC_bounded_vs_leaky": round(float(d_leaky.mean()), 1),
        "participants_favoring_bounded_over_both": int(((d_unb > 0) & (d_leaky > 0)).sum()),
        "best_model_counts": {m: int(c) for m, c in zip(hm.MODELS, np.bincount(np.argmin(np.stack([aic[m] for m in hm.MODELS]), 0), minlength=3))},
        "paper": {"dAIC_vs_unbounded": 82.5, "dAIC_vs_leaky": 72.3, "favor_bounded": "37/39"},
        "median_params": {m: {k: round(float(np.median([f["fits"][m]["params"][k] for f in fits])), 4)
                              for k in hm.MODELS[m]} for m in hm.MODELS},
    }
    name = "e3_model_comparison_with_unsolved.json" if include_unsolved else "e3_model_comparison.json"
    (ROOT / "results/study_a" / name).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def heldout(seed=0):
    """Held-out participants: predict each person's games from the median fitted parameters
    of the other 38 (per-parameter median; nothing from the held-out person is used). Also the
    game-index check: does a static model's per-decision fit drift across a person's 10 games?"""
    import statsmodels.formula.api as smf

    task = hm.Task()
    cache = hm.BoundedCache(CACHE)
    fits = [json.loads(p.read_text()) for p in sorted(OUT.glob("participant_*.json"))]
    people = list(participant_games(task.mm).items())
    order = trial_order()
    ll = {m: np.zeros(len(fits)) for m in hm.MODELS}
    rows = []
    for i, ((uid, games), f) in enumerate(zip(people, fits)):
        assert f["uniqueid"] == uid
        for m in hm.MODELS:
            others = [g["fits"][m]["params"] for j, g in enumerate(fits) if j != i]
            p = {k: float(np.median([o[k] for o in others])) for k in hm.MODELS[m]}
            ll[m][i] = hm.loglik(task, m, p, games, cache)
        tables = hm._tables(task, "bounded", f["fits"]["bounded"]["params"], cache)
        for g in games:
            lg = hm.game_loglik(task, "bounded", f["fits"]["bounded"]["params"], g, cache, tables)
            rows.append({"pid": i, "game_index": order[(uid, tuple(q for q, _ in g))], "ll_per_decision": lg / len(g)})
    n_dec = sum(len(g) for _, gs in people for g in gs)
    rng = np.random.default_rng(seed)
    boot = rng.integers(len(fits), size=(5000, len(fits)))

    def diff(a, b):
        d = ll[a] - ll[b]
        lo, hi = np.percentile(d[boot].sum(1), [2.5, 97.5])
        return {"total": round(float(d.sum()), 1), "ci95": [round(float(lo), 1), round(float(hi), 1)],
                "participants_favoring_first": int((d > 0).sum())}

    import pandas as pd
    trend = smf.ols("ll_per_decision ~ game_index + C(pid)", data=pd.DataFrame(rows)).fit(cov_type="cluster", cov_kwds={"groups": pd.DataFrame(rows)["pid"]})
    out = {
        "method": "leave-one-participant-out; per-parameter median of the other 38 participants' fits; solved games",
        "heldout_loglik_total": {m: round(float(v.sum()), 1) for m, v in ll.items()},
        "heldout_loglik_per_decision": {m: round(float(v.sum() / n_dec), 4) for m, v in ll.items()},
        "best_heldout_counts": {m: int(c) for m, c in zip(hm.MODELS, np.bincount(np.argmax(np.stack([ll[m] for m in hm.MODELS]), 0), minlength=3))},
        "bounded_minus_unbounded": diff("bounded", "unbounded"),
        "bounded_minus_leaky": diff("bounded", "leaky"),
        "game_index_trend_bounded": {"slope_ll_per_decision_per_game": round(float(trend.params["game_index"]), 4),
                                     "se": round(float(trend.bse["game_index"]), 4), "p": round(float(trend.pvalues["game_index"]), 4)},
    }
    (ROOT / "results/study_a/e3_heldout.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def halves():
    """Static vs. game-index model (sensitivity): the bounded model fitted to all solved games
    (4 parameters) against separate fits to games 1-5 and 6-10 (8 parameters), by AIC per participant."""
    static = {json.loads(p.read_text())["participant"]: json.loads(p.read_text()) for p in sorted(OUT.glob("participant_*.json"))}
    rows = []
    for p in sorted(out_dir(half="early").glob("participant_*.json")):
        e = json.loads(p.read_text())
        late_path = out_dir(half="late") / p.name
        if not late_path.exists():
            continue
        lt = json.loads(late_path.read_text())
        i = e["participant"]
        split_aic = e["fits"]["bounded"]["aic"] + lt["fits"]["bounded"]["aic"]
        rows.append({"participant": i, "d_aic_static_minus_split": static[i]["fits"]["bounded"]["aic"] - split_aic,
                     "early": e["fits"]["bounded"]["params"], "late": lt["fits"]["bounded"]["params"]})
    d = np.array([r["d_aic_static_minus_split"] for r in rows])
    out = {
        "participants": len(rows),
        "mean_dAIC_static_minus_split": round(float(d.mean()), 1),
        "median_dAIC_static_minus_split": round(float(np.median(d)), 1),
        "participants_favoring_split": int((d > 0).sum()),
        "sign_test_p": round(float(stats.binomtest(int((d > 0).sum()), len(d)).pvalue), 4),
        "wilcoxon_late_minus_early": {k: {"median_diff": round(float(np.median([r["late"][k] - r["early"][k] for r in rows])), 4),
                                          "p": round(float(stats.wilcoxon([r["late"][k] for r in rows], [r["early"][k] for r in rows]).pvalue), 4)}
                                      for k in hm.MODELS["bounded"]},
        "median_params_early": {k: round(float(np.median([r["early"][k] for r in rows])), 4) for k in hm.MODELS["bounded"]},
        "median_params_late": {k: round(float(np.median([r["late"][k] for r in rows])), 4) for k in hm.MODELS["bounded"]},
    }
    (ROOT / "results/study_a/e3_game_index_halves.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def trial_order():
    """(uniqueid, guess sequence) -> game position 1..10 within the person's session (numeric trial_id)."""
    mm = hm.Task().mm
    games = defaultdict(list)
    for r in csv.DictReader(open(E3 / "mastermind_E3.csv")):
        games[(r["uniqueid"], int(r["trial_id"]))].append(r)
    order, per = {}, defaultdict(list)
    for (uid, t), rows in games.items():
        per[uid].append(t)
        rows.sort(key=lambda r: int(r["guess_number"]))
        order[(uid, tuple(mm.index(ast.literal_eval(r["guess"])) for r in rows))] = t
    rank = {uid: {t: k + 1 for k, t in enumerate(sorted(ts))} for uid, ts in per.items()}
    return {key: rank[key[0]][t] for key, t in order.items()}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--participant", type=int, required=True)
    f.add_argument("--models", nargs="+", default=list(hm.MODELS))
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--include-unsolved", action="store_true", help="sensitivity: fit on all games")
    f.add_argument("--half", choices=["early", "late"], help="sensitivity: fit games 1-5 or 6-10 only")
    s = sub.add_parser("summarize")
    s.add_argument("--include-unsolved", action="store_true")
    sub.add_parser("heldout")
    sub.add_parser("halves")
    args = ap.parse_args()
    if args.cmd == "fit":
        fit_one(args.participant, args.models, args.seed, args.include_unsolved, args.half)
    elif args.cmd == "heldout":
        heldout()
    elif args.cmd == "halves":
        halves()
    else:
        summarize(args.include_unsolved)


if __name__ == "__main__":
    main()
