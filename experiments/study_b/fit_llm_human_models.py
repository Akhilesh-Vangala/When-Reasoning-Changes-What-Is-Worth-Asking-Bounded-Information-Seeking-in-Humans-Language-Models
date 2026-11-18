"""B1b: fit the Study A human models to a language model's free play.

Same task, same three models (unbounded, leaky, bounded with the conservative cache), same fitting
procedure as for the 39 people in Study A. The model at fixed decoding settings is one agent, so the
main fit is on all its solved free-play games. Its result is then placed in the distribution of the
per-participant human fits.

ΔAIC grows with the amount of data (a model plays ~100 games, a person 10), so the comparison with
people uses ΔAIC per decision, plus fits on blocks of 10 games (the same unit as one person).

  python experiments/study_b/fit_llm_human_models.py fit --model gpt-oss-120b                 # pooled fit
  python experiments/study_b/fit_llm_human_models.py fit --model gpt-oss-120b --block 3       # games 30-39
  python experiments/study_b/fit_llm_human_models.py fit --model gpt-oss-120b --boot 17       # bootstrap resample 17
  python experiments/study_b/fit_llm_human_models.py summarize --model gpt-oss-120b
  add --include-unsolved to every call for the sensitivity fit on all games (as in Study A)

Invalid turns (no parseable guess after the re-asks) gave no feedback, so they are dropped from the
game; the rest of the game is unchanged. Applies to models that pass the B1 solve-rate rule;
floor models are reported descriptively in B1.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mastermind import human_models as hm  # noqa: E402

B1 = ROOT / "results/study_b/b1"
CACHE = ROOT / "data/model_cache/bounded_cache_v1.npz"
HUMAN = ROOT / "results/study_a"
OUT = ROOT / "results/study_b/b1b"
BLOCK = 10  # games per block, as one human participant


def llm_games(mm, model, condition, solved_only=True):
    """Free-play games in game-id order: [(game number, [(query index, (black, white)), ...]), ...]."""
    recs = [json.loads(line) for line in open(B1 / f"{model}_free_{condition}.jsonl")]
    recs.sort(key=lambda r: int(r["game_id"].split(":")[1]))
    assert len({r["game_id"] for r in recs}) == len(recs), "duplicate games: rebuild the B1 file first"
    games = []
    for r in recs:
        g = [(mm.index(t["guess"]), (t["black"], t["white"])) for t in r["turns"] if t["guess"] is not None]
        if g and (not solved_only or r["solved"]):
            games.append((int(r["game_id"].split(":")[1]), g))
    return games, len(recs)


def run_dir(model, condition, include_unsolved):
    return OUT / f"{model}_{condition}{'_with_unsolved' if include_unsolved else ''}"


def fit(model, condition, include_unsolved=False, block=None, boot=None, seed=0):
    task = hm.Task()
    cache = hm.BoundedCache(CACHE)
    numbered, n_played = llm_games(task.mm, model, condition, solved_only=not include_unsolved)
    games = [g for _, g in numbered]
    if block is not None:  # games numbered block*10 .. block*10+9 (solved ones, unless --include-unsolved)
        games = [g for i, g in numbered if block * BLOCK <= i < (block + 1) * BLOCK]
        name = f"block_{block:02d}"
    elif boot is not None:  # resample games with replacement
        rng = np.random.default_rng(1000 + boot)
        games = [games[i] for i in rng.integers(len(games), size=len(games))]
        name = f"boot_{boot:03d}"
    else:
        name = "pooled"
    out = {"model": model, "condition": condition, "unit": name, "games_played": n_played, "games": len(games),
           "decisions": sum(len(g) for g in games), "fits": {}}
    for m in hm.MODELS:
        out["fits"][m] = hm.fit(task, m, games, cache=cache, seed=seed)
        print(name, m, round(out["fits"][m]["loglik"], 1), out["fits"][m]["params"], flush=True)
    d = run_dir(model, condition, include_unsolved)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"{name}.json.tmp"
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(d / f"{name}.json")


def d_aic(f):
    a = {m: f["fits"][m]["aic"] for m in hm.MODELS}
    return {"vs_unbounded": a["unbounded"] - a["bounded"], "vs_leaky": a["leaky"] - a["bounded"],
            "best": min(a, key=a.get)}


def percentile(x, dist):
    """Share of people with a value below x (ties count half)."""
    dist = np.asarray(dist)
    return round(float(((dist < x).sum() + 0.5 * (dist == x).sum()) / len(dist)), 3)


def summarize(model, condition, include_unsolved=False):
    d = run_dir(model, condition, include_unsolved)
    pooled = json.loads((d / "pooled.json").read_text())
    human_dir = HUMAN / ("e3_fits_with_unsolved" if include_unsolved else "e3_fits")
    people = [json.loads(p.read_text()) for p in sorted(human_dir.glob("participant_*.json"))]

    def per_decision(f):
        x = d_aic(f)
        return {k: x[k] / f["decisions"] for k in ("vs_unbounded", "vs_leaky")}

    h_pd = [per_decision(f) for f in people]
    p = d_aic(pooled)
    p_pd = per_decision(pooled)
    out = {
        "model": model, "condition": condition, "include_unsolved": include_unsolved,
        "games_played": pooled["games_played"], "games_fitted": pooled["games"], "decisions": pooled["decisions"],
        "pooled": {
            "aic": {m: round(pooled["fits"][m]["aic"], 1) for m in hm.MODELS},
            "best_model": p["best"],
            "dAIC_bounded_vs_unbounded": round(p["vs_unbounded"], 1),
            "dAIC_bounded_vs_leaky": round(p["vs_leaky"], 1),
            "dAIC_per_decision": {k: round(v, 4) for k, v in p_pd.items()},
            "params": {m: pooled["fits"][m]["params"] for m in hm.MODELS},
        },
        "human_comparison": {
            "people": len(people),
            "human_dAIC_per_decision_median": {k: round(float(np.median([h[k] for h in h_pd])), 4) for k in ("vs_unbounded", "vs_leaky")},
            "percentile_dAIC_per_decision": {k: percentile(p_pd[k], [h[k] for h in h_pd]) for k in ("vs_unbounded", "vs_leaky")},
            "human_bounded_params_median": {k: round(float(np.median([f["fits"]["bounded"]["params"][k] for f in people])), 4)
                                            for k in hm.MODELS["bounded"]},
            "percentile_bounded_params": {k: percentile(pooled["fits"]["bounded"]["params"][k],
                                                        [f["fits"]["bounded"]["params"][k] for f in people])
                                          for k in hm.MODELS["bounded"]},
        },
    }
    blocks = [json.loads(q.read_text()) for q in sorted(d.glob("block_*.json"))]
    if blocks:
        bd = [d_aic(b) for b in blocks]
        out["blocks_of_10_games"] = {
            "blocks": len(blocks),
            "best_model_counts": {m: sum(x["best"] == m for x in bd) for m in hm.MODELS},
            "favoring_bounded_over_both": sum(x["vs_unbounded"] > 0 and x["vs_leaky"] > 0 for x in bd),
            "mean_dAIC_vs_unbounded": round(float(np.mean([x["vs_unbounded"] for x in bd])), 1),
            "mean_dAIC_vs_leaky": round(float(np.mean([x["vs_leaky"] for x in bd])), 1),
            "median_bounded_params": {k: round(float(np.median([b["fits"]["bounded"]["params"][k] for b in blocks])), 4)
                                      for k in hm.MODELS["bounded"]},
            "human_reference": "Study A: 34 of 39 people favor bounded (solved games); mean dAIC 104.6 / 83.5",
        }
    boots = [json.loads(q.read_text()) for q in sorted(d.glob("boot_*.json"))]
    if boots:
        bd = [per_decision(b) for b in boots]
        ci = lambda xs: [round(float(v), 4) for v in np.percentile(xs, [2.5, 97.5])]  # noqa: E731
        out["bootstrap_over_games"] = {
            "resamples": len(boots),
            "dAIC_per_decision_ci95": {k: ci([x[k] for x in bd]) for k in ("vs_unbounded", "vs_leaky")},
            "bounded_params_ci95": {k: ci([b["fits"]["bounded"]["params"][k] for b in boots]) for k in hm.MODELS["bounded"]},
            "share_bounded_best": round(float(np.mean([d_aic(b)["best"] == "bounded" for b in boots])), 3),
        }
    path = ROOT / "results/study_b" / f"b1b_{model}_{condition}{'_with_unsolved' if include_unsolved else ''}.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "summarize"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--condition", default="T0")
    ap.add_argument("--include-unsolved", action="store_true")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--block", type=int)
    g.add_argument("--boot", type=int)
    a = ap.parse_args()
    if a.cmd == "fit":
        fit(a.model, a.condition, a.include_unsolved, a.block, a.boot)
    else:
        summarize(a.model, a.condition, a.include_unsolved)


if __name__ == "__main__":
    main()
