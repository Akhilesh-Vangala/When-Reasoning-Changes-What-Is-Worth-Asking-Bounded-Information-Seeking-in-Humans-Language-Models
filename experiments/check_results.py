"""Integrity check of every Study A and Study B result file. Run after each sync from the cluster:

  python experiments/check_results.py            # prints PASS / FAIL per check, exit code 1 on any failure

Checks, recomputed from scratch rather than trusted from the files:
  prompts     the current prompt templates still hash to the frozen values
  stimuli     132 Experiment 1 starting positions; split families are disjoint
  b1          every game file: lines parse, game ids unique and complete, supplied history and secret match the
              stimulus, every black/white/consistent/EIG value matches the oracle, solved flag and guess counts,
              prompt template hash, decoding settings match the registry (when recorded)
  battery     baseline (T0) probe records: unique items, families and histories match the split file, template
              hash, identification scoring
  study_a     per-participant fits: AIC = 2k - 2 log L, all participants present; summaries recomputed
  process     process-model fits: AIC, best-model labels
  b1b         B1b summaries recomputed from the pooled fit
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mastermind import human_models as hm  # noqa: E402
from mastermind import process_models as pm  # noqa: E402
from mastermind import prompts  # noqa: E402
from mastermind.engine import get_game  # noqa: E402

MM = get_game(4, 4)
FREEZE = json.loads((ROOT / "configs/prompt_freeze_v1.json").read_text())["template_hashes"]
REGISTRY = json.loads((ROOT / "configs/model_registry.json").read_text())
STIMULI = {(s["stimulus_id"], s["condition"]): s for s in json.loads((ROOT / "data/stimuli/e1_stimuli.json").read_text())}
FAMILIES = {f["family_id"]: f for f in json.loads((ROOT / "data/splits/families_v1.json").read_text())}
FREE_GAMES = 100
results = []


def check(name, problems, detail=""):
    results.append((name, not problems))
    status = "PASS" if not problems else "FAIL"
    print(f"{status}  {name}{'  ' + detail if detail else ''}")
    for p in problems[:8]:
        print(f"        - {p}")
    if len(problems) > 8:
        print(f"        ... {len(problems) - 8} more")


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def check_prompts():
    bad = [f"{k}: frozen {v}, now {prompts.template_hash(*k.split('/'))}" for k, v in FREEZE.items()
           if prompts.template_hash(*k.split("/")) != v]
    check("prompts: templates match the freeze", bad, f"{len(FREEZE)} templates")


def check_inputs():
    bad = [] if len(STIMULI) == 132 else [f"{len(STIMULI)} stimuli, expected 132"]
    keys = Counter(f["key"] for f in FAMILIES.values())
    bad += [f"family key {k} appears {n} times (a state in two families/splits)" for k, n in keys.items() if n > 1]
    check("inputs: E1 stimuli and split families", bad, f"{len(STIMULI)} stimuli, {len(FAMILIES)} families")


def expected_decoding(model, condition):
    reg = REGISTRY[model]
    extra = reg.get("thinking_on" if condition == "T4" else "thinking_off", {})
    return reg.get("decoding"), extra


def check_b1_file(path):
    model, mode, condition = path.stem.rsplit("_", 2)
    bad, recs = [], []
    for n, line in enumerate(open(path), 1):
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            bad.append(f"line {n} does not parse")
    ids = Counter(r["game_id"] for r in recs)
    bad += [f"game {g} appears {c} times" for g, c in ids.items() if c > 1]
    if mode == "e1":
        per_stim = Counter(g.rsplit(":", 1)[0] for g in ids)
        want = {f"{s}:{c}" for s, c in STIMULI}
        if set(per_stim) != want:
            bad.append(f"{len(want - set(per_stim))} stimuli missing, {len(set(per_stim) - want)} unknown")
        if len(set(per_stim.values())) > 1 or min(per_stim.values(), default=0) < 10:
            bad.append(f"completions per stimulus not equal or < 10: {sorted(set(per_stim.values()))}")
    else:
        if set(ids) != {f"free:{g}" for g in range(FREE_GAMES)}:
            bad.append(f"{len(ids)} free games, expected free:0..{FREE_GAMES - 1}")
    th_want = FREEZE[f"guess/{condition}"]
    dec_want, extra_want = expected_decoding(model, condition)
    for r in recs:
        gid = r["game_id"]
        if (r.get("model"), r.get("mode"), r.get("condition")) != (model, mode, condition):
            bad.append(f"{gid}: record says {r.get('model')}/{r.get('mode')}/{r.get('condition')}")
        if "decoding" in r and (r["decoding"] != dec_want or r.get("extra") != extra_want):
            bad.append(f"{gid}: decoding {r['decoding']} {r.get('extra')} differs from the registry")
        secret = MM.index(r["secret"])
        history = [(MM.index(g), (b, w)) for g, b, w in r["supplied"]]
        if mode == "e1":
            stim = STIMULI[tuple(gid.split(":")[:2])]
            if r["secret"] != stim["secret"] or r["supplied"] != [[s["guess"], s["black"], s["white"]] for s in stim["supplied"]]:
                bad.append(f"{gid}: secret or supplied history differs from the stimulus")
        for t in r["turns"]:
            if t["template"] != th_want:
                bad.append(f"{gid} turn {t['turn']}: template {t['template']} != frozen {th_want}")
            mask = MM.consistent(history) if history else np.ones(MM.N, bool)
            if t["codes_remaining"] != int(mask.sum()):
                bad.append(f"{gid} turn {t['turn']}: codes_remaining {t['codes_remaining']} != {int(mask.sum())}")
            if t["guess"] is None:
                continue
            q = MM.index(t["guess"])
            b, w = MM.decode(MM.table[q, secret])
            if (t["black"], t["white"]) != (b, w):
                bad.append(f"{gid} turn {t['turn']}: feedback {t['black']},{t['white']} != oracle {b},{w}")
            if t["consistent"] != bool(mask[q]):
                bad.append(f"{gid} turn {t['turn']}: consistent flag wrong")
            if not close(t["eig"], float(MM.eig(mask, [q])[0]), 1e-3) or not close(t["max_eig"], float(MM.eig(mask).max()), 1e-3):
                bad.append(f"{gid} turn {t['turn']}: EIG values differ from the oracle")
            history.append((q, (b, w)))
        solved = bool(r["turns"]) and r["turns"][-1].get("black") == MM.P
        if r["solved"] != solved or r["own_guesses"] != len(r["turns"]) or len(r["turns"]) > 15:
            bad.append(f"{gid}: solved/own_guesses inconsistent with the turns")
        if not solved and len(r["turns"]) != 15:
            bad.append(f"{gid}: unsolved game stopped after {len(r['turns'])} guesses")
        if r["invalid_turns"] != sum(t["guess"] is None for t in r["turns"]):
            bad.append(f"{gid}: invalid_turns count wrong")
    solved_rate = np.mean([r["solved"] for r in recs]) if recs else float("nan")
    check(f"b1: {path.name}", bad, f"{len(recs)} games, solve rate {solved_rate:.3f}")


def check_battery(path):
    bad, seen = [], Counter()
    recs = []
    for n, line in enumerate(open(path), 1):
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            bad.append(f"line {n} does not parse")
    for r in recs:
        key = (r["family_id"], r["task"], r["clue_index"], r["sample"])
        seen[key] += 1
        fam = FAMILIES.get(r["family_id"])
        if fam is None or fam["split"] != r["split"]:
            bad.append(f"{key}: family or split not in the split file")
            continue
        full = [list(h) for h in fam["history"]]
        want_hist = [full[r["clue_index"]]] if r["task"] == "classify_single" else full
        if [list(h) for h in r["history"]] != want_hist:
            bad.append(f"{key}: history differs from the family")
        tkey = f"{'classify' if r['task'].startswith('classify') else r['task']}/{r['condition']}"
        if r["template"] != FREEZE[tkey]:
            bad.append(f"{key}: template {r['template']} != frozen {FREEZE[tkey]}")
        if r["task"] == "identify" and r.get("answer"):
            mask = MM.consistent([(MM.index(g), (b, w)) for g, b, w in r["history"]])
            if bool(r["correct"]) != bool(mask[MM.index(r["answer"])]):
                bad.append(f"{key}: identification scored wrong")
    bad += [f"{k} appears {c} times" for k, c in seen.items() if c > 1]
    check(f"battery: {path.name}", bad, f"{len(recs)} records")


def aic_ok(f):
    return close(f["aic"], 2 * f["k"] - 2 * f["loglik"])


def check_study_a():
    base = ROOT / "results/study_a"
    for d in sorted(base.glob("e3_fits*")):
        files = sorted(d.glob("participant_*.json"))
        bad = [] if len(files) == 39 else [f"{len(files)} participants, expected 39"]
        for p in files:
            fit = json.loads(p.read_text())
            for m, f in fit["fits"].items():
                if f["k"] != len(hm.MODELS[m]) or not aic_ok(f):
                    bad.append(f"{p.name} {m}: k or AIC inconsistent")
        check(f"study_a: {d.name}", bad, f"{len(files)} participant files")
    for name, folder in [("e3_model_comparison.json", "e3_fits"), ("e3_model_comparison_with_unsolved.json", "e3_fits_with_unsolved")]:
        s = json.loads((base / name).read_text())
        fits = [json.loads(p.read_text()) for p in sorted((base / folder).glob("participant_*.json"))]
        aic = {m: np.array([f["fits"][m]["aic"] for f in fits]) for m in hm.MODELS}
        bad = []
        if not close(s["mean_dAIC_bounded_vs_unbounded"], round(float((aic["unbounded"] - aic["bounded"]).mean()), 1), 1e-3):
            bad.append("mean dAIC vs unbounded does not match the participant files")
        if not close(s["mean_dAIC_bounded_vs_leaky"], round(float((aic["leaky"] - aic["bounded"]).mean()), 1), 1e-3):
            bad.append("mean dAIC vs leaky does not match the participant files")
        best = np.argmin(np.stack([aic[m] for m in hm.MODELS]), 0)
        if [s["best_model_counts"][m] for m in hm.MODELS] != [int((best == i).sum()) for i in range(3)]:
            bad.append("best-model counts do not match the participant files")
        check(f"study_a: {name} recomputed", bad)


def check_process_fits():
    for p in sorted((ROOT / "results/study_b/process_fits").glob("*_T0.json")):
        if "error_profile" in p.name:
            continue
        d = json.loads(p.read_text())
        bad = [f"{m}: AIC inconsistent" for m, f in d["fits"].items() if not aic_ok(f) or f["k"] != len(pm.MODELS[m])]
        planned = d.get("planned_models", list(d["fits"]))  # primary labels cover the planned models only
        if sorted(planned) != sorted(pm.PLANNED) or any(m not in d["fits"] for m in pm.MODELS):
            bad.append("fit file does not cover the planned and exploratory models")
        if d["best_aic"] != min(planned, key=lambda m: d["fits"][m]["aic"]):
            bad.append("best_aic label wrong")
        if d["best_heldout"] != max(planned, key=lambda m: d["fits"][m]["heldout_loglik"]):
            bad.append("best_heldout label wrong")
        ex = d.get("exploratory", {})
        if ex.get("best_aic_all") != min(d["fits"], key=lambda m: d["fits"][m]["aic"]) or \
                ex.get("best_heldout_all") != max(d["fits"], key=lambda m: d["fits"][m]["heldout_loglik"]):
            bad.append("exploratory labels wrong")
        check(f"process: {p.name}", bad)


def check_b1b():
    for s_path in sorted((ROOT / "results/study_b").glob("b1b_*.json")):
        s = json.loads(s_path.read_text())
        run = s_path.stem[len("b1b_"):]
        pooled = json.loads((ROOT / "results/study_b/b1b" / run / "pooled.json").read_text())
        a = {m: pooled["fits"][m]["aic"] for m in hm.MODELS}
        bad = [f"{m}: AIC inconsistent" for m, f in pooled["fits"].items() if not aic_ok(f)]
        if not close(s["pooled"]["dAIC_bounded_vs_unbounded"], round(a["unbounded"] - a["bounded"], 1), 1e-3):
            bad.append("pooled dAIC vs unbounded does not match pooled.json")
        if s["pooled"]["best_model"] != min(a, key=a.get):
            bad.append("best model label wrong")
        check(f"b1b: {s_path.name}", bad, f"{pooled['games']} games fitted")


def main():
    check_prompts()
    check_inputs()
    skip = set(sys.argv[1:])  # file names to skip (e.g. a file known to be pending a rebuild)
    for p in sorted((ROOT / "results/study_b/b1").glob("*.jsonl")):
        if p.name not in skip:
            check_b1_file(p)
    for p in sorted((ROOT / "results/study_b/battery").glob("*_T0_*.jsonl")):
        check_battery(p)
    check_study_a()
    check_process_fits()
    check_b1b()
    n_fail = sum(not ok for _, ok in results)
    print(f"\n{len(results) - n_fail} passed, {n_fail} failed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
