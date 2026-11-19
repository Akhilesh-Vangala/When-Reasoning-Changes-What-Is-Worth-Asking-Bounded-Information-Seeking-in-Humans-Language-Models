"""B3: fit the process models to the battery and compare them.

  fit       fit M0-M3 on the processing-calibration families; score held-out log likelihood on the
            confirmation families; AIC on calibration. One condition per call.
  profile   error profile: false exclusions (saying a still-possible code is ruled out) and false
            retentions (near misses / far misses called possible), by clue count and query complexity,
            for the multi-clue and single-clue probes separately, plus identification accuracy. Uses the
            processing-calibration and confirmation families only (the same families in every condition;
            the development families were run at T0 only).

  confirm   confirmation test of the exploratory model M1a on the reserve families (never used before it,
            plan in experiments/study_b/B3_CONFIRMATION.md): every model keeps the parameters saved by `fit`
            (processing-calibration families); held-out log likelihood per family, family bootstrap.

  python experiments/study_b/fit_process_models.py fit --model qwen3-32b --condition T0
  python experiments/study_b/fit_process_models.py profile --model qwen3-32b --conditions T0 T1 T2 T3
  python experiments/study_b/fit_process_models.py confirm --model qwen3-32b --condition T0
"""
import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mastermind import human_models as hm  # noqa: E402
from mastermind import process_models as pm  # noqa: E402
from mastermind.engine import get_game  # noqa: E402
from mastermind.states import n_distinct  # noqa: E402

BATTERY = ROOT / "results/study_b/battery"
OUT = ROOT / "results/study_b/process_fits"
PROFILE_SPLITS = ("processing_calibration", "confirmation")
CACHES = {"M2c": ROOT / "data/model_cache/bounded_cache_v1.npz", "M2s": ROOT / "data/model_cache/symmetric_cache_v1.npz"}


def load_records(model, condition):
    rows = []
    for path in sorted(glob.glob(str(BATTERY / f"{model}_{condition}_*.jsonl"))):
        rows += [json.loads(line) for line in open(path)]
    seen, out = set(), []
    for r in rows:  # the same item can appear in two files (e.g. development run)
        key = (r["family_id"], r["task"], r["clue_index"], r["sample"])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def to_items(mm, records):
    items, fmt = [], {"answered": 0, "unparsed": 0}
    for r in records:
        hist = [(mm.index(g), (b, w)) for g, b, w in r["history"]]
        if r["task"] == "identify":
            if r["answer"] is None:
                fmt["unparsed"] += 1
                continue
            items.append(("identify", r["split"], pm.Item(mm, hist, "identify", mm.index(r["answer"])), r["family_id"]))
        else:
            ans = [(mm.index(c), v) for c, v in (r.get("labels") or {}).items() if v is not None]
            if not ans:
                fmt["unparsed"] += 1
                continue
            items.append(("classify", r["split"], pm.Item(mm, hist, "classify", ans), r["family_id"]))
        fmt["answered"] += 1
    return items, fmt


def _exact_gap(cache, alpha, mm):
    """Largest |pi - exact| over every query and outcome: 0 means the representation is exact inference."""
    rep = cache.representation(alpha)
    gap = 0.0
    for q in range(mm.N):
        for o in set(mm.table[q].tolist()):
            gap = max(gap, float(np.abs(rep[q, o] - (mm.table[q] == o)).max()))
    return gap


def boundary_notes(m, params, caches, mm, max_clues=6):
    """Fitted values at a boundary where the model reduces to a simpler one, so that its fit is not evidence
    for its own mechanism. For M2c/M2s this is checked directly: is the fitted representation exact?"""
    notes = []
    if m in ("M2c", "M2s"):
        gap = _exact_gap(caches[m], params["alpha"], mm)
        if gap < 0.01:
            notes.append(f"representation at the fitted alpha is exact (max gap {gap:.3g}): the model reduces to M1")
        elif params["alpha"] <= 1e-3:
            kind = "false retentions" if m == "M2c" else "false exclusions and retentions"
            notes.append(f"alpha at or near its lower bound (<= 1e-3), but the representation is not exact (max gap {gap:.2f}): "
                         f"with the distance weight fixed at 1, near-exact approximations keep some weight as alpha -> 0 ({kind})")
    if m == "M3":
        r = expit(params["k0"] + params["k1"] * np.arange(max_clues))
        if r.min() > 0.999:
            notes.append("every clue retained (r > 0.999): the model reduces to M0")
    for k in ("nu", "nu_x", "nu_r", "zeta"):
        if k in params and (params[k] < 1e-3 or params[k] > 0.999):
            notes.append(f"{k} at a boundary ({params[k]:.4g})")
    return notes


def fit(model, condition):
    mm = get_game(4, 4)
    caches = {k: hm.BoundedCache(v) for k, v in CACHES.items()}
    items, fmt = to_items(mm, load_records(model, condition))
    cal = [it for _, s, it, _ in items if s == "processing_calibration"]
    held = [it for _, s, it, _ in items if s == "confirmation"]
    res = {"model": model, "condition": condition, "items_calibration": len(cal), "items_heldout": len(held), "format": fmt, "fits": {}}
    for m in pm.MODELS:
        f = pm.fit(m, cal, caches)
        f["heldout_loglik"] = pm.loglik(m, f["params"], held, caches)
        f["boundary"] = boundary_notes(m, f["params"], caches, mm)
        res["fits"][m] = f
        print(condition, m, round(f["aic"], 1), round(f["heldout_loglik"], 1), f["params"], flush=True)
    # the planned comparison (M0-M3) is the primary result; exploratory models are labelled separately
    res["planned_models"] = pm.PLANNED
    res["best_aic"] = min(pm.PLANNED, key=lambda m: res["fits"][m]["aic"])
    res["best_heldout"] = max(pm.PLANNED, key=lambda m: res["fits"][m]["heldout_loglik"])
    res["exploratory"] = {"models": pm.EXPLORATORY,
                          "note": "added after the planned comparison, motivated by the error profile; "
                                  "confirmation test on the reserve families: `confirm`",
                          "best_aic_all": min(res["fits"], key=lambda m: res["fits"][m]["aic"]),
                          "best_heldout_all": max(res["fits"], key=lambda m: res["fits"][m]["heldout_loglik"])}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{model}_{condition}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "fits"}, indent=2))


def profile(model, conditions):
    mm = get_game(4, 4)
    out = {}
    for c in conditions:
        every = [r for r in load_records(model, c) if r["split"] in PROFILE_SPLITS]
        recs = [r for r in every if r["task"] in ("classify", "classify_single") and r.get("labels")]
        ident = [r for r in every if r["task"] == "identify" and r.get("parsed")]
        cells = {}
        for r in recs:
            n_clues = len(r["history"])
            newest = mm.index(r["history"][-1][0])
            for (code, kind) in r["items"]:
                said = r["labels"].get(code)
                if said is None:
                    continue
                key = (r["task"], kind, n_clues, n_distinct(mm, newest))
                cells.setdefault(key, []).append(said if kind != "possible" else not said)  # error indicator
        rows = [{"task": t, "kind": k, "clues": n, "query_distinct_digits": d, "n": len(v), "error_rate": round(float(np.mean(v)), 3)}
                for (t, k, n, d), v in sorted(cells.items())]
        def rate(kind, task=None):
            v = [x for (t, k, _, _), vals in cells.items() for x in vals if k == kind and (task is None or t == task)]
            return round(float(np.mean(v)), 3) if v else None
        out[c] = {"multi_clue": {k: rate(k, "classify") for k in ("possible", "near", "far")},
                  "single_clue": {k: rate(k, "classify_single") for k in ("possible", "near", "far")},
                  "identification": round(float(np.mean([bool(r["correct"]) for r in ident])), 3) if ident else None,
                  "note": "rates are error rates: 'possible' = false exclusion, 'near'/'far' = false retention",
                  "cells": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{model}_error_profile_{'+'.join(conditions)}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({c: {k: v for k, v in d.items() if k != "cells"} for c, d in out.items()}, indent=2))


def _p_possible(m, p, item, caches):
    """P(answer "still possible") for every code, as in pm.item_loglik."""
    if m == "M3":
        r = expit(p["k0"] + p["k1"] * np.arange(len(item.clues))[::-1])
        g = np.prod(r[:, None] * item.c + (1 - r[:, None]), axis=0)
    else:
        g = np.prod(pm.clue_weights(m, p, item, caches), axis=0)
    return (1 - p["zeta"]) * g + p["zeta"] / 2


def confirm(model, condition, split="reserve", n_boot=2000, seed=20261118):
    """Pre-stated test (B3_CONFIRMATION.md): with the saved calibration parameters, M1a must have a higher
    held-out log likelihood on the reserve families than every planned model, and the family-bootstrap 95%
    interval of the difference to the best planned model must lie above 0."""
    mm = get_game(4, 4)
    fit_path = OUT / f"{model}_{condition}.json"
    saved = json.loads(fit_path.read_text())
    caches = {k: hm.BoundedCache(v) for k, v in CACHES.items()}
    recs = [r for r in load_records(model, condition) if r["split"] == split]
    if not recs:
        raise SystemExit(f"no {split} records for {model} {condition}")
    items, fmt = to_items(mm, recs)
    fams = sorted({f for *_, f in items})
    models = list(pm.MODELS)
    ll = np.zeros((len(fams), len(models)))  # held-out log likelihood per family and model
    index = {f: i for i, f in enumerate(fams)}
    for _, _, it, fam in items:
        for j, m in enumerate(models):
            ll[index[fam], j] += pm.item_loglik(m, saved["fits"][m]["params"], it, caches)
    total = dict(zip(models, ll.sum(0).round(2).tolist()))
    best_planned = max(pm.PLANNED, key=total.get)
    rng = np.random.default_rng(seed)
    boots = rng.integers(len(fams), size=(n_boot, len(fams)))
    diffs = {}
    for m in pm.PLANNED:
        d = ll[:, models.index("M1a")] - ll[:, models.index(m)]
        bs = d[boots].sum(1)
        diffs[m] = {"difference": round(float(d.sum()), 2), "ci95": np.percentile(bs, [2.5, 97.5]).round(2).tolist(),
                    "families_favoring_M1a": int((d > 0).sum())}
    passed = all(total["M1a"] > total[m] for m in pm.PLANNED) and diffs[best_planned]["ci95"][0] > 0

    # descriptive: observed vs predicted error rates by probe type (no test)
    cells = {}
    for r in recs:
        if r["task"] not in ("classify", "classify_single") or not r.get("labels"):
            continue
        it = pm.Item(mm, [(mm.index(g), (b, w)) for g, b, w in r["history"]], "classify", [])
        p_yes = {m: _p_possible(m, saved["fits"][m]["params"], it, caches) for m in models}
        for code, kind in r["items"]:
            said = r["labels"].get(code)
            if said is None:
                continue
            x, err = mm.index(code), kind == "possible"  # error = "ruled out" for possible codes, "possible" otherwise
            c = cells.setdefault(f"{r['task']}:{kind}", {"n": 0, "observed": 0.0, **{m: 0.0 for m in models}})
            c["n"] += 1
            c["observed"] += (not said) if err else said
            for m in models:
                c[m] += (1 - p_yes[m][x]) if err else p_yes[m][x]
    error_rates = {k: {"n": c["n"], **{m: round(c[m] / c["n"], 3) for m in ["observed"] + models}} for k, c in sorted(cells.items())}
    res = {"model": model, "condition": condition, "split": split, "families": len(fams), "items": len(items),
           "format": fmt, "parameters_from": fit_path.name,
           "parameters_sha256": hashlib.sha256(json.dumps({m: saved["fits"][m]["params"] for m in models},
                                                          sort_keys=True).encode()).hexdigest(),
           "heldout_loglik": total, "best_planned": best_planned, "M1a_minus_planned": diffs,
           "bootstrap": {"resamples": n_boot, "unit": "family", "seed": seed}, "confirmed": bool(passed),
           "error_rates": error_rates}
    (OUT / f"{model}_{condition}_{split}_confirmation.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--model", required=True)
    f.add_argument("--condition", default="T0")
    p = sub.add_parser("profile")
    p.add_argument("--model", required=True)
    p.add_argument("--conditions", nargs="+", default=["T0"])
    c = sub.add_parser("confirm")
    c.add_argument("--model", required=True)
    c.add_argument("--condition", default="T0")
    args = ap.parse_args()
    if args.cmd == "fit":
        fit(args.model, args.condition)
    elif args.cmd == "profile":
        profile(args.model, args.conditions)
    else:
        confirm(args.model, args.condition)


if __name__ == "__main__":
    main()
