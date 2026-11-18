"""B3 and C-a: fit the process models to the battery and compare them.

  fit       fit M0-M3 on the processing-calibration families; score held-out log likelihood on the
            confirmation families; AIC on calibration. One condition per call.
  profile   error profile: false exclusions (saying a still-possible code is ruled out) and false
            retentions (near misses / far misses called possible), by clue count and query complexity.

  python experiments/study_b/fit_process_models.py fit --model qwen3-32b --condition T0
  python experiments/study_b/fit_process_models.py profile --model qwen3-32b --conditions T0 T1 T2 T3
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mastermind import human_models as hm  # noqa: E402
from mastermind import process_models as pm  # noqa: E402
from mastermind.engine import get_game  # noqa: E402
from mastermind.states import n_distinct  # noqa: E402

BATTERY = ROOT / "results/study_b/battery"
OUT = ROOT / "results/study_b/process_fits"
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
            items.append(("identify", r["split"], pm.Item(mm, hist, "identify", mm.index(r["answer"]))))
        else:
            ans = [(mm.index(c), v) for c, v in (r.get("labels") or {}).items() if v is not None]
            if not ans:
                fmt["unparsed"] += 1
                continue
            items.append(("classify", r["split"], pm.Item(mm, hist, "classify", ans)))
        fmt["answered"] += 1
    return items, fmt


def fit(model, condition):
    mm = get_game(4, 4)
    caches = {k: hm.BoundedCache(v) for k, v in CACHES.items()}
    items, fmt = to_items(mm, load_records(model, condition))
    cal = [it for _, s, it in items if s == "processing_calibration"]
    held = [it for _, s, it in items if s == "confirmation"]
    res = {"model": model, "condition": condition, "items_calibration": len(cal), "items_heldout": len(held), "format": fmt, "fits": {}}
    for m in pm.MODELS:
        f = pm.fit(m, cal, caches)
        f["heldout_loglik"] = pm.loglik(m, f["params"], held, caches)
        res["fits"][m] = f
        print(condition, m, round(f["aic"], 1), round(f["heldout_loglik"], 1), f["params"], flush=True)
    aics = {m: f["aic"] for m, f in res["fits"].items()}
    res["best_aic"] = min(aics, key=aics.get)
    res["best_heldout"] = max(res["fits"], key=lambda m: res["fits"][m]["heldout_loglik"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{model}_{condition}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "fits"}, indent=2))


def profile(model, conditions):
    mm = get_game(4, 4)
    out = {}
    for c in conditions:
        recs = [r for r in load_records(model, c) if r["task"] in ("classify", "classify_single") and r.get("labels")]
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
        out[c] = {"false_exclusion": rate("possible"), "false_retention_near": rate("near"), "false_retention_far": rate("far"),
                  "single_clue_false_exclusion": rate("possible", "classify_single"), "cells": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{model}_error_profile.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({c: {k: v for k, v in d.items() if k != "cells"} for c, d in out.items()}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--model", required=True)
    f.add_argument("--condition", default="T0")
    p = sub.add_parser("profile")
    p.add_argument("--model", required=True)
    p.add_argument("--conditions", nargs="+", default=["T0"])
    args = ap.parse_args()
    fit(args.model, args.condition) if args.cmd == "fit" else profile(args.model, args.conditions)


if __name__ == "__main__":
    main()
