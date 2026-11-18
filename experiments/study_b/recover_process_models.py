"""Parameter and model recovery for the process models M0-M3 on the planned B2 design
(processing-calibration families, the battery's item mix, 4 samples per item).

Each replicate draws parameters for every generating model (jittered around plausible values),
simulates the battery, and fits all five models. The confusion matrix counts how often the generating
model has the lowest AIC; models the data cannot tell apart are merged before B3 is interpreted.

  python experiments/study_b/recover_process_models.py run --rep 0
  python experiments/study_b/recover_process_models.py summarize
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/study_b"))
from mastermind import human_models as hm  # noqa: E402
from mastermind import process_models as pm  # noqa: E402
from mastermind.engine import get_game  # noqa: E402
from probe_battery import build_jobs  # noqa: E402

OUT = ROOT / "results/study_b/process_recovery"
CACHES = {"M2c": ROOT / "data/model_cache/bounded_cache_v1.npz", "M2s": ROOT / "data/model_cache/symmetric_cache_v1.npz"}
GEN = {"M0": {"zeta": 0.05}, "M1": {"nu": 0.2, "zeta": 0.05}, "M2c": {"alpha": 1.0, "nu": 0.05, "zeta": 0.05},
       "M2s": {"alpha": 1.0, "nu": 0.05, "zeta": 0.05}, "M3": {"k0": 1.5, "k1": -1.0, "zeta": 0.05}}


def design_items(mm, samples=4):
    fams = [f for f in json.loads((ROOT / "data/splits/families_v1.json").read_text()) if f["split"] == "processing_calibration"]
    items = []
    for j in build_jobs(mm, fams, "T0", samples):
        m = j["meta"]
        hist = [(mm.index(g), (b, w)) for g, b, w in m["history"]]
        if m["task"] == "identify":
            items.append(pm.Item(mm, hist, "identify", 0))
        else:
            items.append(pm.Item(mm, hist, "classify", [(mm.index(c), False) for c, _ in m["items"]]))
    return items


def jitter(p, rng):
    out = {}
    for k, v in p.items():
        if k in ("zeta", "nu"):
            out[k] = float(np.clip(v * np.exp(rng.normal(0, 0.4)), 0.005, 0.45))
        elif k == "alpha":
            out[k] = float(v * np.exp(rng.normal(0, 1.0)))
        else:
            out[k] = float(v + rng.normal(0, 0.5))
    return out


def run(rep):
    mm = get_game(4, 4)
    caches = {k: hm.BoundedCache(v) for k, v in CACHES.items()}
    items = design_items(mm)
    rng = np.random.default_rng(100 + rep)
    out = {"rep": rep, "items": len(items), "generated": {}}
    for gen, base in GEN.items():
        p = jitter(base, rng)
        sim = pm.simulate(gen, p, items, caches, rng)
        fits = {m: pm.fit(m, sim, caches) for m in pm.MODELS}
        out["generated"][gen] = {"true": p, "fits": fits, "best": min(fits, key=lambda m: fits[m]["aic"])}
        print(rep, gen, "->", out["generated"][gen]["best"], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"rep_{rep:02d}.json").write_text(json.dumps(out, indent=1))


def summarize():
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("rep_*.json"))]
    names = list(pm.MODELS)
    conf = np.zeros((len(names), len(names)), int)
    par = {}
    for r in rows:
        for i, gen in enumerate(names):
            g = r["generated"][gen]
            conf[i, names.index(g["best"])] += 1
            for k, v in g["true"].items():
                par.setdefault(f"{gen}.{k}", []).append((v, g["fits"][gen]["params"][k]))
    summary = {
        "replicates": len(rows),
        "confusion_rows_generating_cols_best": {g: dict(zip(names, map(int, conf[i]))) for i, g in enumerate(names)},
        "recovery_rate": {g: round(conf[i, i] / max(1, conf[i].sum()), 3) for i, g in enumerate(names)},
        "parameter_recovery_corr": {k: round(float(np.corrcoef(np.array(v).T)[0, 1]), 3) for k, v in par.items() if len(v) > 2},
    }
    (ROOT / "results/study_b/process_recovery_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--rep", type=int, required=True)
    sub.add_parser("summarize")
    args = ap.parse_args()
    run(args.rep) if args.cmd == "run" else summarize()


if __name__ == "__main__":
    main()
