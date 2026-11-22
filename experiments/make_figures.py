"""Figures for the paper and poster, drawn from the saved result files (no model calls, no refits).

  python experiments/make_figures.py      ->  results/figures/fig*.pdf and .png

  fig1_b1       B1: Max-EIG minus Simple for people, every model and the ideal player; Max-EIG histories that leave
                one code
  fig2_b3       B2/B3: Qwen3-32B's judgment errors by number of clues; held-out log likelihood of each process model
                minus M0
  fig3_ca       C-a: the aid's effect on error rates (T3 minus T0 and minus T1); the noise of evidence use by condition
  fig4_cc       C-c and C-b: predicted against observed Gamma; MSPE of each rival minus the support map; choice shift
                against measured Gamma
  fig5_study_a  Study A: E1 guesses after the supplied history, people and the ideal player; per-person AIC differences

The B3 differences, the per-pair Gamma and the choice shifts are recomputed here from the records. Each is checked
against the result file it came from.
"""
import glob
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
logging.getLogger("fontTools").setLevel(logging.ERROR)  # font-subsetting notices, not errors
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments/study_c"))
import prediction_test as P  # noqa: E402
import primary_tests as PT  # noqa: E402

OUT = ROOT / "results/figures"
R = ROOT / "results"

# Colors: categorical slots in fixed order (validated for color-vision deficiency on white), ink and chrome.
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
PEOPLE, MODEL = S2, S1          # the same entity keeps its color in every figure
FULL_WIDTH, HEIGHT = 6.9, 2.45  # inches; a two-column paper page

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7.5,
    "axes.labelsize": 7.5, "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2, "xtick.labelsize": 7, "ytick.labelsize": 7, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "legend.fontsize": 6.8, "legend.frameon": False, "pdf.fonttype": 42,
    "ps.fonttype": 42, "savefig.dpi": 300, "lines.solid_capstyle": "round",
})


def load(path):
    return json.loads((R / path).read_text())


def grid(ax, axis):
    ax.grid(axis=axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def zero_line(ax, vertical=True, at=0.0):
    (ax.axvline if vertical else ax.axhline)(at, color=AXIS, linewidth=0.8, zorder=1)


def point(ax, x, y, color, hollow=False, marker="o", size=4.2, xerr=None, yerr=None):
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=marker, markersize=size, color=color, ecolor=color,
                elinewidth=1.0, capsize=0, markeredgewidth=1.0,
                markerfacecolor="white" if hollow else color, zorder=3)


def minus(text):
    """Typographic minus signs in annotation text."""
    return text.replace("-", "−")


def ci_err(est, ci):
    return [[est - ci[0]], [ci[1] - est]]


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})  # reproducible bytes
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- Figure 1: B1
def fig1():
    s = load("study_b/b1_summary.json")
    rows = [("People", "humans"), ("gpt-oss-120b", "gpt-oss-120b"), ("Claude Haiku 5.5", "claude-haiku-5-5"),
            ("Claude Opus 5.5", "claude-opus-5-5"), ("Qwen3-32B", "qwen3-32b"), ("Phi-4", "phi-4"),
            ("OLMo-2-32B", "olmo2-32b")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_WIDTH, HEIGHT), gridspec_kw={"wspace": 0.08}, sharey=True)
    ys = np.arange(len(rows))[::-1]
    ideal = s["humans"]["ideal_effect"]
    for y, (label, key) in zip(ys, rows):
        d = s[key]
        if key == "humans":
            point(a, d["signature_effect_maxeig_minus_simple"], y, PEOPLE, xerr=ci_err(d["signature_effect_maxeig_minus_simple"], d["ci95"]))
        elif d.get("eligible_for_inference"):
            point(a, d["signature_effect_maxeig_minus_simple"], y, MODEL, xerr=ci_err(d["signature_effect_maxeig_minus_simple"], d["ci95"]))
        else:
            point(a, d["descriptive"]["maxeig_minus_simple"], y, MUTED, hollow=True)
        oc = d["one_code_subset"]["mean_guesses"]
        point(b, oc, y, PEOPLE if key == "humans" else MODEL if d.get("eligible_for_inference") else MUTED,
              hollow=not (key == "humans" or d.get("eligible_for_inference")))
    zero_line(a)
    a.axvline(ideal, color=MUTED, linewidth=0.8, zorder=1)
    a.text(ideal, len(rows) - 0.45, "ideal player", ha="center", va="bottom", fontsize=6.5, color=INK2)
    a.set_yticks(ys, [r[0] for r in rows])
    a.set_ylim(-0.6, len(rows) - 0.1)
    a.set_xlabel("Extra guesses after Max-EIG vs. Simple histories")
    a.set_title("A  The human signature effect")
    grid(a, "x")
    b.axvline(1, color=MUTED, linewidth=0.8, zorder=1)
    b.text(1, len(rows) - 0.45, "ideal player", ha="left", va="bottom", fontsize=6.5, color=INK2)
    b.set_xlim(0, 16.5)
    b.set_xlabel("Guesses to solve (unsolved = 16)")
    b.set_title("B  Decoding a history that leaves one code")
    b.tick_params(axis="y", length=0)
    grid(b, "x")
    handles = [plt.Line2D([], [], marker="o", color=PEOPLE, linestyle="", markersize=4.2),
               plt.Line2D([], [], marker="o", color=MODEL, linestyle="", markersize=4.2),
               plt.Line2D([], [], marker="o", color=MUTED, markerfacecolor="white", linestyle="", markersize=4.2)]
    fig.legend(handles, ["People", "Models that solve at least half their free-play games",
                         "Models below that rule (descriptive)"], loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.16), handletextpad=0.3, columnspacing=1.6)
    save(fig, "fig1_b1")


# ---------------------------------------------------------------- Figure 2: B2/B3
def fig2():
    ep = load("study_b/process_fits/qwen3-32b_error_profile_T0.json")["T0"]["by_clue_count"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_WIDTH, HEIGHT), gridspec_kw={"wspace": 0.62, "width_ratios": [1, 1]})
    series = [("possible", "Still-possible code ruled out", S1, "o"),
              ("near", "Near miss kept (ruled out by the newest clue)", S2, "s"),
              ("far", "Far miss kept", S3, "D")]
    for key, label, color, marker in series:
        xs = sorted(int(k) for k in ep[key])
        ys = [ep[key][str(x)]["rate"] for x in xs]
        a.plot(xs, ys, color=color, linewidth=1.5, marker=marker, markersize=4.2, label=label, zorder=3)
    a.set_xticks([1, 2, 3])
    a.set_xlim(0.8, 3.2)
    a.set_ylim(0, 0.82)
    a.set_xlabel("Clues in the history")
    a.set_ylabel("Error rate")
    a.set_title("A  Qwen3-32B's judgment errors")
    a.legend(loc="upper left", handlelength=1.6)
    grid(a, "y")

    models, ll, boots, iv = PT.heldout_by_family()
    base = models.index("M0")
    names = {"M1": "M1 noisy evidence", "M2c": "M2c conservative (human)", "M2s": "M2s symmetric",
             "M3": "M3 forgetting", "M1a": "M1a two rates (exploratory)"}
    order = ["M1", "M2c", "M2s", "M3", "M1a"]
    ys = np.arange(len(order))[::-1]
    for y, m in zip(ys, order):
        d = ll[:, models.index(m)] - ll[:, base]
        sums = d[boots].sum(1)
        est, ci = float(d.sum()), np.percentile(sums, [2.5, 97.5])
        if m == "M2s":  # the reported interval of the best planned model against M0
            saved = iv["best_planned_minus_other_planned"]["M0"]
            assert round(est, 2) == saved["difference"] and ci.round(2).tolist() == saved["ci95"], (est, ci, saved)
        point(b, est, y, MUTED if m == "M1a" else MODEL, hollow=(m == "M1a"), xerr=ci_err(est, ci))
    zero_line(b)
    b.set_yticks(ys, [names[m] for m in order])
    b.set_ylim(-0.6, len(order) - 0.4)
    b.set_xlabel("Held-out log likelihood minus M0 (exact + noise)")
    b.set_title("B  Process models on 80 held-out families")
    grid(b, "x")
    save(fig, "fig2_b3")


# ---------------------------------------------------------------- Figure 3: C-a
def fig3():
    ca = load("study_c/ca_aid_effects.json")
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_WIDTH, HEIGHT + 0.2), gridspec_kw={"wspace": 0.55, "width_ratios": [1.35, 1]})
    measures = [("false_exclusion_multi", "Possible code ruled out (multi-clue)"),
                ("false_exclusion_single", "Possible code ruled out (one clue)"),
                ("false_retention_near_multi", "Near miss kept (multi-clue)"),
                ("false_retention_far_multi", "Far miss kept (multi-clue)"),
                ("false_retention_near_single", "Near miss kept (one clue)"),
                ("identification", "Named code possible (finished replies)"),
                ("identify_unfinished", "Reply unfinished")]
    ys = np.arange(len(measures))[::-1]
    for (contrast, color, marker, dy) in (("T3-T0", S1, "o", 0.14), ("T3-T1", S2, "s", -0.14)):
        for y, (m, _) in zip(ys, measures):
            c = ca["contrasts"][contrast][m]
            point(a, c["difference"], y + dy, color, marker=marker, size=3.8, xerr=ci_err(c["difference"], c["ci95"]))
    zero_line(a)
    a.set_yticks(ys, [lab for _, lab in measures])
    a.set_ylim(-0.6, len(measures) - 0.4)
    a.set_xlabel("Change in rate with the aid (95% CI)")
    a.set_title("A  What the aid changes")
    fig.legend([plt.Line2D([], [], marker="o", color=S1, linestyle="", markersize=3.8),
                plt.Line2D([], [], marker="s", color=S2, linestyle="", markersize=3.8)],
               ["Aid minus standard prompt (T3 − T0)", "Aid minus length-matched prompt (T3 − T1)"],
               loc="lower center", ncol=2, bbox_to_anchor=(0.36, -0.1), handletextpad=0.3, columnspacing=1.6)
    grid(a, "x")

    conds = [("T0", "Standard"), ("T1", "Length-\nmatched"), ("T2", "Generic\ncheck"), ("T3", "Structured\naid")]
    for i, (c, _) in enumerate(conds):
        nus = [json.loads(Path(f).read_text())["fits"]["M2s"]["params"]["nu"]
               for f in glob.glob(str(R / f"study_b/process_fits/boot_qwen3-32b_{c}/rep_*.json"))]
        est = ca["parameters"][c]["M2s"]["nu"]
        lo, hi = np.percentile(nus, [2.5, 97.5])
        point(b, i, est, MODEL, yerr=[[est - lo], [hi - est]])
    sh = ca["parameter_shifts"]["T3-T0"]["M2s.nu"]
    b.text(3, 0.6, f"T3 − T0: +{sh['difference']:.2f}\n[{sh['ci95'][0]:.2f}, {sh['ci95'][1]:.2f}]",
           ha="center", va="bottom", fontsize=6.5, color=INK2)
    b.set_xticks(range(len(conds)), [lab for _, lab in conds])
    b.set_xlim(-0.5, len(conds) - 0.5)
    b.set_ylim(0.3, 0.7)
    b.set_ylabel(r"Noise of evidence use, $\nu$ (M2s)")
    b.set_title("B  Fitted noise by prompt")
    grid(b, "y")
    save(fig, "fig3_ca")


# ---------------------------------------------------------------- Figure 4: C-c and C-b
def pair_data():
    """Per confirmation pair: stratum, predicted Gamma (support map), observed Gamma, choice shift."""
    lock = json.loads(P.LOCK.read_text())
    b0, c0, pairs = P.records("qwen3-32b", "T0", P.CONF_SPLIT, P.BASE_CAP)
    b3, c3, _ = P.records("qwen3-32b", "T3", P.CONF_SPLIT, P.BASE_CAP)
    gam = P.observed_gamma(P.group({"T0": b0, "T3": b3}), pairs)
    logit = lambda k, n: float(np.log(((k + 0.5) / (n + 1.0)) / (1 - (k + 0.5) / (n + 1.0))))  # noqa: E731
    rows = []
    for p in lock["predictions"]:
        pid = p["pair_id"]
        ch = {c: [r for r in rows_c if r["pair_id"] == pid and r["parsed"]] for c, rows_c in (("T0", c0), ("T3", c3))}
        k = {c: sum(r["chose"] == "q_simple" for r in ch[c]) for c in ch}
        rows.append({"stratum": p["stratum"], "pred": p["gamma_hat"][lock["primary"]], "obs": gam[pid][0],
                     "shift": logit(k["T3"], len(ch["T3"])) - logit(k["T0"], len(ch["T0"]))})
    return rows


def fig4():
    sc = load("study_c/cc_score.json")
    cb = load("study_c/cb_choice_shift_c_confirmation.json")
    rows = pair_data()
    pred, obs, shift = (np.array([r[k] for r in rows]) for k in ("pred", "obs", "shift"))
    cal = sc["by_predictor"]["support_map"]
    assert abs(np.polyfit(pred, obs, 1)[0] - cal["calibration_slope"]) < 1e-9
    assert abs(np.polyfit(obs, shift, 1)[0] - cb["primary_measured_gamma"]["slope"]) < 1e-9

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(FULL_WIDTH, HEIGHT + 0.1), gridspec_kw={"wspace": 0.95, "width_ratios": [1, 0.9, 1]})
    strata = [("a_matched_vstar", "Equal V*", S1, "o"), ("b_similar_eig_diff_burden", "Similar EIG, different digits", S2, "s"),
              ("c_predicted_zero", "Predicted ≈ 0", S3, "D")]
    for key, label, color, marker in strata:
        sel = np.array([r["stratum"] == key for r in rows])
        a.scatter(pred[sel], obs[sel], s=9, marker=marker, facecolor=color, edgecolor="white", linewidth=0.4,
                  alpha=0.85, label=label, zorder=3)
    zero_line(a)
    zero_line(a, vertical=False)
    a.set_xlabel("Predicted Γ (support map)")
    a.set_ylabel("Observed Γ")
    a.set_title("A  Predicted vs. observed")
    a.text(0.04, 0.02, minus(f"slope {cal['calibration_slope']:.2f}\n[{cal['calibration_slope_ci95'][0]:.2f}, "
           f"{cal['calibration_slope_ci95'][1]:.2f}]"), transform=a.transAxes, ha="left", va="bottom", fontsize=6.5, color=INK2)
    fig.legend(*a.get_legend_handles_labels(), title="Pair stratum (A)", title_fontsize=6.8, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.23), handletextpad=0.2, columnspacing=1.4, markerscale=1.3)
    grid(a, "both")

    rivals = [("difficulty_regression", "Difficulty regression"), ("reporting_only", "Reporting only"),
              ("uniform_gain", "Uniform gain (Γ = 0)"), ("ideal_noise", "Ideal + noise")]
    ys = np.arange(len(rivals))[::-1]
    for y, (k, _) in zip(ys, rivals):
        v = sc["primary_vs_rivals"][k]
        est, ci = v["mspe_rival_minus_model"] * 1e4, [x * 1e4 for x in v["ci95"]]
        point(b, est, y, MODEL, xerr=ci_err(est, ci))
    zero_line(b)
    b.set_yticks(ys, [lab for _, lab in rivals])
    b.set_ylim(-0.6, len(rivals) - 0.2)
    b.set_xlabel("MSPE(rival) − MSPE(support map), $\\times 10^{-4}$")
    b.text(0.99, 0.99, "above 0 favors\nthe support map", transform=b.transAxes, ha="right", va="top", fontsize=6.3, color=INK2)
    b.set_title("B  H3: support map vs. rivals")
    grid(b, "x")

    c.scatter(obs, shift, s=8, facecolor=MODEL, edgecolor="white", linewidth=0.4, alpha=0.6, zorder=3)
    xs = np.linspace(obs.min(), obs.max(), 2)
    k1, k0 = np.polyfit(obs, shift, 1)
    c.plot(xs, k0 + k1 * xs, color=INK, linewidth=1.2, zorder=4)
    zero_line(c)
    zero_line(c, vertical=False)
    pm = cb["primary_measured_gamma"]
    c.text(0.02, 0.99, minus(f"slope {pm['slope']:.2f}\n[{pm['ci95'][0]:.2f}, {pm['ci95'][1]:.2f}]"),
           transform=c.transAxes, ha="left", va="top", fontsize=6.5, color=INK2)
    c.set_xlabel("Observed Γ")
    c.set_ylabel("Choice shift (Δ logit, aid − standard)")
    c.set_title("C  H4: choices vs. Γ")
    grid(c, "both")
    save(fig, "fig4_cc")


# ---------------------------------------------------------------- Figure 5: Study A
def fig5():
    e1 = load("study_a/e1_reproduction.json")["by_condition"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_WIDTH * 0.75, HEIGHT), gridspec_kw={"wspace": 0.45})
    conds = [("Simple", "Simple"), ("Max-EIG", "Max-EIG"), ("None", "No history")]
    for i, (k, _) in enumerate(conds):
        point(a, i - 0.1, e1[k]["human_guesses_solved_games"], PEOPLE)
        point(a, i + 0.1, e1[k]["ideal_guesses"], MUTED, hollow=True)
    a.set_xticks(range(len(conds)), [lab for _, lab in conds])
    a.set_xlim(-0.5, len(conds) - 0.5)
    a.set_ylim(0, 9)
    a.set_ylabel("Guesses after the supplied history")
    a.set_title("A  Experiment 1")
    a.legend([plt.Line2D([], [], marker="o", color=PEOPLE, linestyle="", markersize=4.2),
              plt.Line2D([], [], marker="o", color=MUTED, markerfacecolor="white", linestyle="", markersize=4.2)],
             ["People (solved games)", "Ideal player"], loc="upper left")
    grid(a, "y")

    d = []
    for f in sorted(glob.glob(str(R / "study_a/e3_fits_converged/participant_*.json"))):
        x = json.loads(Path(f).read_text())["fits"]
        d.append((x["leaky"]["aic"] - x["bounded"]["aic"], x["unbounded"]["aic"] - x["bounded"]["aic"]))
    d = np.array(d)
    comp = load("study_a/e3_model_comparison_converged.json")
    assert abs(d[:, 0].mean() - comp["mean_dAIC_bounded_vs_leaky"]) < 0.05
    assert abs(d[:, 1].mean() - comp["mean_dAIC_bounded_vs_unbounded"]) < 0.05
    rng = np.random.default_rng(0)
    for i in range(2):
        b.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d[:, i], s=9, facecolor=PEOPLE, edgecolor="white",
                  linewidth=0.4, alpha=0.8, zorder=3)
        b.plot([i - 0.22, i + 0.22], [d[:, i].mean()] * 2, color=INK, linewidth=1.4, zorder=4)
    zero_line(b, vertical=False)
    b.set_xticks([0, 1], ["Leaky −\nbounded", "Unbounded −\nbounded"])
    b.set_xlim(-0.5, 1.5)
    b.set_ylabel("AIC difference per person (> 0 favors bounded)")
    b.set_title("B  Experiment 3 model fits")
    grid(b, "y")
    save(fig, "fig5_study_a")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5):
        f()
