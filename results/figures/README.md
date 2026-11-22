# Figures

Drawn by `python experiments/make_figures.py` from the result files named in each caption. The same command runs
at the end of `experiments/reproduce_summaries.sh`. Each figure is saved as a PDF (vector, for the paper) and a
PNG (for the poster and slides), sized for the full width of a two-column page. The numbers the script recomputes
from the records (B3 differences, per-pair Γ, choice shifts) are checked against the result files they come
from.

Colors are fixed per entity: people orange, models blue, descriptive or exploratory entries hollow gray. Every
series also differs in marker shape, so the figures read in grayscale.

## Figure 1 (`fig1_b1`): B1, the human signature effect

**(A)** Extra guesses after informative but complex (Max-EIG) histories compared with Simple histories, on the
Experiment 1 starting positions.
- Data: people, 393 games (44 people); each model, 1,320 games (132 positions × 10).
- Unsolved games count as 16.
- Bars: 95% intervals from resampling starting positions within condition.
- Hollow markers: models below the solve-rate rule (point estimates only).
- Vertical line: the ideal player (−0.59).

**(B)** Guesses to solve after Max-EIG histories that leave exactly one possible code. People played 46 such
games and each model 190. The ideal player needs 1.

*Source:* `study_b/b1_summary.json`.

## Figure 2 (`fig2_b3`): B2 and B3, how Qwen3-32B uses evidence (standard prompt)

**(A)** Error rates of Qwen3's candidate judgments by number of clues, on the processing-calibration and
confirmation families:
- a still-possible code ruled out (false exclusion);
- a near miss kept (a code ruled out only by the newest clue);
- a far miss kept.

**(B)** Held-out log likelihood of each process model minus exact inference with reporting noise (M0), on the 80
held-out families.
- 95% family-bootstrap intervals, 2,000 resamples.
- M1, M2c and M2s are almost identical. M1a was added after the planned comparison and is exploratory.

*Sources:* `study_b/process_fits/qwen3-32b_error_profile_T0.json`, `qwen3-32b_T0.json` and
`qwen3-32b_T0_intervals.json`.

## Figure 3 (`fig3_ca`): C-a, the structured aid and evidence use (Qwen3-32B, 140 families)

**(A)** Change in each error rate with the aid (T3), against the standard prompt (T0) and against the
length-matched prompt (T1). 95% intervals from 10,000 paired family resamples.

**(B)** The fitted noise of evidence use (ν in M2s) under each prompt.
- 95% intervals from 1,000 family resamples per prompt.
- The paired difference between the aid and the standard prompt is printed.

*Source:* `study_c/ca_aid_effects.json`.

## Figure 4 (`fig4_cc`): the preregistered tests (360 pairs, 120 fresh families)

**(A)** The aid's change in usefulness (Γ) as predicted by the support map, against the change observed. Pairs
are oriented so that the prediction is at least 0.

**(B)** H3: the family-level mean squared prediction error of each rival minus that of the support map.
- 95% intervals from 10,000 family resamples.
- Every point estimate favors the rival, and no difference is reliable.

**(C)** H4: the shift in the model's choice of q1 (aid minus standard, in logits) against observed Γ, with the
least-squares line. The slope and its 95% family-bootstrap interval are printed.

*Sources:* `study_c/cc_score.json`, `study_c/cb_choice_shift_c_confirmation.json` and
`prereg/prediction_lock.json`.

## Figure 5 (`fig5_study_a`): Study A, the human reproduction

**(A)** Experiment 1: guesses after the supplied history, for people (solved games) and the ideal player.

**(B)** Experiment 3: for each person, the AIC of the leaky and of the unbounded model minus that of the bounded
model (converged fits, 39 people). Bars mark the means, 27.8 and 128.1.

*Sources:* `study_a/e1_reproduction.json`, `study_a/e3_fits_converged/` and
`study_a/e3_model_comparison_converged.json`.
