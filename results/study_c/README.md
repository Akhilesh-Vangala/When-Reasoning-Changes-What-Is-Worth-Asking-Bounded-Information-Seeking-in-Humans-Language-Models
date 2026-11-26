# Study C: capacity manipulation (Qwen3-32B)

Every number below is read from the file named in its section: the JSON summaries and, for counts of replies, the
JSONL records. `bash experiments/reproduce_summaries.sh`
regenerates the summaries from the saved records and fits. The scripts are listed in `experiments/study_c/README.md`.

The confirmation test (C-c and C-b) was preregistered. The analysis plan and the prediction lock
(`prereg/analysis_plan.md`, `prereg/prediction_lock.json`, SHA-256 `582604d8…b0406`) were committed and registered
on OSF before any confirmation data were collected. `prediction_test.py score` checks the six locked code
hashes before scoring. All 14 locked hashes (code, families, pairs, calibration, plan, prompts and model registry)
were also checked against the files before scoring, and all matched.

Conditions (frozen templates in `configs/prompt_freeze_v1.json`). Every condition may reason, and every reply is
capped at 6,144 tokens.
- **T0** standard prompt.
- **T1** length-matched control.
- **T2** generic "Reason carefully and check your work before you answer". This is an extra control, not in the
  proposal.
- **T3** the structured aid. For every candidate and every clue: count the right-position digits, count the
  shared digits, take white = shared − black, and keep a code only if both counts match.

## Summary

- **C-a: the aid does not make evidence use more exact.**
  - Errors on single clues and on multi-clue histories do not fall.
  - The fitted noise of evidence use rises (M2s ν 0.42 → 0.52).
  - When an aided reply finishes, it names a still-possible code more often (0.24 → 0.34), but far more replies
    run out of room (5% → 19% unfinished).
  - The generic "check your work" prompt does at least as well.
- **C-c (H3, preregistered): not supported.**
  - The support map predicts the aid's change in usefulness (Γ) no better than any of the four rivals. On every
    comparison it is slightly worse; no difference is reliable.
  - Γ does vary across pairs beyond sampling noise (true SD about 0.04), but no predictor anticipates how.
- **C-b (H4, preregistered): not supported.** Qwen3's own question choices do not shift toward the queries the
  aid makes more useful (slope −0.63 [−2.50, 1.32]).
- **Design check (post hoc).**
  - Γ was noisier, and its true spread smaller, than the power analysis assumed.
  - At the planned smallest effect the realized design had power of about 0.50, not 0.864.
  - The H3 point estimates favor the rivals, so the null is not a near miss.
- **How informative the nulls are (post hoc).**
  - The locked H3 test could not have detected a correct support map. Its predictions spread about 14 times less
    than Γ does, so its power against Γ = 0 was about 0.04 even if exactly right.
  - The scale-free check is informative. The support map correlates with observed Γ at −0.10 [−0.24, 0.06],
    excluding the +0.23 that a predictor at the smallest effect of interest would show.
  - H4: r = −0.04 [−0.14, 0.07]; BF01 = 3–9.

## C-a: how the aid changes evidence use (`ca_aid_effects.json`)

**Data.** The probe battery of B2 ran under T0–T3 on the same 140 families: 60 processing-calibration and
80 confirmation families, 4 samples per item.

**Error rates.** Intervals: 10,000 paired family-bootstrap resamples, with the same families in every condition.
- *False exclusion:* a still-possible code called ruled out.
- *False retention:* a ruled-out code called possible. Near misses are ruled out only by the newest clue.
- *Identification:* the share of finished replies whose named code is still possible.

| Measure | T0 | T1 | T2 | T3 | T3 − T0 | T3 − T1 |
|---|---|---|---|---|---|---|
| False exclusion, multi-clue | 0.569 | 0.542 | 0.576 | 0.604 | +0.035 [−0.014, 0.083] | +0.062 [0.017, 0.107] |
| False exclusion, single clue | 0.323 | 0.329 | 0.321 | 0.330 | +0.007 [−0.019, 0.034] | +0.001 [−0.027, 0.030] |
| False retention, near miss, multi-clue | 0.245 | 0.235 | 0.182 | 0.245 | −0.000 [−0.036, 0.037] | +0.010 [−0.023, 0.044] |
| False retention, far miss, multi-clue | 0.134 | 0.139 | 0.082 | 0.138 | +0.004 [−0.021, 0.027] | −0.000 [−0.027, 0.026] |
| False retention, near miss, single clue | 0.125 | 0.133 | 0.107 | 0.184 | +0.058 [0.041, 0.077] | +0.051 [0.033, 0.069] |
| Identification (finished replies) | 0.238 | 0.230 | 0.309 | 0.338 | +0.099 [0.048, 0.150] | +0.108 [0.055, 0.161] |
| Identification replies unfinished | 0.048 | 0.052 | 0.082 | 0.186 | +0.138 [0.098, 0.177] | +0.134 [0.098, 0.171] |

- **Clue checking does not improve.** No error rate falls under the aid. Near misses on single clues are kept
  more often, and against the length-matched control multi-clue false exclusions rise.
- **Identification improves, but only partly.** It improves on finished replies, but the aid also leaves 14 more
  percentage points of replies unfinished. Counting those as failures (identification × (1 − unfinished), from
  the same rates), the share of all replies that name a still-possible code is:

  | T0 | T1 | T2 | T3 |
  |---|---|---|---|
  | 0.227 | 0.218 | 0.284 | 0.275 |

- **Against the generic prompt (T3 − T2)**, the aid keeps more ruled-out codes:
  - near miss, multi-clue: +0.063 [0.032, 0.095];
  - far miss, multi-clue: +0.057 [0.033, 0.079];
  - near miss, single clue: +0.077 [0.061, 0.094].

  It identifies no better (+0.028 [−0.025, 0.081]) and leaves more replies unfinished
  (+0.104 [0.064, 0.143]).

**Process-model parameters.** The same models were fitted in every condition. Intervals come from 1,000 paired
family resamples, with the same family draws in every condition.

| Parameter | T0 | T1 | T2 | T3 | T3 − T0 | T3 − T1 |
|---|---|---|---|---|---|---|
| M2s ν (noisy evidence use) | 0.416 | 0.441 | 0.458 | 0.516 | +0.101 [0.035, 0.167] | +0.076 [0.008, 0.157] |
| M2s ζ (reporting noise) | 0.111 | 0.099 | 0.025 | 0.047 | −0.064 [−0.160, 0.033] | −0.052 [−0.153, 0.045] |
| M1 ν | 0.424 | 0.446 | 0.465 | 0.525 | +0.100 [0.036, 0.163] | +0.078 [0.020, 0.131] |
| M1a ν_x (rules out a possible code) | 0.276 | 0.289 | 0.310 | 0.312 | +0.036 [−0.010, 0.080] | +0.024 [−0.021, 0.068] |
| M1a ν_r (keeps a ruled-out code) | 0.067 | 0.075 | 0.073 | 0.097 | +0.029 [−0.008, 0.064] | +0.022 [−0.009, 0.057] |

- **The aid raises the noise of evidence use.** It raises ν by 0.10 against the standard prompt, and
  by 0.08 and 0.06 against the length-matched and generic prompts (M2s T3 − T2 +0.058 [0.001, 0.191]). Reporting noise does not change reliably.
- **The best model barely changes.** By AIC it is M2s under T0, T2 and T3, and M1 under T1. On held-out families
  it is M2s under T0–T2 and M2c under T3. Under T3, M2c and M1 have the same held-out likelihood to 0.1, because
  α sits at its lower limit (`process_fits/qwen3-32b_T*.json`).
- **M3 uses the corrected likelihood in every condition.** In T1–T3 this is the stored fit. A refit of M3 with the
  corrected likelihood reproduced the stored T1–T3 fits, and the 563 resamples refitted before the check was
  stopped, to within 10⁻¹² in log likelihood (a check, not saved as a file).

## Reply cap (`cap_pilot_qwen3-32b_cap15000.json`)

**How often replies were cut off.** At the 6,144-token cap of every Study B run, 25.2% of aided (T3) and 9.8% of
standard (T0) usefulness replies were cut off on the calibration families (`cc_calibration.json`).

**The pilot.** It redrew 200 cut-off T3 replies and 60 cut-off T0 replies at 15,000 tokens, with identical
prompts.
- Of the redraws that ran past 6,144 tokens, 16 of 69 (T3) and 0 of 5 (T0) finished by 15,000.
- None of those named a still-possible code.

**Decision.** The cap stayed at 6,144, and unfinished replies count as naming nothing. Each condition's share of
unfinished replies enters the model-based predictions.

**The discarded first run.** A first run of the pilot was discarded because of the client-timeout problem (see
Disclosures).

## C-c: does the fitted model predict how the aid changes usefulness? (`cc_score.json`; H3, preregistered)

**Design** (fixed before the lock; `prereg/analysis_plan.md`).
- 120 fresh families: none equivalent under symmetry to an earlier family, and none sharing a pair of clues
  with one.
- 360 pairs, one per stratum per family, chosen without outcomes:
  - (a) equal V*;
  - (b) similar EIG, different numbers of digits;
  - (c) predicted change near 0.
- 2,713 answer branches with 8 samples each; 8 choices per pair and condition.

**Collection.**
- 49,168 replies (43,408 usefulness, 5,760 choice), all collected.
- 360 of 360 pairs scored, with no missing branches.
- Unfinished usefulness replies: T0 8.2%, T3 24.4% (calibration 9.8%, 25.2%). Every unparsed reply is one that
  reached the cap.

**Measure.** Usefulness U(q) is the probability that the model names the secret after seeing q's answer,
(1/K) Σ_a P(named code ∈ S_a). The aid's effect on a pair is Γ = [U(q1,T3) − U(q2,T3)] − [U(q1,T0) − U(q2,T0)].
Pairs are oriented so that the support map predicts Γ ≥ 0.

**Observed Γ:** mean +0.0015, SD 0.072. The locked noise estimate is 0.058; see the design check for why it
reads low.

**Primary test.** Family-level mean squared prediction error (MSPE) of the support map against each rival:
MSPE(rival) − MSPE(support map), with 10,000 family-bootstrap resamples and Holm across the four rivals. H3 holds
only if the support map beats all four.

| Rival | MSPE(rival) − MSPE(support map) | 95% CI | One-sided p | Holm p | Families favoring the support map |
|---|---|---|---|---|---|
| difficulty regression | −0.000059 | [−0.000171, 0.000041] | 0.87 | 1.00 | 52 / 120 |
| reporting only | −0.000016 | [−0.000043, 0.000010] | 0.88 | 1.00 | 53 / 120 |
| uniform gain (Γ = 0) | −0.000043 | [−0.000129, 0.000037] | 0.85 | 1.00 | 55 / 120 |
| ideal + noise | −0.000042 | [−0.000129, 0.000038] | 0.84 | 1.00 | 55 / 120 |

**H3 is not supported.** Positive values would favor the support map; every point estimate favors the rival.
The secondary predictor (M2s fitted to the T0 and T3 batteries) and the exploratory one (M1a) do not beat any
rival either: every difference favors the rival (one-sided p 0.89–0.97, unadjusted, as the plan specifies;
`secondary_vs_rivals` in `cc_score.json`).

**Also reported** (as planned):

| Predictor | MSPE | Noise-corrected | Bias | Calibration slope [95% CI] | Sign accuracy, largest quarter of effects |
|---|---|---|---|---|---|
| support map (primary) | 0.00524 | 0.00188 | +0.0011 | −2.86 [−7.15, 1.52] | 0.54 |
| M2s fitted to the batteries (secondary) | 0.00526 | 0.00190 | +0.0017 | −2.34 [−5.71, 0.86] | 0.51 |
| M1a fitted to the batteries (exploratory) | 0.00541 | 0.00204 | +0.0053 | −0.37 [−1.23, 0.47] | 0.46 |
| difficulty regression | 0.00518 | 0.00182 | −0.0016 | +3.06 [−1.25, 7.71] | 0.54 |
| reporting only | 0.00522 | 0.00186 | +0.0003 | −4.39 [−10.68, 2.16] | 0.49 |
| uniform gain | 0.00520 | 0.00184 | −0.0015 | (constant) | (constant) |
| ideal + noise | 0.00520 | 0.00184 | −0.0015 | −2.36 [−23.3, 19.2] | 0.07 |

Calibration-slope intervals come from 2,000 family resamples. The largest quarter of effects is the 98 pairs whose
|Γ| is at or above its 75th percentile.

- **The models predicted almost no change.** Mean |Γ̂| for the support map is 0.004 in stratum (a), 0.003 in (b)
  and 0.0001 in (c) (`prereg/prediction_lock.json`).
- **The aid does change usefulness, unpredictably.** Γ varies beyond sampling noise (true SD about 0.04; design
  check below), but no calibration slope is reliably positive. Sign accuracy on the largest effects is at chance.
  "Ideal + noise", like uniform gain, predicts Γ ≈ 0, so its signs are mostly 0.
- **By stratum.** Mean Γ is −0.004 in (a), −0.002 in (b) and +0.011 in (c). In (a) and (b) every rival has a
  lower MSPE than the support map; in (c) the support map and the four rivals differ by less than 0.00002.
- **Reversals.** In 18.6% [14.4%, 23.1%] of pairs the aid reverses which query is more useful. This rate is raw
  and includes sampling noise.

The plan expected this outcome. Calibration had shown that the process models do not predict baseline usefulness
across queries and that Γ is measured with low reliability. The design was sized so that a null would be
informative (but see the design check).

## C-b: do the model's own choices follow the change in usefulness? (`cb_choice_shift_c_confirmation.json`; H4, preregistered)

**Choices.** The choice prompt carries the same guidance as its condition, so the chooser is the supported
reasoner.
- Parsed: T0 2,818 of 2,880, T3 2,720 of 2,880. Every unparsed reply reached the cap. No pair was dropped.
- P(choose q1), where q1 is the query the support map predicted the aid to favor: 0.475 under T0, 0.452 under T3.
  The mean shift is −0.10 logit, slightly away from the predicted query.

**Test.** The OLS slope of the choice shift (Δ logit P(choose q1), with 0.5 added to the counts) on measured Γ,
one-sided, with 10,000 family-bootstrap resamples. **Slope −0.63 [−2.50, 1.32], p = 0.75; H4 is not supported.**
- Rank correlation: about 0 (−0.001 in the locked script, which breaks ties by order; +0.007 with tie-aware ranks).
- Split-half reliability of measured Γ: 0.17, or 0.30 after the Spearman–Brown correction. The slope corrected
  for it is −2.13, still negative.
- Slope on the support map's Γ̂ (secondary): 5.3 [−51.1, 62.5], p = 0.44.
- The exploratory look on the calibration pairs before the lock showed the same thing: slope 0.06 [−3.08, 3.49]
  (`cb_choice_shift_support_calibration+choice_calibration.json`).

## Design check (post hoc; `cc_noise_check.json`; not part of the locked analysis)

**Why Γ is noisy.** Usefulness is (1/K) Σ_a p_a, where p_a is the share of a branch's samples that name a code in
that branch. So Γ's sampling noise grows with Σ_a p_a(1 − p_a) / (n K²).

**The locked estimate of that noise runs low.** It resamples a branch's n samples, a plug-in estimate that
understates the variance by (n − 1)/n: 25% at the 4 calibration samples, 12.5% at the 8 confirmation samples.

| | Calibration (96 pairs, 4 samples) | Confirmation (360 pairs, 8 samples) |
|---|---|---|
| Noise SD, locked estimator | 0.066 | 0.058 |
| Noise SD, corrected by n/(n − 1) | 0.076 | 0.062 |
| True SD of Γ, locked estimator | 0.052 | 0.043 |
| True SD of Γ, corrected | 0.036 | 0.037 |
| Mean p_a(1 − p_a) per branch | 0.047 | 0.059 |
| Mean K (codes remaining) | 7.21 | 6.83 |

The calibration column re-runs the locked estimator with its own resampling. `cc_calibration.json`, which the power
analysis used, has 0.065 and 0.053; the two differ only by resampling noise.

**What this changed.**
- The power analysis assumed a true SD of 0.053 and, at 8 samples, a noise SD of 0.046.
- With the bias corrected, Γ's true spread is about 0.037 on both sets of families.
- The fresh families were also noisier per sample: per-branch hit rates sit closer to one half (mean
  p_a(1 − p_a) 0.059 against 0.047), and the families have slightly fewer codes. That brought the noise SD to 0.062 rather than the 0.054 that the corrected
  calibration value projects.
- Rerunning the planned power simulation at these values gives **power 0.50** for the planned smallest effect
  (a predictor explaining 20% of Γ's true variance), not 0.864.
- So the confirmation is less sensitive than planned. Even so, all four H3 point estimates favor the rivals, and
  the H4 slope is negative.
- The noise-corrected MSPE above uses the locked noise estimate, as the locked code fixes.

## How informative are the nulls? (post hoc; `cc_informative_nulls.json`; not part of the locked analysis)

Script: `experiments/study_c/informative_nulls.py`.
- It reads the locked predictions and the confirmation records through the locked functions (`records`, `group`,
  `observed_gamma`).
- It reuses the preregistered score's 10,000 family resamples (seed 0) and reproduces its H3 intervals exactly.
- Noise uses the corrected (unbiased) estimator of the design check.
- Nothing here changes a preregistered result.

**1. The aid's effect varies, but less than planned.**
- True variance of Γ: 0.00136 [0.00016, 0.0028]. That is a true SD of 0.037 against a planned 0.053;
  P(true variance ≤ 0) = 0.011.
- Reliability of Γ (true over observed variance): 0.26. This is a variance-based estimate. It differs from the
  split-half figure in C-b (0.30 after Spearman–Brown), which uses half the samples per half.

**2. The locked H3 test could not have detected a correct primary predictor.** This is a design flaw that was
visible in `prediction_lock.json` at lock time. It was not disclosed before this analysis.

| | Value |
|---|---|
| SD of the locked support-map predictions Γ̂ | 0.0026 |
| True SD of Γ | 0.037 (14 times larger) |
| Largest expected MSPE gain over Γ = 0 if Γ̂ were exactly right, mean(Γ̂²) | 1.3 × 10⁻⁵ (1% of the true variance) |
| Smallest effect of interest (SESOI): gain of a predictor explaining 20% of the true variance | 2.7 × 10⁻⁴ (measured τ²); 5.7 × 10⁻⁴ (planned τ²) |
| Power of the locked test against Γ = 0 if the support map were exactly right (2,000 simulations) | **0.04** |
| Planned power | 0.864 |

- **Why the planned power doesn't apply.** The planned power is for a hypothetical predictor whose spread matches a
  20% share of the true variance. The locked support map predicts changes about 14 times too small.
- **Why the gain-based tests below mostly hold by construction.** Its MSPE gain over the rivals cannot reach the SESOI
  whether or not its direction is right. So the MSPE-based equivalence tests and Bayes factors below largely restate
  that, and are not independent evidence.
- **The scale-free correlation is the informative check.** A correlation does not depend on the spread of the
  predictions.

**3. H3, scale-free: the support map's direction is not right either.**
- A predictor explaining 20% of Γ's true variance would correlate with observed Γ at about +0.23, given Γ's
  reliability of 0.26.
- The locked support map correlates at **−0.10 [−0.24, 0.06]**. The upper bound excludes +0.23.

| Predictor | r with observed Γ [95% CI] | SD of predictions |
|---|---|---|
| support_map (primary) | −0.10 [−0.24, 0.06] | 0.0026 |
| fitted_M2s | −0.11 [−0.25, 0.04] | 0.0034 |
| fitted_M1a | −0.05 [−0.16, 0.07] | 0.0103 |
| difficulty_regression | +0.08 [−0.04, 0.19] | 0.0018 |
| reporting_only | −0.11 [−0.25, 0.06] | 0.0017 |
| ideal_noise | −0.01 [−0.11, 0.09] | 0.0003 |

No predictor's interval reaches +0.23.

**4. H3 on the planned scale (equivalence and Bayes factors).** Read these with point 2 in mind.

| Rival | MSPE gain of support map [90% CI] | Below the SESOI? | BF01, prior scale = SESOI (measured / planned τ²) |
|---|---|---|---|
| difficulty_regression | −5.9e-5 [−1.5e-4, 2.7e-5] | yes (p < 0.001) | 10 / 21 |
| reporting_only | −1.6e-5 [−3.8e-5, 6.1e-6] | yes (p < 0.001) | 42 / 88 |
| uniform_gain | −4.3e-5 [−1.2e-4, 2.4e-5] | yes (p < 0.001) | 13 / 26 |
| ideal_noise | −4.2e-5 [−1.2e-4, 2.6e-5] | yes (p < 0.001) | 12 / 25 |

- The upper 95% bound of every gain is below the SESOI.
- BF01 uses a half-normal prior on the gain with the bootstrap SE as a normal approximation. It is 5–176 across
  prior scales of 0.5–2 × SESOI.

**5. H4: no sign that choices follow the aid's effect, but the test is weak.**
- **Correlation.** The correlation between the choice shift and measured Γ is **−0.04 [−0.14, 0.07]** (360 pairs).
- **Low reliability.**
  - Measured Γ: 0.26.
  - Choice shift (split halves, Spearman–Brown): 0.33.
  - So even a perfect true relation would appear as r ≤ 0.29.
- **Disattenuated bound.** Corrected for both reliabilities, the upper 95% bound of the true correlation is 0.24.
  That rules out a strong link, not a modest one.
- **Bayes factor.** BF01 for a positive correlation (Fisher z, half-normal prior) is 3.2, 6.1 and 9.1 at prior
  scales 0.1, 0.2 and 0.3. That is moderate evidence for no positive relation.

**How to report it.**
- **H3** is a preregistered null whose primary test lacked sensitivity by design. The scale-free check is the
  substantive result: the fitted process model, and every rival, fails to anticipate even the direction of the
  aid's item-level effect on usefulness.
- **H4** is a null with moderate evidence against a strong link. Both measures are too unreliable to exclude a modest
  one.
- **A follow-up** needs predictors whose spread matches the effect (or a scale-free endpoint), easier states that
  lift usefulness off the floor, and an aid that changes capacity (for example a candidate-set tool).

## Disclosures and departures from the plan

1. **Conditions.** T2 (generic check) is an extra control, reported as secondary. The detailed plan's T4
   (thinking mode) was not run; it is not in the proposal. Prompt paraphrases (detailed plan, not the proposal)
   were not run.
2. **Families.** C-a ran on the old confirmation families, so C-c used fresh families (disclosed before the lock).
3. **Client timeout.** The language-model client first timed out at 600 s and retried silently, which redrew long
   replies until a short one finished.
   - The first cap-pilot run was discarded for this reason (disclosed before the lock).
   - Every Study C run used the corrected client, which waits up to 4 hours and never retries on its own.
   - The confirmation jobs logged no retry. The plan allows up to 3 retries for transport errors, while the
     client's loop allows 4; none occurred.
4. **How the confirmation was run.**
   - It ran in parts: T3 in 6 jobs and T0 in 4, each with a disjoint share of the calls, so no reply was
     requested twice.
   - One run per condition then read every reply from the cache and wrote the record file. It ran with no model
     server, so a missing reply would have stopped it.
5. **A stale docstring in a locked file.** The docstring of the locked `choice_shift.py` shows `--cap 15000` in
   its example. The confirmatory run used the locked 6,144 cap (`cap` in its output file).
6. **The exploratory choice-shift file in the lock commit came from a draft of `choice_shift.py`.** The data are
   unchanged (hashes in `cc_calibration.json`) and so is the slope. Regenerated with the locked script, its
   interval moves from [−3.04, 3.49] to [−3.08, 3.49] and its p from 0.478 to 0.481. The confirmatory output is
   reproduced exactly on every run.
7. **Design check.** The design check above is post hoc and labeled so. It changes no preregistered result.
8. **Hash check.** The plan says `score` checks the hashes in the lock; the locked code checks only the six code
   hashes. The other eight (families, pairs, calibration, plan, prompts, model registry) were checked separately
   before scoring, and all 14 matched.
