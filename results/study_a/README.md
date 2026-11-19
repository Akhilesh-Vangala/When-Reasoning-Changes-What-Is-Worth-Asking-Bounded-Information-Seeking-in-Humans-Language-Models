# Study A: human reproduction and model comparison

Data: Cheyette et al., OSF `vjrqu` (E1, E2, E3).
Every number below is read from the JSON file named in its row. Rerun the scripts in `experiments/study_a/` to regenerate.

## Behavior

| Measure | Ours | Paper | File |
|---|---|---|---|
| E1 sample | 44 people, 393 games, 132 distinct starting positions | same | `e1_reproduction.json` |
| E1 guesses after the supplied history (solved games): Simple / Max-EIG / none | 4.74 / 6.32 / 7.15 | same | `e1_reproduction.json` |
| E1 ideal continuation: Simple / Max-EIG / none | 2.49 / 1.91 / 3.68 | (not reported) | `e1_reproduction.json` |
| E1 Max-EIG histories that leave exactly one code (ideal needs 1 guess) | people need 5.41 (46 games) | (not reported) | `e1_reproduction.json` |
| E1 Max-EIG minus Simple, unsolved = 16, clustered by person | 1.29 [0.29, 2.29] | | `e1_reproduction.json` |
| … controlling for supplied guesses and log2 codes remaining | **2.41 [1.34, 3.48]** | | `e1_reproduction.json` |
| E2 (descriptive only; 33 of 40 people released) | easy choices rise from 0.39 (trial 1) to 0.69 (trial 10); easy choices predict solving (logit slope 1.79 [0.95, 2.63]) and fewer guesses (−3.84 [−5.15, −2.52]) | same direction (authors' R script) | `e2_descriptive.json` |
| E3 sample after cleaning | 39 people, 390 games | 39 | `e3_reproduction.json` |
| E3 solve rate | 0.864 | 0.86 | `e3_reproduction.json` |
| E3 guesses to solve | **7.72** (SD 3.16) | **7.5** | `e3_reproduction.json` |
| Ideal (greedy Max-EIG), all 256 secrets | 3.56 (max 4) | 3.6 | `e3_reproduction.json` |
| Random-query benchmark | 4.81 (SD 1.26, max 12) | 4.8 (SD 1.0, max 7) | `e3_reproduction.json` |
| Guesses still possible given earlier feedback | 0.31 | | `e3_reproduction.json` |
| First guesses using a single color | 0.45 | | `e3_reproduction.json` |

## Model comparison (E3, solved games, per participant)

| Measure | Ours | Paper | File |
|---|---|---|---|
| Mean ΔAIC, bounded vs. unbounded | **104.6** | 82.5 | `e3_model_comparison.json` |
| Mean ΔAIC, bounded vs. leaky | **83.5** | 72.3 | `e3_model_comparison.json` |
| People best fit by bounded / leaky / unbounded | 34 / 5 / 0 | 37 of 39 favor bounded | `e3_model_comparison.json` |
| Median bounded parameters | ε 0.30, α 3.34, λ 9.5, τ 0.089 | | `e3_model_comparison.json` |
| **Held-out participants** (each person predicted from the median parameters of the other 38) | bounded best for 28 of 39 (sign test p = 0.009); total log likelihood bounded −13,082 vs. unbounded −13,407 and leaky −13,429; bootstrap CI of the total difference includes 0 | | `e3_heldout.json` |
| Game-index drift (static bounded fit, per-decision log likelihood vs. game 1–10) | later games fit better: +0.075 per game (SE 0.023, p = 0.0009) | | `e3_heldout.json` |
| Sensitivity: refit including unsolved games | mean ΔAIC 98.4 vs. unbounded, 90.2 vs. leaky; bounded / leaky / unbounded best for 37 / 2 / 0. The ordering holds, and the count matches the paper's 37 of 39 | 82.5 / 72.3; 37 of 39 | `e3_model_comparison_with_unsolved.json` |
| Sensitivity: bounded fitted to games 1–5 and 6–10 separately (8 parameters) vs. one static fit (4) | separate fits are better for 21 of 39 people (sign test p = 0.75); median AIC gain 0.7 (mean 9.7, driven by a few people). Late minus early: α +0.85 (Wilcoxon p = 0.04), ε −0.06 (p = 0.06), λ and τ flat; no shift survives Holm correction across the 4 parameters | | `e3_game_index_halves.json` |
| Cache seed: bounded refitted with an independently rebuilt cache (new search seeds) | model comparison unchanged: ΔAIC 108.4 / 87.2 (vs. 104.6 / 83.5), bounded best for 35 of 39 (vs. 34). Per-person bounded AIC correlates r = 0.99 across caches (median change 11.4). Individual α is less stable (Spearman 0.55; group median 3.14 vs. 3.34), so α is interpreted at the group level | | `e3_cache_seed.json` |
| Recovery: each model simulates every participant (their fitted parameters, same number of solved games), then all three are refitted | generating model wins by AIC: bounded 35/39 (0.90), unbounded 29/39 (0.74), leaky 28/39 (0.72). Bounded wins falsely for 2/39 unbounded and 8/39 leaky simulations, far below the 34/39 observed in people. α recovered with Spearman 0.91 | | `e3_recovery_summary.json` |

## Discrepancies to report

1. **E3 guesses to solve:** 7.72 vs. the paper's 7.5, from the same cleaned 39 participants.
2. **ΔAIC magnitudes** are larger than the paper's (104.6 / 83.5 vs. 82.5 / 72.3), and 34 rather than 37 of 39 people favor bounded on solved games. Including unsolved games gives 37 of 39, as in the paper. The ordering bounded > leaky > unbounded holds in both. The gap is not randomness in our cache search: a second, independent search gives 108.4 / 87.2. It is systematic, most likely from search choices the paper does not specify (chain settings, proposal correction, size cap; recorded in the cache manifest), since the authors' cache is unreleased.
3. **Held-out generalization is weaker than the in-sample result.** Bounded predicts most held-out people best, but the total-likelihood advantage is not reliable across people: a few are predicted much worse from the group's typical parameters.
4. **People change over the session** (game-index drift): the static bounded model fits later games better. Splitting the fit into early and late halves does not improve it for most people, so the static model is an adequate summary, with this caveat stated.
5. **Random-query benchmark:** mean matches; the spread does not. No simple rule reproduces SD 1.0 and max 7 (without repeats: 4.74 / 1.11 / 10; possible codes only: 3.78 / 0.78 / 6).
6. **E1 analysis:** our pipeline is Python. The authors' E1 R script calls `mm_plot` (line 63) before defining it (line 109); we did not use it.

## Scripts

| Script | Produces |
|---|---|
| `experiments/study_a/e1_reproduction.py` | `data/stimuli/e1_stimuli.json`, `e1_reproduction.json` |
| `experiments/study_a/e2_descriptive.py` | `e2_descriptive.json` |
| `experiments/study_a/e3_reproduction.py` | `e3_reproduction.json` |
| `experiments/study_a/build_cache.py` | `data/model_cache/bounded_cache_v1.npz` (+ symmetric variant for Study B; `--rep 2` for the independent replicate) |
| `experiments/study_a/fit_human_models.py` | `e3_fits/`, `e3_model_comparison.json`, `e3_heldout.json`, sensitivity fits |
| `experiments/study_a/recover_human_models.py` | recovery check (cluster array job) |
