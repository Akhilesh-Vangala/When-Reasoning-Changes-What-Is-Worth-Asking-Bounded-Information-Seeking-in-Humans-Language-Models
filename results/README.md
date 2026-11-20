# Results

Outputs of the scripts in `experiments/`, one folder per study. The findings, with every number traced to its
file, are in `study_a/README.md` and `study_b/README.md`.

File and folder suffixes: none = the paper's greedy search; `_converged` = converged fits (primary);
`_with_unsolved`, `_early` / `_late`, `_cacherep2` and `_wide` = sensitivity fits.

| Folder | Contents |
|---|---|
| `study_a/` | Experiment 1–3 reproduction summaries; per-participant fits of the three human models (`e3_fits*/`), recovery (`e3_recovery*/`) and their summaries |
| `study_b/screening/` | screening of the candidate models (records and summaries) |
| `study_b/b1/` | B1 game records, one file per model, task and condition |
| `study_b/b1b/` | B1b fits: pooled, 10 blocks and 200 resamples per model; summaries in `b1b_<model>_T0*.json` |
| `study_b/battery/` | B2 probe-battery records (Qwen3-32B, T0) |
| `study_b/process_fits/` | B3 fits, family-bootstrap resamples, intervals, error profile, product-rule check and M1a confirmation |
| `study_b/process_recovery*/` | B3 recovery replicates (planned design; at the fitted values) |
| `study_b/usefulness_choice/`, `study_b/native_choice/` | native question choice: records and analysis; usefulness and choice records under T0 and T3 on the calibration families (Study C calibration) |
| `study_b/battery/*_T1–T3_*`, `study_b/process_fits/*T1–T3*` | Study C aid conditions: probe-battery records, process-model fits and family-bootstrap resamples |
| `study_c/` | C-a aid effects (`ca_aid_effects.json`), reply-cap pilot, C-c calibration and power analysis, exploratory choice-shift check on the calibration pairs |
