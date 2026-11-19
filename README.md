# When Reasoning Changes What Is Worth Asking

**Bounded Information Seeking in Humans and Language Models**

## Overview

People playing Mastermind trade informativity for interpretability: they favor queries whose feedback they can interpret, even when other queries are more informative (Cheyette et al., 2025). This project extends that work to ask whether language models show the same trade-off, and whether independent measures of how they process evidence can predict how a reasoning aid changes which questions are worth asking.

The project has three parts:

1. **Study A, human modeling:** reproduce the published behavioral findings and model comparison.
2. **Study B, language models:** run language models on the same game positions people saw and on full games, fit the same human models to their play, and compare process models of how they use feedback.
3. **Study C, capacity manipulation:** give a model a structured reasoning aid and test whether the fitted models predict how the aid changes the usefulness of each question.

## Repository layout

```
mastermind/          core package: game engine, ideal observer, human models, process models,
                     prompts, language-model client and symmetry utilities
experiments/
  study_a/           human reproduction, representation cache, model fits, recovery
  study_b/           LLM game play (B1), human models fitted to LLM play (B1b),
                     probe battery (B2), process-model comparison (B3)
  hpc/               SLURM job scripts (see experiments/hpc/README.md)
configs/             model registry (versions, decoding settings) and frozen prompt templates
data/
  stimuli/           the Experiment 1 starting positions, derived from the human data
  splits/            state families and fixed development / calibration / confirmation splits
  model_cache/       rebuilt representation cache for the bounded model
  human/             where to put the human data (not redistributed; see its README)
results/             outputs of every script, one folder per study
tests/               unit tests (pytest)
```

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # API keys, only needed for hosted models; .env is git-ignored
pytest                        # unit tests
```

Download the human data as described in `data/human/README.md`. Open-weight models are served with vLLM on a GPU cluster (`experiments/hpc/`); everything else runs on a laptop.

## Reproducing Study A

```
python experiments/study_a/e1_reproduction.py
python experiments/study_a/e2_descriptive.py
python experiments/study_a/e3_reproduction.py
python experiments/study_a/fit_human_models.py fit --participant 0      # repeat for 0-38, or use the array job
python experiments/study_a/fit_human_models.py summarize
```

Results and the comparison with the published values are in `results/study_a/README.md`.

## Status

This repository contains Studies A and B. Study A is complete; Study B is in progress.

## References

- Cheyette, S. J., Callaway, F., Bramley, N. R., Nelson, J. D., & Tenenbaum, J. B. (2025). Representational constraints shape human information-seeking. *PsyArXiv*. https://doi.org/10.31234/osf.io/w4b3y
- Rothe, A., Lake, B. M., & Gureckis, T. M. (2018). Do people ask good questions? *Computational Brain & Behavior, 1*, 69–89.
