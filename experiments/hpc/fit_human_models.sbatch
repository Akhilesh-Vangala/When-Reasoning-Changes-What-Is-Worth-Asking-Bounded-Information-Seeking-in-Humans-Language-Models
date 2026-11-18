#!/bin/bash
# Fit the 3 human models to each Experiment 3 participant (one array task each).
#   sbatch experiments/hpc/fit_human_models.sbatch
# Then: python experiments/study_a/fit_human_models.py summarize
#SBATCH --job-name=mm-fit-human
#SBATCH --array=0-38
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm_%x_%a.log

cd "${CCM_ROOT:-$SLURM_SUBMIT_DIR}"
CCM_ENV=${CCM_ENV:-$PWD/env}  # Python environment on the cluster
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
"$CCM_ENV"/bin/python -u experiments/study_a/fit_human_models.py fit --participant $SLURM_ARRAY_TASK_ID "$@"
