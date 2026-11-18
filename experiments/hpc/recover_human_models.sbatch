#!/bin/bash
# Model recovery for the 3 human models (one array task per participant).
#   sbatch experiments/hpc/recover_human_models.sbatch
# Then: python experiments/study_a/recover_human_models.py summarize
#SBATCH --job-name=mm-recover-human
#SBATCH --array=0-38
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm_%x_%a.log

cd "${CCM_ROOT:-$SLURM_SUBMIT_DIR}"
CCM_ENV=${CCM_ENV:-$PWD/env}  # Python environment on the cluster
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
"$CCM_ENV"/bin/python -u experiments/study_a/recover_human_models.py run --participant $SLURM_ARRAY_TASK_ID "$@"
