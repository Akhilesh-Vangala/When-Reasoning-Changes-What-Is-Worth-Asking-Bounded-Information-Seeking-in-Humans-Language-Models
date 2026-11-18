#!/bin/bash
# Recovery for the process models M0-M3 on the B2 design (one array task per replicate).
#   sbatch experiments/hpc/recover_process_models.sbatch
# Then: python experiments/study_b/recover_process_models.py summarize
#SBATCH --job-name=mm-recover-process
#SBATCH --array=0-19
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm_%x_%a.log

cd "${CCM_ROOT:-$SLURM_SUBMIT_DIR}"
CCM_ENV=${CCM_ENV:-$PWD/env}  # Python environment on the cluster
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
"$CCM_ENV"/bin/python -u experiments/study_b/recover_process_models.py run --rep $SLURM_ARRAY_TASK_ID "$@"
