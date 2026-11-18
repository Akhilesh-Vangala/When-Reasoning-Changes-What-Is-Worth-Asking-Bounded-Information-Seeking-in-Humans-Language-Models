# Cluster job scripts (SLURM)

Open-weight models are served with vLLM on a GPU cluster; CPU-heavy fits run as array jobs.
Submit every script from the repository root.

Before submitting, set the allocation (and, if your layout differs from the defaults, the paths):

| Variable | Meaning | Default |
|---|---|---|
| `SBATCH_ACCOUNT` | SLURM allocation to charge (read by `sbatch` itself) | none: required |
| `CCM_ROOT` | repository path on the cluster | the submit directory |
| `CCM_ENV` | Python environment with the requirements and vLLM | `$CCM_ROOT/env` |
| `HF_HOME` | Hugging Face cache holding the model weights | `hf_cache/` next to the repository |

Put the exports in `experiments/hpc/site.env` (git-ignored; copy `site.env.example`) and
`source experiments/hpc/site.env` before `sbatch`. `chain.sh` sources it automatically.

| Script | Use |
|---|---|
| `download_models.sbatch` | download model weights (CPU only) |
| `serve_and_run.sbatch` | serve one model from `configs/model_registry.json` and run an experiment script against it |
| `chain.sh` | submit N short GPU jobs in sequence; replies are cached, so each job resumes where the last stopped |
| `build_cache.sbatch` | approximation search for the bounded model's representation cache |
| `fit_human_models.sbatch`, `recover_human_models.sbatch` | Study A fits and recovery, one array task per participant |
| `recover_process_models.sbatch` | process-model recovery, one array task per replicate |
