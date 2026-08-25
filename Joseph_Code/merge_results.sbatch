#!/bin/bash
#SBATCH --job-name=merge-cellpose
#SBATCH --account=yuanlab
#SBATCH --partition=common
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/hpc/group/jjk79/logs/merge-%j.out
#SBATCH --error=/hpc/group/jjk79/logs/merge-%j.err

set -euo pipefail

: "${PROJECT:?PROJECT must be exported by the submission job}"
: "${CODE_DIR:?CODE_DIR must be exported by the submission job}"
: "${RESULTS_ROOT:?RESULTS_ROOT must be exported by the submission job}"
: "${MERGED_OUTPUT:?MERGED_OUTPUT must be exported by the submission job}"

NETID="${NETID:-jjk79}"
CONDA_ROOT="/hpc/group/yuanlab/${NETID}/miniforge3"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLBACKEND=Agg

source "${CONDA_ROOT}/bin/activate" "${PROJECT}"

python "${CODE_DIR}/dcc_cluster/merge_cellpose_results.py" \
    --results-root "${RESULTS_ROOT}" \
    --output "${MERGED_OUTPUT}"
