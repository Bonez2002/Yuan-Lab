#!/bin/bash
#SBATCH --job-name=cellpose3d
#SBATCH --account=yuanlab
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/hpc/group/jjk79/logs/arr-%A_%a.out
#SBATCH --error=/hpc/group/jjk79/logs/arr-%A_%a.err

set -euo pipefail

: "${PROJECT:?PROJECT must be exported by the submission job}"
: "${MANIFEST:?MANIFEST must be exported by the submission job}"
: "${CODE_DIR:?CODE_DIR must be exported by the submission job}"
: "${CACHE_ROOT:?CACHE_ROOT must be exported by the submission job}"
: "${SLURM_ARRAY_TASK_ID:?This script must run as a SLURM array}"
: "${TMPDIR:?DCC must provide node-local TMPDIR}"

NETID="${NETID:-jjk79}"
CONDA_ROOT="/hpc/group/yuanlab/${NETID}/miniforge3"

# Prevent numerical libraries from multiplying the allocated CPU count.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Headless execution and permanent, group-backed caches.
export MPLBACKEND=Agg
export QT_QPA_PLATFORM=offscreen
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TORCH_HOME="${CACHE_ROOT}/torch"
export HF_HOME="${CACHE_ROOT}/huggingface"
export CELLPOSE_LOCAL_MODELS_PATH="${CACHE_ROOT}/cellpose/models"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
mkdir -p \
    "${XDG_CACHE_HOME}" \
    "${TORCH_HOME}" \
    "${HF_HOME}" \
    "${CELLPOSE_LOCAL_MODELS_PATH}" \
    "${PIP_CACHE_DIR}"

source "${CONDA_ROOT}/bin/activate" "${PROJECT}"

# Header is line 1, so zero-based array task N reads manifest line N+2.
ROW="$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")"
if [[ -z "${ROW}" ]]; then
    echo "No manifest row for array task ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
fi

IFS=$'\t' read -r EXPERIMENT PAIR_NAME GROUP1_PATH GROUP2_PATH OUTPUT_DIR <<< "${ROW}"

if [[ -f "${OUTPUT_DIR}/_SUCCESS" ]]; then
    echo "Already complete; skipping ${EXPERIMENT}/${PAIR_NAME}"
    exit 0
fi

for INPUT_PATH in "${GROUP1_PATH}" "${GROUP2_PATH}"; do
    if [[ ! -f "${INPUT_PATH}" ]]; then
        echo "Missing input: ${INPUT_PATH}" >&2
        exit 3
    fi
done

# Stage large TIFF reads to private node-local storage.
STAGED_ROOT="${TMPDIR}/cellpose_input"
LOCAL_OUTPUT="${TMPDIR}/cellpose_output"
mkdir -p "${STAGED_ROOT}/group_1" "${STAGED_ROOT}/group_2" "${LOCAL_OUTPUT}"
cp -- "${GROUP1_PATH}" "${STAGED_ROOT}/group_1/$(basename "${GROUP1_PATH}")"
cp -- "${GROUP2_PATH}" "${STAGED_ROOT}/group_2/$(basename "${GROUP2_PATH}")"

export CELLPOSE_SAM_BATCH_EXPERIMENT_ROOT="${STAGED_ROOT}"
export CELLPOSE_SAM_SINGLE_PAIR="${PAIR_NAME}"
export CELLPOSE_SAM_OUTPUT_FOLDER="${LOCAL_OUTPUT}"
export CELLPOSE_SAM_SKIP_EXISTING=1

python "${CODE_DIR}/CellposeSAM_3D_analysis_diffuse_floor.py"

EXPECTED_XLSX="${LOCAL_OUTPUT}/3D_cell_intensity_results_CellposeSAM.xlsx"
if [[ ! -s "${EXPECTED_XLSX}" ]]; then
    echo "Expected workbook was not produced: ${EXPECTED_XLSX}" >&2
    exit 4
fi

# Only publish outputs after the analysis succeeds. A missing _SUCCESS marker
# identifies a partial copy and makes the task safely resumable.
mkdir -p "${OUTPUT_DIR}"
cp -a "${LOCAL_OUTPUT}/." "${OUTPUT_DIR}/"
touch "${OUTPUT_DIR}/_SUCCESS"

echo "Completed ${EXPERIMENT}/${PAIR_NAME}"
echo "Output: ${OUTPUT_DIR}"
