#!/bin/bash
#SBATCH --job-name=submit-cellpose
#SBATCH --account=yuanlab
#SBATCH --partition=common
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/hpc/group/jjk79/logs/submit-%j.out
#SBATCH --error=/hpc/group/jjk79/logs/submit-%j.err

set -euo pipefail

# EDIT THIS ONE VALUE. It must also be the name of the existing conda env.
PROJECT="REPLACE_WITH_PROJECT"
NETID="jjk79"

if [[ "${PROJECT}" == "REPLACE_WITH_PROJECT" ]]; then
    echo "Edit PROJECT in submit_pipeline.sbatch before submission." >&2
    exit 2
fi

GROUP_ROOT="/hpc/group/${NETID}"
YUANLAB_ROOT="/hpc/group/yuanlab/${NETID}/${PROJECT}"
CODE_DIR="${GROUP_ROOT}/code"
CONFIG_DIR="${GROUP_ROOT}/config"
RESULTS_ROOT="${GROUP_ROOT}/results/${PROJECT}"
LOG_DIR="${YUANLAB_ROOT}/logs"
CACHE_ROOT="/hpc/group/yuanlab/${NETID}/caches/${PROJECT}"
EXPERIMENTS_TSV="${CONFIG_DIR}/${PROJECT}_experiments.tsv"
MANIFEST="${CONFIG_DIR}/${PROJECT}_cellpose_manifest.tsv"
MERGED_OUTPUT="${RESULTS_ROOT}/${PROJECT}_CellposeSAM_Combined.xlsx"

mkdir -p "${CONFIG_DIR}" "${RESULTS_ROOT}" "${LOG_DIR}" "${CACHE_ROOT}"

if [[ ! -f "${EXPERIMENTS_TSV}" ]]; then
    echo "Missing experiment configuration: ${EXPERIMENTS_TSV}" >&2
    exit 3
fi

source "/hpc/group/yuanlab/${NETID}/miniforge3/bin/activate" "${PROJECT}"

python "${CODE_DIR}/dcc_cluster/make_dcc_manifest.py" \
    --experiments "${EXPERIMENTS_TSV}" \
    --output "${MANIFEST}" \
    --results-root "${RESULTS_ROOT}"

TASK_COUNT=$(( $(wc -l < "${MANIFEST}") - 1 ))
if (( TASK_COUNT < 1 )); then
    echo "Manifest contains no image tasks." >&2
    exit 4
fi

# Limit simultaneous GPU tasks to eight by default. The old unlimited-array
# behavior is intentionally avoided to reduce shared-filesystem and cache load.
ARRAY_JOB_ID="$(sbatch --parsable \
    --array="0-$((TASK_COUNT - 1))%8" \
    --output="${LOG_DIR}/arr-%A_%a.out" \
    --error="${LOG_DIR}/arr-%A_%a.err" \
    --export="ALL,PROJECT=${PROJECT},NETID=${NETID},MANIFEST=${MANIFEST},CODE_DIR=${CODE_DIR},CACHE_ROOT=${CACHE_ROOT}" \
    "${CODE_DIR}/dcc_cluster/cellpose_array.sbatch")"

# afterany intentionally merges every task that reached _SUCCESS even if a
# different array element failed; failed elements can then be resubmitted.
MERGE_JOB_ID="$(sbatch --parsable \
    --dependency="afterany:${ARRAY_JOB_ID}" \
    --output="${LOG_DIR}/merge-%j.out" \
    --error="${LOG_DIR}/merge-%j.err" \
    --export="ALL,PROJECT=${PROJECT},NETID=${NETID},CODE_DIR=${CODE_DIR},RESULTS_ROOT=${RESULTS_ROOT},MERGED_OUTPUT=${MERGED_OUTPUT}" \
    "${CODE_DIR}/dcc_cluster/merge_results.sbatch")"

echo "Manifest tasks: ${TASK_COUNT}"
echo "Array job: ${ARRAY_JOB_ID}"
echo "Merge job: ${MERGE_JOB_ID}"
echo "Array logs: ${LOG_DIR}/arr-${ARRAY_JOB_ID}_TASK_ID.out/.err"
echo "Combined workbook: ${MERGED_OUTPUT}"
