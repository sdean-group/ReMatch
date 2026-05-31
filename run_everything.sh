#!/usr/bin/env bash
set -euo pipefail

TOPIC_PREFIX="rematch_yujin_a8f3k29q"
NTFY_URL="https://ntfy.sh"

mkdir -p logs

notify() {
    local topic="$1"
    local title="$2"
    local message="$3"
    local priority="${4:-default}"

    curl -s \
        -H "Title: ${title}" \
        -H "Priority: ${priority}" \
        -d "${message}" \
        "${NTFY_URL}/${topic}" >/dev/null || true
}

run_job() {
    local name="$1"
    local script="$2"
    local topic="${TOPIC_PREFIX}_${name}"
    local log_file="logs/${name}.log"

    echo "Run ${name}"
    notify "${topic}" "Started: ${name}" "${name} started on $(hostname)" "default"
    notify "${TOPIC_PREFIX}_all" "Started: ${name}" "${name} started on $(hostname)" "default"

    set +e
    GPUS=0,1,2,3,4,5,6,7 NPROC=8 bash "${script}" 2>&1 | tee "${log_file}"
    status=${PIPESTATUS[0]}
    set -e

    if [ "${status}" -eq 0 ]; then
        notify "${topic}" "Finished: ${name}" "${name} finished successfully on $(hostname). Log: ${log_file}" "default"
        notify "${TOPIC_PREFIX}_all" "Finished: ${name}" "${name} finished successfully on $(hostname). Log: ${log_file}" "default"
    else
        notify "${topic}" "Failed: ${name}" "${name} failed on $(hostname) with exit code ${status}. Log: ${log_file}" "urgent"
        notify "${TOPIC_PREFIX}_all" "Failed: ${name}" "${name} failed on $(hostname) with exit code ${status}. Log: ${log_file}" "urgent"
        exit "${status}"
    fi
}

notify "${TOPIC_PREFIX}_all" "Started: full pipeline" "Full pipeline started on $(hostname)" "default"

# run_job "run_rematchs" "scripts/run_rematchs.sh"
run_job "run_rematchu" "scripts/run_rematchu.sh"
run_job "run_corrdiff" "scripts/run_corrdiff.sh"
run_job "run_swinir" "scripts/run_swinir.sh"

notify "${TOPIC_PREFIX}_all" "Finished: full pipeline" "Full pipeline finished successfully on $(hostname)" "default"