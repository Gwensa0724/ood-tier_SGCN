#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DEVICE="${DEVICE:-0}"
DATA_DIR="${DATA_DIR:-/mnt/energy_ood/data/}"
OUT_ROOT="${OUT_ROOT:-/mnt/energy_ood/outputs/mixed_test_amazon_main_budget16}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS="${SEEDS:-123 124 125}"

RUN_ROOT="${OUT_ROOT}"
LOG_DIR="${RUN_ROOT}/logs"
RESULT_DIR="${RUN_ROOT}/results"
NOHUP_LOG="${RUN_ROOT}/nohup.log"

mkdir -p "${LOG_DIR}" "${RESULT_DIR}"
cd "${SCRIPT_DIR}"

set -- ${SEEDS}
nohup "${PYTHON_BIN}" "${SCRIPT_DIR}/two_stage_mixed_test.py" \
  --dataset amazon-photo \
  --ood_type label \
  --backbone gcn \
  --use_bn \
  --use_prop \
  --device "${DEVICE}" \
  --data_dir "${DATA_DIR}" \
  --results_dir "${RESULT_DIR}" \
  --contam_strategies random_attach \
  --contam_ratios 1.00 \
  --attach_budget 16 \
  --seeds "$@" \
  > "${NOHUP_LOG}" 2>&1 &

PID=$!
printf 'Started mixed-test amazon run with PID=%s\n' "${PID}"
printf 'Nohup log: %s\n' "${NOHUP_LOG}"
