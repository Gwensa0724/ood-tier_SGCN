#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DEVICE="${DEVICE:-0}"
DATA_DIR="${DATA_DIR:-/mnt/energy_ood/data/}"
OUT_ROOT="${OUT_ROOT:-/mnt/energy_ood/outputs/mixed_test_amazon_full_matrix}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/gnnsafe/bin/python}"
SEEDS="${SEEDS:-123 124 125 126 127}"
CONTAM_RATIOS="${CONTAM_RATIOS:-0.50 1.00}"
ATTACH_BUDGETS="${ATTACH_BUDGETS:-8 12 16 24 32}"
BACKBONES="${BACKBONES:-gcn sage sgcn}"
CONTAM_STRATEGIES="${CONTAM_STRATEGIES:-natural random_attach targeted_attach}"
ENERGY_WEIGHTINGS="${ENERGY_WEIGHTINGS:-none rank sigmoid hard}"
ENERGY_AGGREGATIONS="${ENERGY_AGGREGATIONS:-mean median trimmed_mean bottomk_mean}"

RUN_ROOT="${OUT_ROOT}"
LOG_ROOT="${RUN_ROOT}/logs"
mkdir -p "${LOG_ROOT}"
cd "${SCRIPT_DIR}"

join_csv() {
  printf '%s' "$*" | sed 's/ /,/g'
}

run_one() {
  energy_weighting="$1"
  energy_aggregation="$2"
  run_dir="${RUN_ROOT}/agg_${energy_aggregation}_weight_${energy_weighting}"
  result_dir="${run_dir}/results"
  log_dir="${run_dir}/logs"
  mkdir -p "${result_dir}" "${log_dir}"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/two_stage_mixed_test.py" \
    --dataset amazon-photo \
    --ood_type label \
    --backbones "$(join_csv ${BACKBONES})" \
    --use_bn \
    --use_prop \
    --device "${DEVICE}" \
    --data_dir "${DATA_DIR}" \
    --results_dir "${result_dir}" \
    --contam_strategies ${CONTAM_STRATEGIES} \
    --contam_ratios ${CONTAM_RATIOS} \
    --attach_budgets "$(join_csv ${ATTACH_BUDGETS})" \
    --seeds ${SEEDS} \
    --sgcn_energy_weighting "${energy_weighting}" \
    --sgcn_energy_aggregation "${energy_aggregation}" \
    > "${log_dir}/run.log" 2>&1
}

(
  for energy_aggregation in ${ENERGY_AGGREGATIONS}; do
    for energy_weighting in ${ENERGY_WEIGHTINGS}; do
      printf 'Running amazon matrix combo: aggregation=%s weighting=%s\n' "${energy_aggregation}" "${energy_weighting}"
      run_one "${energy_weighting}" "${energy_aggregation}"
    done
  done
) > "${RUN_ROOT}/nohup.log" 2>&1 &

PID=$!
printf 'Started amazon matrix run with PID=%s\n' "${PID}"
printf 'Nohup log: %s\n' "${RUN_ROOT}/nohup.log"
