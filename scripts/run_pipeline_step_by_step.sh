#!/usr/bin/env bash
set -euo pipefail

# Demonstrate the three independently runnable pipeline stages:
#   1. generate train/test datasets
#   2. train and checkpoint a model
#   3. evaluate the checkpoint
#
# Usage:
#   scripts/run_pipeline_step_by_step.sh configs/smoke.yaml artifacts/stepwise cpu
#
# Set RUN_ID to place checkpoints below RUN_DIR/checkpoints/RUN_ID. Set
# RESUME=1 to continue an existing run and reuse already generated datasets.

RELEASE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${RELEASE_ROOT}"

CONFIG=${1:-configs/smoke.yaml}
RUN_DIR=${2:-artifacts/stepwise}
DEVICE=${3:-auto}
RUN_ID=${RUN_ID:-}
RESUME=${RESUME:-0}

if [[ "${RESUME}" != "0" && "${RESUME}" != "1" ]]; then
  echo "RESUME must be 0 or 1" >&2
  exit 2
fi

for command_name in grans-generate grans-train grans-evaluate; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "required command not found: ${command_name}" >&2
    echo "install the package first with: python -m pip install -e '.[dev]'" >&2
    exit 1
  fi
done

DATA_DIR="${RUN_DIR}/data"
CHECKPOINT_ROOT="${RUN_DIR}/checkpoints"
RESULT_DIR="${RUN_DIR}/results"
mkdir -p "${DATA_DIR}" "${CHECKPOINT_ROOT}" "${RESULT_DIR}"

echo "[1/3] generating the training dataset"
if [[ "${RESUME}" == "1" && -f "${DATA_DIR}/train.pt" ]]; then
  echo "reusing ${DATA_DIR}/train.pt"
else
  grans-generate \
    --config "${CONFIG}" \
    --split train \
    --output "${DATA_DIR}/train.pt"
fi

echo "[1/3] generating the test dataset"
if [[ "${RESUME}" == "1" && -f "${DATA_DIR}/test.pt" ]]; then
  echo "reusing ${DATA_DIR}/test.pt"
else
  grans-generate \
    --config "${CONFIG}" \
    --split test \
    --output "${DATA_DIR}/test.pt"
fi

TRAIN_ARGS=(
  --config "${CONFIG}"
  --data "${DATA_DIR}/train.pt"
  --output-dir "${CHECKPOINT_ROOT}"
  --device "${DEVICE}"
)
CHECKPOINT="${CHECKPOINT_ROOT}/final_model.pt"
if [[ -n "${RUN_ID}" ]]; then
  TRAIN_ARGS+=(--run-id "${RUN_ID}")
  CHECKPOINT="${CHECKPOINT_ROOT}/${RUN_ID}/final_model.pt"
fi
if [[ "${RESUME}" == "1" ]]; then
  TRAIN_ARGS+=(--resume)
fi

echo "[2/3] training the model"
grans-train "${TRAIN_ARGS[@]}"

echo "[3/3] evaluating the checkpoint"
grans-evaluate \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --data "${DATA_DIR}/test.pt" \
  --output "${RESULT_DIR}/evaluation.json" \
  --device "${DEVICE}"

echo "pipeline complete"
echo "full report: ${RESULT_DIR}/evaluation.json"
echo "summary:     ${RESULT_DIR}/evaluation_summary.json"
