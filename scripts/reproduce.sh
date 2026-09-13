#!/usr/bin/env bash
set -euo pipefail

RELEASE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${RELEASE_ROOT}"

CONFIG=${1:-configs/paper/poisson_n1000.yaml}
RUN_DIR=${2:-artifacts/reproduction}
DEVICE=${DEVICE:-auto}
RUN_ID=${RUN_ID:-}
RESUME=${RESUME:-0}

mkdir -p "${RUN_DIR}/data" "${RUN_DIR}/checkpoints" "${RUN_DIR}/results"

if [[ "${RESUME}" != "1" || ! -f "${RUN_DIR}/data/train.pt" ]]; then
  grans-generate \
    --config "${CONFIG}" \
    --split train \
    --output "${RUN_DIR}/data/train.pt"
fi

if [[ "${RESUME}" != "1" || ! -f "${RUN_DIR}/data/test.pt" ]]; then
  grans-generate \
    --config "${CONFIG}" \
    --split test \
    --output "${RUN_DIR}/data/test.pt"
fi

TRAIN_ARGS=(
  --config "${CONFIG}"
  --data "${RUN_DIR}/data/train.pt"
  --output-dir "${RUN_DIR}/checkpoints"
  --device "${DEVICE}"
)
CHECKPOINT="${RUN_DIR}/checkpoints/final_model.pt"
if [[ -n "${RUN_ID}" ]]; then
  TRAIN_ARGS+=(--run-id "${RUN_ID}")
  CHECKPOINT="${RUN_DIR}/checkpoints/${RUN_ID}/final_model.pt"
fi
if [[ "${RESUME}" == "1" ]]; then
  TRAIN_ARGS+=(--resume)
fi

grans-train "${TRAIN_ARGS[@]}"

grans-evaluate \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --data "${RUN_DIR}/data/test.pt" \
  --output "${RUN_DIR}/results/evaluation.json" \
  --device "${DEVICE}"
