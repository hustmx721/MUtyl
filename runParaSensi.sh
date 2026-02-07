#!/usr/bin/env bash
set -euo pipefail

# Example batch runner for parameter sensitivity experiments.
# Adjust dataset/model/gpuid as needed before running.

DATASET="${DATASET:-MI}"
MODEL="${MODEL:-EEGNet}"
GPUID="${GPUID:-0}"

python para_sensi.py \
  --dataset "${DATASET}" \
  --model "${MODEL}" \
  --gpuid "${GPUID}" \
  --sensi_params "dice_lambda_cf,dice_lambda_m,dice_lambda_sub,dice_temperature"
