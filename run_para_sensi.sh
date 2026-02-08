#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <dataset> <model> [extra args...]"
  echo "Example: $0 MI EEGNet --is_task True"
  exit 1
fi

DATASET="$1"
MODEL="$2"
shift 2

GPU_IDS="${GPU_IDS:-0 1 2 3}"
read -r -a GPU_ARRAY <<< "${GPU_IDS}"

python para_sensi.py --dataset "${DATASET}" --model "${MODEL}" --gpuid "${GPU_ARRAY[0]:-0}" \
  --sensi_params dice_lambda_cf "$@" &
python para_sensi.py --dataset "${DATASET}" --model "${MODEL}" --gpuid "${GPU_ARRAY[1]:-0}" \
  --sensi_params dice_lambda_m "$@" &
python para_sensi.py --dataset "${DATASET}" --model "${MODEL}" --gpuid "${GPU_ARRAY[2]:-0}" \
  --sensi_params dice_lambda_sub "$@" &
python para_sensi.py --dataset "${DATASET}" --model "${MODEL}" --gpuid "${GPU_ARRAY[3]:-0}" \
  --sensi_params dice_temperature "$@" &

wait
echo "All sensitivity sweeps completed."
