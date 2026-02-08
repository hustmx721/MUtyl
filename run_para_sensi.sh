#!/usr/bin/env bash
set -euo pipefail

python para_sensi.py --is_task True --dataset "004" --model "EEGNet" --gpuid "3" \
  --sensi_params dice_lambda_cf "$@" &
python para_sensi.py --is_task True --dataset "004" --model "EEGNet" --gpuid "4" \
  --sensi_params dice_lambda_m "$@" &
python para_sensi.py --is_task True --dataset "004" --model "EEGNet" --gpuid "5" \
  --sensi_params dice_lambda_sub "$@" &
python para_sensi.py --is_task True --dataset "004" --model "EEGNet" --gpuid "6" \
  --sensi_params dice_temperature "$@" &

