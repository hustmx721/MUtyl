#!/usr/bin/env bash
set -euo pipefail

# 批量运行 DiCE checkpoint 的黑盒 MIA 评估。
# 默认会遍历 datasets/models，并将结果写到 csv/<dataset>/DiCE_MIA_<model>.csv

datasets=("001" "004" "MI" "SSVEP" "ERP")
models=("EEGNet" "Conformer")
gpus=(5 6)

# 并发上限（建议 <= 可用GPU数 * 每卡可承载任务数）
max_jobs=10
jobs=()
failed=0
job_idx=0

mia_methods="correctness,confidence,entropy,modified_entropy"

cleanup() {
  if ((${#jobs[@]} > 0)); then
    echo "Cleaning up ${#jobs[@]} running jobs..."
    for pid in "${jobs[@]}"; do
      kill "$pid" 2>/dev/null || true
    done
  fi
}
trap cleanup EXIT INT TERM

wait_one() {
  local pid="$1"
  if ! wait "$pid"; then
    echo "WARNING: job failed (pid=$pid)" >&2
    failed=1
  fi
}

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    gpu_id=${gpus[$((job_idx % ${#gpus[@]}))]}
    job_idx=$((job_idx + 1))

    echo "Launch MIA: dataset=${dataset}, model=${model}, gpu=${gpu_id}"
    python -u main_MIA.py \
      --dataset "${dataset}" \
      --model "${model}" \
      --gpuid "${gpu_id}" \
      --mia_methods "${mia_methods}" &

    pid=$!
    jobs+=("$pid")

    if ((${#jobs[@]} >= max_jobs)); then
      wait_one "${jobs[0]}"
      jobs=("${jobs[@]:1}")
    fi
  done
done

for pid in "${jobs[@]}"; do
  wait_one "$pid"
done

if ((failed == 1)); then
  echo "All MIA experiments completed, BUT some jobs failed." >&2
  exit 1
fi

echo "All MIA batch experiments completed successfully."
