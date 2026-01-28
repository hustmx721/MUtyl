#!/usr/bin/env bash
set -euo pipefail

# Batch runner for clean-data baseline (leave-one-subject-out style).
# Runs baseline training for each dataset/model pair.

echo "Baseline LOOCV experiments (datasets: 001, 004, MI, SSVEP, ERP)"

datasets=("001" "004" "MI" "SSVEP" "ERP")
models=("DeepConvNet" "EEGNet" "ShallowConvNet")
# Update this list if subject IDs differ in your .mat metadata.
gpus=(2 3 4 5 6)

max_jobs=15
jobs=()
job_idx=0
failed=0

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

extra_args=()
if [[ -n "${SUBJECTS:-}" ]]; then
  extra_args+=("--subjects" "${SUBJECTS}")
fi

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    gpu_id=${gpus[$(( job_idx % ${#gpus[@]} ))]}
    job_idx=$((job_idx + 1))

    echo "Launch: dataset=${dataset}, model=${model}, gpu=${gpu_id}"
    python -u main_baseline.py \
      --dataset "${dataset}" \
      --model "${model}" \
      --gpuid "${gpu_id}" \
      --repeats 3 \
      --seed 2024 \
      "${extra_args[@]}" &

    pid=$!
    jobs+=("$pid")

    if (( ${#jobs[@]} >= max_jobs )); then
      wait_one "${jobs[0]}"
      jobs=("${jobs[@]:1}")
    fi
  done
done

for pid in "${jobs[@]}"; do
  wait_one "$pid"
done

if (( failed == 1 )); then
  echo "All experiments completed, BUT some jobs failed." >&2
  exit 1
fi

echo "All baseline experiments completed."
