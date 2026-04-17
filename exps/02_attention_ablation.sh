#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

COMMON_ARGS=(
  --dl
  --headless
  --speed 0
  --scenario dubai_marina
  --num-vehicles 25
  --rounds 200
  --target_acc 1.01
  --dl-dataset CIFAR10
  --dl-model CNN
  --verbose result
)

declare -a EXPERIMENTS=()

run_cli_experiment "DANTE attention model" "${COMMON_ARGS[@]}" --dl-algorithm DANTE
EXPERIMENTS+=("$LAST_EXPERIMENT_DIR")

run_cli_experiment "IPPO ablation" "${COMMON_ARGS[@]}" --dl-algorithm IPPO
EXPERIMENTS+=("$LAST_EXPERIMENT_DIR")

run_comparison "attention ablation" "${EXPERIMENTS[@]}"

copy_figure "$LAST_COMPARISON_DIR" "accuracy_vs_rounds_comparison" "attention_accuracy_vs_rounds_comparison"
copy_figure "$LAST_COMPARISON_DIR" "accuracy_vs_time_comparison" "attention_accuracy_vs_time_comparison"
copy_figure "$LAST_COMPARISON_DIR" "loss_vs_time_comparison" "attention_loss_vs_time_comparison"
copy_figure "$LAST_COMPARISON_DIR" "ppo_vs_rounds_comparison" "attention_ppo_vs_rounds_comparison"

echo "Attention ablation figures are ready in paper/assets/."
