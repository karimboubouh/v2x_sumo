#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

COMMON_ARGS=(
  --dl
  --headless
  --scenario khalifa_university
  --num-vehicles 25
  --rounds 200
  --stop-on eval_acc
  --target_acc 0.80
  --dl-dataset CIFAR10
  --dl-model CNN
  --verbose result
  --save-logs
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
copy_figure "$LAST_COMPARISON_DIR" "communication_energy_comparison" "attention_communication_energy_comparison"

echo "Attention ablation figures are ready in paper/assets/."
