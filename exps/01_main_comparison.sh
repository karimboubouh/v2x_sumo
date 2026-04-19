#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

COMMON_ARGS=(
  --dl
  --headless
  --scenario khalifa_university
  --num-vehicles 50
  --rounds 150
  --target_acc 1.01
  --dl-dataset CIFAR10
  --dl-model CNN
  --verbose debug
)

declare -a EXPERIMENTS=()

for algo in DANTE DPFL; do
#for algo in DANTE DPFL IPPO pFedGraph FedAvg D-PSGD; do
  run_cli_experiment "$algo main comparison" "${COMMON_ARGS[@]}" --dl-algorithm "$algo"
  EXPERIMENTS+=("$LAST_EXPERIMENT_DIR")
done

run_comparison "main comparison" "${EXPERIMENTS[@]}"

copy_figure "$LAST_COMPARISON_DIR" "accuracy_vs_time_comparison" "main_accuracy_vs_time_comparison"
copy_figure "$LAST_COMPARISON_DIR" "accuracy_vs_rounds_comparison" "main_accuracy_vs_rounds_comparison"
copy_figure "$LAST_COMPARISON_DIR" "loss_vs_time_comparison" "main_loss_vs_time_comparison"
copy_figure "$LAST_COMPARISON_DIR" "energy_comparison" "main_energy_comparison"
copy_figure "$LAST_COMPARISON_DIR" "ppo_vs_rounds_comparison" "main_reward_vs_rounds_comparison"
copy_figure "$LAST_COMPARISON_DIR" "ppo_vs_time_comparison" "main_reward_vs_time_comparison"

echo "Main comparison figures are ready in paper/assets/."
