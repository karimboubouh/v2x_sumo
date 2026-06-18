#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

COMMON_ARGS=(
  --dl
  --headless
  --scenario khalifa_university
  --num-vehicles 40
  --rounds 200
  --stop-on eval_acc
  --target_acc 0.80
  --dl-dataset CIFAR10
  --dl-model CNN
  --dl-algorithm DANTE
  --verbose result
  --save-logs
)

declare -a EXPERIMENTS=()

run_wrapped_experiment "DANTE" "" "${COMMON_ARGS[@]}"
relabel_experiment "$LAST_EXPERIMENT_DIR" "DANTE"
EXPERIMENTS+=("$LAST_EXPERIMENT_DIR")

run_wrapped_experiment \
  "DANTE sidelink only" \
  "from algorithms.dante import config as dante_config; from algorithms.dante import algorithm as dante_algorithm; dante_config.MAX_INTERNET_NEIGHBORS = 0; dante_config.MAX_INTERNET_EXPLORATION_NEIGHBORS = 0; dante_algorithm.MAX_INTERNET_NEIGHBORS = 0; dante_algorithm.MAX_INTERNET_EXPLORATION_NEIGHBORS = 0" \
  "${COMMON_ARGS[@]}"
relabel_experiment "$LAST_EXPERIMENT_DIR" "DANTE-Sidelink"
EXPERIMENTS+=("$LAST_EXPERIMENT_DIR")

run_wrapped_experiment \
  "DANTE internet only" \
  "from algorithms.dante import config as dante_config; from algorithms.dante import algorithm as dante_algorithm; dante_config.MAX_SIDELINK_NEIGHBORS = 0; dante_algorithm.MAX_SIDELINK_NEIGHBORS = 0" \
  "${COMMON_ARGS[@]}"
relabel_experiment "$LAST_EXPERIMENT_DIR" "DANTE-Internet"
EXPERIMENTS+=("$LAST_EXPERIMENT_DIR")

run_comparison "dual interface ablation" "${EXPERIMENTS[@]}"

copy_figure "$LAST_COMPARISON_DIR" "accuracy_vs_rounds_comparison" "dual_interface_accuracy_vs_rounds_comparison"
copy_figure "$LAST_COMPARISON_DIR" "accuracy_vs_time_comparison" "dual_interface_accuracy_vs_time_comparison"
copy_figure "$LAST_COMPARISON_DIR" "loss_vs_rounds_comparison" "dual_interface_loss_vs_rounds_comparison"
copy_figure "$LAST_COMPARISON_DIR" "loss_vs_time_comparison" "dual_interface_loss_vs_time_comparison"
copy_figure "$LAST_COMPARISON_DIR" "energy_comparison" "dual_interface_energy_comparison"
copy_figure "$LAST_COMPARISON_DIR" "communication_energy_comparison" "dual_interface_communication_energy_comparison"

echo "Dual-interface ablation figures are ready in paper/assets/."
