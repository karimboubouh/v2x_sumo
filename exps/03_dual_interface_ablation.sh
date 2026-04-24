#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

COMMON_ARGS=(
  --dl
  --headless
  --speed 0
  --scenario sheikh_zayed_road
  --num-vehicles 40
  --rounds 200
  --target_acc 1.01
  --dl-dataset CIFAR10
  --dl-model CNN
  --dl-algorithm DANTE
  --verbose result
)

declare -a EXPERIMENTS=()

run_wrapped_experiment "DANTE hybrid" "" "${COMMON_ARGS[@]}"
relabel_experiment "$LAST_EXPERIMENT_DIR" "DANTE-Hybrid"
EXPERIMENTS+=("$LAST_EXPERIMENT_DIR")

run_wrapped_experiment \
  "DANTE PC5 only" \
  "from algorithms.dante import config as dante_config; dante_config.MAX_INTERNET_NEIGHBORS = 0" \
  "${COMMON_ARGS[@]}"
relabel_experiment "$LAST_EXPERIMENT_DIR" "DANTE-PC5"
EXPERIMENTS+=("$LAST_EXPERIMENT_DIR")

run_wrapped_experiment \
  "DANTE Uu only" \
  "from algorithms.dante import config as dante_config; dante_config.MAX_SIDELINK_NEIGHBORS = 0; config.COMM_RANGE = 0.0" \
  "${COMMON_ARGS[@]}"
relabel_experiment "$LAST_EXPERIMENT_DIR" "DANTE-Uu"
EXPERIMENTS+=("$LAST_EXPERIMENT_DIR")

run_comparison "dual interface ablation" "${EXPERIMENTS[@]}"

copy_figure "$LAST_COMPARISON_DIR" "accuracy_vs_rounds_comparison" "dual_interface_accuracy_vs_rounds_comparison"
copy_figure "$LAST_COMPARISON_DIR" "accuracy_vs_time_comparison" "dual_interface_accuracy_vs_time_comparison"
copy_figure "$LAST_COMPARISON_DIR" "loss_vs_time_comparison" "dual_interface_loss_vs_time_comparison"
copy_figure "$LAST_COMPARISON_DIR" "energy_comparison" "dual_interface_energy_comparison"

echo "Dual-interface ablation figures are ready in paper/assets/."
