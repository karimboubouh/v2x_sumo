#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

BASE_ARGS=(
  --dl
  --headless
  --speed 0
  --num-vehicles 25
  --rounds 200
  --target_acc 1.01
  --dl-dataset CIFAR10
  --dl-model CNN
  --dl-algorithm DANTE
  --verbose result
)

declare -a EXPERIMENTS=()

for scenario in dubai_marina sheikh_zayed_road sharjah_university; do
  run_cli_experiment "DANTE scenario $scenario" "${BASE_ARGS[@]}" --scenario "$scenario"
  case "$scenario" in
    dubai_marina) label="DANTE | Marina" ;;
    sheikh_zayed_road) label="DANTE | Highway" ;;
    sharjah_university) label="DANTE | Campus" ;;
    *) label="DANTE | $scenario" ;;
  esac
  relabel_experiment "$LAST_EXPERIMENT_DIR" "$label"
  EXPERIMENTS+=("$LAST_EXPERIMENT_DIR")
done

run_comparison "scenario transfer" "${EXPERIMENTS[@]}"

copy_figure "$LAST_COMPARISON_DIR" "accuracy_vs_rounds_comparison" "scenario_accuracy_vs_rounds_comparison"
copy_figure "$LAST_COMPARISON_DIR" "accuracy_vs_time_comparison" "scenario_accuracy_vs_time_comparison"
copy_figure "$LAST_COMPARISON_DIR" "energy_comparison" "scenario_energy_comparison"

echo "Scenario-transfer figures are ready in paper/assets/."
