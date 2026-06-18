#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

BASE_ARGS=(
  --dl
  --headless
  --scenario khalifa_university
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

for n in 10 25 50 100; do
  run_cli_experiment "DANTE density n=$n" "${BASE_ARGS[@]}" --num-vehicles "$n"
  relabel_experiment "$LAST_EXPERIMENT_DIR" "DANTE | n=$n"
  EXPERIMENTS+=("$LAST_EXPERIMENT_DIR")
done

run_comparison "vehicle density sweep" "${EXPERIMENTS[@]}"

copy_figure "$LAST_COMPARISON_DIR" "accuracy_vs_rounds_comparison" "density_accuracy_vs_rounds_comparison"
copy_figure "$LAST_COMPARISON_DIR" "accuracy_vs_time_comparison" "density_accuracy_vs_time_comparison"
copy_figure "$LAST_COMPARISON_DIR" "energy_comparison" "density_energy_comparison"
copy_figure "$LAST_COMPARISON_DIR" "communication_energy_comparison" "density_communication_energy_comparison"

echo "Vehicle-density figures are ready in paper/assets/."
