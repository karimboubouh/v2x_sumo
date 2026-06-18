#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

BASE_ARGS=(
  --dl
  --headless
  --scenario khalifa_university
  --num-vehicles 25
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

for concentration in 0.3 0.5 1.0; do
  case "$concentration" in
    0.3) label="High non-IID" ;;
    0.5) label="Moderate non-IID" ;;
    1.0) label="Mild non-IID" ;;
    *) label="Dirichlet concentration=$concentration" ;;
  esac

  run_wrapped_experiment \
    "DANTE non-IID Dirichlet concentration=$concentration" \
    "config.DATA_ALPHA = $concentration" \
    "${BASE_ARGS[@]}"
  relabel_experiment "$LAST_EXPERIMENT_DIR" "$label"
  EXPERIMENTS+=("$LAST_EXPERIMENT_DIR")
done

run_comparison "non-iid sweep" "${EXPERIMENTS[@]}"

copy_figure "$LAST_COMPARISON_DIR" "accuracy_vs_rounds_comparison" "noniid_accuracy_vs_rounds_comparison"
copy_figure "$LAST_COMPARISON_DIR" "accuracy_vs_time_comparison" "noniid_accuracy_vs_time_comparison"
copy_figure "$LAST_COMPARISON_DIR" "energy_comparison" "noniid_energy_comparison"
copy_figure "$LAST_COMPARISON_DIR" "communication_energy_comparison" "noniid_communication_energy_comparison"

echo "Non-IID sweep figures are ready in paper/assets/."
