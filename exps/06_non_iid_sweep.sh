#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

BASE_ARGS=(
  --dl
  --headless
  --speed 0
  --scenario dubai_marina
  --num-vehicles 25
  --rounds 150
  --target_acc 1.01
  --dl-dataset MNIST
  --dl-model DNN
  --dl-algorithm DANTE
  --verbose result
)

declare -a EXPERIMENTS=()

for alpha in 0.1 0.3 0.5 1.0; do
  run_wrapped_experiment \
    "DANTE non-IID alpha=$alpha" \
    "config.DATA_ALPHA = $alpha" \
    "${BASE_ARGS[@]}"
  relabel_experiment "$LAST_EXPERIMENT_DIR" "DANTE | alpha=$alpha"
  EXPERIMENTS+=("$LAST_EXPERIMENT_DIR")
done

run_comparison "non-iid sweep" "${EXPERIMENTS[@]}"

copy_figure "$LAST_COMPARISON_DIR" "accuracy_vs_rounds_comparison" "noniid_accuracy_vs_rounds_comparison"
copy_figure "$LAST_COMPARISON_DIR" "accuracy_vs_time_comparison" "noniid_accuracy_vs_time_comparison"
copy_figure "$LAST_COMPARISON_DIR" "energy_comparison" "noniid_energy_comparison"

echo "Non-IID sweep figures are ready in paper/assets/."
