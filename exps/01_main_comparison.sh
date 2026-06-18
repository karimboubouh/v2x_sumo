#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

COMMON_ARGS=(
  --dl
  --headless
  --scenario khalifa_university
  --num-vehicles 50
  --rounds 200
  --stop-on eval_acc
  --target_acc 0.80
  --dl-dataset CIFAR10
  --dl-model CNN
  --verbose info
  --save-logs
)

declare -a EXPERIMENTS=()

# for algo in DANTE DPFL LocalOnly pFedGraph; do
for algo in DANTE DPFL IPPO pFedGraph FedAvg D-PSGD; do
  run_wrapped_experiment \
    "$algo main comparison" \
    $'config.TRAIN_AUGMENTATION_POLICY = "dataset_default"\nconfig.SHARED_INITIAL_MODEL = True\nconfig.LOCAL_OPTIMIZER = "sgd"\nconfig.LOCAL_LR = 5e-2\nconfig.LOCAL_MOMENTUM = 0.9\nconfig.LOCAL_WEIGHT_DECAY = 5e-4\nconfig.BATCHES_PER_ROUND = 4\nconfig.EVAL_BATCHES_PER_ROUND = 10' \
    "${COMMON_ARGS[@]}" \
    --dl-algorithm "$algo"
  if [[ "$algo" == "LocalOnly" ]]; then
    relabel_experiment "$LAST_EXPERIMENT_DIR" "Local Only"
  fi
  EXPERIMENTS+=("$LAST_EXPERIMENT_DIR")
done

run_comparison "main comparison" "${EXPERIMENTS[@]}"

copy_figure "$LAST_COMPARISON_DIR" "accuracy_vs_time_comparison" "main_accuracy_vs_time_comparison"
copy_figure "$LAST_COMPARISON_DIR" "accuracy_vs_rounds_comparison" "main_accuracy_vs_rounds_comparison"
copy_figure "$LAST_COMPARISON_DIR" "loss_vs_time_comparison" "main_loss_vs_time_comparison"
copy_figure "$LAST_COMPARISON_DIR" "energy_comparison" "main_energy_comparison"
copy_figure "$LAST_COMPARISON_DIR" "communication_energy_comparison" "main_communication_energy_comparison"
copy_figure "$LAST_COMPARISON_DIR" "ppo_vs_rounds_comparison" "main_reward_vs_rounds_comparison"
copy_figure "$LAST_COMPARISON_DIR" "ppo_vs_time_comparison" "main_reward_vs_time_comparison"

echo "Main comparison figures are ready in paper/assets/."
