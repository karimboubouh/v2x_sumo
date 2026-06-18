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
  --verbose result
  --save-logs
)

ATTACK_OVERRIDES=$'config.BYZANTINE_FRACTION = 0.20\nconfig.BYZANTINE_ATTACK = "sign_flip"\nconfig.BYZANTINE_START_ROUND = 20\nconfig.BYZANTINE_SIGN_FLIP_SCALE = 2000.0\nimport algorithms.dpfl.config as dpfl_config\ndpfl_config.DPFL_UPDATE_EVERY = 200'

declare -a EXPERIMENTS=()

run_wrapped_experiment \
  "DANTE under sign-flip attack" \
  "$ATTACK_OVERRIDES" \
  "${COMMON_ARGS[@]}" \
  --dl-algorithm DANTE
relabel_experiment "$LAST_EXPERIMENT_DIR" "DANTE-SignFlip"
DANTE_EXPERIMENT_DIR="$LAST_EXPERIMENT_DIR"
EXPERIMENTS+=("$LAST_EXPERIMENT_DIR")

run_wrapped_experiment \
  "DPFL under sign-flip attack" \
  "$ATTACK_OVERRIDES" \
  "${COMMON_ARGS[@]}" \
  --dl-algorithm DPFL
relabel_experiment "$LAST_EXPERIMENT_DIR" "DPFL-SignFlip"
DPFL_EXPERIMENT_DIR="$LAST_EXPERIMENT_DIR"
EXPERIMENTS+=("$LAST_EXPERIMENT_DIR")

run_wrapped_experiment \
  "pFedGraph under sign-flip attack" \
  "$ATTACK_OVERRIDES" \
  "${COMMON_ARGS[@]}" \
  --dl-algorithm pFedGraph
relabel_experiment "$LAST_EXPERIMENT_DIR" "pFedGraph-SignFlip"
PFEDGRAPH_EXPERIMENT_DIR="$LAST_EXPERIMENT_DIR"
EXPERIMENTS+=("$LAST_EXPERIMENT_DIR")

"$PYTHON_EXE" "$EXPS_DIR/verify_byzantine_experiment.py" \
  --dante "$DANTE_EXPERIMENT_DIR" \
  --dpfl "$DPFL_EXPERIMENT_DIR" \
  --pfedgraph "$PFEDGRAPH_EXPERIMENT_DIR" \
  --attack sign_flip \
  --fraction 0.20 \
  --start-round 20 \
  --max-dpfl-post-acc 0.50 \
  --max-dpfl-final-acc 0.50 \
  --min-dante-final-acc 0.70

run_comparison "byzantine sign-flip attack" "${EXPERIMENTS[@]}"

copy_figure "$LAST_COMPARISON_DIR" "accuracy_vs_rounds_comparison" "byzantine_signflip_accuracy_vs_rounds_comparison"
copy_figure "$LAST_COMPARISON_DIR" "accuracy_vs_time_comparison" "byzantine_signflip_accuracy_vs_time_comparison"
copy_figure "$LAST_COMPARISON_DIR" "loss_vs_rounds_comparison" "byzantine_signflip_loss_vs_rounds_comparison"
copy_figure "$LAST_COMPARISON_DIR" "loss_vs_time_comparison" "byzantine_signflip_loss_vs_time_comparison"
copy_figure "$LAST_COMPARISON_DIR" "attack_train_accuracy_vs_rounds_comparison" "byzantine_signflip_train_accuracy_vs_rounds_comparison"
copy_figure "$LAST_COMPARISON_DIR" "attack_train_accuracy_vs_time_comparison" "byzantine_signflip_train_accuracy_vs_time_comparison"
copy_figure "$LAST_COMPARISON_DIR" "attack_train_loss_vs_rounds_comparison" "byzantine_signflip_train_loss_vs_rounds_comparison"
copy_figure "$LAST_COMPARISON_DIR" "attack_train_loss_vs_time_comparison" "byzantine_signflip_train_loss_vs_time_comparison"
copy_figure "$LAST_COMPARISON_DIR" "energy_comparison" "byzantine_signflip_energy_comparison"
copy_figure "$LAST_COMPARISON_DIR" "communication_energy_comparison" "byzantine_signflip_communication_energy_comparison"

echo "Byzantine sign-flip attack comparison figures are ready in paper/assets/."
