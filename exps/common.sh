#!/usr/bin/env bash
set -euo pipefail

EXPS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$EXPS_DIR/.." && pwd)"
OUT_DIR="$REPO_ROOT/out"
PAPER_ASSETS_DIR="$REPO_ROOT/paper/assets"

mkdir -p "$OUT_DIR" "$PAPER_ASSETS_DIR"

LAST_EXPERIMENT_DIR=""
LAST_COMPARISON_DIR=""

warn_if_zero_link_experiment() {
  local experiment_dir="$1"
  local label="${2:-}"

  python - "$experiment_dir/experiment.pkl" "$label" <<'PY'
import pickle
import re
import sys

path = sys.argv[1]
label = sys.argv[2].strip() or "experiment"
with open(path, "rb") as fh:
    experiment = pickle.load(fh)

def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())

max_links = max(
    (int(point.get("total_links", 0)) for point in experiment.get("train_history", [])),
    default=0,
)
cfg = dict(experiment.get("config", {}))
metadata = dict(experiment.get("metadata", {}))
labels = {
    normalize(label),
    normalize(metadata.get("algorithm", "")),
    normalize(cfg.get("ALGORITHM", "")),
}
zero_caps = all(
    int(cfg.get(key, 0) or 0) <= 0
    for key in ("MAX_SIDELINK_NEIGHBORS", "MAX_INTERNET_NEIGHBORS", "MAX_COLLAB_NEIGHBORS")
)
is_local_only = "localonly" in labels or "localonlymaincomparison" in labels or "localonlymaincomparison" in normalize(label)
if (is_local_only or zero_caps) and max_links > 0:
    raise SystemExit(
        f"{label} declared zero collaboration capacity but recorded {max_links} active links."
    )
if max_links == 0 and not is_local_only:
    print(
        f"    WARNING: {label} recorded zero collaboration links across all rounds; "
        "treat it as a local baseline, not an active collaborative comparator.",
        file=sys.stderr,
    )
PY
}

snapshot_experiments() {
  find "$OUT_DIR" -mindepth 1 -maxdepth 1 -type d -exec test -f "{}/experiment.pkl" ";" -print | sort
}

snapshot_comparisons() {
  find "$OUT_DIR" -mindepth 1 -maxdepth 1 -type d -name 'comparison_*' -print | sort
}

detect_new_path() {
  local before="$1"
  local after="$2"
  local newest

  newest="$(comm -13 "$before" "$after" | tail -n1 || true)"
  if [[ -n "$newest" ]]; then
    printf '%s\n' "$newest"
    return
  fi

  tail -n1 "$after" || true
}

run_cli_experiment() {
  local label="$1"
  shift

  local before after
  before="$(mktemp)"
  after="$(mktemp)"
  snapshot_experiments >"$before"

  echo "==> Running: $label"
  (
    cd "$REPO_ROOT"
    python main.py "$@"
  )

  snapshot_experiments >"$after"
  LAST_EXPERIMENT_DIR="$(detect_new_path "$before" "$after")"
  rm -f "$before" "$after"

  if [[ -z "$LAST_EXPERIMENT_DIR" ]]; then
    echo "Failed to locate saved experiment folder for: $label" >&2
    exit 1
  fi

  echo "    Saved experiment: $LAST_EXPERIMENT_DIR"
  warn_if_zero_link_experiment "$LAST_EXPERIMENT_DIR" "$label"
}

run_wrapped_experiment() {
  local label="$1"
  local overrides="$2"
  shift 2

  local before after
  before="$(mktemp)"
  after="$(mktemp)"
  snapshot_experiments >"$before"

  echo "==> Running: $label"
  (
    cd "$REPO_ROOT"
    EXPS_REPO_ROOT="$REPO_ROOT" \
    EXPS_OVERRIDES="$overrides" \
    python - "$@" <<'PY'
import os
import sys

repo = os.environ["EXPS_REPO_ROOT"]
overrides = os.environ.get("EXPS_OVERRIDES", "")

sys.path.insert(0, repo)

import config

if overrides:
    exec(overrides, {"config": config})

import main

sys.argv = ["main.py", *sys.argv[1:]]
main.main()
PY
  )

  snapshot_experiments >"$after"
  LAST_EXPERIMENT_DIR="$(detect_new_path "$before" "$after")"
  rm -f "$before" "$after"

  if [[ -z "$LAST_EXPERIMENT_DIR" ]]; then
    echo "Failed to locate saved experiment folder for: $label" >&2
    exit 1
  fi

  echo "    Saved experiment: $LAST_EXPERIMENT_DIR"
  warn_if_zero_link_experiment "$LAST_EXPERIMENT_DIR" "$label"
}

relabel_experiment() {
  local experiment_dir="$1"
  local new_label="$2"

  python - "$experiment_dir/experiment.pkl" "$new_label" <<'PY'
import pickle
import sys

path, label = sys.argv[1], sys.argv[2]
with open(path, "rb") as fh:
    experiment = pickle.load(fh)

experiment.setdefault("metadata", {})["algorithm"] = label

with open(path, "wb") as fh:
    pickle.dump(experiment, fh)
PY
}

run_comparison() {
  local label="$1"
  shift

  local before after
  before="$(mktemp)"
  after="$(mktemp)"
  snapshot_comparisons >"$before"

  echo "==> Building comparison: $label"
  (
    cd "$REPO_ROOT"
    MPLBACKEND=Agg python run_plots.py "$@"
  )

  snapshot_comparisons >"$after"
  LAST_COMPARISON_DIR="$(detect_new_path "$before" "$after")"
  rm -f "$before" "$after"

  if [[ -z "$LAST_COMPARISON_DIR" ]]; then
    echo "Failed to locate comparison folder for: $label" >&2
    exit 1
  fi

  echo "    Saved comparison: $LAST_COMPARISON_DIR"
}

copy_figure() {
  local comparison_dir="$1"
  local stem="$2"
  local paper_name="$3"

  cp "$comparison_dir/$stem.pdf" "$PAPER_ASSETS_DIR/$paper_name.pdf"
  if [[ -f "$comparison_dir/$stem.png" ]]; then
    cp "$comparison_dir/$stem.png" "$PAPER_ASSETS_DIR/$paper_name.png"
  fi
  echo "    Copied: paper/assets/$paper_name.pdf"
}
