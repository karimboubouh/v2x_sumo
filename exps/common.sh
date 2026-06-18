#!/usr/bin/env bash
set -euo pipefail

EXPS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$EXPS_DIR/.." && pwd)"
OUT_DIR="$REPO_ROOT/out"
PAPER_ASSETS_DIR="$REPO_ROOT/paper/assets"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_EXE="$PYTHON_BIN"
elif [[ -x "$REPO_ROOT/venv/bin/python3" ]]; then
  PYTHON_EXE="$REPO_ROOT/venv/bin/python3"
else
  PYTHON_EXE="python"
fi

mkdir -p "$OUT_DIR" "$PAPER_ASSETS_DIR"

LAST_EXPERIMENT_DIR=""
LAST_COMPARISON_DIR=""

warn_if_zero_link_experiment() {
  local experiment_dir="$1"
  local label="${2:-}"

  "$PYTHON_EXE" - "$experiment_dir/experiment.pkl" "$label" <<'PY'
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

  "$PYTHON_EXE" - "$before" "$after" <<'PY'
import os
import sys

before_path, after_path = sys.argv[1], sys.argv[2]
with open(before_path, encoding="utf-8") as fh:
    before = {line.strip() for line in fh if line.strip()}
with open(after_path, encoding="utf-8") as fh:
    after = {line.strip() for line in fh if line.strip()}

candidates = sorted(after - before) or sorted(after)
if not candidates:
    raise SystemExit(0)

print(max(candidates, key=lambda path: os.path.getmtime(path)))
PY
}

detect_new_experiment_path() {
  local before="$1"
  local after="$2"
  shift 2

  "$PYTHON_EXE" - "$before" "$after" "$@" <<'PY'
import os
import pickle
import sys

before_path, after_path, *cli_args = sys.argv[1:]
with open(before_path, encoding="utf-8") as fh:
    before = {line.strip() for line in fh if line.strip()}
with open(after_path, encoding="utf-8") as fh:
    after = {line.strip() for line in fh if line.strip()}

candidates = sorted(after - before)
if not candidates:
    candidates = sorted(after)

value_options = {
    "--scenario": ("scenario", str),
    "--num-vehicles": ("num_vehicles", int),
    "--rounds": ("rounds", int),
    "--target_acc": ("target_acc", float),
    "--dl-algorithm": ("dl_algorithm", str),
    "--dl-dataset": ("dl_dataset", str),
    "--dl-model": ("dl_model", str),
}

expected = {}
idx = 0
while idx < len(cli_args):
    token = cli_args[idx]
    option, sep, inline_value = token.partition("=")
    if option in value_options:
        key, caster = value_options[option]
        if sep:
            raw_value = inline_value
        else:
            idx += 1
            if idx >= len(cli_args):
                break
            raw_value = cli_args[idx]
        expected[key] = caster(raw_value)
    idx += 1

def load_experiment(path: str) -> dict | None:
    try:
        with open(os.path.join(path, "experiment.pkl"), "rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None

def observed_values(experiment: dict) -> dict:
    metadata = dict(experiment.get("metadata", {}))
    args = dict(metadata.get("args", {}))
    observed = dict(args)
    observed.setdefault("scenario", metadata.get("scenario"))
    observed.setdefault("num_vehicles", metadata.get("num_vehicles"))
    observed.setdefault("dl_algorithm", metadata.get("algorithm"))
    observed.setdefault("dl_dataset", metadata.get("dataset"))
    observed.setdefault("dl_model", metadata.get("model"))
    return observed

def values_match(key: str, expected_value, observed_value) -> bool:
    if observed_value is None:
        return False
    if key in {"num_vehicles", "rounds"}:
        return int(observed_value) == int(expected_value)
    if key == "target_acc":
        return abs(float(observed_value) - float(expected_value)) <= 1e-12
    return str(observed_value) == str(expected_value)

matches = []
for path in candidates:
    experiment = load_experiment(path)
    if experiment is None:
        continue
    observed = observed_values(experiment)
    if all(values_match(key, value, observed.get(key)) for key, value in expected.items()):
        matches.append(path)

if matches:
    print(max(matches, key=lambda path: os.path.getmtime(os.path.join(path, "experiment.pkl"))))
    raise SystemExit(0)

if len(candidates) == 1:
    print(candidates[0])
    raise SystemExit(0)

candidate_names = "\n  ".join(os.path.basename(path) for path in candidates) or "(none)"
expected_text = ", ".join(f"{key}={value!r}" for key, value in sorted(expected.items()))
print(
    "Failed to identify the experiment created by this run.\n"
    f"Expected metadata: {expected_text}\n"
    f"New candidates:\n  {candidate_names}",
    file=sys.stderr,
)
raise SystemExit(2)
PY
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
    "$PYTHON_EXE" main.py "$@"
  )

  snapshot_experiments >"$after"
  LAST_EXPERIMENT_DIR="$(detect_new_experiment_path "$before" "$after" "$@")"
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
    "$PYTHON_EXE" - "$@" <<'PY'
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
  LAST_EXPERIMENT_DIR="$(detect_new_experiment_path "$before" "$after" "$@")"
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

  "$PYTHON_EXE" - "$experiment_dir/experiment.pkl" "$new_label" <<'PY'
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
    MPLBACKEND=Agg "$PYTHON_EXE" run_plots.py "$@"
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
