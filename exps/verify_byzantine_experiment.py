#!/usr/bin/env python3
"""Validate Byzantine attack experiment outputs before plotting."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def _load(path: str) -> dict:
    pkl = Path(path) / "experiment.pkl"
    if not pkl.is_file():
        raise SystemExit(f"Missing experiment.pkl in {path}")
    with pkl.open("rb") as fh:
        return pickle.load(fh)


def _label(experiment: dict) -> str:
    return str(experiment.get("metadata", {}).get("algorithm", "experiment"))


def _validate_common(
    experiment: dict,
    expected_fraction: float,
    expected_start: int,
    expected_attack: str,
) -> None:
    label = _label(experiment)
    cfg = dict(experiment.get("config", {}))
    attack = dict(experiment.get("attack", {}))
    fraction = float(attack.get("fraction", cfg.get("BYZANTINE_FRACTION", 0.0)) or 0.0)
    attack_name = str(attack.get("attack", cfg.get("BYZANTINE_ATTACK", ""))).lower()
    start_round = int(attack.get("start_round", cfg.get("BYZANTINE_START_ROUND", -1)) or -1)
    byzantine_ids = list(attack.get("byzantine_ids", []))
    num_vehicles = int(experiment.get("metadata", {}).get("num_vehicles", 0) or 0)
    expected_byzantine = int(num_vehicles * expected_fraction)

    if abs(fraction - expected_fraction) > 1e-12:
        raise SystemExit(f"{label}: expected Byzantine fraction {expected_fraction}, got {fraction}")
    if attack_name != expected_attack:
        raise SystemExit(f"{label}: expected {expected_attack!r} attack, got {attack_name!r}")
    if start_round != expected_start:
        raise SystemExit(f"{label}: expected attack start round {expected_start}, got {start_round}")
    if len(byzantine_ids) != expected_byzantine:
        raise SystemExit(
            f"{label}: expected {expected_byzantine} Byzantine vehicles, got {len(byzantine_ids)}"
        )

    history = sorted(experiment.get("train_history", []), key=lambda p: p["round"])
    active_points = [p for p in history if int(p.get("round", -1)) >= expected_start]
    inactive_after_start = [
        int(p.get("round", -1))
        for p in active_points
        if not bool(p.get("byzantine_active", False))
    ]
    if not active_points:
        raise SystemExit(f"{label}: no training points at or after attack start round")
    if inactive_after_start:
        raise SystemExit(
            f"{label}: attack is not continuously active after start; inactive rounds "
            f"{inactive_after_start[:8]}"
        )

    print(
        f"{label}: {expected_attack} active from round {expected_start}; "
        f"{len(byzantine_ids)}/{num_vehicles} Byzantine vehicles"
    )


def _validate_dpfl_effect(
    experiment: dict,
    start_round: int,
    max_post_acc: float,
    max_final_acc: float,
) -> None:
    label = _label(experiment)
    history = sorted(experiment.get("train_history", []), key=lambda p: p["round"])
    pre = [p for p in history if int(p.get("round", -1)) <= start_round]
    post = [
        p
        for p in history
        if start_round < int(p.get("round", -1)) <= start_round + 10
    ]
    if not pre or not post:
        raise SystemExit(f"{label}: insufficient history around attack start")

    byzantine_links = sum(int(p.get("byzantine_links", 0) or 0) for p in post)
    if byzantine_links <= 0:
        raise SystemExit(
            f"{label}: no active Byzantine aggregation links after attack start; "
            "the attack cannot affect DPFL."
        )

    baseline_loss = float(pre[-1].get("loss", 0.0))
    peak_loss = max(float(p.get("loss", 0.0)) for p in post)
    min_acc = min(float(p.get("acc", 1.0)) for p in post)
    if baseline_loss <= 0.0 or peak_loss < 1.20 * baseline_loss:
        raise SystemExit(
            f"{label}: post-attack loss jump is too small "
            f"({baseline_loss:.4f} -> {peak_loss:.4f}); check attack strength/exposure."
        )
    if min_acc > max_post_acc:
        raise SystemExit(
            f"{label}: post-attack accuracy did not collapse below "
            f"{100.0 * max_post_acc:.1f}% (minimum={100.0 * min_acc:.2f}%)."
        )
    summary = dict(experiment.get("summary", {}))
    final_acc = float(summary.get("final_test_acc", 0.0) or 0.0)
    if final_acc > max_final_acc:
        raise SystemExit(
            f"{label}: final test accuracy remained above "
            f"{100.0 * max_final_acc:.1f}% ({100.0 * final_acc:.2f}%)."
        )

    print(
        f"{label}: Byzantine links after start={byzantine_links}, "
        f"loss jump {baseline_loss:.4f}->{peak_loss:.4f}, "
        f"minimum post-attack accuracy={100.0 * min_acc:.2f}%, "
        f"final test accuracy={100.0 * final_acc:.2f}%"
    )


def _validate_dante_resilience(experiment: dict, min_final_acc: float) -> None:
    label = _label(experiment)
    summary = dict(experiment.get("summary", {}))
    final_acc = float(summary.get("final_test_acc", 0.0) or 0.0)
    if final_acc < min_final_acc:
        raise SystemExit(
            f"{label}: final test accuracy {100.0 * final_acc:.2f}% is below "
            f"the resilience threshold {100.0 * min_final_acc:.1f}%."
        )
    print(f"{label}: final test accuracy={100.0 * final_acc:.2f}%")


def _report_final_accuracy(experiment: dict) -> None:
    label = _label(experiment)
    summary = dict(experiment.get("summary", {}))
    final_acc = float(summary.get("final_test_acc", 0.0) or 0.0)
    print(f"{label}: final test accuracy={100.0 * final_acc:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dante", required=True, help="DANTE experiment directory")
    parser.add_argument("--dpfl", required=True, help="DPFL experiment directory")
    parser.add_argument("--pfedgraph", help="Optional pFedGraph experiment directory")
    parser.add_argument("--attack", default="lie", help="Expected attack name")
    parser.add_argument("--fraction", type=float, default=0.20)
    parser.add_argument("--start-round", type=int, default=20)
    parser.add_argument("--max-dpfl-post-acc", type=float, default=1.0)
    parser.add_argument("--max-dpfl-final-acc", type=float, default=1.0)
    parser.add_argument("--min-dante-final-acc", type=float, default=0.0)
    args = parser.parse_args()

    dante = _load(args.dante)
    dpfl = _load(args.dpfl)
    pfedgraph = _load(args.pfedgraph) if args.pfedgraph else None
    _validate_common(dante, args.fraction, args.start_round, args.attack)
    _validate_common(dpfl, args.fraction, args.start_round, args.attack)
    if pfedgraph is not None:
        _validate_common(pfedgraph, args.fraction, args.start_round, args.attack)
    _validate_dante_resilience(dante, args.min_dante_final_acc)
    _validate_dpfl_effect(
        dpfl,
        args.start_round,
        args.max_dpfl_post_acc,
        args.max_dpfl_final_acc,
    )
    if pfedgraph is not None:
        _report_final_accuracy(pfedgraph)
    print("Byzantine experiment verification passed.")


if __name__ == "__main__":
    main()
