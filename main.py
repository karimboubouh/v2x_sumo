#!/usr/bin/env python3
"""SUMO V2V Communication Dashboard - Main entry point."""

from __future__ import annotations

import os
import signal
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import logger
from event_stream import EventStream
from parser import parse_args, validate_args
from runtime_tuning import configure_runtime
from simulation_runtime import SimulationRuntime


def _event_stream_limits() -> tuple[int | None, int]:
    max_lines = getattr(config, "LOG_MAX_LINES", 2000)
    if max_lines is None:
        return None, 256
    max_lines = max(int(max_lines), 1)
    return max(max_lines * 4, 1024), max(128, min(max_lines // 2, 512))


def main() -> None:
    args = parse_args()
    args.ui_fps = max(int(args.ui_fps), 1)
    logger.set_level(args.verbose)

    tuning = configure_runtime(args)
    try:
        validate_args(args)
    except ValueError as exc:
        logger.log(str(exc), "error")
        sys.exit(2)

    config.SEED = None if args.seed is None or int(args.seed) < 0 else int(args.seed)
    config.STOP_ON = args.stop_on
    config.COMM_RANGE = args.comm_range
    config.SAVE_LOGS = bool(args.save_logs)

    if config.SEED is not None:
        if args.dl:
            from dl.helpers import set_global_seed

            set_global_seed(config.SEED)
        else:
            import random

            import numpy as np

            random.seed(config.SEED)
            np.random.seed(config.SEED)

    if args.plot_experiment:
        from dl.experiment import plot_saved_experiment

        plotted = plot_saved_experiment(
            args.plot_experiment,
            out_root=config.OUT_DIR,
            show=True,
            block=True,
        )
        logger.log(f"Experiment plots regenerated from {plotted['pickle_path']}", "success")
        logger.log(f"Plot folder: {plotted['experiment_dir']}")
        return

    scenario_info = config.SCENARIOS[args.scenario]
    experiment_paths: dict = {}

    def build_experiment_metadata(sim_time_value: float | None = None) -> dict:
        metadata = {
            "scenario": args.scenario,
            "scenario_name": scenario_info["name"],
            "algorithm": args.dl_algorithm,
            "dataset": args.dl_dataset,
            "model": args.dl_model,
            "num_vehicles": args.num_vehicles,
            "args": vars(args),
        }
        if sim_time_value is not None:
            metadata["sim_time"] = sim_time_value
        if experiment_paths:
            metadata["experiment_id"] = experiment_paths["experiment_id"]
        return metadata

    if args.save_logs and args.dl:
        from dl.experiment import prepare_experiment_dir

        experiment_paths.update(
            prepare_experiment_dir(
                build_experiment_metadata(),
                out_root=config.OUT_DIR,
            )
        )
        logger.start_file_logging(experiment_paths["log_path"])

    logger.log("Starting SUMO V2V Dashboard", "info")
    logger.log(f"Scenario: {scenario_info['name']} ({args.scenario})")
    logger.log(f"Vehicles: {args.num_vehicles}")
    logger.log(f"Comm range: {args.comm_range}m")
    logger.log(
        f"Speed: {args.speed}x "
        f"{'(real-time)' if args.speed == 1.0 else '(unlimited)' if args.speed == 0 else ''}"
    )
    if args.force_speed:
        logger.log(f"Force speed: {args.force_speed:.0f} km/h ({args.force_speed / 3.6:.1f} m/s)")
    else:
        logger.log("Force speed: off (SUMO default car-following model)")
    if args.dl:
        logger.log(f"DPL: {args.dl_algorithm} | {args.dl_dataset} | {args.dl_model}")
        logger.log(
            "DPL stop: "
            f"rounds={args.rounds} | "
            f"target_acc={'off' if args.target_acc > 1.0 else f'{args.target_acc:.2%}'} | "
            f"mode={args.stop_on}"
        )
        logger.log(
            "Runtime: "
            f"train_workers={tuning.train_workers} | "
            f"torch_threads={tuning.torch_threads} | "
            f"torch_interop={tuning.torch_interop_threads}",
            "info",
        )
        logger.log(f"Seed: {'off' if config.SEED is None else config.SEED}", "info")
    else:
        logger.log("DPL: off")

    event_stream_max, event_drain_batch = _event_stream_limits()
    event_stream = EventStream(max_events=event_stream_max)
    runtime = SimulationRuntime(
        args,
        scenario_info,
        event_stream,
        build_experiment_metadata,
    )
    dashboard = None

    def signal_handler(sig, frame):
        del sig, frame
        logger.log("Shutting down...", "warning")
        runtime.stop()

    signal.signal(signal.SIGINT, signal_handler)

    try:
        net_bounds, edge_shapes = runtime.prepare()
        if args.headless:
            logger.log("Running in headless mode (no dashboard).", "info")
            runtime.run_forever()
        else:
            from dashboard.app import DashboardApp

            dashboard = DashboardApp(net_bounds, edge_shapes, scenario_info["name"])
            dashboard.initialize()
            logger.log("Dashboard ready. Press ESC or Q to quit.", "success")
            runtime.start_background()
            dashboard.run_with_runtime(
                runtime,
                event_stream,
                ui_fps=args.ui_fps,
                event_drain_batch=event_drain_batch,
            )
    except FileNotFoundError as exc:
        logger.log(str(exc), "error")
        sys.exit(1)
    finally:
        runtime.stop()
        runtime.join(timeout=5.0)
        runtime.cleanup()
        if dashboard is not None:
            dashboard.cleanup()
        logger.stop_file_logging()


if __name__ == "__main__":
    main()
