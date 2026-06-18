"""CLI argument parser for the SUMO V2V Dashboard."""

import argparse
import config


def parse_args():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description="SUMO V2V Communication Dashboard")
    parser.add_argument(
        "--scenario", "-s",
        default=config.DEFAULT_SCENARIO,
        choices=list(config.SCENARIOS.keys()),
        help=f"Scenario to run (default: {config.DEFAULT_SCENARIO})",
    )
    parser.add_argument(
        "--num-vehicles", "-n",
        type=int,
        default=config.NUM_VEHICLES,
        help=f"Target number of vehicles (default: {config.NUM_VEHICLES})",
    )
    parser.add_argument(
        "--comm-range", "-r",
        type=float,
        default=config.COMM_RANGE,
        help=f"Communication range in meters (default: {config.COMM_RANGE})",
    )
    parser.add_argument(
        "--speed", "-x",
        type=float,
        default=1.0,
        help="Simulation speed multiplier: 1.0=real-time, 2.0=2× faster, 0=unlimited (default: 1.0)",
    )
    parser.add_argument(
        "--force-speed",
        type=float,
        default=config.VEHICLE_FORCE_SPEED,
        dest="force_speed",
        metavar="KM/H",
        help="Force all vehicles to this speed in km/h, overriding road limits (e.g. 50, 120, 280); omit to use SUMO default model",
    )
    parser.add_argument(
        "--verbose", "-v",
        default=config.LOG_LEVEL,
        choices=["debug", "info", "success", "result", "warning", "error"],
        metavar="LEVEL",
        help="Minimum log level to display: debug|info|success|result|warning|error (default: %(default)s)",
    )
    parser.add_argument(
        "--save-logs",
        action="store_true",
        default=config.SAVE_LOGS,
        dest="save_logs",
        help="Save all plain logger output into out/<experiment_id>/run.log for DPL runs, regardless of --verbose",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=config.HEADLESS,
        help="Disable dashboard and graphics; run with console output only",
    )
    parser.add_argument(
        "--plot-experiment",
        metavar="PKL",
        dest="plot_experiment",
        help="Load a saved DPL experiment pickle, regenerate the plots, and show them",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.SEED,
        help=f"Reproducibility seed; pass a negative value to leave RNGs unseeded (default: {config.SEED})",
    )
    parser.add_argument(
        "--perf-log",
        action="store_true",
        help="Log lightweight runtime performance metrics every few seconds",
    )
    parser.add_argument(
        "--ui-fps",
        type=int,
        default=config.UI_FPS,
        metavar="FPS",
        help=f"Dashboard refresh rate when graphics are enabled (default: {config.UI_FPS})",
    )
    parser.add_argument(
        "--train-workers",
        type=int,
        default=None,
        metavar="N",
        help="Override DPL training worker threads; dashboard mode defaults to a conservative CPU cap",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=None,
        metavar="N",
        help=f"Override torch intra-op threads for DPL runs (default: {config.TORCH_NUM_THREADS})",
    )
    parser.add_argument(
        "--torch-interop-threads",
        type=int,
        default=None,
        metavar="N",
        help=f"Override torch inter-op threads for DPL runs (default: {config.TORCH_NUM_INTEROP_THREADS})",
    )
    # ── Decentralized Personalized Learning ────────────────────────────────
    parser.add_argument(
        "--dl",
        action="store_true",
        help="Enable decentralized personalized learning",
    )
    parser.add_argument(
        "--dl-algorithm",
        default=config.ALGORITHM,
        dest="dl_algorithm",
        help=f"DPL algorithm (default: {config.ALGORITHM})",
    )
    parser.add_argument(
        "--dl-dataset",
        default=config.DATASET,
        choices=["MNIST", "FEMNIST", "CIFAR10", "CIFAR100"],
        dest="dl_dataset",
        help=f"DPL training dataset (default: {config.DATASET})",
    )
    parser.add_argument(
        "--dl-model",
        default=config.MODEL_ARCH,
        choices=["DNN", "CNN", "LSTM", "Transformer", "ResNet"],
        dest="dl_model",
        help=f"DPL model architecture (default: {config.MODEL_ARCH})",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=config.MAX_TR_ROUNDS,
        metavar="ROUNDS",
        help=f"Max rounds per vehicle (default: {config.MAX_TR_ROUNDS})",
    )
    parser.add_argument(
        "--target_acc",
        type=float,
        default=config.TARGET_ACCURACY,
        metavar="TARGET_ACC",
        help=f"Target accuracy for automatic stop (default: {config.TARGET_ACCURACY})",
    )
    parser.add_argument(
        "--stop-on",
        default=config.STOP_ON,
        choices=["rounds", "train_acc", "eval_acc"],
        dest="stop_on",
        help=f"DPL stop criterion: rounds, train_acc, or eval_acc (default: {config.STOP_ON})",
    )
    return parser.parse_args()


def validate_args(args) -> None:
    """Validate options whose choices require delayed optional imports."""
    if not getattr(args, "dl", False):
        return

    from algorithms import get_available_algorithms

    available = get_available_algorithms()
    if args.dl_algorithm not in available:
        choices = ", ".join(available)
        raise ValueError(
            f"Unknown DPL algorithm {args.dl_algorithm!r}. Available: {choices}"
        )
