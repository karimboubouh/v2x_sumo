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
        "--fl-demo",
        action="store_true",
        help="Enable periodic FL weight exchange messages",
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
    return parser.parse_args()
