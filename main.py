#!/usr/bin/env python3
"""SUMO V2V Communication Dashboard - Main entry point."""

import os
import random
import signal
import sys
import time

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parser import parse_args

import config
import logger
from communication.comm_manager import CommManager
from dashboard.app import DashboardApp
from fl_interface.fl_payload import FLPayload
from simulation.sumo_manager import SumoManager


def main():
    args = parse_args()
    logger.set_level(args.verbose)

    scenario_info = config.SCENARIOS[args.scenario]
    logger.log("Starting SUMO V2V Dashboard", "info")
    logger.log(f"Scenario: {scenario_info['name']} ({args.scenario})")
    logger.log(f"Vehicles: {args.num_vehicles}")
    logger.log(f"Comm range: {args.comm_range}m")
    logger.log(f"Speed: {args.speed}x {'(real-time)' if args.speed == 1.0 else '(unlimited)' if args.speed == 0 else ''}")
    if args.force_speed:
        logger.log(f"Force speed: {args.force_speed:.0f} km/h ({args.force_speed / 3.6:.1f} m/s)")
    else:
        logger.log("Force speed: off (SUMO default car-following model)")
    logger.log(f"FL demo: {'on' if args.fl_demo else 'off'}")

    sumo = SumoManager(args.scenario, args.num_vehicles, args.force_speed)
    comm = CommManager(comm_range=args.comm_range)
    dashboard = None
    running = True

    # Speed multiplier: 1.0 = real-time, 2.0 = 2x faster, 0 = unlimited
    speed_mult = args.speed

    def signal_handler(sig, frame):
        nonlocal running
        logger.log("Shutting down...", "warning")
        running = False

    signal.signal(signal.SIGINT, signal_handler)

    try:
        logger.log("Starting SUMO simulation...", "info")
        sumo.start()

        net_bounds = sumo.get_network_bounds()
        edge_shapes = sumo.get_edge_shapes()
        logger.log(f"Network bounds: {net_bounds}")
        logger.log(f"Road segments: {len(edge_shapes)}")

        dashboard = DashboardApp(net_bounds, edge_shapes, scenario_info["name"])
        dashboard.initialize()
        logger.log("Dashboard ready. Press ESC or Q to quit.", "success")

        fl_payload = FLPayload() if args.fl_demo else None
        fl_interval = 10.0
        last_fl_time = 0.0
        step_count = 0
        vehicle_states = {}
        sim_time = 0.0
        active_links = []
        new_messages = []

        # Accumulator: tracks how much sim-time we owe
        sim_accumulator = 0.0
        last_frame_time = time.perf_counter()

        while running:
            now = time.perf_counter()
            dt = now - last_frame_time
            last_frame_time = now

            # --- Simulation stepping (decoupled from render) ---
            if not dashboard.paused:
                if speed_mult == 0:
                    # Unlimited: one SUMO step per render frame
                    vehicle_states = sumo.step()
                    sim_time = sumo.get_sim_time()
                    new_messages += comm.update(vehicle_states, sim_time)
                    step_count += 1
                else:
                    sim_accumulator += dt * speed_mult
                    # Cap to prevent spiral-of-death after lag spikes
                    sim_accumulator = min(sim_accumulator, config.SIM_STEP_LENGTH * 3)
                    while sim_accumulator >= config.SIM_STEP_LENGTH:
                        sim_accumulator -= config.SIM_STEP_LENGTH
                        vehicle_states = sumo.step()
                        sim_time = sumo.get_sim_time()
                        new_messages += comm.update(vehicle_states, sim_time)
                        step_count += 1

                        # FL weight exchange demo
                        if fl_payload and sim_time - last_fl_time >= fl_interval:
                            last_fl_time = sim_time
                            veh_ids = list(vehicle_states.keys())
                            if len(veh_ids) >= 2:
                                sender = random.choice(veh_ids)
                                neighbors = comm.get_neighbors(sender)
                                if neighbors:
                                    receiver = random.choice(neighbors)
                                    weights = fl_payload.dummy_weights()
                                    payload = fl_payload.serialize_weights(weights)
                                    comm.send_message(sender, receiver, "fl_weights", payload, sim_time)

                active_links = comm.get_active_links()

            # --- Render (always, at FPS rate — Clock.tick handles pacing) ---
            if not dashboard.render(vehicle_states, active_links, new_messages, sim_time):
                break
            new_messages = []  # clear after handing to dashboard

            # Periodic console status
            if step_count > 0 and step_count % 100 == 0:
                stats = comm.get_stats()
                logger.log(
                    f"Step {step_count} | Time: {sim_time:.0f}s | "
                    f"Vehicles: {len(vehicle_states)} | "
                    f"Links: {len(active_links)} | "
                    f"Msgs sent: {stats['sent']} delivered: {stats['delivered']}",
                    "result",
                )

    except FileNotFoundError as e:
        logger.log(str(e), "error")
        sys.exit(1)
    except Exception as e:
        logger.log(str(e), "error")
        import traceback
        traceback.print_exc()
    finally:
        logger.log("Cleaning up...", "info")
        if dashboard:
            dashboard.cleanup()
        sumo.stop()
        logger.log("Done.", "success")


if __name__ == "__main__":
    main()
