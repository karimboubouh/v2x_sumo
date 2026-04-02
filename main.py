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
from event_stream import EventStream
from fl_interface.fl_payload import DLPayload
from simulation.sumo_manager import SumoManager


def main():
    args = parse_args()
    logger.set_level(args.verbose)

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
    logger.log("Starting SUMO V2V Dashboard", "info")
    logger.log(f"Scenario: {scenario_info['name']} ({args.scenario})")
    logger.log(f"Vehicles: {args.num_vehicles}")
    logger.log(f"Comm range: {args.comm_range}m")
    logger.log(
        f"Speed: {args.speed}x {'(real-time)' if args.speed == 1.0 else '(unlimited)' if args.speed == 0 else ''}"
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
            f"target_acc={'off' if args.target_acc > 1.0 else f'{args.target_acc:.2%}'}"
        )
    else:
        logger.log("DPL: off")
    logger.log(f"DPL demo: {'on' if args.dl_demo else 'off'}")

    if config.LOG_MAX_LINES is None:
        event_stream_max = None
        event_drain_batch = 256
    else:
        event_stream_max = max(config.LOG_MAX_LINES * 80, 1024)
        event_drain_batch = max(config.LOG_MAX_LINES * 4, 256)

    event_stream = EventStream(max_events=event_stream_max)
    sumo = SumoManager(args.scenario, args.num_vehicles, args.force_speed)
    comm = CommManager(comm_range=args.comm_range, event_stream=event_stream)
    dashboard = None
    dl_env = None
    running = True
    training_status = None
    plots_generated = False

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

        # ── DL initialization (only when --dl is passed) ──────────────
        dl_env = None
        if args.dl:
            from dl.data import partition_dataset
            from dl.env import DLEnvironment

            config.DL_CFG["ALGORITHM"] = args.dl_algorithm
            config.DL_CFG["DATASET"] = args.dl_dataset
            config.DL_CFG["MODEL_ARCH"] = args.dl_model
            config.DL_CFG["MAX_TR_ROUNDS"] = args.rounds
            config.DL_CFG["TARGET_ACCURACY"] = args.target_acc

            logger.log(f"Partitioning {args.dl_dataset} (non-IID) for {args.num_vehicles} vehicles...", "info")
            sumo_ids = [f"mv_{i}" for i in range(args.num_vehicles)]
            train_loaders, test_loader = partition_dataset(
                config.DL_CFG["DATASET"],
                args.num_vehicles,
                alpha=config.DL_CFG["DATA_ALPHA"],
                batch_size=config.DL_CFG["BATCH_SIZE"],
            )
            logger.log("Initializing DPL environment...", "info")
            dl_env = DLEnvironment(
                train_loaders,
                net_bounds,
                sumo_ids,
                test_loader=test_loader,
                event_stream=event_stream,
            )
            training_status = dl_env.get_progress_snapshot()
            logger.log(
                f"DPL ready: {args.dl_algorithm} | {args.dl_dataset}/{args.dl_model} | {args.num_vehicles} vehicles",
                "success",
            )

        dl_payload = DLPayload() if args.dl_demo else None
        dl_complete_logged = False
        dl_interval = 10.0
        last_dl_time = 0.0
        step_count = 0
        last_status_step = -1
        vehicle_states = {}
        sim_time = 0.0
        active_links = []
        new_messages = []

        # Accumulator: tracks how much sim-time we owe
        sim_accumulator = 0.0
        last_frame_time = time.perf_counter()

        def finish_dl(stop_reason, avg_loss=None, avg_acc=None, tr_round=None):
            nonlocal dl_complete_logged

            if dashboard is not None:
                dashboard.mark_simulation_done()
            if dl_complete_logged:
                return

            event_stream.publish(sim_time, "status", f"DPL complete: {stop_reason}")
            logger.log(f"DPL complete: {stop_reason}", "success")
            if avg_loss is not None and avg_acc is not None and tr_round is not None:
                logger.log(
                    f"DPL Final | Round {tr_round} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Acc: {avg_acc:.2%}",
                    "result",
                )
            dl_complete_logged = True

        def finalize_experiment_outputs():
            nonlocal plots_generated

            if plots_generated or dl_env is None:
                return

            from dl.experiment import save_and_plot_experiment

            experiment = dl_env.export_experiment({
                "scenario": args.scenario,
                "scenario_name": scenario_info["name"],
                "algorithm": config.DL_CFG["ALGORITHM"],
                "dataset": config.DL_CFG["DATASET"],
                "model": config.DL_CFG["MODEL_ARCH"],
                "num_vehicles": args.num_vehicles,
                "sim_time": sim_time,
                "args": vars(args),
            })
            saved = save_and_plot_experiment(
                experiment,
                out_root=config.OUT_DIR,
                show=True,
                block=False,
            )
            event_stream.publish(
                sim_time,
                "status",
                f"DPL plots saved to {saved['experiment_dir']}",
            )
            logger.log(f"DPL outputs saved to {saved['experiment_dir']}", "success")
            logger.log(f"Experiment pickle: {saved['pickle_path']}")
            plots_generated = True

        def run_dl_step(current_vehicle_states, current_sim_time):
            if dl_env is None or not current_vehicle_states:
                return False

            dl_info = dl_env.step(current_vehicle_states, current_sim_time)
            if dl_info["new_tr_data"]:
                logger.log(
                    f"DPL Round {dl_info['tr_round']} | "
                    f"Loss: {dl_info['avg_loss']:.4f} | "
                    f"Acc: {dl_info['avg_acc']:.2%}",
                    "result",
                )
            if dl_info["new_test_data"]:
                logger.log(
                    f"Test Round {dl_info['test_round']} | "
                    f"Loss: {dl_info['test_loss']:.4f} | "
                    f"Test Acc: {dl_info['test_acc']:.2%}",
                    "success",
                )
            if dl_info["done"]:
                finish_dl(
                    dl_info["stop_reason"],
                    avg_loss=dl_info["avg_loss"],
                    avg_acc=dl_info["avg_acc"],
                    tr_round=dl_info["tr_round"],
                )
                return True
            return False

        if dl_env is not None and dl_env.is_done():
            finish_dl(
                dl_env.get_stop_reason(),
                avg_loss=dl_env.global_loss,
                avg_acc=dl_env.global_acc,
                tr_round=dl_env.tr_round,
            )

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
                    comm.update(vehicle_states, sim_time)
                    step_count += 1
                    run_dl_step(vehicle_states, sim_time)
                else:
                    sim_accumulator += dt * speed_mult
                    # Cap to prevent spiral-of-death after lag spikes
                    sim_accumulator = min(sim_accumulator, config.SIM_STEP_LENGTH * 3)
                    while sim_accumulator >= config.SIM_STEP_LENGTH:
                        sim_accumulator -= config.SIM_STEP_LENGTH
                        vehicle_states = sumo.step()
                        sim_time = sumo.get_sim_time()
                        comm.update(vehicle_states, sim_time)
                        step_count += 1

                        if run_dl_step(vehicle_states, sim_time):
                            break

                        # DL weight exchange demo
                        if dl_payload and sim_time - last_dl_time >= dl_interval:
                            last_dl_time = sim_time
                            veh_ids = list(vehicle_states.keys())
                            if len(veh_ids) >= 2:
                                sender = random.choice(veh_ids)
                                neighbors = comm.get_neighbors(sender)
                                if neighbors:
                                    receiver = random.choice(neighbors)
                                    weights = dl_payload.dummy_weights()
                                    payload = dl_payload.serialize_weights(weights)
                                    comm.send_message(sender, receiver, "dl_weights", payload, sim_time)

                active_links = comm.get_active_links()

            if dl_env is not None:
                training_status = dl_env.get_progress_snapshot()
                if training_status["done"] and not training_status["test_running"]:
                    finalize_experiment_outputs()
            new_messages += event_stream.drain(max_items=event_drain_batch)

            # --- Render (always, at FPS rate — Clock.tick handles pacing) ---
            if not dashboard.render(
                vehicle_states,
                active_links,
                new_messages,
                sim_time,
                training_status=training_status,
            ):
                break
            new_messages = []  # clear after handing to dashboard

            # Periodic console status
            if step_count > 0 and step_count % 100 == 0 and step_count != last_status_step:
                stats = comm.get_stats()
                logger.log(
                    f"Step {step_count} | Time: {sim_time:.0f}s | "
                    f"Vehicles: {len(vehicle_states)} | "
                    f"Links: {len(active_links)} | "
                    f"Msgs sent: {stats['sent']} delivered: {stats['delivered']}",
                    "result",
                )
                last_status_step = step_count

    except FileNotFoundError as e:
        logger.log(str(e), "error")
        sys.exit(1)
    except Exception as e:
        logger.log(str(e), "error")
        import traceback

        traceback.print_exc()
    finally:
        logger.log("Cleaning up...", "info")
        if dl_env is not None:
            dl_env.executor.shutdown(wait=False)
            if dl_env.eval_executor is not None:
                dl_env.eval_executor.shutdown(wait=False)
        if dashboard:
            dashboard.cleanup()
        sumo.stop()
        logger.log("Done.", "success")


if __name__ == "__main__":
    main()
