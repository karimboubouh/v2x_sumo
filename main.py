#!/usr/bin/env python3
"""SUMO V2V Communication Dashboard - Main entry point."""

import os
import signal
import sys
import time

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parser import parse_args

import config
import logger
from communication.comm_manager import CommManager
from event_stream import EventStream
from simulation.sumo_manager import SumoManager


def main():
    args = parse_args()
    logger.set_level(args.verbose)
    if not args.headless:
        from dashboard.app import DashboardApp
    config.COMM_RANGE = args.comm_range

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

    if config.LOG_MAX_LINES is None:
        event_stream_max = None
        event_drain_batch = 256
    else:
        event_stream_max = max(config.LOG_MAX_LINES * 80, 1024)
        event_drain_batch = max(config.LOG_MAX_LINES * 4, 256)

    event_stream = EventStream(max_events=event_stream_max)
    sumo = SumoManager(args.scenario, args.num_vehicles, args.force_speed)
    # In headless mode pass no event stream: skips all event publishing (link
    # connect/disconnect, weight serialisation, etc.) which is display-only work.
    _viz_stream = None if args.headless else event_stream
    comm = CommManager(comm_range=args.comm_range, event_stream=_viz_stream)
    dashboard = None
    dl_env = None
    running = True
    training_status = None
    plots_generated = False
    plot_windows_active = False
    last_logged_eval_round = -1
    final_eval_logged = False

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

        if not args.headless:
            dashboard = DashboardApp(net_bounds, edge_shapes, scenario_info["name"])
            dashboard.initialize()
            logger.log("Dashboard ready. Press ESC or Q to quit.", "success")
        else:
            logger.log("Running in headless mode (no dashboard).", "info")

        # ── DL initialization (only when --dl is passed) ──────────────
        dl_env = None
        if args.dl:
            from dl.data import partition_dataset
            from dl.env import DLEnvironment

            config.ALGORITHM = args.dl_algorithm
            config.DATASET = args.dl_dataset
            config.MODEL_ARCH = args.dl_model
            config.MAX_TR_ROUNDS = args.rounds
            config.TARGET_ACCURACY = args.target_acc

            logger.log(f"Partitioning {args.dl_dataset} (non-IID) for {args.num_vehicles} vehicles...", "info")
            sumo_ids = [f"mv_{i}" for i in range(args.num_vehicles)]
            train_loaders, train_eval_loaders, val_loaders, local_test_loaders, test_loader = partition_dataset(
                config.DATASET,
                args.num_vehicles,
                alpha=config.DATA_ALPHA,
                batch_size=config.BATCH_SIZE,
            )
            logger.log("Initializing DPL environment...", "info")
            dl_env = DLEnvironment(
                train_loaders,
                net_bounds,
                sumo_ids,
                train_eval_loaders=train_eval_loaders,
                val_loaders=val_loaders,
                test_loader=test_loader,
                local_test_loaders=local_test_loaders,
                event_stream=_viz_stream,
            )
            training_status = dl_env.get_progress_snapshot()
            logger.log(
                f"DPL ready: {args.dl_algorithm} | {args.dl_dataset}/{args.dl_model} | {args.num_vehicles} vehicles",
                "success",
            )
            logger.log(
                f"Evaluation: {dl_env.eval_label} split | mode={dl_env.evaluation_mode}",
                "info",
            )
            if dl_env.reward_source is not None:
                logger.log(
                    f"PPO reward source: {dl_env.reward_source}",
                    "info",
                )
            if dl_env.eval_split == "validation":
                logger.log(
                    f"Validation used: {config.VALIDATION_FRACTION:.0%} of each vehicle shard",
                    "info",
                )

        dl_complete_logged = False
        step_count = 0
        last_status_step = -1
        vehicle_states = {}
        sim_time = 0.0
        render_links = []
        log_links = []
        vehicle_overlays = None
        new_messages = []

        # Accumulator: tracks how much sim-time we owe
        sim_accumulator = 0.0
        last_frame_time = time.perf_counter()
        render_interval = 1.0 / max(config.FPS, 1)
        # Reserve a small wall-clock slice for unlimited mode so expensive
        # repaints do not directly throttle SUMO progression.
        unlimited_sim_budget = render_interval
        last_bar_update = 0.0
        logger.enable_progress_bar()

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

        def log_eval_metrics(eval_round, eval_loss, eval_acc, eval_loss_std=None, eval_acc_std=None):
            nonlocal last_logged_eval_round

            if eval_round is None or eval_loss is None or eval_acc is None:
                return
            if eval_round <= last_logged_eval_round:
                return
            last_logged_eval_round = int(eval_round)
            label = dl_env.eval_label if dl_env is not None else "Evaluation"
            loss_text = f"{eval_loss:.4f}"
            if eval_loss_std is not None:
                loss_text += f" ± {eval_loss_std:.4f}"
            acc_text = f"{eval_acc:.2%}"
            if eval_acc_std is not None:
                acc_text += f" ± {eval_acc_std:.2%}"
            logger.log(
                f"{label} Round {eval_round} | "
                f"Loss: {loss_text} | "
                f"Acc: {acc_text}",
                "warning",
            )

        def log_final_eval_metrics(training_snapshot):
            nonlocal final_eval_logged

            if final_eval_logged:
                return

            eval_round = training_snapshot.get("eval_round", training_snapshot.get("test_round"))
            eval_loss = training_snapshot.get("eval_loss", training_snapshot.get("test_loss"))
            eval_loss_std = training_snapshot.get("eval_loss_std", training_snapshot.get("test_loss_std"))
            eval_acc = training_snapshot.get("eval_acc", training_snapshot.get("test_acc"))
            eval_acc_std = training_snapshot.get("eval_acc_std", training_snapshot.get("test_acc_std"))
            if eval_round is None or eval_loss is None or eval_acc is None:
                return

            label = training_snapshot.get("eval_label", "Evaluation")
            loss_text = f"{eval_loss:.4f}"
            if eval_loss_std is not None:
                loss_text += f" ± {eval_loss_std:.4f}"
            acc_text = f"{eval_acc:.2%}"
            if eval_acc_std is not None:
                acc_text += f" ± {eval_acc_std:.2%}"
            elapsed = float(training_snapshot.get("elapsed_time", 0.0) or 0.0)
            h, rem = divmod(int(elapsed), 3600)
            m, s = divmod(rem, 60)
            time_text = f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

            comp_e = float(training_snapshot.get("computation_energy_j", 0.0) or 0.0)
            sl_e = float(training_snapshot.get("sidelink_tx_energy_j", 0.0) or 0.0)
            in_e = float(training_snapshot.get("internet_tx_energy_j", 0.0) or 0.0)
            total_e = comp_e + sl_e + in_e

            def _fmt_j(x: float) -> str:
                if x >= 1000.0:
                    return f"{x/1000.0:.2f} kJ"
                if x >= 1.0:
                    return f"{x:.2f} J"
                return f"{x*1000.0:.2f} mJ"

            logger.log(
                f"DPL Final {label} | Round {eval_round} | "
                f"Loss: {loss_text} | "
                f"Acc: {acc_text} | "
                f"Time: {time_text} | "
                f"Comp: {_fmt_j(comp_e)} | "
                f"SL: {_fmt_j(sl_e)} | "
                f"IN: {_fmt_j(in_e)} | "
                f"TL: {_fmt_j(total_e)}",
                "result",
            )
            final_eval_logged = True

        if training_status is not None:
            log_eval_metrics(
                training_status.get("eval_round", training_status.get("test_round")),
                training_status.get("eval_loss", training_status.get("test_loss")),
                training_status.get("eval_acc", training_status.get("test_acc")),
                training_status.get("eval_loss_std", training_status.get("test_loss_std")),
                training_status.get("eval_acc_std", training_status.get("test_acc_std")),
            )

        def finalize_experiment_outputs():
            nonlocal plots_generated, plot_windows_active

            if plots_generated or dl_env is None:
                return

            from dl.experiment import save_and_plot_experiment

            experiment = dl_env.export_experiment({
                "scenario": args.scenario,
                "scenario_name": scenario_info["name"],
                "algorithm": config.ALGORITHM,
                "dataset": config.DATASET,
                "model": config.MODEL_ARCH,
                "num_vehicles": args.num_vehicles,
                "sim_time": sim_time,
                "args": vars(args),
            })
            saved = save_and_plot_experiment(
                experiment,
                out_root=config.OUT_DIR,
                show=not args.headless,
                block=False,
            )
            event_stream.publish(
                sim_time,
                "status",
                f"DPL plots saved to {saved['experiment_dir']}",
            )
            logger.log(f"DPL outputs saved to {saved['experiment_dir']}", "success")
            logger.log(f"Experiment pickle: {saved['pickle_path']}")
            if saved.get("shown"):
                logger.log(f"Plot window backend: {saved['backend']}")
                plot_windows_active = True
            else:
                logger.log(
                    f"Plots were saved but not shown because backend '{saved['backend']}' is non-interactive",
                    "warning",
                )
            plots_generated = True

        def run_dl_step(current_vehicle_states, current_sim_time):
            if dl_env is None or not current_vehicle_states:
                return False

            dl_info = dl_env.step(current_vehicle_states, current_sim_time)
            if dl_info["new_tr_data"]:
                logger.log(
                    f"DPL Round {dl_info['tr_round']} | "
                    f"Loss: {dl_info['avg_loss']:.4f} | "
                    f"Acc: {dl_info['avg_acc']:.2%} | "
                    f"SL/IN: {dl_info.get('sidelink_links', 0)}/{dl_info.get('internet_links', 0)}",
                    "result",
                )
            if dl_info["new_test_data"]:
                log_eval_metrics(
                    dl_info.get("eval_round", dl_info.get("test_round")),
                    dl_info.get("eval_loss", dl_info.get("test_loss")),
                    dl_info.get("eval_acc", dl_info.get("test_acc")),
                    dl_info.get("eval_loss_std", dl_info.get("test_loss_std")),
                    dl_info.get("eval_acc_std", dl_info.get("test_acc_std")),
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

        def advance_simulation_step() -> bool:
            nonlocal vehicle_states, sim_time, step_count

            vehicle_states = sumo.step(headless=args.headless)
            sim_time = sumo.get_sim_time()
            # In headless mode skip V2V link computation (O(n²) per step) because
            # it is only needed for rendering.
            if not args.headless:
                comm.update(vehicle_states, sim_time)
            step_count += 1

            dl_done = run_dl_step(vehicle_states, sim_time)
            return dl_done

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

            # --- Simulation stepping ---
            if dashboard is None or not dashboard.paused:
                if speed_mult == 0:
                    # Unlimited: burst-advance SUMO for a bounded wall-clock slice,
                    # then do housekeeping once.  Headless uses a much larger window
                    # (no render cost) so more steps are batched per housekeeping
                    # cycle, keeping lock-acquisition rate on DL state low.
                    budget = 0.1 if args.headless else unlimited_sim_budget
                    burst_deadline = time.perf_counter() + budget
                    while True:
                        if advance_simulation_step():
                            break
                        if time.perf_counter() >= burst_deadline:
                            break
                else:
                    sim_accumulator += dt * speed_mult
                    # Cap to prevent spiral-of-death after lag spikes
                    sim_accumulator = min(sim_accumulator, config.SIM_STEP_LENGTH * 3)
                    did_step = False
                    while sim_accumulator >= config.SIM_STEP_LENGTH:
                        sim_accumulator -= config.SIM_STEP_LENGTH
                        did_step = True
                        if advance_simulation_step():
                            break
                    # Headless: sleep when no step is due to avoid a busy-wait
                    # loop that hammers DL training locks at millions of Hz.
                    if args.headless and not did_step:
                        time.sleep(0.001)

                log_links = comm.get_active_links()

            if dl_env is not None:
                training_status = dl_env.get_progress_snapshot()
                # Collaboration links and vehicle overlays are purely for map
                # rendering — skip them in headless mode to avoid the per-step
                # loop over all vehicles and their connection lists.
                if args.headless:
                    render_links = []
                    vehicle_overlays = None
                else:
                    render_links = dl_env.get_collaboration_links()
                    vehicle_overlays = dl_env.get_vehicle_overlays()
                if training_status.get("eval_round", training_status.get("test_round", 0)) > last_logged_eval_round:
                    log_eval_metrics(
                        training_status.get("eval_round", training_status.get("test_round")),
                        training_status.get("eval_loss", training_status.get("test_loss")),
                        training_status.get("eval_acc", training_status.get("test_acc")),
                        training_status.get("eval_loss_std", training_status.get("test_loss_std")),
                        training_status.get("eval_acc_std", training_status.get("test_acc_std")),
                    )
                if (
                    training_status["done"]
                    and not training_status.get("eval_running", training_status["test_running"])
                    and not training_status.get("eval_pending", training_status.get("test_pending", False))
                ):
                    log_final_eval_metrics(training_status)
                    finalize_experiment_outputs()
                    if plots_generated and not plot_windows_active:
                        running = False
            else:
                render_links = log_links
                vehicle_overlays = None
            if not args.headless:
                new_messages += event_stream.drain(max_items=event_drain_batch)

            # --- Render latest state and process Qt events ---
            if not args.headless:
                if not dashboard.render(
                        vehicle_states,
                        render_links,
                        new_messages,
                        sim_time,
                        training_status=training_status,
                        vehicle_overlays=vehicle_overlays,
                        log_links=log_links,
                ):
                    break
                if plot_windows_active:
                    from dl.experiment import pump_plot_events

                    pump_plot_events()

                frame_sleep = render_interval - (time.perf_counter() - now)
                if frame_sleep > 0:
                    time.sleep(frame_sleep)
            new_messages = []  # clear after handing to dashboard (or draining)

            # --- Console progress bar (throttled to ~10 Hz) ---
            if now - last_bar_update >= 0.1:
                logger.update_progress_bar(
                    training_status, sim_time,
                    len(vehicle_states), len(render_links), step_count,
                )
                last_bar_update = now

            # Periodic console status
            if step_count > 0 and step_count % 100 == 0 and step_count != last_status_step:
                stats = comm.get_stats()
                logger.log(
                    f"Step {step_count} | Time: {sim_time:.0f}s | "
                    f"Vehicles: {len(vehicle_states)} | "
                    f"Links: {len(render_links)} | "
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
        logger.clear_progress_bar()
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
