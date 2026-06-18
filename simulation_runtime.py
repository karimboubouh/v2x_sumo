"""Background simulation runtime for SUMO, communication, and DPL orchestration."""

from __future__ import annotations

import threading
import time
import traceback
from typing import Callable

import config
import logger
from runtime_state import FrameSnapshot, LatestFrameBuffer, PerfStats


class SimulationRuntime:
    """Owns the simulator and publishes latest-frame snapshots for the GUI."""

    def __init__(
        self,
        args,
        scenario_info: dict,
        event_stream,
        metadata_builder: Callable[[float | None], dict],
    ) -> None:
        self.args = args
        self.scenario_info = scenario_info
        self.event_stream = event_stream
        self._metadata_builder = metadata_builder
        self._viz_stream = None if args.headless else event_stream

        self.frame_buffer = LatestFrameBuffer()
        self.perf = PerfStats(bool(getattr(args, "perf_log", False)))

        self._stop_event = threading.Event()
        self._paused = False
        self._pause_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._prepared = False
        self._cleaned = False
        self._exception: BaseException | None = None

        self.sumo = None
        self.comm = None
        self.dl_env = None
        self.net_bounds = None
        self.edge_shapes = None

        self.training_status = None
        self.vehicle_states = {}
        self.render_links = []
        self.log_links = []
        self.vehicle_overlays = None
        self.sim_time = 0.0
        self.step_count = 0

        self._dl_initialized = False
        self._dl_complete_logged = False
        self._plots_generated = False
        self._last_logged_eval_round = -1
        self._final_eval_logged = False
        self._last_status_step = -1
        self._last_training_snapshot_at = 0.0
        self._last_overlay_snapshot_at = 0.0
        self._last_frame_publish_at = 0.0

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def prepare(self) -> tuple[tuple, list]:
        """Start SUMO and return static geometry needed by the dashboard."""
        if self._prepared:
            return self.net_bounds, self.edge_shapes

        from communication.comm_manager import CommManager
        from simulation.sumo_manager import SumoManager

        logger.log("Starting SUMO simulation...", "info")
        self.sumo = SumoManager(
            self.args.scenario,
            self.args.num_vehicles,
            self.args.force_speed,
        )
        self.sumo.start()
        self.comm = CommManager(
            comm_range=self.args.comm_range,
            event_stream=self._viz_stream,
        )
        self.net_bounds = self.sumo.get_network_bounds()
        self.edge_shapes = self.sumo.get_edge_shapes()
        logger.log(f"Network bounds: {self.net_bounds}")
        logger.log(f"Road segments: {len(self.edge_shapes)}")
        self._prepared = True
        self._publish_frame(force=True)
        return self.net_bounds, self.edge_shapes

    def start_background(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self.run_forever,
            name="SimulationRuntime",
            daemon=True,
        )
        self._thread.start()

    def run_forever(self) -> None:
        """Run the simulation loop until stopped or completed."""
        try:
            if not self._prepared:
                self.prepare()
            logger.enable_progress_bar()
            self._initialize_dl_if_needed()
            if self.dl_env is not None and self.dl_env.is_done():
                self._finish_dl(
                    self.dl_env.get_stop_reason(),
                    avg_loss=self.dl_env.global_loss,
                    avg_acc=self.dl_env.global_acc,
                    tr_round=self.dl_env.tr_round,
                )

            self._run_loop()
        except BaseException as exc:
            self._exception = exc
            logger.log(str(exc), "error")
            traceback.print_exc()
        finally:
            self.cleanup()

    def stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def exception(self) -> BaseException | None:
        return self._exception

    def set_paused(self, paused: bool) -> None:
        with self._pause_lock:
            self._paused = bool(paused)

    def _is_paused(self) -> bool:
        with self._pause_lock:
            return bool(self._paused)

    def cleanup(self) -> None:
        if self._cleaned:
            return
        self._cleaned = True
        logger.clear_progress_bar()
        logger.log("Cleaning up...", "info")
        if self.dl_env is not None:
            self.dl_env.executor.shutdown(wait=False)
            if self.dl_env.eval_executor is not None:
                self.dl_env.eval_executor.shutdown(wait=False)
        if self.sumo is not None:
            self.sumo.stop()
        logger.log("Done.", "success")

    # ── Initialization ──────────────────────────────────────────────────────

    def _initialize_dl_if_needed(self) -> None:
        if self._dl_initialized or not self.args.dl:
            return
        self._dl_initialized = True

        from dl.data import describe_train_augmentation, partition_dataset
        from dl.env import DLEnvironment

        config.ALGORITHM = self.args.dl_algorithm
        config.DATASET = self.args.dl_dataset
        config.MODEL_ARCH = self.args.dl_model
        config.MAX_TR_ROUNDS = self.args.rounds
        config.TARGET_ACCURACY = self.args.target_acc
        config.STOP_ON = self.args.stop_on

        logger.log(
            f"Partitioning {self.args.dl_dataset} (non-IID) for {self.args.num_vehicles} vehicles...",
            "info",
        )
        sumo_ids = [f"mv_{i}" for i in range(self.args.num_vehicles)]
        train_loaders, train_eval_loaders, val_loaders, local_test_loaders, test_loader = partition_dataset(
            config.DATASET,
            self.args.num_vehicles,
            alpha=config.DATA_ALPHA,
            batch_size=config.BATCH_SIZE,
        )
        logger.log("Initializing DPL environment...", "info")
        self.dl_env = DLEnvironment(
            train_loaders,
            self.net_bounds,
            sumo_ids,
            train_eval_loaders=train_eval_loaders,
            val_loaders=val_loaders,
            test_loader=test_loader,
            local_test_loaders=local_test_loaders,
            event_stream=self._viz_stream,
        )
        self._refresh_training_status(force=True)
        logger.log(
            f"DPL ready: {self.args.dl_algorithm} | {self.args.dl_dataset}/{self.args.dl_model} | "
            f"{self.args.num_vehicles} vehicles",
            "success",
        )
        logger.log(
            f"Train augmentation: {describe_train_augmentation(config.DATASET)}",
            "info",
        )
        logger.log(
            f"Evaluation: {self.dl_env.eval_label} split | mode={self.dl_env.evaluation_mode}",
            "info",
        )
        if self.dl_env.reward_source is not None:
            logger.log(f"PPO reward source: {self.dl_env.reward_source}", "info")
        if self.dl_env.eval_split == "validation":
            logger.log(
                f"Validation used: {config.VALIDATION_FRACTION:.0%} of each vehicle shard",
                "info",
            )
        if self.training_status is not None:
            self._log_eval_metrics(
                self.training_status.get("eval_round", self.training_status.get("test_round")),
                self.training_status.get("eval_loss", self.training_status.get("test_loss")),
                self.training_status.get("eval_acc", self.training_status.get("test_acc")),
                self.training_status.get("eval_loss_std", self.training_status.get("test_loss_std")),
                self.training_status.get("eval_acc_std", self.training_status.get("test_acc_std")),
            )

    # ── Main loop ───────────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        speed_mult = self.args.speed
        sim_accumulator = 0.0
        last_frame_time = time.perf_counter()
        ui_interval = 1.0 / max(int(getattr(self.args, "ui_fps", config.UI_FPS)), 1)
        unlimited_sim_budget = min(ui_interval, 0.02)
        last_bar_update = 0.0

        while not self._stop_event.is_set():
            now = time.perf_counter()
            dt = now - last_frame_time
            last_frame_time = now
            did_step = False

            if self._is_paused():
                self._service_completed_dl()
                time.sleep(0.02)
            elif speed_mult == 0:
                budget = 0.1 if self.args.headless else unlimited_sim_budget
                burst_deadline = time.perf_counter() + budget
                while not self._stop_event.is_set():
                    did_step = True
                    if self._advance_simulation_step():
                        break
                    if time.perf_counter() >= burst_deadline:
                        break
            else:
                sim_accumulator += dt * speed_mult
                sim_accumulator = min(sim_accumulator, config.SIM_STEP_LENGTH * 3)
                while sim_accumulator >= config.SIM_STEP_LENGTH and not self._stop_event.is_set():
                    sim_accumulator -= config.SIM_STEP_LENGTH
                    did_step = True
                    if self._advance_simulation_step():
                        break
                if not did_step:
                    self._service_completed_dl()

            self.log_links = self.comm.get_active_links() if self.comm is not None else []
            self._refresh_training_status(force=False)
            self._refresh_render_overlays(force=False)
            self._publish_frame(force=did_step)
            self._maybe_finalize_completed_dl()

            if now - last_bar_update >= 0.1:
                logger.update_progress_bar(
                    self.training_status,
                    self.sim_time,
                    len(self.vehicle_states),
                    len(self.render_links),
                    self.step_count,
                )
                last_bar_update = now

            if (
                self.step_count > 0
                and self.step_count % 100 == 0
                and self.step_count != self._last_status_step
            ):
                stats = self.comm.get_stats() if self.comm is not None else {"sent": 0, "delivered": 0}
                logger.log(
                    f"Step {self.step_count} | Time: {self.sim_time:.0f}s | "
                    f"Vehicles: {len(self.vehicle_states)} | "
                    f"Links: {len(self.render_links)} | "
                    f"Msgs sent: {stats['sent']} delivered: {stats['delivered']}",
                    "result",
                )
                self._last_status_step = self.step_count

            self.perf.set_latest(
                event_backlog=self.event_stream.depth() if self.event_stream is not None else 0,
                active_trainers=(self.training_status or {}).get("active_trainers", 0),
            )
            self.perf.maybe_log()

            if not did_step and not self._stop_event.is_set():
                time.sleep(0.001)

    def _advance_simulation_step(self) -> bool:
        started = time.perf_counter()
        self.vehicle_states = self.sumo.step(headless=self.args.headless)
        self.sim_time = self.sumo.get_sim_time()
        if not self.args.headless:
            self.comm.update(self.vehicle_states, self.sim_time)
        self.step_count += 1
        dl_done = self._run_dl_step(self.vehicle_states, self.sim_time)
        self.perf.record("sim_step_s", time.perf_counter() - started)
        return dl_done

    # ── DPL ─────────────────────────────────────────────────────────────────

    def _run_dl_step(self, current_vehicle_states, current_sim_time) -> bool:
        if self.dl_env is None or not current_vehicle_states:
            return False

        started = time.perf_counter()
        dl_info = self.dl_env.step(current_vehicle_states, current_sim_time)
        self.perf.record("dl_step_s", time.perf_counter() - started)

        force_snapshot = bool(
            dl_info.get("new_tr_data")
            or dl_info.get("new_test_data")
            or dl_info.get("done")
        )
        if force_snapshot:
            self._refresh_training_status(force=True)

        if dl_info["new_tr_data"]:
            logger.log(
                f"DPL Round {dl_info['tr_round']} | "
                f"Loss: {dl_info['avg_loss']:.4f} | "
                f"Acc: {dl_info['avg_acc']:.2%} | "
                f"SL/IN: {dl_info.get('sidelink_links', 0)}/{dl_info.get('internet_links', 0)}",
                "result",
            )
            ts = self.training_status or {}
            vehicle_count = max(int(ts.get("vehicle_count", len(current_vehicle_states))), 1)
            total_links = int(dl_info.get("total_links", 0))
            sl_links = int(dl_info.get("sidelink_links", 0))
            in_links = int(dl_info.get("internet_links", 0))
            avg_reward = ts.get("avg_reward")
            total_tx_energy_j = float(ts.get("total_tx_energy_j", 0.0))
            budget_j = getattr(getattr(self.dl_env, "algo", None), "round_energy_budget_j", None)
            info_parts = [
                f"Links/veh: {total_links / vehicle_count:.2f}",
                f"SL/veh: {sl_links / vehicle_count:.2f}",
                f"IN/veh: {in_links / vehicle_count:.2f}",
                f"TX energy: {total_tx_energy_j:.2f} J",
            ]
            if avg_reward is not None:
                info_parts.append(f"Avg reward: {float(avg_reward):.4f}")
            if budget_j is not None:
                info_parts.append(f"Round energy budget: {float(budget_j):.3f} J")
            info_parts.append(f"Batches/round: {config.BATCHES_PER_ROUND}")
            info_parts.append(f"Eval batches: {config.EVAL_BATCHES_PER_ROUND}")
            logger.log(" | ".join(info_parts), "info")
            consume_debug = getattr(getattr(self.dl_env, "algo", None), "consume_debug_logs", None)
            if callable(consume_debug):
                for debug_line in consume_debug():
                    logger.log(debug_line, "debug")

        if dl_info["new_test_data"]:
            self._log_eval_metrics(
                dl_info.get("eval_round", dl_info.get("test_round")),
                dl_info.get("eval_loss", dl_info.get("test_loss")),
                dl_info.get("eval_acc", dl_info.get("test_acc")),
                dl_info.get("eval_loss_std", dl_info.get("test_loss_std")),
                dl_info.get("eval_acc_std", dl_info.get("test_acc_std")),
            )

        if dl_info["done"]:
            self._finish_dl(
                dl_info["stop_reason"],
                avg_loss=dl_info["avg_loss"],
                avg_acc=dl_info["avg_acc"],
                tr_round=dl_info["tr_round"],
            )
            return True
        return False

    def _service_completed_dl(self) -> None:
        if self.dl_env is None or not self._dl_complete_logged:
            return
        service = getattr(self.dl_env, "service_background_eval", None)
        if callable(service):
            service(self.sim_time, stop_reason=self.dl_env.get_stop_reason())

    def _refresh_training_status(self, force: bool) -> None:
        if self.dl_env is None:
            return
        now = time.perf_counter()
        if not force and now - self._last_training_snapshot_at < 0.25:
            return
        started = time.perf_counter()
        self.training_status = self.dl_env.get_progress_snapshot()
        self.perf.record("snapshot_s", time.perf_counter() - started)
        self._last_training_snapshot_at = now

        eval_round = self.training_status.get("eval_round", self.training_status.get("test_round", 0))
        if eval_round is not None and eval_round > self._last_logged_eval_round:
            self._log_eval_metrics(
                eval_round,
                self.training_status.get("eval_loss", self.training_status.get("test_loss")),
                self.training_status.get("eval_acc", self.training_status.get("test_acc")),
                self.training_status.get("eval_loss_std", self.training_status.get("test_loss_std")),
                self.training_status.get("eval_acc_std", self.training_status.get("test_acc_std")),
            )

    def _refresh_render_overlays(self, force: bool) -> None:
        if self.dl_env is None:
            self.render_links = self.log_links
            self.vehicle_overlays = None
            return
        if self.args.headless:
            self.render_links = []
            self.vehicle_overlays = None
            return
        now = time.perf_counter()
        if not force and now - self._last_overlay_snapshot_at < 0.10:
            return
        self.render_links = self.dl_env.get_collaboration_links()
        self.vehicle_overlays = self.dl_env.get_vehicle_overlays()
        self._last_overlay_snapshot_at = now

    def _finish_dl(self, stop_reason, avg_loss=None, avg_acc=None, tr_round=None) -> None:
        if self._dl_complete_logged:
            return
        if self.event_stream is not None:
            self.event_stream.publish(self.sim_time, "status", f"DPL complete: {stop_reason}")
        logger.log(f"DPL complete: {stop_reason}", "success")
        if avg_loss is not None and avg_acc is not None and tr_round is not None:
            logger.log(
                f"DPL Final | Round {tr_round} | "
                f"Loss: {avg_loss:.4f} | "
                f"Acc: {avg_acc:.2%}",
                "result",
            )
        self._dl_complete_logged = True

    def _maybe_finalize_completed_dl(self) -> None:
        if self.dl_env is None or not self._dl_complete_logged or self.training_status is None:
            return
        if self._plots_generated:
            self.stop()
            return
        if (
            self.training_status["done"]
            and not self.training_status.get("eval_running", self.training_status["test_running"])
            and not self.training_status.get("eval_pending", self.training_status.get("test_pending", False))
        ):
            self._log_final_eval_metrics(self.training_status)
            self._finalize_experiment_outputs()
            self.stop()

    def _finalize_experiment_outputs(self) -> None:
        if self._plots_generated or self.dl_env is None:
            return
        from dl.experiment import save_and_plot_experiment

        experiment = self.dl_env.export_experiment(self._metadata_builder(self.sim_time))
        saved = save_and_plot_experiment(
            experiment,
            out_root=config.OUT_DIR,
            show=False,
            block=False,
        )
        if self.event_stream is not None:
            self.event_stream.publish(
                self.sim_time,
                "status",
                f"DPL plots saved to {saved['experiment_dir']}",
            )
        logger.log(f"DPL outputs saved to {saved['experiment_dir']}", "success")
        logger.log(f"Experiment pickle: {saved['pickle_path']}")
        if saved.get("log_path"):
            logger.log(f"Run log: {saved['log_path']}")
        logger.log(
            f"Plots were saved but not shown because runtime finalization runs off the Qt thread "
            f"(backend '{saved['backend']}')",
            "warning",
        )
        self._plots_generated = True

    # ── Logging helpers ─────────────────────────────────────────────────────

    def _log_eval_metrics(self, eval_round, eval_loss, eval_acc, eval_loss_std=None, eval_acc_std=None) -> None:
        if eval_round is None or eval_loss is None or eval_acc is None:
            return
        if eval_round <= self._last_logged_eval_round:
            return
        self._last_logged_eval_round = int(eval_round)
        label = self.dl_env.eval_label if self.dl_env is not None else "Evaluation"
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

    def _log_final_eval_metrics(self, training_snapshot: dict) -> None:
        if self._final_eval_logged:
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
        self._final_eval_logged = True

    # ── Frame publishing ────────────────────────────────────────────────────

    def _publish_frame(self, force: bool) -> None:
        if self.args.headless:
            return
        now = time.perf_counter()
        interval = 1.0 / max(int(getattr(self.args, "ui_fps", config.UI_FPS)), 1)
        if not force and now - self._last_frame_publish_at < interval:
            return
        frame = FrameSnapshot(
            vehicle_states=self.vehicle_states,
            render_links=self.render_links,
            log_links=self.log_links,
            training_status=self.training_status,
            vehicle_overlays=self.vehicle_overlays,
            sim_time=self.sim_time,
            step_count=self.step_count,
            simulation_done=self._dl_complete_logged,
            overlay_text="SIMULATION DONE" if self._dl_complete_logged else None,
        )
        self.frame_buffer.publish(frame)
        self._last_frame_publish_at = now
