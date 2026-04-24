"""
dl/env.py — DPL environment for the SUMO V2V Dashboard.

Manages Vehicle objects, neighbor discovery, algorithm dispatch,
and background training. Position updates come from SUMO TraCI
(not self-managed road movement).

Adapted from v2x_sim/env.py.
"""

import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait as futures_wait
from dataclasses import dataclass

import numpy as np
import torch

import config
import logger
from algorithms import build_algorithm, LINK_SIDELINK, LINK_INTERNET
from algorithms import get_algorithm_config
from dl.helpers import (
    eval_model_on_loader,
    eval_vehicles,
    eval_weight_snapshots,
    inet_tx_energy_j,
    inet_tx_time_s,
    sl_tx_energy_j,
    sl_tx_time_s,
    synchronize_vehicle_initial_models,
    tx_payload_bits,
)
from dl.vehicle import Vehicle


@dataclass
class CollaborationLinkOverlay:
    """Render-friendly directed FL collaboration edge."""

    sender_id: str
    receiver_id: str
    link_type: float
    alpha: float
    parallel_offset: float = 0.0


class DLEnvironment:
    """
    Orchestrates decentralized personalized learning training for SUMO vehicles.

    Public attributes
    -----------------
    vehicles    : list of Vehicle
    tr_round    : synchronized completed training round across all vehicles
    global_loss : avg current_loss across all vehicles
    global_acc  : avg current_acc across all vehicles

    Public methods
    --------------
    step(vehicle_states, sim_time) -> dict : one DPL step
    is_done() -> bool            : True when termination condition is met
    """

    def __init__(
        self,
        train_loaders: list,
        network_bounds: tuple,
        sumo_ids: list,
        train_eval_loaders=None,
        val_loaders=None,
        test_loader=None,
        local_test_loaders=None,
        event_stream=None,
    ):
        """
        Args:
            train_loaders: one training DataLoader per vehicle (Dirichlet-partitioned)
            network_bounds: (x_min, y_min, x_max, y_max) from SumoManager
            sumo_ids: list of SUMO managed vehicle string IDs (e.g. ["mv_0", ...])
        """
        self.step_n = 0
        self.tr_round = 0
        self.max_tr_round = 0
        self.round_skew = 0
        self._event_stream = event_stream
        self._last_sim_time = 0.0
        self._wall_started = time.perf_counter()
        self._round_wall_mark = self._wall_started
        self.last_round_time = 0.0
        self.algo_config = get_algorithm_config(config.ALGORITHM)
        self.eval_split = self._normalize_eval_split(
            self.algo_config.get("EVAL_SPLIT", config.EVAL_SPLIT)
        )
        self.eval_label = self._evaluation_label(self.eval_split)
        self.eval_source = None
        self.evaluation_mode = "global"
        self.reward_source = None
        self.eval_loss = None
        self.eval_loss_std = None
        self.eval_acc = None
        self.eval_acc_std = None
        self.init_eval_loss = None
        self.init_eval_loss_std = None
        self.init_eval_acc = None
        self.init_eval_acc_std = None
        self._eval_future = None
        self._eval_running_round = 0
        self._last_eval_round = 0
        self._last_eval_requested_round = 0
        self._eval_request_times = {}
        self._pending_eval_jobs = deque()
        self._scheduled_eval_rounds = set()
        self.train_history = []
        self.eval_history = []
        self.reward_history = []
        self._latest_avg_reward = 0.0
        self.async_eval = bool(getattr(config, "ASYNC_EVAL", True))

        # Create all vehicles
        n = len(sumo_ids)
        train_eval_loaders = train_eval_loaders or train_loaders
        val_loaders = val_loaders or train_eval_loaders
        local_test_loaders = local_test_loaders or [test_loader] * n
        if len(train_loaders) != n or len(train_eval_loaders) != n or len(val_loaders) != n or len(local_test_loaders) != n:
            raise ValueError("Per-vehicle train/eval loader counts must match the number of SUMO vehicle IDs.")
        self.vehicles = [
            Vehicle(
                i,
                sumo_ids[i],
                train_loaders[i],
                network_bounds,
                train_eval_loader=train_eval_loaders[i],
                val_loader=val_loaders[i],
                test_loader=local_test_loaders[i],
                event_stream=event_stream,
            )
            for i in range(n)
        ]
        self._apply_initial_model_policy()

        # Mark Byzantine adversaries (Gaussian-noise weight poisoners).
        byz_frac = float(getattr(config, "BYZANTINE_FRACTION", 0.0))
        n_byz = int(round(byz_frac * n))
        if n_byz > 0:
            import random as _random
            byz_ids = _random.sample(range(n), n_byz)
            for i in byz_ids:
                self.vehicles[i].is_byzantine = True
            logger.log(
                f"Byzantine vehicles ({n_byz}/{n}, {byz_frac*100:.0f}%): "
                + ", ".join(str(i) for i in sorted(byz_ids)),
                "warning",
            )

        # Build algorithm before configuring evaluation so test routing can
        # distinguish global-model baselines from personalized methods.
        self.algo = build_algorithm(config.ALGORITHM)
        self.evaluation_mode = self._normalize_evaluation_mode(
            getattr(self.algo, "evaluation_mode", "global")
        )
        self.reward_source = getattr(self.algo, "reward_source", None)

        for vehicle in self.vehicles:
            if self.eval_split == "train":
                vehicle.eval_loader = vehicle.train_eval_loader
            elif self.eval_split == "validation":
                vehicle.eval_loader = vehicle.val_loader
            else:
                vehicle.eval_loader = (
                    vehicle.test_loader
                    if self.evaluation_mode == "personalized"
                    else test_loader
                )

        if self.eval_split == "test":
            self.eval_source = (
                [vehicle.test_loader for vehicle in self.vehicles]
                if self.evaluation_mode == "personalized"
                else test_loader
            )
        else:
            self.eval_source = [vehicle.eval_loader for vehicle in self.vehicles]

        # SUMO ID -> integer ID mapping
        self._sumo_to_int = {sid: i for i, sid in enumerate(sumo_ids)}

        # Thread pool for background training
        self.executor = ThreadPoolExecutor(max_workers=config.N_TRAIN_WORKERS)
        self.eval_executor = (
            ThreadPoolExecutor(max_workers=1)
            if self.eval_source is not None and self.async_eval
            else None
        )

        # Inject the active algorithm into each vehicle now that evaluation is configured.
        for v in self.vehicles:
            v._algo = self.algo
        self.algo.setup(self.vehicles)

        # ── Baseline metrics (before any training) ───────────────────────────
        # Evaluate each vehicle's random-init model on one batch using a
        # temporary iterator so _inf_iter is not advanced.
        for v in self.vehicles:
            v.model.eval()
            with torch.no_grad():
                images, labels = next(iter(v.train_loader))
                logits = v.model(images)
                v.current_loss = v.criterion(logits, labels).item()
                v.current_acc = (
                    (logits.argmax(1) == labels).sum().item() / len(labels)
                )
                v.current_reward_loss = v.current_loss
                v.current_reward_acc = v.current_acc
                v._prev_reward_loss = v.current_loss

            if self.reward_source == "validation":
                val_metrics = eval_model_on_loader(
                    v.model,
                    v.val_loader,
                    criterion=v.criterion,
                )
                v.current_val_loss = float(val_metrics["loss"])
                v.current_val_acc = float(val_metrics["acc"])
                v._prev_val_loss = v.current_val_loss
                v.current_reward_loss = v.current_val_loss
                v.current_reward_acc = v.current_val_acc
                v._prev_reward_loss = v.current_val_loss
        self.global_loss = float(np.mean([v.current_loss for v in self.vehicles]))
        self.global_acc = float(np.mean([v.current_acc for v in self.vehicles]))
        self.train_history.append({
            "round": 0,
            "time": 0.0,
            "loss": self.global_loss,
            "acc": self.global_acc,
            **self._collect_energy_totals(),
        })
        _init_eval = ""
        if self.eval_source is not None:
            init_eval = eval_vehicles(self.vehicles, self.eval_source)
            self.init_eval_loss = init_eval["loss"]
            self.init_eval_loss_std = init_eval["loss_std"]
            self.init_eval_acc = init_eval["acc"]
            self.init_eval_acc_std = init_eval["acc_std"]
            self.eval_history.append({
                "round": 0,
                "time": 0.0,
                "loss": self.init_eval_loss,
                "loss_std": self.init_eval_loss_std,
                "acc": self.init_eval_acc,
                "acc_std": self.init_eval_acc_std,
                "n_vehicles": init_eval["n_vehicles"],
            })
            self._scheduled_eval_rounds.add(0)
            _init_eval = (
                f" | {self.eval_label} loss={self.init_eval_loss:.4f} ± {self.init_eval_loss_std:.4f}"
                f" | {self.eval_label} acc={self.init_eval_acc:.2%} ± {self.init_eval_acc_std:.2%}"
            )
        logger.log(
            f"Initial model (before training) | Loss: {self.global_loss:.4f} | "
            f"Acc: {self.global_acc:.2%}{_init_eval}",
            "result",
        )

        # ── Initial synchronous DPL training round ────────────────────────────
        futs = []
        for v in self.vehicles:
            v.prepare_training_round(0.0, [])
            v.training_done.clear()
            futs.append(self.executor.submit(v.train_local))
        futures_wait(futs)
        self._refresh_metrics()
        self._record_train_metrics()
        now = time.perf_counter()
        self.last_round_time = now - self._round_wall_mark
        self._round_wall_mark = now
        self._maybe_schedule_eval(0.0, stop_reason=self.get_stop_reason())
        self._training_rounds_done = 0  # rounds completed in main loop (excludes init)
        self._train_wall_start = time.perf_counter()  # wall clock after all init overhead

    # ── Topology ──────────────────────────────────────────────────────────────

    def _apply_initial_model_policy(self) -> None:
        """Optionally align all vehicles to one shared random initialization."""
        if bool(getattr(config, "SHARED_INITIAL_MODEL", False)):
            synchronize_vehicle_initial_models(self.vehicles)

    def neighbors_of(self, v: Vehicle) -> list:
        """
        Return list of (Vehicle, distance_m, link_type) candidates for vehicle
        ``v``, combining physical sidelink discovery and internet relay links.

        Link preference is:
        1. use sidelink for close peers when the algorithm supports sidelink
        2. otherwise fall back to internet for any remaining peer when internet
           is supported, prioritizing geographically closer peers first

        A peer appears at most once in the final candidate list.
        """
        v2x_range = float(config.COMM_RANGE)
        max_sl = max(int(getattr(self.algo, "max_sidelink_neighbors", 0)), 0)
        max_inet = max(int(getattr(self.algo, "max_internet_neighbors", 0)), 0)

        close_neighbors = []
        internet_candidates = []

        for other in self.vehicles:
            if other.id == v.id:
                continue
            dist = float(np.linalg.norm(v.pos - other.pos))

            if dist <= v2x_range and max_sl > 0:
                close_neighbors.append((other, dist))
            elif max_inet > 0:
                internet_candidates.append((other, dist))

        close_neighbors.sort(key=lambda x: x[1])
        sidelink = [
            (other, dist, LINK_SIDELINK)
            for other, dist in close_neighbors[:max_sl]
        ]

        if max_inet > 0:
            internet_candidates.extend(close_neighbors[max_sl:])
        internet_candidates.sort(key=lambda x: x[1])
        internet = [
            (other, dist, LINK_INTERNET)
            for other, dist in internet_candidates[:max_inet]
        ]

        return sidelink + internet

    def sidelink_neighbors_of(self, v: Vehicle) -> list:
        """Return only physical sidelink neighbors for the active algorithm."""
        v2x_range = float(config.COMM_RANGE)
        max_sl = int(getattr(self.algo, "max_sidelink_neighbors", 0))
        if max_sl <= 0:
            return []

        close_neighbors = []
        for other in self.vehicles:
            if other.id == v.id:
                continue
            dist = float(np.linalg.norm(v.pos - other.pos))
            if dist <= v2x_range:
                close_neighbors.append((other, dist))

        close_neighbors.sort(key=lambda x: x[1])
        return [
            (other, dist, LINK_SIDELINK)
            for other, dist in close_neighbors[:max_sl]
        ]

    def neighbor_features(self, v: Vehicle, nbrs: list) -> np.ndarray:
        """Build the algorithm-specific neighbor feature matrix from V2X beacons."""
        feature_dim = int(getattr(self.algo, "neighbor_feature_dim", 6))
        if not nbrs:
            return np.zeros((0, feature_dim), dtype=np.float32)

        v2x_range = float(config.COMM_RANGE)
        trust_score_fn = getattr(self.algo, "get_trust_score", None)
        robust_score_fn = getattr(self.algo, "get_last_robust_score", None)
        energy_cost_fn = getattr(self.algo, "feature_energy_cost", None)
        bandwidth_cost_fn = getattr(self.algo, "feature_bandwidth_cost", None)
        latency_cost_fn = getattr(self.algo, "feature_latency_cost", None)
        feats = []
        u_v = v.get_update_vec()
        update_norm_v = np.linalg.norm(u_v)
        default_energy_ref = max(
            float(inet_tx_energy_j()),
            float(sl_tx_energy_j(v2x_range)),
            1e-8,
        )
        default_bandwidth_ref = max(float(tx_payload_bits()), 1e-8)
        default_latency_ref = max(
            float(inet_tx_time_s()),
            float(sl_tx_time_s(v2x_range)),
            1e-8,
        )

        for nbr, dist, link_type in nbrs:
            u_n = nbr.get_update_vec()
            update_align = float(np.clip(
                np.dot(u_v, u_n) / (update_norm_v * np.linalg.norm(u_n) + 1e-8),
                -1.0, 1.0,
            ))
            tx_energy = (
                float(sl_tx_energy_j(dist))
                if link_type == LINK_SIDELINK
                else float(inet_tx_energy_j())
            )
            tx_latency = (
                float(sl_tx_time_s(dist))
                if link_type == LINK_SIDELINK
                else float(inet_tx_time_s())
            )
            if callable(energy_cost_fn):
                energy_cost = float(energy_cost_fn(v, link_type, dist))
            else:
                energy_cost = float(np.clip(tx_energy / default_energy_ref, 0.0, 4.0))
            if callable(latency_cost_fn):
                latency_cost = float(latency_cost_fn(v, link_type, dist))
            else:
                latency_cost = float(np.clip(tx_latency / default_latency_ref, 0.0, 4.0))

            trust = 1.0
            if callable(trust_score_fn):
                trust = float(np.clip(trust_score_fn(v, nbr.id), 0.0, 1.0))
            robust_score = 1.0
            if callable(robust_score_fn):
                robust_score = float(np.clip(robust_score_fn(v.id, nbr.id), 0.0, 1.0))
            row = [
                update_align,
                energy_cost,
                latency_cost,
                float(link_type),
                trust,
                robust_score,
            ]

            feats.append(row)

        return np.array(feats, dtype=np.float32)

    # ── Metrics ───────────────────────────────────────────────────────────────

    def _refresh_metrics(self):
        """Recompute global_loss, global_acc, and tr_round from all vehicles.

        DANTE is evaluated in communication rounds.  A reported round therefore
        advances only when every vehicle has completed that local round.  The
        scheduler below allows a small bounded lead so faster clients do not
        waste energy racing far ahead of the evaluation frontier.
        """
        valid = [v.current_loss for v in self.vehicles
                 if np.isfinite(v.current_loss)]
        self.global_loss = float(np.mean(valid)) if valid else 0.0
        self.global_acc = float(np.mean([v.current_acc for v in self.vehicles]))
        rounds = [int(v.tr_rounds) for v in self.vehicles]
        self.tr_round = min(rounds) if rounds else 0
        self.max_tr_round = max(rounds) if rounds else 0
        self.round_skew = self.max_tr_round - self.tr_round

    def _collect_energy_totals(self) -> dict:
        """Return cumulative energy totals summed across all vehicles."""
        computation = 0.0
        sidelink = 0.0
        internet = 0.0

        for vehicle in self.vehicles:
            snapshot = vehicle.get_energy_snapshot()
            computation += snapshot["computation_energy_j"]
            sidelink += snapshot["sidelink_tx_energy_j"]
            internet += snapshot["internet_tx_energy_j"]

        return {
            "computation_energy_j": computation,
            "sidelink_tx_energy_j": sidelink,
            "internet_tx_energy_j": internet,
            "total_tx_energy_j": sidelink + internet,
        }

    def _count_active_collaboration_links(self) -> dict:
        """Return directed active collaboration-link counts by link type."""
        sidelink = 0
        internet = 0

        for vehicle in self.vehicles:
            for nid in vehicle.connections:
                if float(vehicle.alphas.get(nid, 0.0)) <= 0.0:
                    continue
                link_type = vehicle.link_types.get(nid)
                if link_type == LINK_SIDELINK:
                    sidelink += 1
                elif link_type == LINK_INTERNET:
                    internet += 1

        return {
            "sidelink_links": int(sidelink),
            "internet_links": int(internet),
            "total_links": int(sidelink + internet),
        }

    def _record_train_metrics(self) -> None:
        """Append one global training history point per completed shared round."""
        if self.train_history and self.train_history[-1]["round"] == self.tr_round:
            return

        elapsed = max(time.perf_counter() - self._wall_started, 0.0)
        energies = self._collect_energy_totals()
        links = self._count_active_collaboration_links()
        self.train_history.append({
            "round": self.tr_round,
            "time": elapsed,
            "loss": self.global_loss,
            "acc": self.global_acc,
            **links,
            **energies,
        })

    def _capture_eval_snapshot(self) -> tuple[list[dict], object]:
        """Return (weight_snapshots, loaders) for honest (non-Byzantine) vehicles.

        Byzantine vehicles are excluded so their near-random accuracy (due to
        Gaussian-noise weights) does not pull down the reported mean metric.
        The returned loaders are matched 1-to-1 with the returned snapshots.
        """
        honest = [v for v in self.vehicles if not v.is_byzantine]
        snapshots = [v.get_shared_weights() for v in honest]

        if isinstance(self.eval_source, list):
            # Personalized mode: each vehicle has its own loader — filter in lock-step.
            id_to_loader = {v.id: loader for v, loader in zip(self.vehicles, self.eval_source)}
            loaders = [id_to_loader[v.id] for v in honest]
        else:
            # Global mode: single shared loader — same loader for all honest vehicles.
            loaders = self.eval_source

        return snapshots, loaders

    def _evaluate_models(self, eval_round: int, payload: tuple) -> tuple[int, dict]:
        """Run configured evaluation from thread-safe model weight snapshots."""
        weight_snapshots, loaders = payload
        return eval_round, eval_weight_snapshots(weight_snapshots, loaders)

    def _commit_eval_result(
        self,
        eval_round: int,
        eval_metrics: dict,
        sim_time: float | None = None,
    ) -> None:
        """Store completed evaluation metrics on the environment state."""
        eval_loss = float(eval_metrics["loss"])
        eval_loss_std = float(eval_metrics["loss_std"])
        eval_acc = float(eval_metrics["acc"])
        eval_acc_std = float(eval_metrics["acc_std"])
        self.eval_loss = eval_loss
        self.eval_loss_std = eval_loss_std
        self.eval_acc = eval_acc
        self.eval_acc_std = eval_acc_std
        self._last_eval_round = eval_round
        eval_time = self._eval_request_times.pop(
            eval_round,
            max(time.perf_counter() - self._wall_started, 0.0),
        )
        if (
            not self.eval_history
            or self.eval_history[-1]["round"] != eval_round
        ):
            self.eval_history.append({
                "round": eval_round,
                "time": eval_time,
                "loss": eval_loss,
                "loss_std": eval_loss_std,
                "acc": eval_acc,
                "acc_std": eval_acc_std,
                "n_vehicles": int(eval_metrics.get("n_vehicles", len(self.vehicles))),
            })
        if sim_time is not None:
            self._emit_event(
                sim_time,
                "status",
                f"DPL {self.eval_label.lower()} metrics at round {eval_round}: "
                f"loss={eval_loss:.4f} ± {eval_loss_std:.4f}, "
                f"acc={eval_acc:.2%} ± {eval_acc_std:.2%}",
            )

    def _run_eval_now(self, eval_round: int, sim_time: float | None = None) -> None:
        """Execute one evaluation job synchronously on the current thread."""
        self._last_eval_requested_round = max(self._last_eval_requested_round, eval_round)
        self._eval_request_times[eval_round] = max(time.perf_counter() - self._wall_started, 0.0)
        if sim_time is not None:
            msg = f"running {self.eval_label.lower()} evaluation at round {eval_round}"
            self._emit_event(sim_time, "status", msg)
            logger.log(f"DPL {msg}", "info")
        eval_round, eval_metrics = self._evaluate_models(
            eval_round,
            self._capture_eval_snapshot(),
        )
        self._commit_eval_result(eval_round, eval_metrics, sim_time)

    def _dispatch_pending_eval(self, sim_time: float | None = None) -> None:
        """Start the next queued evaluation if the worker is idle."""
        if (
            self._eval_future is not None
            or self.eval_executor is None
            or not self._pending_eval_jobs
        ):
            return

        eval_round, eval_time, weight_snapshots = self._pending_eval_jobs.popleft()
        self._eval_running_round = eval_round
        self._last_eval_requested_round = max(self._last_eval_requested_round, eval_round)
        self._eval_request_times[eval_round] = eval_time
        self._eval_future = self.eval_executor.submit(
            self._evaluate_models,
            eval_round,
            weight_snapshots,
        )
        if sim_time is not None:
            msg = f"running {self.eval_label.lower()} evaluation at round {eval_round}"
            self._emit_event(
                sim_time,
                "status",
                msg,
            )
            logger.log(f"DPL {msg}", "info")

    def _poll_eval_future(self, sim_time: float | None = None) -> None:
        """Commit completed async evaluation metrics back onto the environment."""
        if self._eval_future is None or not self._eval_future.done():
            return

        try:
            eval_round, eval_metrics = self._eval_future.result()
            self._commit_eval_result(eval_round, eval_metrics, sim_time)
        except Exception as exc:
            msg = f"DPL {self.eval_label.lower()} evaluation failed: {exc}"
            if sim_time is not None:
                self._emit_event(sim_time, "warning", msg)
            logger.log(msg, "warning")
        finally:
            self._eval_future = None
            self._eval_running_round = 0
            self._dispatch_pending_eval(sim_time)

    def _maybe_schedule_eval(self, sim_time: float, stop_reason: str | None = None) -> None:
        """Launch async evaluation every EVAL_ROUNDS and once on final stop."""
        if self.eval_source is None:
            return

        eval_every = max(int(config.EVAL_ROUNDS), 1)
        next_due_round = (
            ((self._last_eval_requested_round // eval_every) + 1) * eval_every
            if self._last_eval_requested_round > 0
            else eval_every
        )
        if not self.async_eval:
            while next_due_round <= self.tr_round:
                if next_due_round not in self._scheduled_eval_rounds:
                    self._scheduled_eval_rounds.add(next_due_round)
                    self._run_eval_now(next_due_round, sim_time)
                next_due_round += eval_every

            if stop_reason is not None and self.tr_round not in self._scheduled_eval_rounds:
                self._scheduled_eval_rounds.add(self.tr_round)
                self._run_eval_now(self.tr_round, sim_time)
            return

        while next_due_round <= self.tr_round:
            if next_due_round not in self._scheduled_eval_rounds:
                self._scheduled_eval_rounds.add(next_due_round)
                self._pending_eval_jobs.append((
                    next_due_round,
                    max(time.perf_counter() - self._wall_started, 0.0),
                    self._capture_eval_snapshot(),
                ))
            next_due_round += eval_every

        if stop_reason is not None and self.tr_round not in self._scheduled_eval_rounds:
            self._scheduled_eval_rounds.add(self.tr_round)
            self._pending_eval_jobs.append((
                self.tr_round,
                max(time.perf_counter() - self._wall_started, 0.0),
                self._capture_eval_snapshot(),
            ))

        self._dispatch_pending_eval(sim_time)

    def get_progress_snapshot(self) -> dict:
        """Return a render-safe DPL progress summary for the dashboard."""
        self._poll_eval_future(self._last_sim_time)
        self._refresh_metrics()
        stop_reason = self.get_stop_reason()
        if stop_reason is not None:
            self._maybe_schedule_eval(self._last_sim_time, stop_reason=stop_reason)
            self._poll_eval_future(self._last_sim_time)

        max_rounds = max(int(config.MAX_TR_ROUNDS), 1)
        elapsed = max(time.perf_counter() - self._wall_started, 0.0)
        avg_round_time = elapsed / max(self.tr_round, 1)
        rounds_remaining = max(max_rounds - self.tr_round, 0)
        # ETA uses post-init throughput for the synchronized communication-round
        # metric shown in the progress bar.
        if self._training_rounds_done > 0:
            training_elapsed = max(time.perf_counter() - self._train_wall_start, 1e-6)
            eta_round_time = training_elapsed / self._training_rounds_done
        else:
            eta_round_time = avg_round_time
        active_trainers = sum(not v.training_done.is_set() for v in self.vehicles)
        done_vehicles = sum(self._vehicle_is_done(v) for v in self.vehicles)
        stop_reason = self.get_stop_reason()
        energies = self._collect_energy_totals()
        links = self._count_active_collaboration_links()
        current_eval_loss = self.eval_loss if self.eval_loss is not None else self.init_eval_loss
        current_eval_loss_std = (
            self.eval_loss_std if self.eval_loss_std is not None else self.init_eval_loss_std
        )
        current_eval_acc = self.eval_acc if self.eval_acc is not None else self.init_eval_acc
        current_eval_acc_std = (
            self.eval_acc_std if self.eval_acc_std is not None else self.init_eval_acc_std
        )
        current_eval_round = (
            self._last_eval_round
            if current_eval_loss is not None or current_eval_acc is not None
            else None
        )
        target_metrics = self._target_reach_summary()

        return {
            "enabled": True,
            "algorithm": str(self.algo),
            "round": self.tr_round,
            "max_vehicle_round": self.max_tr_round,
            "round_skew": self.round_skew,
            "max_rounds": max_rounds,
            "progress": min(self.tr_round / max_rounds, 1.0),
            "round_time": self.last_round_time or avg_round_time,
            "avg_round_time": avg_round_time,
            "estimated_round_time": eta_round_time,
            "elapsed_time": elapsed,
            "remaining_time": eta_round_time * rounds_remaining,
            "rounds_remaining": rounds_remaining,
            "train_loss": self.global_loss,
            "train_acc": self.global_acc,
            "eval_split": self.eval_split,
            "eval_label": self.eval_label,
            "evaluation_mode": self.evaluation_mode,
            "reward_source": self.reward_source,
            "eval_loss": current_eval_loss,
            "eval_loss_std": current_eval_loss_std,
            "eval_acc": current_eval_acc,
            "eval_acc_std": current_eval_acc_std,
            "eval_round": current_eval_round,
            "init_eval_loss": self.init_eval_loss,
            "init_eval_loss_std": self.init_eval_loss_std,
            "init_eval_acc": self.init_eval_acc,
            "init_eval_acc_std": self.init_eval_acc_std,
            "test_loss": current_eval_loss,
            "test_loss_std": current_eval_loss_std,
            "test_acc": current_eval_acc,
            "test_acc_std": current_eval_acc_std,
            "test_round": current_eval_round,
            "init_test_loss": self.init_eval_loss,
            "init_test_loss_std": self.init_eval_loss_std,
            "init_test_acc": self.init_eval_acc,
            "init_test_acc_std": self.init_eval_acc_std,
            "eval_every": max(int(config.EVAL_ROUNDS), 1),
            "eval_running": self._eval_future is not None,
            "eval_pending": bool(self._pending_eval_jobs),
            "test_running": self._eval_future is not None,
            "test_pending": bool(self._pending_eval_jobs),
            "eval_running_round": self._eval_running_round,
            "active_trainers": active_trainers,
            "done_vehicles": done_vehicles,
            "vehicle_count": len(self.vehicles),
            "target_acc": float(config.TARGET_ACCURACY),
            **target_metrics,
            "avg_reward": self._latest_avg_reward,
            "done": stop_reason is not None,
            "stop_reason": stop_reason,
            **links,
            **energies,
        }

    def export_experiment(self, metadata: dict | None = None) -> dict:
        """Build a serializable experiment bundle for saving and replotting."""
        snapshot = self.get_progress_snapshot()
        algo_diagnostics = {}
        export_diag = getattr(self.algo, "export_diagnostics", None)
        if callable(export_diag):
            algo_diagnostics = dict(export_diag())
        experiment_cfg = {
            "ALGORITHM": config.ALGORITHM,
            "MAX_TR_ROUNDS": config.MAX_TR_ROUNDS,
            "TARGET_ACCURACY": config.TARGET_ACCURACY,
            "STOP_ON": getattr(config, "STOP_ON", "rounds"),
            "SEED": getattr(config, "SEED", None),
            "EVAL_ROUNDS": config.EVAL_ROUNDS,
            "EVAL_SPLIT": self.eval_split,
            "EVALUATION_MODE": self.evaluation_mode,
            "REWARD_SOURCE": self.reward_source,
            "EVAL_BATCHES_PER_ROUND": config.EVAL_BATCHES_PER_ROUND,
            "ASYNC_EVAL": self.async_eval,
            "DATASET": config.DATASET,
            "MODEL_ARCH": config.MODEL_ARCH,
            "SHARED_INITIAL_MODEL": getattr(config, "SHARED_INITIAL_MODEL", False),
            "LOCAL_LR": config.LOCAL_LR,
            "LOCAL_LR_SCHEDULE": getattr(config, "LOCAL_LR_SCHEDULE", "constant"),
            "LOCAL_LR_MIN": getattr(config, "LOCAL_LR_MIN", config.LOCAL_LR),
            "LABEL_SMOOTHING": getattr(config, "LABEL_SMOOTHING", 0.0),
            "BATCH_SIZE": config.BATCH_SIZE,
            "BATCHES_PER_ROUND": config.BATCHES_PER_ROUND,
            "MAX_ROUND_SKEW": getattr(config, "MAX_ROUND_SKEW", 1),
            "TRAIN_AUGMENTATION_POLICY": getattr(config, "TRAIN_AUGMENTATION_POLICY", "none"),
            "CNN_DROPOUT": getattr(config, "CNN_DROPOUT", 0.0),
            "CNN_CHANNELS": getattr(config, "CNN_CHANNELS", 16),
            "CNN_HIDDEN": getattr(config, "CNN_HIDDEN", 64),
            "DATA_ALPHA": config.DATA_ALPHA,
            "VALIDATION_FRACTION": config.VALIDATION_FRACTION,
            "KAPPA": config.KAPPA,
            "CPU_FREQ_HZ": config.CPU_FREQ_HZ,
            "CPU_CYCLES_PER_SAMPLE": config.CPU_CYCLES_PER_SAMPLE,
            "COMPRESSION_RATIO": config.COMPRESSION_RATIO,
            "COMM_RANGE": config.COMM_RANGE,
            "SL_BANDWIDTH_HZ": config.SL_BANDWIDTH_HZ,
            "SL_TX_POWER_W": config.SL_TX_POWER_W,
            "SL_SNR_AT_MAX_RANGE_DB": config.SL_SNR_AT_MAX_RANGE_DB,
            "INET_BANDWIDTH_HZ": config.INET_BANDWIDTH_HZ,
            "INET_TX_POWER_W": config.INET_TX_POWER_W,
            "INET_SNR_DB": config.INET_SNR_DB,
            "N_TRAIN_WORKERS": config.N_TRAIN_WORKERS,
        }
        experiment_cfg.update(self.algo_config)
        return {
            "format_version": 3,
            "config": experiment_cfg,
            "metadata": dict(metadata or {}),
            "train_history": list(self.train_history),
            "eval_history": list(self.eval_history),
            "test_history": list(self.eval_history),
            "reward_history": list(self.reward_history),
            "summary": {
                "final_round": self.tr_round,
                "max_vehicle_round": snapshot.get("max_vehicle_round", self.tr_round),
                "round_skew": snapshot.get("round_skew", 0),
                "final_train_loss": self.global_loss,
                "final_train_acc": self.global_acc,
                "eval_split": self.eval_split,
                "eval_label": self.eval_label,
                "evaluation_mode": self.evaluation_mode,
                "reward_source": self.reward_source,
                "final_eval_loss": snapshot["eval_loss"],
                "final_eval_loss_std": snapshot["eval_loss_std"],
                "final_eval_acc": snapshot["eval_acc"],
                "final_eval_acc_std": snapshot["eval_acc_std"],
                "final_test_loss": snapshot["test_loss"],
                "final_test_loss_std": snapshot["test_loss_std"],
                "final_test_acc": snapshot["test_acc"],
                "final_test_acc_std": snapshot["test_acc_std"],
                "rounds_to_target": snapshot["rounds_to_target"],
                "wall_time_to_target_s": snapshot["wall_time_to_target_s"],
                "energy_to_target_j": snapshot["energy_to_target_j"],
                "elapsed_time": snapshot["elapsed_time"],
                "stop_reason": snapshot["stop_reason"],
            },
            "diagnostics": algo_diagnostics,
            "energy_totals": self._collect_energy_totals(),
            "vehicles": [
                {
                    "id": vehicle.id,
                    "sumo_id": vehicle.sumo_id,
                    "rounds": vehicle.tr_rounds,
                    "current_loss": vehicle.current_loss,
                    "current_acc": vehicle.current_acc,
                    "loss_hist": list(vehicle.loss_hist),
                    "acc_hist": list(vehicle.acc_hist),
                    "reward_hist": list(vehicle.reward_hist),
                    "round_time_hist": list(vehicle.round_time_hist),
                    "computation_energy_hist": list(vehicle.computation_energy_hist),
                    **vehicle.get_energy_snapshot(),
                }
                for vehicle in self.vehicles
            ],
        }

    @staticmethod
    def _normalize_eval_split(split: str) -> str:
        value = str(split).strip().lower()
        if value not in {"train", "validation", "test"}:
            raise ValueError(
                f"Unsupported evaluation split {split!r}. Expected train, validation, or test."
            )
        return value

    @staticmethod
    def _evaluation_label(split: str) -> str:
        return {
            "train": "Train Eval",
            "validation": "Validation",
            "test": "Test",
        }[split]

    @staticmethod
    def _normalize_evaluation_mode(mode: str) -> str:
        value = str(mode).strip().lower()
        if value not in {"global", "personalized"}:
            raise ValueError(
                f"Unsupported evaluation mode {mode!r}. Expected global or personalized."
            )
        return value

    def _vehicle_reached_target(self, v: Vehicle) -> bool:
        """True when a vehicle has reached the configured local target."""
        return (
            str(getattr(config, "STOP_ON", "rounds")).strip().lower() == "train_acc"
            and config.TARGET_ACCURACY <= 1.0
            and v.current_acc >= config.TARGET_ACCURACY
        )

    def _vehicle_reached_round_cap(self, v: Vehicle) -> bool:
        """True when a vehicle has reached the configured training-round cap."""
        return config.MAX_TR_ROUNDS > 0 and v.tr_rounds >= config.MAX_TR_ROUNDS

    def _vehicle_is_done(self, v: Vehicle) -> bool:
        """True when either local stop condition has been met.

        Accuracy and round count are independent stop conditions.  A finite
        round cap must remain active even when target-accuracy early stopping is
        enabled, otherwise asynchronous training can continue past ``--rounds``
        when the target is not reached.
        """
        return self._vehicle_reached_target(v) or self._vehicle_reached_round_cap(v)

    def _eval_reached_target(self) -> bool:
        """True when configured evaluation accuracy has reached the target."""
        return (
            str(getattr(config, "STOP_ON", "rounds")).strip().lower() == "eval_acc"
            and config.TARGET_ACCURACY <= 1.0
            and self.eval_acc is not None
            and float(self.eval_acc) >= float(config.TARGET_ACCURACY)
        )

    def get_stop_reason(self) -> str | None:
        """Human-readable explanation when a DPL stop condition has been met."""
        stop_mode = str(getattr(config, "STOP_ON", "rounds")).strip().lower()
        if self._eval_reached_target():
            return (
                f"{self.eval_label.lower()} reached target accuracy "
                f"({float(self.eval_acc):.2%} >= {config.TARGET_ACCURACY:.2%})"
            )

        if all(self._vehicle_reached_round_cap(v) for v in self.vehicles):
            return f"all vehicles completed {config.MAX_TR_ROUNDS} training rounds"

        if stop_mode == "train_acc" and all(self._vehicle_is_done(v) for v in self.vehicles):
            if all(self._vehicle_reached_target(v) for v in self.vehicles):
                return f"all vehicles reached target accuracy ({config.TARGET_ACCURACY:.2%})"
            return (
                f"all vehicles reached target accuracy ({config.TARGET_ACCURACY:.2%}) "
                f"or completed {config.MAX_TR_ROUNDS} training rounds"
            )
        return None

    def is_done(self) -> bool:
        """True when either termination condition is satisfied."""
        return self.get_stop_reason() is not None

    def _link_name(self, link_type: float) -> str:
        """Human-readable name for the link used between two vehicles."""
        if link_type == LINK_SIDELINK:
            return "5G sidelink"
        return "Internet"

    def _emit_event(self, sim_time: float, category: str, text: str) -> None:
        """Publish an interaction-log event if a stream is configured."""
        if self._event_stream is not None:
            self._event_stream.publish(sim_time, category, text)

    def _publish_connection_changes(
        self,
        vehicle: Vehicle,
        prev_connections: set,
        prev_link_types: dict,
        sim_time: float,
    ) -> None:
        """Emit connect/disconnect events for DPL collaboration links."""
        if self._event_stream is None:
            return
        current_connections = set(vehicle.connections)
        changed = {
            nid for nid in prev_connections & current_connections
            if prev_link_types.get(nid) != vehicle.link_types.get(nid)
        }

        removed = (prev_connections - current_connections) | changed
        added = (current_connections - prev_connections) | changed

        for nid in sorted(removed):
            peer = self.vehicles[nid].sumo_id
            self._emit_event(
                sim_time,
                "link",
                f"vehicle {vehicle.sumo_id} disconnected from vehicle {peer}",
            )

        for nid in sorted(added):
            peer = self.vehicles[nid].sumo_id
            link_name = self._link_name(vehicle.link_types.get(nid, LINK_INTERNET))
            self._emit_event(
                sim_time,
                "link",
                f"vehicle {vehicle.sumo_id} connected to vehicle {peer} via {link_name}",
            )

    def _build_peer_transfers(self, vehicle: Vehicle) -> list:
        """Describe the neighbor weights the next local round will use."""
        transfers = []
        for nid in sorted(vehicle.connections):
            if nid >= len(self.vehicles):
                continue
            peer = self.vehicles[nid]
            link_type = vehicle.link_types.get(nid, LINK_INTERNET)
            dist = float(np.linalg.norm(vehicle.pos - peer.pos))
            if link_type == LINK_SIDELINK:
                tx_energy = float(sl_tx_energy_j(dist))
            else:
                tx_energy = float(inet_tx_energy_j())
            peer.add_transmission_energy(link_type, tx_energy)
            transfers.append({
                "peer_id": peer.sumo_id,
                "size_bytes": peer.shared_weights_bytes,
                "link_name": self._link_name(link_type),
                "tx_energy_j": tx_energy,
            })
        return transfers

    def _record_reward_metrics(self, rewards: dict) -> None:
        """Append a PPO reward datapoint when one or more rounds complete."""
        if not rewards:
            return

        avg_reward = float(np.mean(list(rewards.values())))
        self._latest_avg_reward = avg_reward
        self.reward_history.append({
            "step": self.step_n,
            "time": max(time.perf_counter() - self._wall_started, 0.0),
            "reward": avg_reward,
        })

    def _target_reach_summary(self) -> dict:
        """Return rounds/time/energy-to-target metrics from evaluation history."""
        target_acc = float(config.TARGET_ACCURACY)
        summary = {
            "rounds_to_target": None,
            "wall_time_to_target_s": None,
            "energy_to_target_j": None,
        }
        if target_acc > 1.0:
            return summary

        target_eval = next(
            (
                point
                for point in sorted(self.eval_history, key=lambda p: p["round"])
                if point["acc"] >= target_acc
            ),
            None,
        )
        if target_eval is None:
            return summary

        summary["rounds_to_target"] = int(target_eval["round"])
        summary["wall_time_to_target_s"] = float(target_eval["time"])
        target_train = next(
            (
                point
                for point in sorted(self.train_history, key=lambda p: p["round"])
                if point["round"] >= target_eval["round"]
            ),
            None,
        )
        if target_train is not None:
            summary["energy_to_target_j"] = float(
                target_train.get("computation_energy_j", 0.0)
                + target_train.get("total_tx_energy_j", 0.0)
            )
        return summary

    def get_vehicle_overlays(self) -> dict:
        """Return per-vehicle visualization metadata for the dashboard map."""
        overlays = {}
        for vehicle in self.vehicles:
            overlays[vehicle.sumo_id] = {
                "accuracy": float(np.clip(vehicle.current_acc, 0.0, 1.0)),
                "byzantine": bool(vehicle.is_byzantine),
                "training_active": not vehicle.training_done.is_set(),
            }
        return overlays

    def get_collaboration_links(self) -> list:
        """Return directed FL collaboration links for map rendering."""
        overlays = []
        pair_groups = {}

        for vehicle in self.vehicles:
            for nid in sorted(vehicle.connections):
                if nid >= len(self.vehicles):
                    continue

                alpha = float(np.clip(vehicle.alphas.get(nid, 0.0), 0.0, 1.0))
                if alpha <= 0.0:
                    continue

                peer = self.vehicles[nid]
                overlay = CollaborationLinkOverlay(
                    sender_id=vehicle.sumo_id,
                    receiver_id=peer.sumo_id,
                    link_type=vehicle.link_types.get(nid, LINK_INTERNET),
                    alpha=alpha,
                )
                overlays.append(overlay)
                pair_groups.setdefault(tuple(sorted((vehicle.id, nid))), []).append(overlay)

        for group in pair_groups.values():
            if len(group) == 2:
                group[0].parallel_offset = -1.0
                group[1].parallel_offset = 1.0
            elif group:
                group[0].parallel_offset = 0.0

        return overlays

    # ── Main simulation step ──────────────────────────────────────────────────

    def step(self, vehicle_states: dict, sim_time: float) -> dict:
        """
        Execute one DPL step.

        1. Update vehicle positions from SUMO vehicle states.
        2. Let the active algorithm build its candidate neighbor set.
        3. Run algorithm neighbor selection and model aggregation.
        4. Refresh metrics and evaluate stop conditions.
        5. Submit background training for eligible idle vehicles.
        6. Return current metrics and completion status.

        Args:
            vehicle_states: dict[str, VehicleState] from SumoManager.step()
            sim_time: current SUMO simulation time in seconds

        Returns:
            dict with avg_loss, avg_acc, tr_round, new_tr_data, step, done,
            and stop_reason.
        """
        self.step_n += 1
        self._last_sim_time = float(sim_time)
        prev_eval_round = self._last_eval_round
        self._poll_eval_future(sim_time)
        prev_tr_round = self.tr_round
        transitions = {}
        rewards = {}

        # 1. Update positions from SUMO
        for v in self.vehicles:
            if v.sumo_id in vehicle_states:
                v.update_from_sumo(vehicle_states[v.sumo_id], sim_time)

        # RL algorithms whose aggregation feedback belongs to the just-finished
        # local round must consume it before the next selection can overwrite it.
        if bool(getattr(self.algo, "finalize_rewards_before_selection", False)):
            rewards.update(self.algo.post_step(self.vehicles, {}, self.step_n))

        # 2. Select neighbors (algorithm decides)
        for v in self.vehicles:
            prev_connections = set(v.connections)
            prev_link_types = dict(v.link_types)
            candidates = (
                self.algo.build_candidates(v, self)
                if self.algo.needs_dynamic_neighbors
                else []
            )
            v.connections, v.alphas, v.link_types, t = \
                self.algo.select_neighbors(v, candidates, self)
            self._publish_connection_changes(
                v,
                prev_connections,
                prev_link_types,
                sim_time,
            )
            if t is not None:
                transitions[v.id] = t

        # 3. Aggregate neighbor models
        for v in self.vehicles:
            if v.training_done.is_set() and not self._vehicle_is_done(v):
                self.algo.aggregate(v, self.vehicles)

        # 4. Refresh metrics before scheduling more work so training stops cleanly.
        self._refresh_metrics()
        if self.tr_round > prev_tr_round:
            now = time.perf_counter()
            self.last_round_time = now - self._round_wall_mark
            self._round_wall_mark = now
            self._training_rounds_done += self.tr_round - prev_tr_round
            self._record_train_metrics()
        stop_reason = self.get_stop_reason()
        self._maybe_schedule_eval(sim_time, stop_reason=stop_reason)

        if stop_reason is None:
            round_frontier = min((v.tr_rounds for v in self.vehicles), default=0)
            max_round_skew = max(int(getattr(config, "MAX_ROUND_SKEW", 1)), 1)
            for v in self.vehicles:
                if (
                    v.training_done.is_set()
                    and not self._vehicle_is_done(v)
                    and not self._vehicle_reached_round_cap(v)
                    and int(v.tr_rounds) < int(round_frontier) + max_round_skew
                ):
                    v.prepare_training_round(sim_time, self._build_peer_transfers(v))
                    v.training_done.clear()
                    self.executor.submit(v.train_local)

        # 5. Rewards (no-op for non-RL algorithms)
        post_rewards = self.algo.post_step(self.vehicles, transitions, self.step_n)
        rewards.update(post_rewards)
        self._record_reward_metrics(rewards)

        new_eval_data = self._last_eval_round > prev_eval_round

        return {
            "rewards": rewards,
            "avg_loss": self.global_loss,
            "avg_acc": self.global_acc,
            "tr_round": self.tr_round,
            "new_tr_data": self.tr_round > prev_tr_round,
            "new_eval_data": new_eval_data,
            "new_test_data": new_eval_data,
            "eval_label": self.eval_label,
            "eval_split": self.eval_split,
            "eval_acc": self.eval_acc if new_eval_data else None,
            "eval_acc_std": self.eval_acc_std if new_eval_data else None,
            "eval_loss": self.eval_loss if new_eval_data else None,
            "eval_loss_std": self.eval_loss_std if new_eval_data else None,
            "eval_round": self._last_eval_round if new_eval_data else None,
            "test_acc": self.eval_acc if new_eval_data else None,
            "test_acc_std": self.eval_acc_std if new_eval_data else None,
            "test_loss": self.eval_loss if new_eval_data else None,
            "test_loss_std": self.eval_loss_std if new_eval_data else None,
            "test_round": self._last_eval_round if new_eval_data else None,
            "step": self.step_n,
            "done": stop_reason is not None,
            "stop_reason": stop_reason,
            **self._count_active_collaboration_links(),
            "training_status": self.get_progress_snapshot(),
        }
