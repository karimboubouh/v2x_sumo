"""
dl/env.py — DPL environment for the SUMO V2V Dashboard.

Manages Vehicle objects, neighbor discovery, algorithm dispatch,
and background training. Position updates come from SUMO TraCI
(not self-managed road movement).

Adapted from v2x_sim/env.py.
"""

import sys
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
from algorithms.dante.config import NBR_DIM, TYPICAL_ROUND_ENERGY_J
from dl.helpers import (
    eval_model_on_loader,
    eval_vehicles,
    eval_weight_snapshots,
    inet_tx_energy_j,
    sl_tx_energy_j,
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
    tr_round    : max completed training rounds across all vehicles
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
        print(
            f" -> Initial model (before training) — loss={self.global_loss:.4f} | "
            f"acc={self.global_acc:.2%}{_init_eval}",
            file=sys.stderr,
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

    def neighbors_of(self, v: Vehicle) -> list:
        """
        Return list of (Vehicle, distance_m, link_type) for all reachable
        neighbors, combining sidelink and internet links.

        Link preference is:
        1. use sidelink for close peers when the algorithm supports sidelink
        2. otherwise fall back to internet for the same peer when internet is
           supported and the peer passes the internet-quality filter

        A peer appears at most once in the final candidate list.
        """
        v2x_range = float(config.COMM_RANGE)
        inet_range = float(config.INTERNET_RANGE)
        inet_thresh = float(config.INTERNET_QUALITY_THRESHOLD)
        max_sl = int(getattr(self.algo, "max_sidelink_neighbors", 0))
        max_inet = int(getattr(self.algo, "max_internet_neighbors", 0))

        close_neighbors = []
        internet_candidates = []

        for other in self.vehicles:
            if other.id == v.id:
                continue
            dist = float(np.linalg.norm(v.pos - other.pos))
            if dist > max(v2x_range, inet_range):
                continue

            if dist <= v2x_range and max_sl > 0:
                close_neighbors.append((other, dist))
            elif dist <= inet_range and max_inet > 0:
                quality = self._link_quality(v, other)
                if quality >= inet_thresh:
                    internet_candidates.append((other, dist, quality, LINK_INTERNET))

        close_neighbors.sort(key=lambda x: x[1])
        sidelink = [
            (other, dist, LINK_SIDELINK)
            for other, dist in close_neighbors[:max(max_sl, 0)]
        ]

        close_overflow = (
            close_neighbors[max_sl:]
            if max_sl > 0
            else close_neighbors
        )
        if max_inet > 0:
            for other, dist in close_overflow:
                if dist > inet_range:
                    continue
                quality = self._link_quality(v, other)
                if quality >= inet_thresh:
                    internet_candidates.append((other, dist, quality, LINK_INTERNET))

        internet_candidates.sort(key=lambda x: x[2], reverse=True)
        internet = [
            (other, dist, LINK_INTERNET)
            for other, dist, _, lt in internet_candidates[:max_inet]
        ]

        return sidelink + internet

    def _link_quality(self, v: Vehicle, other: Vehicle) -> float:
        """
        Quality score: cosine_similarity(first_layer_params) * accuracy_other.
        """
        p_v = v.get_param_vec()
        p_o = other.get_param_vec()
        cos_sim = float(np.clip(
            np.dot(p_v, p_o) / (np.linalg.norm(p_v) * np.linalg.norm(p_o) + 1e-8),
            0.0, 1.0,
        ))
        return cos_sim * float(np.clip(other.current_reward_acc, 0.0, 1.0))

    def neighbor_features(self, v: Vehicle, nbrs: list) -> np.ndarray:
        """Build the (N, NBR_DIM=7) feature matrix from V2X beacon data."""
        if not nbrs:
            return np.zeros((0, NBR_DIM), dtype=np.float32)

        v2x_range = float(config.COMM_RANGE)
        inet_range = float(config.INTERNET_RANGE)
        feats = []
        p_v = v.get_param_vec()
        g_v = v.get_grad_vec()
        norm_v = np.linalg.norm(p_v)
        grad_norm_v = np.linalg.norm(g_v)
        typical_energy = max(float(TYPICAL_ROUND_ENERGY_J), 1e-8)

        for nbr, dist, link_type in nbrs:
            p_n = nbr.get_param_vec()
            g_n = nbr.get_grad_vec()
            cos_sim = float(np.clip(
                np.dot(p_v, p_n) / (norm_v * np.linalg.norm(p_n) + 1e-8),
                -1.0, 1.0,
            ))
            grad_align = float(np.clip(
                np.dot(g_v, g_n) / (grad_norm_v * np.linalg.norm(g_n) + 1e-8),
                -1.0, 1.0,
            ))

            ref_range = v2x_range if link_type == LINK_SIDELINK else inet_range
            nd = float(np.clip(dist / max(ref_range, 1.0), 0.0, 1.0))
            tx_energy = float(sl_tx_energy_j(dist)) if link_type == LINK_SIDELINK else float(inet_tx_energy_j())
            tx_cost = float(np.clip(tx_energy / typical_energy, 0.0, 4.0))

            dh = abs(v.heading - nbr.heading)
            rel_spd = float(np.clip(min(dh, 2 * np.pi - dh) / np.pi, 0.0, 1.0))
            nbr_acc = float(np.clip(nbr.current_reward_acc, 0.0, 1.0))

            feats.append([cos_sim, grad_align, nd, tx_cost, rel_spd, nbr_acc, link_type])

        return np.array(feats, dtype=np.float32)

    # ── Metrics ───────────────────────────────────────────────────────────────

    def _refresh_metrics(self):
        """Recompute global_loss, global_acc, and tr_round from all vehicles.

        tr_round uses max so the status bar advances whenever *any* vehicle
        completes a new round, rather than being pinned to the slowest vehicle.
        """
        valid = [v.current_loss for v in self.vehicles
                 if np.isfinite(v.current_loss)]
        self.global_loss = float(np.mean(valid)) if valid else 0.0
        self.global_acc = float(np.mean([v.current_acc for v in self.vehicles]))
        self.tr_round = max(v.tr_rounds for v in self.vehicles)

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

        max_rounds = max(int(config.MAX_TR_ROUNDS), 1)
        elapsed = max(time.perf_counter() - self._wall_started, 0.0)
        avg_round_time = elapsed / max(self.tr_round, 1)
        rounds_remaining = max(max_rounds - self.tr_round, 0)
        # ETA uses post-init training throughput for the same frontier-round metric
        # shown in the progress bar. This is a heuristic because "round" here is
        # the max completed round across vehicles, not a synchronized all-vehicles
        # barrier round.
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

        return {
            "enabled": True,
            "algorithm": str(self.algo),
            "round": self.tr_round,
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
            "avg_reward": self._latest_avg_reward,
            "done": stop_reason is not None,
            "stop_reason": stop_reason,
            **links,
            **energies,
        }

    def export_experiment(self, metadata: dict | None = None) -> dict:
        """Build a serializable experiment bundle for saving and replotting."""
        snapshot = self.get_progress_snapshot()
        experiment_cfg = {
            "ALGORITHM": config.ALGORITHM,
            "MAX_TR_ROUNDS": config.MAX_TR_ROUNDS,
            "TARGET_ACCURACY": config.TARGET_ACCURACY,
            "EVAL_ROUNDS": config.EVAL_ROUNDS,
            "EVAL_SPLIT": self.eval_split,
            "EVALUATION_MODE": self.evaluation_mode,
            "REWARD_SOURCE": self.reward_source,
            "EVAL_BATCHES_PER_ROUND": config.EVAL_BATCHES_PER_ROUND,
            "ASYNC_EVAL": self.async_eval,
            "DATASET": config.DATASET,
            "MODEL_ARCH": config.MODEL_ARCH,
            "LOCAL_LR": config.LOCAL_LR,
            "BATCH_SIZE": config.BATCH_SIZE,
            "BATCHES_PER_ROUND": config.BATCHES_PER_ROUND,
            "DATA_ALPHA": config.DATA_ALPHA,
            "VALIDATION_FRACTION": config.VALIDATION_FRACTION,
            "KAPPA": config.KAPPA,
            "CPU_FREQ_HZ": config.CPU_FREQ_HZ,
            "CPU_CYCLES_PER_SAMPLE": config.CPU_CYCLES_PER_SAMPLE,
            "COMPRESSION_RATIO": config.COMPRESSION_RATIO,
            "COMM_RANGE": config.COMM_RANGE,
            "INTERNET_RANGE": config.INTERNET_RANGE,
            "INTERNET_QUALITY_THRESHOLD": config.INTERNET_QUALITY_THRESHOLD,
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
                "elapsed_time": snapshot["elapsed_time"],
                "stop_reason": snapshot["stop_reason"],
            },
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

    def _vehicle_is_done(self, v: Vehicle) -> bool:
        """True when a vehicle has hit its local training stop condition.

        Modes are mutually exclusive:
          TARGET_ACCURACY < 1.0  → accuracy mode: stop on accuracy, ignore MAX_TR_ROUNDS
          TARGET_ACCURACY ≥ 1.0  → rounds mode:   stop on MAX_TR_ROUNDS, ignore accuracy
          MAX_TR_ROUNDS = 0      → no round cap (only meaningful in rounds mode; in
                                   accuracy mode MAX_TR_ROUNDS is already ignored)
        """
        if config.TARGET_ACCURACY < 1.0:
            return v.current_acc >= config.TARGET_ACCURACY
        if config.MAX_TR_ROUNDS > 0:
            return v.tr_rounds >= config.MAX_TR_ROUNDS
        return False  # both sentinels disabled — never auto-stops

    def get_stop_reason(self) -> str | None:
        """Human-readable explanation when a DPL stop condition has been met."""
        if all(self._vehicle_is_done(v) for v in self.vehicles):
            if config.TARGET_ACCURACY < 1.0:
                return f"all vehicles reached target accuracy ({config.TARGET_ACCURACY:.2%})"
            return f"all vehicles completed {config.MAX_TR_ROUNDS} training rounds"
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
        2. Discover neighbors (sidelink + internet) for dynamic algorithms.
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

        # 1. Update positions from SUMO
        for v in self.vehicles:
            if v.sumo_id in vehicle_states:
                v.update_from_sumo(vehicle_states[v.sumo_id], sim_time)

        # 2. Select neighbors (algorithm decides)
        for v in self.vehicles:
            prev_connections = set(v.connections)
            prev_link_types = dict(v.link_types)
            candidates = (
                self.neighbors_of(v) if self.algo.needs_dynamic_neighbors else []
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
            for v in self.vehicles:
                if v.training_done.is_set() and not self._vehicle_is_done(v):
                    v.prepare_training_round(sim_time, self._build_peer_transfers(v))
                    v.training_done.clear()
                    self.executor.submit(v.train_local)

        # 5. Rewards (no-op for non-RL algorithms)
        rewards = self.algo.post_step(self.vehicles, transitions, self.step_n)
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
