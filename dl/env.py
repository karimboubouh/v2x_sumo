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
from algorithms import build_algorithm, LINK_SIDELINK, LINK_INTERNET
from algorithms import get_algorithm_config
from algorithms.gat_ppo.config import NBR_DIM
from dl.helpers import eval_vehicles, eval_weight_snapshots, inet_tx_energy_j, sl_tx_energy_j
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

    def __init__(self, train_loaders: list, network_bounds: tuple, sumo_ids: list,
                 test_loader=None, event_stream=None):
        """
        Args:
            train_loaders: one DataLoader per vehicle (Dirichlet-partitioned)
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
        self.test_loader = test_loader
        self.test_loss = None
        self.test_acc = None
        self.init_test_loss = None
        self.init_test_acc = None
        self._eval_future = None
        self._eval_running_round = 0
        self._last_eval_round = 0
        self._last_eval_requested_round = 0
        self._eval_request_times = {}
        self._pending_eval_jobs = deque()
        self._scheduled_eval_rounds = set()
        self.train_history = []
        self.test_history = []
        self.reward_history = []
        self._latest_avg_reward = 0.0

        # Create all vehicles
        n = len(sumo_ids)
        self.vehicles = [
            Vehicle(i, sumo_ids[i], train_loaders[i], network_bounds, event_stream=event_stream)
            for i in range(n)
        ]

        # SUMO ID -> integer ID mapping
        self._sumo_to_int = {sid: i for i, sid in enumerate(sumo_ids)}

        # Thread pool for background training
        self.executor = ThreadPoolExecutor(max_workers=config.N_TRAIN_WORKERS)
        self.eval_executor = ThreadPoolExecutor(max_workers=1) if test_loader is not None else None

        # Build algorithm and inject into vehicles
        self.algo = build_algorithm(config.ALGORITHM)
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
        self.global_loss = float(np.mean([v.current_loss for v in self.vehicles]))
        self.global_acc = float(np.mean([v.current_acc for v in self.vehicles]))
        self.train_history.append({
            "round": 0,
            "time": 0.0,
            "loss": self.global_loss,
            "acc": self.global_acc,
            **self._collect_energy_totals(),
        })
        _init_test = ""
        if self.test_loader is not None:
            self.init_test_loss, self.init_test_acc = eval_vehicles(self.vehicles, self.test_loader)
            self.test_history.append({
                "round": 0,
                "time": 0.0,
                "loss": self.init_test_loss,
                "acc": self.init_test_acc,
            })
            self._scheduled_eval_rounds.add(0)
            _init_test = f" | test_loss={self.init_test_loss:.4f} | test_acc={self.init_test_acc:.2%}"
        print(
            f" -> Initial model (before training) — loss={self.global_loss:.4f} | "
            f"acc={self.global_acc:.2%}{_init_test}",
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

    # ── Topology ──────────────────────────────────────────────────────────────

    def neighbors_of(self, v: Vehicle) -> list:
        """
        Return list of (Vehicle, distance_m, link_type) for all reachable
        neighbors, combining sidelink and internet links.
        """
        v2x_range = float(config.COMM_RANGE)
        inet_range = float(config.INTERNET_RANGE)
        inet_thresh = float(config.INTERNET_QUALITY_THRESHOLD)
        max_sl = int(config.MAX_NEIGHBORS)
        max_inet = int(config.MAX_INTERNET_NEIGHBORS)

        sidelink = []
        internet_candidates = []

        for other in self.vehicles:
            if other.id == v.id:
                continue
            dist = float(np.linalg.norm(v.pos - other.pos))

            if dist <= v2x_range:
                sidelink.append((other, dist, LINK_SIDELINK))
            elif dist <= inet_range:
                quality = self._link_quality(v, other)
                if quality >= inet_thresh:
                    internet_candidates.append((other, dist, quality, LINK_INTERNET))

        sidelink.sort(key=lambda x: x[1])
        sidelink = sidelink[:max_sl]

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
        return cos_sim * float(np.clip(other.current_acc, 0.0, 1.0))

    def neighbor_features(self, v: Vehicle, nbrs: list) -> np.ndarray:
        """Build the (N, NBR_DIM=6) feature matrix from V2X beacon data."""
        if not nbrs:
            return np.zeros((0, NBR_DIM), dtype=np.float32)

        v2x_range = float(config.COMM_RANGE)
        inet_range = float(config.INTERNET_RANGE)
        feats = []
        p_v = v.get_param_vec()
        norm_v = np.linalg.norm(p_v)

        for nbr, dist, link_type in nbrs:
            p_n = nbr.get_param_vec()
            cos_sim = float(np.clip(
                np.dot(p_v, p_n) / (norm_v * np.linalg.norm(p_n) + 1e-8),
                -1.0, 1.0,
            ))

            ref_range = v2x_range if link_type == LINK_SIDELINK else inet_range
            nd = float(np.clip(dist / max(ref_range, 1.0), 0.0, 1.0))
            tx_cost = nd ** 2

            dh = abs(v.heading - nbr.heading)
            rel_spd = float(np.clip(min(dh, 2 * np.pi - dh) / np.pi, 0.0, 1.0))
            nbr_acc = float(np.clip(nbr.current_acc, 0.0, 1.0))

            feats.append([cos_sim, nd, tx_cost, rel_spd, nbr_acc, link_type])

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

    def _record_train_metrics(self) -> None:
        """Append one global training history point per completed shared round."""
        if self.train_history and self.train_history[-1]["round"] == self.tr_round:
            return

        elapsed = max(time.perf_counter() - self._wall_started, 0.0)
        energies = self._collect_energy_totals()
        self.train_history.append({
            "round": self.tr_round,
            "time": elapsed,
            "loss": self.global_loss,
            "acc": self.global_acc,
            **energies,
        })

    def _capture_eval_snapshot(self) -> list[dict]:
        """Return a thread-safe weight snapshot for every vehicle."""
        return [v.get_shared_weights() for v in self.vehicles]

    def _evaluate_models(self, eval_round: int, weight_snapshots: list[dict]) -> tuple[int, float, float]:
        """Run global test evaluation from thread-safe model weight snapshots."""
        test_loss, test_acc = eval_weight_snapshots(weight_snapshots, self.test_loader)
        return eval_round, test_loss, test_acc

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
            self._emit_event(
                sim_time,
                "status",
                f"running test evaluation at round {eval_round}",
            )

    def _poll_eval_future(self, sim_time: float | None = None) -> None:
        """Commit completed async test metrics back onto the environment."""
        if self._eval_future is None or not self._eval_future.done():
            return

        try:
            eval_round, test_loss, test_acc = self._eval_future.result()
            self.test_loss = test_loss
            self.test_acc = test_acc
            self._last_eval_round = eval_round
            eval_time = self._eval_request_times.pop(
                eval_round,
                max(time.perf_counter() - self._wall_started, 0.0),
            )
            if (
                not self.test_history
                or self.test_history[-1]["round"] != eval_round
            ):
                self.test_history.append({
                    "round": eval_round,
                    "time": eval_time,
                    "loss": test_loss,
                    "acc": test_acc,
                })
            if sim_time is not None:
                self._emit_event(
                    sim_time,
                    "status",
                    f"DPL test metrics at round {eval_round}: "
                    f"loss={test_loss:.4f}, acc={test_acc:.2%}",
                )
        except Exception as exc:
            if sim_time is not None:
                self._emit_event(sim_time, "warning", f"DPL evaluation failed: {exc}")
        finally:
            self._eval_future = None
            self._eval_running_round = 0
            self._dispatch_pending_eval(sim_time)

    def _maybe_schedule_eval(self, sim_time: float, stop_reason: str | None = None) -> None:
        """Launch async evaluation every EVAL_ROUNDS and once on final stop."""
        if self.test_loader is None or self.eval_executor is None:
            return

        eval_every = max(int(config.EVAL_ROUNDS), 1)
        next_due_round = (
            ((self._last_eval_requested_round // eval_every) + 1) * eval_every
            if self._last_eval_requested_round > 0
            else eval_every
        )
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
        active_trainers = sum(not v.training_done.is_set() for v in self.vehicles)
        done_vehicles = sum(self._vehicle_is_done(v) for v in self.vehicles)
        stop_reason = self.get_stop_reason()
        energies = self._collect_energy_totals()

        return {
            "enabled": True,
            "algorithm": str(self.algo),
            "round": self.tr_round,
            "max_rounds": max_rounds,
            "progress": min(self.tr_round / max_rounds, 1.0),
            "round_time": self.last_round_time or avg_round_time,
            "avg_round_time": avg_round_time,
            "elapsed_time": elapsed,
            "remaining_time": avg_round_time * rounds_remaining,
            "rounds_remaining": rounds_remaining,
            "train_loss": self.global_loss,
            "train_acc": self.global_acc,
            "test_loss": self.test_loss,
            "test_acc": self.test_acc,
            "test_round": self._last_eval_round,
            "init_test_loss": self.init_test_loss,
            "init_test_acc": self.init_test_acc,
            "eval_every": max(int(config.EVAL_ROUNDS), 1),
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
            "DATASET": config.DATASET,
            "MODEL_ARCH": config.MODEL_ARCH,
            "LOCAL_LR": config.LOCAL_LR,
            "BATCH_SIZE": config.BATCH_SIZE,
            "BATCHES_PER_ROUND": config.BATCHES_PER_ROUND,
            "DATA_ALPHA": config.DATA_ALPHA,
            "KAPPA": config.KAPPA,
            "CPU_FREQ_HZ": config.CPU_FREQ_HZ,
            "CPU_CYCLES_PER_SAMPLE": config.CPU_CYCLES_PER_SAMPLE,
            "COMPRESSION_RATIO": config.COMPRESSION_RATIO,
            "COMM_RANGE": config.COMM_RANGE,
            "MAX_NEIGHBORS": config.MAX_NEIGHBORS,
            "INTERNET_RANGE": config.INTERNET_RANGE,
            "MAX_INTERNET_NEIGHBORS": config.MAX_INTERNET_NEIGHBORS,
            "INTERNET_QUALITY_THRESHOLD": config.INTERNET_QUALITY_THRESHOLD,
            "SL_BANDWIDTH_HZ": config.SL_BANDWIDTH_HZ,
            "SL_TX_POWER_W": config.SL_TX_POWER_W,
            "SL_SNR_AT_MAX_RANGE_DB": config.SL_SNR_AT_MAX_RANGE_DB,
            "INET_BANDWIDTH_HZ": config.INET_BANDWIDTH_HZ,
            "INET_TX_POWER_W": config.INET_TX_POWER_W,
            "INET_SNR_DB": config.INET_SNR_DB,
            "N_TRAIN_WORKERS": config.N_TRAIN_WORKERS,
        }
        experiment_cfg.update(get_algorithm_config(config.ALGORITHM))
        return {
            "format_version": 1,
            "config": experiment_cfg,
            "metadata": dict(metadata or {}),
            "train_history": list(self.train_history),
            "test_history": list(self.test_history),
            "reward_history": list(self.reward_history),
            "summary": {
                "final_round": self.tr_round,
                "final_train_loss": self.global_loss,
                "final_train_acc": self.global_acc,
                "final_test_loss": self.test_loss,
                "final_test_acc": self.test_acc,
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

        new_test_data = self._last_eval_round > prev_eval_round

        return {
            "rewards": rewards,
            "avg_loss": self.global_loss,
            "avg_acc": self.global_acc,
            "tr_round": self.tr_round,
            "new_tr_data": self.tr_round > prev_tr_round,
            "new_test_data": new_test_data,
            "test_acc": self.test_acc if new_test_data else None,
            "test_loss": self.test_loss if new_test_data else None,
            "test_round": self._last_eval_round if new_test_data else None,
            "step": self.step_n,
            "done": stop_reason is not None,
            "stop_reason": stop_reason,
            "training_status": self.get_progress_snapshot(),
        }
