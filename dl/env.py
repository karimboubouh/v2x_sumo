"""
dl/env.py — FL Environment for SUMO V2V Dashboard.

Manages Vehicle objects, neighbor discovery, algorithm dispatch,
and background training. Position updates come from SUMO TraCI
(not self-managed road movement).

Adapted from v2x_sim/env.py.
"""

import sys
from concurrent.futures import ThreadPoolExecutor, wait as futures_wait

import numpy as np

from algorithms import build_algorithm, LINK_SIDELINK, LINK_INTERNET
from dl.config import DL_CFG as CFG
from dl.helpers import sl_tx_cost_norm
from dl.vehicle import Vehicle


class DLEnvironment:
    """
    Orchestrates decentralized FL training for SUMO vehicles.

    Public attributes
    -----------------
    vehicles    : list of Vehicle
    tr_round    : min completed training rounds across all vehicles
    global_loss : avg current_loss across all vehicles
    global_acc  : avg current_acc across all vehicles

    Public methods
    --------------
    step(vehicle_states) -> dict : one FL step (update pos + select + aggregate + train)
    is_done() -> bool            : True when termination condition is met
    """

    def __init__(self, train_loaders: list, network_bounds: tuple, sumo_ids: list):
        """
        Args:
            train_loaders: one DataLoader per vehicle (Dirichlet-partitioned)
            network_bounds: (x_min, y_min, x_max, y_max) from SumoManager
            sumo_ids: list of SUMO managed vehicle string IDs (e.g. ["mv_0", ...])
        """
        self.step_n = 0
        self.tr_round = 0

        # Create all vehicles
        n = len(sumo_ids)
        self.vehicles = [
            Vehicle(i, sumo_ids[i], train_loaders[i], network_bounds)
            for i in range(n)
        ]

        # SUMO ID -> integer ID mapping
        self._sumo_to_int = {sid: i for i, sid in enumerate(sumo_ids)}

        # Thread pool for background training
        self.executor = ThreadPoolExecutor(max_workers=CFG["N_TRAIN_WORKERS"])

        # Build algorithm and inject into vehicles
        self.algo = build_algorithm(CFG)
        for v in self.vehicles:
            v._algo = self.algo
        self.algo.setup(self.vehicles)

        self.global_loss = float("inf")
        self.global_acc = 0.0

        # Initial synchronous training round
        futs = []
        for v in self.vehicles:
            v.training_done.clear()
            futs.append(self.executor.submit(v.train_local))
        futures_wait(futs)
        self._refresh_metrics()
        print(
            f" -> Initial FL round done — loss={self.global_loss:.4f} | "
            f"acc={self.global_acc:.2%}",
            file=sys.stderr,
        )

    # ── Topology ──────────────────────────────────────────────────────────────

    def neighbors_of(self, v: Vehicle) -> list:
        """
        Return list of (Vehicle, distance_m, link_type) for all reachable
        neighbors, combining sidelink and internet links.
        """
        v2x_range = float(CFG["V2X_RANGE"])
        inet_range = float(CFG["INTERNET_RANGE"])
        inet_thresh = float(CFG["INTERNET_QUALITY_THRESHOLD"])
        max_sl = int(CFG["MAX_NEIGHBORS"])
        max_inet = int(CFG["MAX_INTERNET_NEIGHBORS"])

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
            return np.zeros((0, CFG["NBR_DIM"]), dtype=np.float32)

        v2x_range = float(CFG["V2X_RANGE"])
        feats = []
        p_v = v.get_param_vec()
        norm_v = np.linalg.norm(p_v)

        for nbr, dist, link_type in nbrs:
            p_n = nbr.get_param_vec()
            cos_sim = float(np.clip(
                np.dot(p_v, p_n) / (norm_v * np.linalg.norm(p_n) + 1e-8),
                -1.0, 1.0,
            ))

            if link_type == LINK_SIDELINK:
                nd = float(np.clip(dist / v2x_range, 0.0, 1.0))
                tx_cost = float(sl_tx_cost_norm(dist))
            else:
                nd = 1.0
                tx_cost = 1.0

            dh = abs(v.heading - nbr.heading)
            rel_spd = float(np.clip(min(dh, 2 * np.pi - dh) / np.pi, 0.0, 1.0))
            nbr_acc = float(np.clip(nbr.current_acc, 0.0, 1.0))

            feats.append([cos_sim, nd, tx_cost, rel_spd, nbr_acc, link_type])

        return np.array(feats, dtype=np.float32)

    # ── Metrics ───────────────────────────────────────────────────────────────

    def _refresh_metrics(self):
        """Recompute global_loss, global_acc, and tr_round from all vehicles."""
        valid = [v.current_loss for v in self.vehicles
                 if np.isfinite(v.current_loss)]
        self.global_loss = float(np.mean(valid)) if valid else 0.0
        self.global_acc = float(np.mean([v.current_acc for v in self.vehicles]))
        self.tr_round = min(v.tr_rounds for v in self.vehicles)

    def is_done(self) -> bool:
        """True when either termination condition is satisfied."""
        return (
            self.tr_round >= CFG["MAX_TR_ROUNDS"]
            or self.global_acc >= CFG["TARGET_ACCURACY"]
        )

    # ── Main simulation step ──────────────────────────────────────────────────

    def step(self, vehicle_states: dict) -> dict:
        """
        Execute one FL step.

        1. Update vehicle positions from SUMO vehicle states.
        2. Discover neighbors (sidelink + internet) for dynamic algorithms.
        3. Run algorithm neighbor selection and model aggregation.
        4. Submit background training for idle vehicles.
        5. Refresh global metrics.

        Args:
            vehicle_states: dict[str, VehicleState] from SumoManager.step()

        Returns:
            dict with avg_loss, avg_acc, tr_round, new_tr_data, step.
        """
        self.step_n += 1
        prev_tr_round = self.tr_round
        transitions = {}

        # 1. Update positions from SUMO
        for v in self.vehicles:
            if v.sumo_id in vehicle_states:
                v.update_from_sumo(vehicle_states[v.sumo_id])

        # 2. Select neighbors (algorithm decides)
        for v in self.vehicles:
            candidates = (
                self.neighbors_of(v) if self.algo.needs_dynamic_neighbors else []
            )
            v.connections, v.alphas, v.link_types, t = \
                self.algo.select_neighbors(v, candidates, self)
            if t is not None:
                transitions[v.id] = t

        # 3. Aggregate neighbor models
        for v in self.vehicles:
            self.algo.aggregate(v, self.vehicles)

        # 4. Submit training for idle vehicles
        for v in self.vehicles:
            if v.training_done.is_set():
                v.training_done.clear()
                self.executor.submit(v.train_local)

        # 5. Rewards (no-op for non-RL algorithms)
        rewards = self.algo.post_step(self.vehicles, transitions, self.step_n)

        # 6. Refresh global metrics
        self._refresh_metrics()

        return {
            "rewards": rewards,
            "avg_loss": self.global_loss,
            "avg_acc": self.global_acc,
            "tr_round": self.tr_round,
            "new_tr_data": self.tr_round > prev_tr_round,
            "step": self.step_n,
        }
