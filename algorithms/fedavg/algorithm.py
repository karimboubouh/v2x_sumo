"""
FedAvg — Decentralized FedAvg baseline.

Static random neighbor graph, equal-weight aggregation.
"""

import random

from algorithms.base import DLAlgorithm, LINK_INTERNET
from algorithms.fedavg.config import (
    MAX_COLLAB_NEIGHBORS,
    MAX_INTERNET_NEIGHBORS,
    MAX_SIDELINK_NEIGHBORS,
    SELF_WEIGHT,
)
from dl.helpers import clone_state_dict


class FedAvgAlgorithm(DLAlgorithm):
    """
    Decentralized FedAvg — static random neighbor graph, equal-weight aggregation.

    new_theta = (theta_own + sum_j theta_j) / (1 + |neighbors|)
    """

    name = "FedAvg"
    needs_dynamic_neighbors = False

    def __init__(self):
        self.max_sidelink_neighbors = int(MAX_SIDELINK_NEIGHBORS)
        self.max_internet_neighbors = int(MAX_INTERNET_NEIGHBORS)
        self.max_collab_neighbors = int(MAX_COLLAB_NEIGHBORS)

    def setup(self, vehicles: list) -> None:
        """Assign each vehicle up to self.max_collab_neighbors random peers."""
        max_n = int(self.max_collab_neighbors)
        for v in vehicles:
            others = [o.id for o in vehicles if o.id != v.id]
            k = min(max_n, len(others))
            v.static_neighbors = random.sample(others, k)

    def select_neighbors(self, v, candidates: list, env) -> tuple:
        """Return the fixed static neighbor set via LINK_INTERNET."""
        connections = set(v.static_neighbors)
        if connections:
            self_w = self._resolve_self_weight(len(connections))
            nbr_w = (1.0 - self_w) / len(connections)
            alphas = {nid: nbr_w for nid in connections}
        else:
            alphas = {}
        link_types = {nid: LINK_INTERNET for nid in v.static_neighbors}
        return connections, alphas, link_types, None

    def aggregate(self, v, vehicles: list) -> None:
        """Aggregate the local model with neighbor models.

        If SELF_WEIGHT is None, use classic FedAvg with equal weighting across
        the local model and all received neighbor models.

        If SELF_WEIGHT is numeric, retain that fraction of the local model and
        split the remaining mass equally across neighbors.
        """
        nbr_sds = [
            vehicles[nid].get_shared_weights()
            for nid in v.connections
            if nid < len(vehicles)
        ]
        if not nbr_sds:
            return

        self_w = self._resolve_self_weight(len(nbr_sds))
        nbr_w = (1.0 - self_w) / len(nbr_sds)

        with v._lock:
            own_sd = v.model.state_dict()
            new_sd = clone_state_dict(own_sd)
            for key in new_sd:
                if not new_sd[key].is_floating_point():
                    continue
                agg = self_w * own_sd[key].float()
                for sd in nbr_sds:
                    agg = agg + nbr_w * sd[key].float()
                new_sd[key] = agg
            v.model.load_state_dict(new_sd)

    def _resolve_self_weight(self, n_neighbors: int) -> float:
        """Resolve FedAvg self-weight with the shared personalized semantics."""
        if SELF_WEIGHT is None:
            return 1.0 / (n_neighbors + 1.0)
        self_w = float(SELF_WEIGHT)
        if not 0.0 <= self_w <= 1.0:
            raise ValueError(
                f"FedAvg SELF_WEIGHT must be None or in [0, 1], got {SELF_WEIGHT!r}"
            )
        return self_w
