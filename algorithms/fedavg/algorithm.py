"""
FedAvg — Decentralized FedAvg baseline.

Static random neighbor graph, equal-weight aggregation.
Adapted from v2x_sim/algorithms/fedavg/algorithm.py.
"""

import random

from algorithms.base import DLAlgorithm, LINK_INTERNET
from algorithms.fedavg.config import SELF_WEIGHT
import config
from dl.helpers import clone_state_dict


class FedAvgAlgorithm(DLAlgorithm):
    """
    Decentralized FedAvg — static random neighbor graph, equal-weight aggregation.

    new_theta = (theta_own + sum_j theta_j) / (1 + |neighbors|)
    """

    name = "FedAvg"
    needs_dynamic_neighbors = False

    def setup(self, vehicles: list) -> None:
        """Assign each vehicle MAX_NEIGHBORS randomly chosen peers."""
        max_n = int(config.MAX_NEIGHBORS)
        for v in vehicles:
            others = [o.id for o in vehicles if o.id != v.id]
            k = min(max_n, len(others))
            v.static_neighbors = random.sample(others, k)

    def select_neighbors(self, v, candidates: list, env) -> tuple:
        """Return the fixed static neighbor set via LINK_INTERNET."""
        connections = set(v.static_neighbors)
        alphas = {nid: 1.0 for nid in v.static_neighbors}
        link_types = {nid: LINK_INTERNET for nid in v.static_neighbors}
        return connections, alphas, link_types, None

    def aggregate(self, v, vehicles: list) -> None:
        """Personalized FedAvg: vehicle retains SELF_WEIGHT of its own model,
        the remaining (1 - SELF_WEIGHT) is split equally across neighbors."""
        nbr_sds = [
            vehicles[nid].get_shared_weights()
            for nid in v.connections
            if nid < len(vehicles)
        ]
        if not nbr_sds:
            return

        self_w = float(SELF_WEIGHT)
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
