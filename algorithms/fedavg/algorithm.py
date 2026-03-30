"""
FedAvg — Decentralized FedAvg baseline.

Static random neighbor graph, equal-weight aggregation.
Adapted from v2x_sim/algorithms/fedavg/algorithm.py.
"""

import random

from algorithms.base import DLAlgorithm, LINK_INTERNET
from dl.config import DL_CFG as CFG
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
        max_n = int(CFG["MAX_NEIGHBORS"])
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
        """Standard equal-weight FedAvg."""
        nbr_sds = [
            vehicles[nid].get_shared_weights()
            for nid in v.connections
            if nid < len(vehicles)
        ]
        if not nbr_sds:
            return

        with v._lock:
            own_sd = v.model.state_dict()
            new_sd = clone_state_dict(own_sd)
            n_total = 1 + len(nbr_sds)
            for key in new_sd:
                if not new_sd[key].is_floating_point():
                    continue
                agg = own_sd[key].float()
                for sd in nbr_sds:
                    agg = agg + sd[key].float()
                new_sd[key] = agg / n_total
            v.model.load_state_dict(new_sd)
