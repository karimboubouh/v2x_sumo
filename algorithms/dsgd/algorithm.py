"""
D-PSGD — Decentralized Parallel SGD with Metropolis-Hastings gossip averaging.

Dynamic topology, doubly-stochastic mixing matrix.
Adapted from v2x_sim/algorithms/dsgd/algorithm.py.
"""

from algorithms.base import DLAlgorithm, LINK_INTERNET
from dl.helpers import clone_state_dict


class DSGDAlgorithm(DLAlgorithm):
    """
    Decentralized Parallel SGD with Metropolis-Hastings gossip averaging.

    W_ij = 1 / (max(deg_i, deg_j) + 1)
    W_ii = 1 - sum_{j != i} W_ij
    """

    name = "D-PSGD"
    needs_dynamic_neighbors = True

    def select_neighbors(self, v, candidates: list, env) -> tuple:
        """Accept all current internet neighbors."""
        connections = set()
        link_types = {}
        alphas = {}

        for other, dist, lt in candidates:
            if lt == LINK_INTERNET:
                connections.add(other.id)
                link_types[other.id] = LINK_INTERNET
                alphas[other.id] = 1.0  # overwritten in aggregate

        return connections, alphas, link_types, None

    def aggregate(self, v, vehicles: list) -> None:
        """Metropolis-Hastings gossip averaging."""
        accepted = [vehicles[nid] for nid in v.connections if nid < len(vehicles)]
        if not accepted:
            return

        d_i = len(v.connections)
        nbr_sds = []
        nbr_weights = []

        for nbr in accepted:
            d_j = len(nbr.connections)
            w_ij = 1.0 / (max(d_i, d_j) + 1)
            nbr_weights.append(w_ij)
            nbr_sds.append(nbr.get_shared_weights())

        w_self = max(0.0, 1.0 - sum(nbr_weights))

        with v._lock:
            own_sd = v.model.state_dict()
            new_sd = clone_state_dict(own_sd)
            for key in new_sd:
                if not new_sd[key].is_floating_point():
                    continue
                agg = own_sd[key].float() * w_self
                for w, sd in zip(nbr_weights, nbr_sds):
                    agg = agg + sd[key].float() * w
                new_sd[key] = agg
            v.model.load_state_dict(new_sd)
