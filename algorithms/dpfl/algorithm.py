"""
DPFL — Decentralized Personalized Federated Learning.

Greedy Graph Construction for optimal collaboration graph.
Adapted from v2x_sim/algorithms/dpfl/algorithm.py.
"""

import numpy as np
import torch
import torch.nn.functional as F

from algorithms.base import DLAlgorithm, LINK_INTERNET
from dl.config import DL_CFG as CFG
from dl.helpers import clone_state_dict


class DPFLAlgorithm(DLAlgorithm):
    """
    Decentralized Personalized FL via Greedy Graph Construction.

    The collaboration graph is rebuilt every DPFL_UPDATE_EVERY FL rounds
    using GGC. Between rebuilds the cached set is used, pruned to
    whatever neighbors are currently in range.
    """

    name = "DPFL"
    needs_dynamic_neighbors = True

    def __init__(self, update_every: int = None):
        self._update_every = update_every or int(CFG.get("DPFL_UPDATE_EVERY", 10))
        self._temp_model = None

    def setup(self, vehicles: list) -> None:
        """Initialise per-vehicle DPFL state and the shared evaluation model."""
        from dl.models import build_model

        for v in vehicles:
            v._dpfl_collab = set()
            v._dpfl_alphas = {}
            v._dpfl_last_update = -(self._update_every + 1)

        self._temp_model = build_model(CFG["DATASET"], CFG["MODEL_ARCH"])
        self._temp_model.eval()

    def select_neighbors(self, v, candidates: list, env) -> tuple:
        """Return the (possibly cached) collaboration set for vehicle v."""
        needs_update = v.tr_rounds - v._dpfl_last_update >= self._update_every
        if needs_update:
            self._run_ggc(v, candidates)

        available = {c.id: lt for c, _, lt in candidates}
        connections = {nid for nid in v._dpfl_collab if nid in available}
        link_types = {nid: available[nid] for nid in connections}
        w = 1.0 / (len(connections) + 1) if connections else 1.0
        alphas = {nid: w for nid in connections}
        return connections, alphas, link_types, None

    def aggregate(self, v, vehicles: list) -> None:
        """Uniform-weight FedAvg over the collaboration set."""
        accepted = [vehicles[nid] for nid in v.connections if nid < len(vehicles)]
        if not accepted:
            return

        nbr_sds = [nbr.get_shared_weights() for nbr in accepted]
        n = len(nbr_sds) + 1

        with v._lock:
            own_sd = v.model.state_dict()
            new_sd = clone_state_dict(own_sd)
            for key in new_sd:
                if not new_sd[key].is_floating_point():
                    continue
                agg = own_sd[key].float()
                for sd in nbr_sds:
                    agg = agg + sd[key].float()
                new_sd[key] = agg / n
            v.model.load_state_dict(new_sd)

    def _run_ggc(self, v, candidates: list) -> None:
        """Greedy Graph Construction (Algorithm 2, Kharrat et al. 2025)."""
        val_images, val_labels = next(v._inf_iter)

        candidate_dict = {c.id: c for c, _, lt in candidates if lt == LINK_INTERNET}
        if not candidate_dict:
            v._dpfl_collab = set()
            v._dpfl_alphas = {}
            v._dpfl_last_update = v.tr_rounds
            return

        budget = min(int(CFG["MAX_NEIGHBORS"]), len(candidate_dict))
        X = {}
        base_reward = self._eval_reward(v, [], val_images, val_labels)
        remaining = dict(candidate_dict)

        for _ in range(budget):
            best_gain = 0.0
            best_nid = None

            for nid, nbr in remaining.items():
                if nid in X:
                    continue
                reward = self._eval_reward(
                    v, list(X.values()) + [nbr], val_images, val_labels
                )
                gain = reward - base_reward
                if gain > best_gain:
                    best_gain = gain
                    best_nid = nid

            if best_nid is None:
                break

            X[best_nid] = remaining.pop(best_nid)
            base_reward += best_gain

        n = len(X) + 1
        v._dpfl_collab = set(X.keys())
        v._dpfl_alphas = {nid: 1.0 / n for nid in X}
        v._dpfl_last_update = v.tr_rounds

    def _eval_reward(self, v, peers: list,
                     val_images: torch.Tensor,
                     val_labels: torch.Tensor) -> float:
        """Compute R(S) = -F_k^V(w_{S u {k}})."""
        all_sds = [v.get_shared_weights()] + [p.get_shared_weights() for p in peers]
        n = len(all_sds)
        ref = all_sds[0]

        avg_sd = {
            key: (
                sum(sd[key].float() for sd in all_sds) / n
                if ref[key].is_floating_point()
                else ref[key].clone()
            )
            for key in ref
        }

        self._temp_model.load_state_dict(avg_sd)
        with torch.no_grad():
            logits = self._temp_model(val_images)
            loss = F.cross_entropy(logits, val_labels).item()

        return -loss

    def __repr__(self) -> str:
        return f"DPFLAlgorithm[{self.name}, update_every={self._update_every}]"
