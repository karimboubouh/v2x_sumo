"""
DPFL — Decentralized Personalized Learning.

Greedy Graph Construction for optimal collaboration graph.
"""
import numpy as np
import torch
import torch.nn.functional as F

import config
from algorithms.base import DLAlgorithm, LINK_INTERNET
from algorithms.dpfl.config import (
    DPFL_UPDATE_EVERY,
    MAX_COLLAB_NEIGHBORS,
    MAX_INTERNET_NEIGHBORS,
    MAX_SIDELINK_NEIGHBORS,
    SELF_WEIGHT,
)
from dl.helpers import clone_state_dict


class DPFLAlgorithm(DLAlgorithm):
    """
    Decentralized Personalized Learning via Greedy Graph Construction.

    The collaboration graph is rebuilt every DPFL_UPDATE_EVERY DPL rounds
    using GGC. Between rebuilds the cached set is used, pruned to
    whatever neighbors are currently in range.
    """

    name = "DPFL"
    needs_dynamic_neighbors = True
    evaluation_mode = "personalized"

    def __init__(self):
        self._update_every = DPFL_UPDATE_EVERY
        self._temp_model = None
        self.max_sidelink_neighbors = int(MAX_SIDELINK_NEIGHBORS)
        self.max_internet_neighbors = int(MAX_INTERNET_NEIGHBORS)
        self.max_collab_neighbors = int(MAX_COLLAB_NEIGHBORS)

    def setup(self, vehicles: list) -> None:
        """Initialise per-vehicle DPFL state and the shared evaluation model."""
        from dl.models import build_model

        for v in vehicles:
            v._dpfl_collab = set()
            v._dpfl_alphas = {}
            v._dpfl_last_update = -(self._update_every + 1)

        self._temp_model = build_model(config.DATASET, config.MODEL_ARCH)
        self._temp_model.eval()

    def select_neighbors(self, v, candidates: list, env) -> tuple:
        """Return the (possibly cached) collaboration set for vehicle v."""
        needs_update = v.tr_rounds - v._dpfl_last_update >= self._update_every
        if needs_update:
            self._run_ggc(v, candidates)

        available = {c.id: lt for c, _, lt in candidates}
        connections = {nid for nid in v._dpfl_collab if nid in available}
        link_types = {nid: available[nid] for nid in connections}
        if connections:
            self_w = self._resolve_self_weight(len(connections))
            nbr_w = (1.0 - self_w) / len(connections)
            alphas = {nid: nbr_w for nid in connections}
        else:
            alphas = {}
        return connections, alphas, link_types, None

    def aggregate(self, v, vehicles: list) -> None:
        """Personalized aggregation over the GGC collaboration set.
        If SELF_WEIGHT is None, the local model is treated like one more peer.
        If SELF_WEIGHT is numeric, that fraction is retained locally and the
        remaining mass is split equally across selected neighbors."""
        accepted = [vehicles[nid] for nid in v.connections if nid < len(vehicles)]
        if not accepted:
            return

        nbr_sds = [nbr.get_shared_weights() for nbr in accepted]
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

    def _run_ggc(self, v, candidates: list) -> None:
        """Greedy Graph Construction following the paper's double-greedy rule."""
        eval_loader = getattr(v, "val_loader", None) or getattr(v, "eval_loader", None) or v.train_eval_loader
        val_images, val_labels = next(iter(eval_loader))

        candidate_dict = {c.id: c for c, _, lt in candidates if lt == LINK_INTERNET}
        if not candidate_dict:
            v._dpfl_collab = set()
            v._dpfl_alphas = {}
            v._dpfl_last_update = v.tr_rounds
            return

        budget = min(int(self.max_collab_neighbors), len(candidate_dict))
        if budget <= 0:
            v._dpfl_collab = set()
            v._dpfl_alphas = {}
            v._dpfl_last_update = v.tr_rounds
            return

        # Paper-style GGC keeps two sets: X (selected collaborators) and
        # Y (remaining feasible collaborators). Self is implicit in
        # _eval_reward(), so X/Y only track peers.
        X = {}
        Y = dict(candidate_dict)
        shuffled_ids = np.random.permutation(list(candidate_dict.keys()))

        for nid in shuffled_ids:
            nbr = candidate_dict[int(nid)]

            reward_x = self._eval_reward(v, list(X.values()), val_images, val_labels)
            reward_x_with_j = self._eval_reward(
                v, list(X.values()) + [nbr], val_images, val_labels
            )
            reward_y = self._eval_reward(v, list(Y.values()), val_images, val_labels)
            reward_y_without_j = self._eval_reward(
                v,
                [peer for peer_id, peer in Y.items() if peer_id != nid],
                val_images,
                val_labels,
            )

            a = max(reward_x_with_j - reward_x, 0.0)
            b = max(reward_y_without_j - reward_y, 0.0)
            p_add = 1.0 if (a + b) == 0.0 else a / (a + b)

            if np.random.random() <= p_add:
                X[int(nid)] = nbr
            else:
                Y.pop(int(nid), None)

            if len(X) >= budget:
                break

        v._dpfl_collab = set(X.keys())
        if X:
            self_w = self._resolve_self_weight(len(X))
            nbr_w = (1.0 - self_w) / len(X)
            v._dpfl_alphas = {nid: nbr_w for nid in X}
        else:
            v._dpfl_alphas = {}
        v._dpfl_last_update = v.tr_rounds

    def _resolve_self_weight(self, n_neighbors: int) -> float:
        """Resolve DPFL self-weight with the shared personalized semantics."""
        if SELF_WEIGHT is None:
            return 1.0 / (n_neighbors + 1.0)
        self_w = float(SELF_WEIGHT)
        if not 0.0 <= self_w <= 1.0:
            raise ValueError(
                f"DPFL SELF_WEIGHT must be None or in [0, 1], got {SELF_WEIGHT!r}"
            )
        return self_w

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

    def __str__(self) -> str:
        return self.__repr__()
