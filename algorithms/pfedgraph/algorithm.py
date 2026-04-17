"""
pFedGraph — Personalized Federated Learning with Inferred Collaboration Graphs.

Paper:
  Ye et al., "Personalized Federated Learning with Inferred Collaboration
  Graphs", ICML 2023.

The original method is server-based. Within this project's peer-to-peer hook
structure, we keep the paper's two core ideas intact:
  1. infer collaboration weights from dataset size + model similarity
  2. train from the aggregated model with cosine regularization

The feasible collaboration set is constrained to the neighbors currently
reachable through the simulator.
"""

import numpy as np
import torch

import config
from algorithms.base import DLAlgorithm
from algorithms.pfedgraph.config import (
    EVAL_SPLIT,
    GRAPH_ALPHA_SCALE,
    MAX_COLLAB_NEIGHBORS,
    MAX_INTERNET_NEIGHBORS,
    MAX_SIDELINK_NEIGHBORS,
    REG_LAMBDA,
    SIMILARITY_CAP,
)
from dl.helpers import clone_state_dict


def _synchronize_initial_models(vehicles: list) -> dict:
    """Align all vehicles to a common initial model, matching FL assumptions."""
    anchor_state = clone_state_dict(vehicles[0].model.state_dict())
    for vehicle in vehicles:
        vehicle.model.load_state_dict(anchor_state)
        vehicle._shared_weights = clone_state_dict(anchor_state)
        vehicle._ref_weights = clone_state_dict(anchor_state)
        vehicle.shared_weights_bytes = vehicle._state_dict_nbytes(anchor_state)
        vehicle._param_vec = None
    return anchor_state


def _project_to_simplex(values: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the probability simplex."""
    if values.size == 1:
        return np.array([1.0], dtype=np.float32)

    u = np.sort(values)[::-1]
    cssv = np.cumsum(u)
    rho_candidates = u * np.arange(1, len(u) + 1) > (cssv - 1.0)
    rho = int(np.nonzero(rho_candidates)[0][-1])
    theta = (cssv[rho] - 1.0) / float(rho + 1)
    return np.maximum(values - theta, 0.0).astype(np.float32)


class PFedGraphAlgorithm(DLAlgorithm):
    """Graph-inferred personalized aggregation plus local cosine regularization."""

    name = "pFedGraph"
    needs_dynamic_neighbors = True
    evaluation_mode = "personalized"

    def __init__(self):
        self._alpha_scale = float(GRAPH_ALPHA_SCALE)
        self._reg_lambda = float(REG_LAMBDA)
        self._similarity_cap = float(SIMILARITY_CAP)
        self._initial_state = None
        self._dataset_sizes = {}
        self._similarity_cache_step = -1
        self._similarity_cache = {}
        self.max_sidelink_neighbors = int(MAX_SIDELINK_NEIGHBORS)
        self.max_internet_neighbors = int(MAX_INTERNET_NEIGHBORS)
        self.max_collab_neighbors = int(MAX_COLLAB_NEIGHBORS)

    def setup(self, vehicles: list) -> None:
        self._initial_state = _synchronize_initial_models(vehicles)
        self._dataset_sizes = {
            vehicle.id: max(int(len(vehicle.train_loader.dataset)), 1)
            for vehicle in vehicles
        }
        for vehicle in vehicles:
            vehicle._pfedgraph_self_weight = 1.0
            vehicle._pfedgraph_peer_weights = {}
            vehicle._pfedgraph_anchor_params = {
                name: param.detach().clone()
                for name, param in vehicle.model.named_parameters()
            }

    def select_neighbors(self, v, candidates: list, env) -> tuple:
        self._ensure_similarity_cache(env)

        candidate_ids = []
        link_types = {}
        for other, _, link_type in candidates:
            if other.id == v.id:
                continue
            candidate_ids.append(other.id)
            link_types[other.id] = link_type

        feasible_ids = [v.id] + candidate_ids
        feasible_sizes = np.array(
            [self._dataset_sizes[nid] for nid in feasible_ids],
            dtype=np.float32,
        )
        size_prior = feasible_sizes / max(feasible_sizes.sum(), 1.0)

        alpha = self._alpha_scale * len(feasible_ids)
        sim_vec = np.array(
            [1.0 if nid == v.id else self._similarity_cache[(v.id, nid)] for nid in feasible_ids],
            dtype=np.float32,
        )

        weights = _project_to_simplex(size_prior + 0.5 * alpha * sim_vec)

        max_k = int(self.max_collab_neighbors)
        peer_pairs = [
            (nid, float(weight))
            for nid, weight in zip(feasible_ids[1:], weights[1:])
            if weight > 1e-8
        ]
        peer_pairs.sort(key=lambda item: item[1], reverse=True)
        peer_pairs = peer_pairs[:max_k]

        kept_ids = [v.id] + [nid for nid, _ in peer_pairs]
        kept_weights = np.array(
            [weights[0]] + [weight for _, weight in peer_pairs],
            dtype=np.float32,
        )
        kept_weights = kept_weights / max(kept_weights.sum(), 1e-12)

        v._pfedgraph_self_weight = float(kept_weights[0])
        v._pfedgraph_peer_weights = {
            nid: float(weight)
            for nid, weight in zip(kept_ids[1:], kept_weights[1:])
        }

        alphas = dict(v._pfedgraph_peer_weights)
        return set(alphas.keys()), alphas, {nid: link_types[nid] for nid in alphas}, None

    def aggregate(self, v, vehicles: list) -> None:
        self_weight = float(getattr(v, "_pfedgraph_self_weight", 1.0))
        peer_weights = dict(getattr(v, "_pfedgraph_peer_weights", {}))

        own_state = v.get_shared_weights()
        aggregated = clone_state_dict(own_state)

        for key, value in aggregated.items():
            if not value.is_floating_point():
                continue

            mixed = own_state[key].float() * self_weight
            for nid, weight in peer_weights.items():
                if nid >= len(vehicles):
                    continue
                mixed = mixed + vehicles[nid].get_shared_weights()[key].float() * float(weight)
            aggregated[key] = mixed.to(dtype=value.dtype)

        with v._lock:
            v.model.load_state_dict(aggregated)

        # The paper reports that using the original aggregated model instead of
        # the normalized aggregated model gives comparable performance while
        # avoiding an extra download.
        v._pfedgraph_anchor_params = {
            name: aggregated[name].detach().clone()
            for name, _ in v.model.named_parameters()
        }

    def extra_loss(self, v) -> torch.Tensor | None:
        anchor_params = getattr(v, "_pfedgraph_anchor_params", None)
        if not anchor_params:
            return None

        dot = None
        sq_model = None
        sq_anchor = None

        for name, param in v.model.named_parameters():
            anchor = anchor_params[name].to(device=param.device, dtype=param.dtype)
            prod = torch.sum(param * anchor)
            dot = prod if dot is None else dot + prod
            model_sq = torch.sum(param * param)
            anchor_sq = torch.sum(anchor * anchor)
            sq_model = model_sq if sq_model is None else sq_model + model_sq
            sq_anchor = anchor_sq if sq_anchor is None else sq_anchor + anchor_sq

        cosine = dot / (torch.sqrt(sq_model * sq_anchor) + 1e-12)
        return -0.5 * self._reg_lambda * cosine

    def _ensure_similarity_cache(self, env) -> None:
        if self._similarity_cache_step == env.step_n:
            return

        self._similarity_cache_step = env.step_n
        self._similarity_cache = {}

        deltas = {}
        norms = {}
        for vehicle in env.vehicles:
            current = vehicle.get_shared_weights()
            parts = []
            for key, tensor in current.items():
                if not tensor.is_floating_point():
                    continue
                base = self._initial_state[key].to(dtype=tensor.dtype)
                parts.append((tensor - base).reshape(-1).float().numpy())

            vec = np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)
            deltas[vehicle.id] = vec
            norms[vehicle.id] = float(np.linalg.norm(vec))

        for i in deltas:
            for j in deltas:
                if i == j:
                    self._similarity_cache[(i, j)] = 1.0
                    continue

                norm_i = norms[i]
                norm_j = norms[j]
                if norm_i <= 1e-12 or norm_j <= 1e-12:
                    similarity = 0.0
                else:
                    similarity = float(np.dot(deltas[i], deltas[j]) / (norm_i * norm_j))
                similarity = float(np.clip(similarity, -1.0, 1.0))
                if similarity >= self._similarity_cap:
                    similarity = 1.0
                self._similarity_cache[(i, j)] = similarity

    def __repr__(self) -> str:
        return (
            f"PFedGraphAlgorithm[{self.name}, alpha_scale={self._alpha_scale}, "
            f"lambda={self._reg_lambda}, eval_split={EVAL_SPLIT}]"
        )
