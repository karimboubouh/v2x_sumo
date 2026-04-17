"""
CHOCO-SGD — compressed decentralized SGD with error-feedback style public states.

This implementation follows Algorithm 1 from:
  Koloskova et al., "Decentralized Deep Learning with Arbitrary Communication
  Compression", ICLR 2020.

The project loop already performs "aggregate then local train", which matches:
  1. modified gossip averaging
  2. stochastic gradient update
"""

import math

from algorithms.base import DLAlgorithm
from algorithms.choco_sgd.config import (
    COMPRESSION,
    COMPRESSION_RATIO,
    CONSENSUS_STEPSIZE,
    MAX_COLLAB_NEIGHBORS,
    MAX_INTERNET_NEIGHBORS,
    MAX_SIDELINK_NEIGHBORS,
)
from dl.helpers import clone_state_dict


def _synchronize_initial_models(vehicles: list) -> dict:
    """Align all vehicles to the same initialization, as assumed by the paper."""
    anchor_state = clone_state_dict(vehicles[0].model.state_dict())
    for vehicle in vehicles:
        vehicle.model.load_state_dict(anchor_state)
        vehicle._shared_weights = clone_state_dict(anchor_state)
        vehicle._ref_weights = clone_state_dict(anchor_state)
        vehicle.shared_weights_bytes = vehicle._state_dict_nbytes(anchor_state)
        vehicle._param_vec = None
    return anchor_state


def _compress_tensor(tensor, mode: str, keep_ratio: float):
    """Apply the configured compression operator to one tensor."""
    if not tensor.is_floating_point():
        return tensor.clone()

    mode = str(mode).strip().lower()
    if mode == "identity" or keep_ratio >= 1.0:
        return tensor.clone()

    flat = tensor.reshape(-1)
    if flat.numel() == 0:
        return tensor.clone()

    if mode == "sign":
        scale = flat.abs().mean()
        return (scale * flat.sign()).reshape_as(tensor)

    if mode != "topk":
        raise ValueError(f"Unsupported CHOCO compression mode: {mode!r}")

    k = max(1, int(math.ceil(flat.numel() * max(float(keep_ratio), 0.0))))
    if k >= flat.numel():
        return tensor.clone()

    _, idx = flat.abs().topk(k, sorted=False)
    compressed = flat.new_zeros(flat.shape)
    compressed[idx] = flat[idx]
    return compressed.reshape_as(tensor)


class CHOCOSGDAlgorithm(DLAlgorithm):
    """Compressed decentralized SGD with persistent public copies of local models."""

    name = "CHOCO-SGD"
    needs_dynamic_neighbors = True

    def __init__(self):
        self._gamma = float(CONSENSUS_STEPSIZE)
        self._compression = str(COMPRESSION)
        self._compression_ratio = float(COMPRESSION_RATIO)
        self._pending_step_id = -1
        self._cached_step_id = -1
        self._private_snapshot = {}
        self._public_snapshot = {}
        self._mutual_neighbors = {}
        self.max_sidelink_neighbors = int(MAX_SIDELINK_NEIGHBORS)
        self.max_internet_neighbors = int(MAX_INTERNET_NEIGHBORS)
        self.max_collab_neighbors = int(MAX_COLLAB_NEIGHBORS)

    def setup(self, vehicles: list) -> None:
        initial_state = _synchronize_initial_models(vehicles)
        zero_public = {
            key: tensor.clone().zero_() if tensor.is_floating_point() else tensor.clone()
            for key, tensor in initial_state.items()
        }
        for vehicle in vehicles:
            vehicle._choco_public_state = clone_state_dict(zero_public)

    def select_neighbors(self, v, candidates: list, env) -> tuple:
        self._pending_step_id = env.step_n

        max_k = int(self.max_collab_neighbors)
        selected = []
        link_types = {}
        for other, _, link_type in candidates:
            if len(selected) >= max_k:
                break
            selected.append(other.id)
            link_types[other.id] = link_type

        # Exact gossip weights depend on the mutual graph and are filled in
        # during aggregation once all vehicles have selected their neighbors.
        alphas = {nid: 1.0 for nid in selected}
        return set(selected), alphas, link_types, None

    def aggregate(self, v, vehicles: list) -> None:
        self._ensure_step_cache(vehicles)

        neighbors = sorted(self._mutual_neighbors.get(v.id, set()))
        private_state = self._private_snapshot[v.id]
        public_state = self._public_snapshot[v.id]
        d_i = len(neighbors)

        new_state = clone_state_dict(private_state)
        alphas = {}

        for key, value in new_state.items():
            if not value.is_floating_point():
                continue

            updated = private_state[key].float().clone()
            for nid in neighbors:
                d_j = len(self._mutual_neighbors.get(nid, set()))
                w_ij = 1.0 / (max(d_i, d_j) + 1.0)
                updated = updated + self._gamma * w_ij * (
                    self._public_snapshot[nid][key].float() - public_state[key].float()
                )
                alphas[nid] = w_ij

            new_state[key] = updated.to(dtype=value.dtype)

        with v._lock:
            v.model.load_state_dict(new_state)

        compressed_delta = {}
        updated_public_state = clone_state_dict(public_state)
        for key, value in new_state.items():
            if not value.is_floating_point():
                continue
            delta = value.float() - public_state[key].float()
            q = _compress_tensor(delta, self._compression, self._compression_ratio)
            compressed_delta[key] = q
            updated_public_state[key] = (public_state[key].float() + q).to(dtype=value.dtype)

        v._choco_public_state = updated_public_state
        v.alphas = alphas

    def _ensure_step_cache(self, vehicles: list) -> None:
        if self._cached_step_id == self._pending_step_id:
            return

        self._cached_step_id = self._pending_step_id
        self._private_snapshot = {
            vehicle.id: vehicle.get_shared_weights()
            for vehicle in vehicles
        }
        self._public_snapshot = {
            vehicle.id: clone_state_dict(vehicle._choco_public_state)
            for vehicle in vehicles
        }
        self._mutual_neighbors = {vehicle.id: set() for vehicle in vehicles}

        for vehicle in vehicles:
            for nid in vehicle.connections:
                if nid >= len(vehicles):
                    continue
                if vehicle.id in vehicles[nid].connections:
                    self._mutual_neighbors[vehicle.id].add(nid)

    def __repr__(self) -> str:
        return (
            f"CHOCOSGDAlgorithm[{self.name}, gamma={self._gamma}, "
            f"compression={self._compression}, ratio={self._compression_ratio}]"
        )
