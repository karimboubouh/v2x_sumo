"""
dl/vehicle.py — Vehicle (Node) class for decentralized federated learning.

Each vehicle owns a local ML model, a local dataset partition, and
participates in decentralized training via neighbor collaboration.

Position is updated from SUMO TraCI each simulation step (not self-managed).

Adapted from v2x_sim/vehicle.py (road state removed, SUMO integration added).
"""

import math
import threading

import numpy as np
import torch
import torch.nn as nn

from dl.config import DL_CFG as CFG
from dl.data import get_n_classes
from dl.models import build_model
from dl.helpers import _inf_loader, clone_state_dict


class Vehicle:
    """
    One participant in the V2X federated learning network.

    SUMO state (updated every step from TraCI via update_from_sumo)
    ---------------------------------------------------------------
    pos        : np.ndarray (x, y) in SUMO metres
    heading    : float in radians

    FL state
    --------
    model           : personalized nn.Module
    optimizer       : Adam optimizer
    train_loader    : local training data (non-IID via Dirichlet)
    tr_rounds       : completed training rounds
    current_loss    : latest avg mini-batch loss
    current_acc     : latest avg mini-batch accuracy in [0, 1]

    Threading
    ---------
    _lock           : protects model + metrics during concurrent read/write
    training_done   : Event: SET = idle, CLEAR = training in progress
    """

    def __init__(self, vid: int, sumo_id: str, train_loader, network_bounds: tuple):
        """
        Args:
            vid: integer ID (0-based, used for list indexing)
            sumo_id: SUMO managed vehicle string ID (e.g. "mv_0")
            train_loader: DataLoader for this vehicle's local data partition
            network_bounds: (x_min, y_min, x_max, y_max) for feature normalization
        """
        self.id = vid
        self.sumo_id = sumo_id

        # SUMO position — updated each step by update_from_sumo()
        self.pos = np.array([0.0, 0.0])
        self.heading = 0.0

        # Network size for feature normalization
        x_min, y_min, x_max, y_max = network_bounds
        self._network_size = max(x_max - x_min, y_max - y_min, 1.0)

        # FL model
        self.model = build_model(CFG["DATASET"], CFG["MODEL_ARCH"])
        self._lock = threading.Lock()
        self.train_loader = train_loader
        self._inf_iter = _inf_loader(train_loader)
        self.n_classes = get_n_classes(CFG["DATASET"])
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=CFG["LOCAL_LR"]
        )
        self.criterion = nn.CrossEntropyLoss()

        _init_sd = self.model.state_dict()
        self._shared_weights = clone_state_dict(_init_sd)
        self._ref_weights = clone_state_dict(_init_sd)

        # Neighbor state
        self.connections = set()       # accepted neighbor IDs this step
        self.alphas = {}               # {nid: alpha} aggregation weights
        self.link_types = {}           # {nid: LINK_SIDELINK | LINK_INTERNET}
        self.static_neighbors = []     # populated by FedAvg setup()

        # Reference to the active DLAlgorithm (injected by DLEnvironment)
        self._algo = None

        # Metrics histories
        self.tr_rounds = 0
        self.current_loss = 2.3        # CE for random 10-class model ~ ln(10)
        self.current_acc = 0.0
        self._prev_loss = 2.3
        self.loss_hist = []
        self.acc_hist = []

        # Cached flattened first-layer parameters for cosine-similarity
        self._param_vec: np.ndarray | None = None

        # Threading — starts SET so the vehicle is eligible immediately
        self.training_done = threading.Event()
        self.training_done.set()

    # ── SUMO integration ──────────────────────────────────────────────────────

    def update_from_sumo(self, vehicle_state) -> None:
        """Update position and heading from a SumoManager VehicleState.

        Args:
            vehicle_state: VehicleState with x, y, angle attributes.
                          angle is SUMO's degrees clockwise from north.
        """
        self.pos = np.array([vehicle_state.x, vehicle_state.y])
        # Convert SUMO angle (degrees CW from north) to radians
        self.heading = math.radians(vehicle_state.angle)

    # ── Feature vector ────────────────────────────────────────────────────────

    def get_param_vec(self) -> np.ndarray:
        """Cached flattened first-layer parameters for cosine-similarity."""
        if self._param_vec is None:
            with self._lock:
                self._param_vec = np.concatenate([
                    p.detach().numpy().flatten()
                    for p in list(self.model.parameters())[:2]
                ])
        return self._param_vec

    def own_features(self) -> np.ndarray:
        """
        Compact state vector (OWN_DIM=6).
        [0] loss / 5
        [1] current_acc
        [2] |connections| / MAX_NEIGHBORS
        [3] pos_x / network_size
        [4] pos_y / network_size
        [5] 0.0 (reserved)
        """
        ns = self._network_size
        return np.array([
            float(np.clip(self.current_loss, 0.0, 5.0)) / 5.0,
            float(np.clip(self.current_acc, 0.0, 1.0)),
            len(self.connections) / max(CFG["MAX_NEIGHBORS"], 1),
            float(np.clip(self.pos[0] / ns, 0.0, 1.0)),
            float(np.clip(self.pos[1] / ns, 0.0, 1.0)),
            0.0,
        ], dtype=np.float32)

    # ── Background training round ─────────────────────────────────────────────

    def train_local(self) -> None:
        """
        Process BATCHES_PER_ROUND mini-batches and update metrics.

        Runs in a background thread. Wrapped in try/finally to GUARANTEE
        training_done.set() is called even if an exception occurs.
        """
        try:
            self._ref_weights = clone_state_dict(self.model.state_dict())
            self.model.train()
            total_loss, total_correct, total_n = 0.0, 0, 0

            for _ in range(CFG["BATCHES_PER_ROUND"]):
                images, labels = next(self._inf_iter)
                self.optimizer.zero_grad()
                logits = self.model(images)
                loss = self.criterion(logits, labels)

                # Algorithm-specific regularization (e.g. FedProx proximal term)
                if self._algo is not None:
                    extra = self._algo.extra_loss(self)
                    if extra is not None:
                        loss = loss + extra

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                n = len(labels)
                total_loss += loss.item() * n
                total_correct += (logits.argmax(1) == labels).sum().item()
                total_n += n

            avg_loss = total_loss / max(total_n, 1)
            avg_acc = total_correct / max(total_n, 1)

            with self._lock:
                self._prev_loss = self.current_loss
                self.current_loss = avg_loss
                self.current_acc = avg_acc
                self.tr_rounds += 1

                self._shared_weights = clone_state_dict(self.model.state_dict())

                # Refresh param cache for neighbor feature computation
                self._param_vec = np.concatenate([
                    p.detach().numpy().flatten()
                    for p in list(self.model.parameters())[:2]
                ])

            self.loss_hist.append(avg_loss)
            self.acc_hist.append(avg_acc)

        finally:
            self.training_done.set()

    def get_shared_weights(self) -> dict:
        """Thread-safe copy of the weights broadcast over V2X."""
        with self._lock:
            return clone_state_dict(self._shared_weights)

    # ── Representation ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (f"Vehicle(id={self.id}, sumo_id={self.sumo_id}, "
                f"rounds={self.tr_rounds}, acc={self.current_acc:.2%})")

    def __str__(self) -> str:
        return self.__repr__()
