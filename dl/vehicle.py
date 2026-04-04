"""
dl/vehicle.py — Vehicle (node) class for decentralized learning.

Each vehicle owns a local ML model, a local dataset partition, and
participates in DPL training via neighbor collaboration.

Position is updated from SUMO TraCI each simulation step (not self-managed).

Adapted from v2x_sim/vehicle.py (road state removed, SUMO integration added).
"""

import math
import threading
import time

import numpy as np
import torch
import torch.nn as nn

from algorithms.base import LINK_INTERNET, LINK_SIDELINK
import config
from dl.data import get_n_classes
from dl.models import build_model
from dl.helpers import _inf_loader, clone_state_dict


class Vehicle:
    """
    One participant in the V2X decentralized learning network.

    SUMO state (updated every step from TraCI via update_from_sumo)
    ---------------------------------------------------------------
    pos        : np.ndarray (x, y) in SUMO metres
    heading    : float in radians

    DPL state
    ---------
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

    def __init__(self, vid: int, sumo_id: str, train_loader, network_bounds: tuple,
                 event_stream=None):
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
        self.speed = 0.0

        # Network size for feature normalization
        x_min, y_min, x_max, y_max = network_bounds
        self._network_size = max(x_max - x_min, y_max - y_min, 1.0)

        # DPL model
        self.model = build_model(config.DATASET, config.MODEL_ARCH)
        self._lock = threading.Lock()
        self.train_loader = train_loader
        self._inf_iter = _inf_loader(train_loader)
        self.n_classes = get_n_classes(config.DATASET)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.LOCAL_LR
        )
        self.criterion = nn.CrossEntropyLoss()

        _init_sd = self.model.state_dict()
        self._shared_weights = clone_state_dict(_init_sd)
        self._ref_weights = clone_state_dict(_init_sd)
        self.shared_weights_bytes = self._state_dict_nbytes(_init_sd)

        # Neighbor state
        self.connections = set()       # accepted neighbor IDs this step
        self.alphas = {}               # {nid: alpha} aggregation weights
        self.link_types = {}           # {nid: LINK_SIDELINK | LINK_INTERNET}
        self.static_neighbors = []     # populated by FedAvg setup()
        self.is_byzantine = False

        # Reference to the active DLAlgorithm (injected by DLEnvironment)
        self._algo = None
        self._event_stream = event_stream

        # Metrics histories
        self.tr_rounds = 0
        init_loss = float(math.log(max(self.n_classes, 1)))
        self.current_loss = init_loss
        self.current_acc = 0.0
        self._prev_loss = init_loss
        self.loss_hist = []
        self.acc_hist = []
        self.reward_hist = []
        self.round_time_hist = []
        self.computation_energy_hist = []
        self.computation_energy_j = 0.0
        self.sidelink_tx_energy_j = 0.0
        self.internet_tx_energy_j = 0.0

        # Cached flattened first-layer parameters for cosine-similarity
        self._param_vec: np.ndarray | None = None
        self.last_sim_time = 0.0
        self._round_started_at = 0.0
        self._pending_transfers = []
        self._target_accuracy_announced = False
        self._training_finished_announced = False

        # Threading — starts SET so the vehicle is eligible immediately
        self.training_done = threading.Event()
        self.training_done.set()

    # ── SUMO integration ──────────────────────────────────────────────────────

    def update_from_sumo(self, vehicle_state, sim_time=None) -> None:
        """Update position and heading from a SumoManager VehicleState.

        Args:
            vehicle_state: VehicleState with x, y, angle attributes.
                          angle is SUMO's degrees clockwise from north.
        """
        self.pos = np.array([vehicle_state.x, vehicle_state.y])
        self.speed = float(vehicle_state.speed)
        if sim_time is not None:
            self.last_sim_time = float(sim_time)
        # Convert SUMO angle (degrees CW from north) to radians
        self.heading = math.radians(vehicle_state.angle)

    def prepare_training_round(self, sim_time: float, peer_transfers: list) -> None:
        """Store metadata for the next background training submission."""
        self._round_started_at = float(sim_time)
        self._pending_transfers = list(peer_transfers)

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
        Compact state vector (6 features).
        [0] loss / 5
        [1] current_acc
        [2] |connections| / MAX_NEIGHBORS
        [3] pos_x / network_size
        [4] pos_y / network_size
        [5] Byzantine flag
        """
        ns = self._network_size
        return np.array([
            float(np.clip(self.current_loss, 0.0, 5.0)) / 5.0,
            float(np.clip(self.current_acc, 0.0, 1.0)),
            len(self.connections) / max(config.MAX_NEIGHBORS, 1),
            float(np.clip(self.pos[0] / ns, 0.0, 1.0)),
            float(np.clip(self.pos[1] / ns, 0.0, 1.0)),
            float(self.is_byzantine),
        ], dtype=np.float32)

    # ── Background training round ─────────────────────────────────────────────

    def train_local(self) -> None:
        """
        Process BATCHES_PER_ROUND mini-batches and update metrics.

        Runs in a background thread. Wrapped in try/finally to GUARANTEE
        training_done.set() is called even if an exception occurs.
        """
        round_started = time.perf_counter()
        try:
            self._publish_transfer_events()
            self._ref_weights = clone_state_dict(self.model.state_dict())
            self.model.train()
            total_loss, total_correct, total_n = 0.0, 0, 0

            for _ in range(config.BATCHES_PER_ROUND):
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
            round_time_s = time.perf_counter() - round_started
            # Theoretical DVFS computation energy: E = κ · I·|D_k| · L_k · f_k²
            # total_n = I × |D_k| (actual samples processed this round)
            kappa = float(config.KAPPA)
            f_k = float(config.CPU_FREQ_HZ)
            L_k = float(config.CPU_CYCLES_PER_SAMPLE)
            computation_energy_j = kappa * total_n * L_k * (f_k ** 2)

            with self._lock:
                self._prev_loss = self.current_loss
                self.current_loss = avg_loss
                self.current_acc = avg_acc
                self.tr_rounds += 1
                round_n = self.tr_rounds

                self._shared_weights = clone_state_dict(self.model.state_dict())

                # Refresh param cache for neighbor feature computation
                self._param_vec = np.concatenate([
                    p.detach().numpy().flatten()
                    for p in list(self.model.parameters())[:2]
                ])
                self.computation_energy_j += computation_energy_j

            self.loss_hist.append(avg_loss)
            self.acc_hist.append(avg_acc)
            self.round_time_hist.append(round_time_s)
            self.computation_energy_hist.append(computation_energy_j)
            self._emit_event(
                "training",
                f"vehicle {self.sumo_id} completed training round {round_n} "
                f"(loss={avg_loss:.4f}, acc={avg_acc:.2%}, time={round_time_s:.2f}s)",
            )

            if (
                config.TARGET_ACCURACY <= 1.0
                and avg_acc >= config.TARGET_ACCURACY
                and not self._target_accuracy_announced
            ):
                self._target_accuracy_announced = True
                self._emit_event(
                    "training",
                    f"vehicle {self.sumo_id} reached target accuracy "
                    f"({avg_acc:.2%} >= {config.TARGET_ACCURACY:.2%})",
                )

            if config.TARGET_ACCURACY <= 1.0:
                _finished = avg_acc >= config.TARGET_ACCURACY
            elif config.MAX_TR_ROUNDS > 0:
                _finished = round_n >= config.MAX_TR_ROUNDS
            else:
                _finished = False

            if _finished and not self._training_finished_announced:
                self._training_finished_announced = True
                self._emit_event(
                    "training",
                    f"vehicle {self.sumo_id} finished training after {round_n} rounds",
                )

        except Exception as exc:
            self._emit_event(
                "warning",
                f"vehicle {self.sumo_id} training failed: {exc}",
            )
            raise

        finally:
            self._pending_transfers = []
            self.training_done.set()

    def get_shared_weights(self) -> dict:
        """Thread-safe copy of the weights broadcast over V2X."""
        with self._lock:
            return clone_state_dict(self._shared_weights)

    def add_transmission_energy(self, link_type: float, energy_j: float) -> None:
        """Accumulate transmission energy spent sending weights to peers."""
        with self._lock:
            if link_type == LINK_SIDELINK:
                self.sidelink_tx_energy_j += float(energy_j)
            else:
                self.internet_tx_energy_j += float(energy_j)

    def get_energy_snapshot(self) -> dict:
        """Return current cumulative energy totals for serialization/plotting."""
        with self._lock:
            return {
                "computation_energy_j": self.computation_energy_j,
                "sidelink_tx_energy_j": self.sidelink_tx_energy_j,
                "internet_tx_energy_j": self.internet_tx_energy_j,
                "total_tx_energy_j": (
                    self.sidelink_tx_energy_j + self.internet_tx_energy_j
                ),
            }

    def _publish_transfer_events(self) -> None:
        """Emit send/receive log events for the model updates used this round."""
        for transfer in self._pending_transfers:
            size_text = self._format_bytes(transfer["size_bytes"])
            peer_id = transfer["peer_id"]
            link_name = transfer["link_name"]
            self._emit_event(
                "weight",
                f"vehicle {peer_id} sent model weights of size {size_text} "
                f"to vehicle {self.sumo_id} via {link_name}",
            )
            self._emit_event(
                "weight",
                f"vehicle {self.sumo_id} received model weights from vehicle {peer_id} "
                f"({size_text}) via {link_name}",
            )

    def _emit_event(self, category: str, text: str) -> None:
        """Publish an interaction-log event if a stream is configured."""
        if self._event_stream is not None:
            ts = self._round_started_at or self.last_sim_time
            self._event_stream.publish(ts, category, text)

    def _state_dict_nbytes(self, state_dict: dict) -> int:
        """Estimate serialized tensor size for human-readable logging."""
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in state_dict.values()
        )

    def _format_bytes(self, size_bytes: int) -> str:
        """Format a byte count for the interaction log."""
        units = ["B", "KB", "MB", "GB"]
        size = float(size_bytes)
        for unit in units:
            if size < 1024.0 or unit == units[-1]:
                if unit == "B":
                    return f"{int(size)} {unit}"
                return f"{size:.1f} {unit}"
            size /= 1024.0

    # ── Representation ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (f"Vehicle(id={self.id}, sumo_id={self.sumo_id}, "
                f"rounds={self.tr_rounds}, acc={self.current_acc:.2%})")

    def __str__(self) -> str:
        return self.__repr__()
