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
from dl.helpers import _inf_loader, clone_state_dict, eval_model_on_loader


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
    current_reward_loss : latest PPO reward loss source (train or validation)
    current_reward_acc  : latest PPO reward accuracy source (train or validation)

    Threading
    ---------
    _lock           : protects model + metrics during concurrent read/write
    training_done   : Event: SET = idle, CLEAR = training in progress
    """

    def __init__(
        self,
        vid: int,
        sumo_id: str,
        train_loader,
        network_bounds: tuple,
        train_eval_loader=None,
        val_loader=None,
        test_loader=None,
        event_stream=None,
    ):
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
        self.train_eval_loader = train_eval_loader or train_loader
        self.val_loader = val_loader or self.train_eval_loader
        self.test_loader = test_loader or self.val_loader
        self.eval_loader = self.train_eval_loader
        self._inf_iter = _inf_loader(train_loader)
        self.n_classes = get_n_classes(config.DATASET)
        self.optimizer = self._build_optimizer()
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=float(getattr(config, "LABEL_SMOOTHING", 0.0))
        )

        _init_sd = self.model.state_dict()
        self._shared_weights = clone_state_dict(_init_sd)
        self._ref_weights = clone_state_dict(_init_sd)
        self._shared_update = self._zero_update_state(_init_sd)
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
        self.current_val_loss = init_loss
        self.current_val_acc = 0.0
        self._prev_val_loss = init_loss
        self._prev_val_acc = 0.0
        self.current_reward_loss = init_loss
        self.current_reward_acc = 0.0
        self._prev_reward_loss = init_loss
        self._prev_reward_acc = 0.0
        self.loss_hist = []
        self.acc_hist = []
        self.reward_hist = []
        self.round_time_hist = []
        self.computation_energy_hist = []
        self.computation_energy_j = 0.0
        self.sidelink_tx_energy_j = 0.0
        self.internet_tx_energy_j = 0.0

        # Cached compact parameter/update views for neighbor scoring
        self._param_vec: np.ndarray | None = None
        self.last_update_vec: np.ndarray = self._zero_update_vec()
        self.last_sim_time = 0.0
        self._round_started_at = 0.0
        self._pending_transfers = []
        self._target_accuracy_announced = False
        self._training_finished_announced = False

        # Threading — starts SET so the vehicle is eligible immediately
        self.training_done = threading.Event()
        self.training_done.set()

    def _build_optimizer(self):
        optimizer_name = str(getattr(config, "LOCAL_OPTIMIZER", "adam")).strip().lower()
        lr = float(config.LOCAL_LR)
        if optimizer_name == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=float(getattr(config, "LOCAL_MOMENTUM", 0.0)),
                weight_decay=float(getattr(config, "LOCAL_WEIGHT_DECAY", 0.0)),
                nesterov=bool(getattr(config, "LOCAL_NESTEROV", False)),
            )
        if optimizer_name == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=float(getattr(config, "LOCAL_WEIGHT_DECAY", 0.0)),
            )
        raise ValueError(
            f"Unsupported LOCAL_OPTIMIZER={getattr(config, 'LOCAL_OPTIMIZER', None)!r}; "
            "expected 'sgd' or 'adam'."
        )

    def reset_optimizer(self) -> None:
        """Reset local optimizer state after externally loading aggregated weights."""
        self.optimizer = self._build_optimizer()

    def _scheduled_lr(self) -> float:
        base_lr = float(config.LOCAL_LR)
        schedule = str(getattr(config, "LOCAL_LR_SCHEDULE", "constant")).strip().lower()
        if schedule != "cosine":
            return base_lr
        min_lr = float(getattr(config, "LOCAL_LR_MIN", base_lr))
        max_rounds = max(int(getattr(config, "MAX_TR_ROUNDS", 1)), 1)
        progress = min(max(float(self.tr_rounds) / max(max_rounds - 1, 1), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (base_lr - min_lr) * cosine

    def _apply_lr_schedule(self) -> None:
        lr = self._scheduled_lr()
        for group in self.optimizer.param_groups:
            group["lr"] = lr

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

    def _flatten_first_two_params(self) -> np.ndarray:
        return np.concatenate([
            p.detach().numpy().ravel()
            for p in list(self.model.parameters())[:2]
        ]).astype(np.float32, copy=False)

    def _flatten_all_grads(self) -> np.ndarray:
        parts = []
        for param in self.model.parameters():
            grad = param.grad
            if grad is None:
                parts.append(np.zeros(param.numel(), dtype=np.float32))
            else:
                parts.append(grad.detach().numpy().ravel().astype(np.float32, copy=False))
        return np.concatenate(parts).astype(np.float32, copy=False)

    def _flatten_all_param_delta(self, reference_state: dict) -> np.ndarray:
        parts = []
        for name, param in self.model.state_dict().items():
            if not param.is_floating_point():
                continue
            ref_tensor = reference_state.get(name, param)
            delta = param.detach().cpu().numpy().ravel() - ref_tensor.detach().cpu().numpy().ravel()
            parts.append(delta.astype(np.float32, copy=False))
        return np.concatenate(parts).astype(np.float32, copy=False)

    def _zero_update_vec(self) -> np.ndarray:
        parts = []
        for param in self.model.parameters():
            parts.append(np.zeros(param.numel(), dtype=np.float32))
        return np.concatenate(parts).astype(np.float32, copy=False)

    def _zero_update_state(self, state_dict: dict) -> dict:
        update = {}
        for key, tensor in state_dict.items():
            update[key] = torch.zeros_like(tensor) if tensor.is_floating_point() else tensor.clone()
        return update

    def _state_delta(self, current_state: dict, reference_state: dict) -> dict:
        delta = {}
        for key, tensor in current_state.items():
            if not tensor.is_floating_point():
                delta[key] = tensor.clone()
                continue
            ref_tensor = reference_state.get(key, tensor)
            delta[key] = tensor.detach().clone().float() - ref_tensor.detach().clone().float()
        return delta

    def _zero_param_vec(self) -> np.ndarray:
        return np.zeros_like(self._flatten_first_two_params(), dtype=np.float32)

    def get_param_vec(self) -> np.ndarray:
        """Cached flattened first-layer parameters for cosine-similarity."""
        if self._param_vec is None:
            with self._lock:
                self._param_vec = self._flatten_first_two_params()
        return self._param_vec

    def get_update_vec(self) -> np.ndarray:
        """Return the most recent round-level local update direction."""
        with self._lock:
            return self.last_update_vec.copy()

    def get_grad_vec(self) -> np.ndarray:
        """Backward-compatible alias for the latest round-level update vector."""
        return self.get_update_vec()

    def own_features(self) -> np.ndarray:
        """
        Compact state vector (5 features).
        [0] validation loss / 5
        [1] validation accuracy
        [2] remaining energy budget ratio
        [3] remaining bandwidth budget ratio
        [4] latency slack ratio from the last completed round
        """
        budget_features = (1.0, 1.0, 1.0)
        budget_feature_fn = getattr(self._algo, "get_budget_features", None)
        if callable(budget_feature_fn):
            budget_features = tuple(float(x) for x in budget_feature_fn(self))

        return np.array([
            float(np.clip(self.current_reward_loss, 0.0, 5.0)) / 5.0,
            float(np.clip(self.current_reward_acc, 0.0, 1.0)),
            float(np.clip(budget_features[0], 0.0, 1.0)),
            float(np.clip(budget_features[1], 0.0, 1.0)),
            float(np.clip(budget_features[2], 0.0, 1.0)),
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
            self._apply_lr_schedule()
            self.model.train()
            total_loss, total_correct, total_n = 0.0, 0, 0
            last_grad_vec = self._zero_update_vec()

            _bpr = config.BATCHES_PER_ROUND
            batch_iter = (
                iter(self.train_loader)          # full epoch
                if not _bpr                      # 0 or None → all batches
                else (next(self._inf_iter) for _ in range(_bpr))
            )
            for images, labels in batch_iter:
                self.optimizer.zero_grad()
                logits = self.model(images)
                loss = self.criterion(logits, labels)

                # Algorithm-specific regularization (e.g. FedProx proximal term)
                if self._algo is not None:
                    extra = self._algo.extra_loss(self)
                    if extra is not None:
                        loss = loss + extra

                loss.backward()
                last_grad_vec = self._flatten_all_grads()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                n = len(labels)
                total_loss += loss.item() * n
                total_correct += (logits.argmax(1) == labels).sum().item()
                total_n += n

            avg_loss = total_loss / max(total_n, 1)
            avg_acc = total_correct / max(total_n, 1)
            round_time_s = time.perf_counter() - round_started
            reward_source = str(
                getattr(self._algo, "reward_source", "training")
            ).strip().lower()
            reward_eval = None
            if reward_source == "validation":
                reward_eval = eval_model_on_loader(
                    self.model,
                    self.val_loader,
                    criterion=self.criterion,
                )
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
                self._prev_reward_loss = self.current_reward_loss
                self._prev_reward_acc = self.current_reward_acc
                if reward_eval is not None:
                    self._prev_val_loss = self.current_val_loss
                    self._prev_val_acc = self.current_val_acc
                    self.current_val_loss = float(reward_eval["loss"])
                    self.current_val_acc = float(reward_eval["acc"])
                    self.current_reward_loss = self.current_val_loss
                    self.current_reward_acc = self.current_val_acc
                else:
                    self.current_reward_loss = avg_loss
                    self.current_reward_acc = avg_acc
                self.tr_rounds += 1
                round_n = self.tr_rounds

                current_state = self.model.state_dict()
                self._shared_weights = clone_state_dict(current_state)
                self._shared_update = self._state_delta(current_state, self._ref_weights)

                # Refresh param cache for neighbor feature computation
                self._param_vec = self._flatten_first_two_params()
                round_update_vec = self._flatten_all_param_delta(self._ref_weights)
                if np.all(np.isfinite(round_update_vec)):
                    self.last_update_vec = round_update_vec.astype(np.float32, copy=True)
                elif np.all(np.isfinite(last_grad_vec)):
                    self.last_update_vec = last_grad_vec.astype(np.float32, copy=True)
                else:
                    self.last_update_vec = self._zero_update_vec()
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
                str(getattr(config, "STOP_ON", "rounds")).lower() == "train_acc"
                and config.TARGET_ACCURACY <= 1.0
                and avg_acc >= config.TARGET_ACCURACY
                and not self._target_accuracy_announced
            ):
                self._target_accuracy_announced = True
                self._emit_event(
                    "training",
                    f"vehicle {self.sumo_id} reached target accuracy "
                    f"({avg_acc:.2%} >= {config.TARGET_ACCURACY:.2%})",
                )

            if (
                str(getattr(config, "STOP_ON", "rounds")).lower() == "train_acc"
                and config.TARGET_ACCURACY <= 1.0
            ):
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
        """Thread-safe copy of the weights broadcast over V2X.

        Byzantine vehicles send Gaussian-noise weights instead of their real
        model to poison the aggregation of any neighbor that selects them.
        """
        with self._lock:
            if self.is_byzantine:
                return self._byzantine_weights(self._shared_weights)
            return clone_state_dict(self._shared_weights)

    def get_shared_update(self) -> dict:
        """Thread-safe copy of the most recent broadcast update delta."""
        with self._lock:
            if self.is_byzantine:
                return self._byzantine_update(self._shared_update)
            return clone_state_dict(self._shared_update)

    def _byzantine_weights(self, sd: dict) -> dict:
        """Return a state dict where every floating-point tensor is replaced
        by i.i.d. Gaussian noise with the same shape and dtype.  Non-floating
        tensors (e.g. BatchNorm num_batches_tracked) are cloned unchanged."""
        corrupted = {}
        for key, tensor in sd.items():
            if tensor.is_floating_point():
                corrupted[key] = torch.randn_like(tensor)
            else:
                corrupted[key] = tensor.clone()
        return corrupted

    def _byzantine_update(self, update: dict) -> dict:
        """Return an arbitrary floating-point update for Byzantine simulation."""
        corrupted = {}
        for key, tensor in update.items():
            if tensor.is_floating_point():
                corrupted[key] = torch.randn_like(tensor.float())
            else:
                corrupted[key] = tensor.clone()
        return corrupted

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
