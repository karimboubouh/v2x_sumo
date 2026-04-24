"""
dl/helpers.py — Utility functions for the DPL subsystem.

Adapted from v2x_sim/helpers.py.
"""

import math
import random

import numpy as np
import torch
import torch.nn as nn

import config
from dl.models import build_model


def set_global_seed(seed: int | None) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible experiments."""
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _resolve_eval_batch_limit() -> int | None:
    """Return the configured evaluation batch cap, or None for full-loader eval."""
    limit = getattr(config, "EVAL_BATCHES_PER_ROUND", None)
    if limit is None:
        return None
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return None
    return limit if limit > 0 else None


def eval_model_on_loader(
    model,
    loader,
    *,
    criterion=None,
    batch_limit: int | None = None,
) -> dict:
    """Evaluate one model on one loader and return loss/accuracy summary."""
    if loader is None:
        raise ValueError("loader must not be None")

    criterion = criterion or nn.CrossEntropyLoss()
    was_training = bool(model.training)
    total_loss, total_correct, total_n = 0.0, 0, 0

    model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            if batch_limit is not None and batch_idx >= batch_limit:
                break
            logits = model(images)
            loss = criterion(logits, labels)
            n = len(labels)
            total_loss += loss.item() * n
            total_correct += int((logits.argmax(1) == labels).sum())
            total_n += n

    if was_training:
        model.train()

    return {
        "loss": total_loss / max(total_n, 1),
        "acc": total_correct / max(total_n, 1),
        "n_samples": int(total_n),
    }


def eval_weight_snapshots(weight_snapshots: list[dict], eval_loaders) -> dict:
    """Evaluate model-weight snapshots on shared or per-vehicle evaluation data."""
    if eval_loaders is None:
        raise ValueError("eval_loaders must not be None")

    if isinstance(eval_loaders, (list, tuple)):
        per_model_loaders = list(eval_loaders)
        if len(per_model_loaders) != len(weight_snapshots):
            raise ValueError(
                "per-vehicle eval loader count must match weight snapshot count: "
                f"{len(per_model_loaders)} != {len(weight_snapshots)}"
            )
    else:
        per_model_loaders = [eval_loaders] * len(weight_snapshots)

    criterion = nn.CrossEntropyLoss()
    per_loss = []
    per_acc = []
    batch_limit = _resolve_eval_batch_limit()

    for weights, loader in zip(weight_snapshots, per_model_loaders):
        model = build_model(config.DATASET, config.MODEL_ARCH)
        model.load_state_dict(weights)
        metrics = eval_model_on_loader(
            model,
            loader,
            criterion=criterion,
            batch_limit=batch_limit,
        )
        per_loss.append(metrics["loss"])
        per_acc.append(metrics["acc"])

    per_loss = np.asarray(per_loss, dtype=np.float64)
    per_acc = np.asarray(per_acc, dtype=np.float64)
    return {
        "loss": float(per_loss.mean()) if per_loss.size else 0.0,
        "loss_std": float(per_loss.std()) if per_loss.size else 0.0,
        "acc": float(per_acc.mean()) if per_acc.size else 0.0,
        "acc_std": float(per_acc.std()) if per_acc.size else 0.0,
        "per_vehicle_loss": per_loss.astype(np.float32).tolist(),
        "per_vehicle_acc": per_acc.astype(np.float32).tolist(),
        "n_vehicles": int(per_acc.size),
    }


def eval_vehicles(vehicles, eval_loaders) -> dict:
    """Evaluate every vehicle's model on the configured evaluation data.

    Returns:
        Evaluation summary containing mean/std across vehicles.
    """
    return eval_weight_snapshots(
        [v.get_shared_weights() for v in vehicles],
        eval_loaders,
    )


def clone_state_dict(state_dict: dict) -> dict:
    """Clone a model state_dict using tensor.clone() instead of deepcopy."""
    return {k: v.clone() for k, v in state_dict.items()}


def synchronize_vehicle_initial_models(vehicles: list) -> dict:
    """Clone the first vehicle's initial weights across every vehicle."""
    if not vehicles:
        return {}

    anchor_state = clone_state_dict(vehicles[0].model.state_dict())
    for vehicle in vehicles:
        vehicle.model.load_state_dict(anchor_state)
        vehicle._shared_weights = clone_state_dict(anchor_state)
        vehicle._ref_weights = clone_state_dict(anchor_state)
        if hasattr(vehicle, "_zero_update_state"):
            vehicle._shared_update = vehicle._zero_update_state(anchor_state)
        vehicle.shared_weights_bytes = vehicle._state_dict_nbytes(anchor_state)
        vehicle._param_vec = None
    return anchor_state


def _inf_loader(loader):
    """Yield batches forever without caching every batch in memory."""
    while True:
        yield from loader


# ── Shannon-capacity TX helpers ───────────────────────────────────────────────

def _snr_linear(snr_db: float) -> float:
    return 10.0 ** (snr_db / 10.0)


_MODEL_SIZE_BITS: float = 0.0


def _get_model_size_bits() -> float:
    """Return payload size in bits = num_float32_params x 32."""
    global _MODEL_SIZE_BITS
    if _MODEL_SIZE_BITS == 0.0:
        model = build_model(config.DATASET, config.MODEL_ARCH)
        n_params = sum(p.numel() for p in model.parameters())
        _MODEL_SIZE_BITS = float(n_params * 32)
    return _MODEL_SIZE_BITS


def tx_payload_bits() -> float:
    """Return the transmitted payload size in bits after compression."""
    gamma = float(config.COMPRESSION_RATIO)
    return gamma * _get_model_size_bits()


def sl_tx_energy_j(dist_m: float) -> float:
    """Sidelink (PC5) TX energy in Joules for one model-parameter exchange.

    Formula: E = p_k × T,  T = γ·S / C_{k,j}
    where C_{k,j} = B·log2(1+ρ) and γ is the compression ratio.
    """
    v2x_range = float(config.COMM_RANGE)
    snr_0 = _snr_linear(float(config.SL_SNR_AT_MAX_RANGE_DB))
    snr_d = snr_0 * (v2x_range / max(float(dist_m), 1.0)) ** 2
    C = float(config.SL_BANDWIDTH_HZ) * math.log2(1.0 + snr_d)
    T = tx_payload_bits() / C
    return float(config.SL_TX_POWER_W) * T


def inet_tx_energy_j() -> float:
    """Internet (5G Uu relay) TX energy in Joules for one model-parameter exchange.

    Formula: E = 2 × p_k × T,  T = γ·S / C  (×2 for uplink + downlink relay legs)
    """
    snr = _snr_linear(float(config.INET_SNR_DB))
    C = float(config.INET_BANDWIDTH_HZ) * math.log2(1.0 + snr)
    T = tx_payload_bits() / C
    return 2.0 * float(config.INET_TX_POWER_W) * T


def sl_tx_time_s(dist_m: float) -> float:
    """Sidelink transmission time in seconds for one compressed payload."""
    v2x_range = float(config.COMM_RANGE)
    snr_0 = _snr_linear(float(config.SL_SNR_AT_MAX_RANGE_DB))
    snr_d = snr_0 * (v2x_range / max(float(dist_m), 1.0)) ** 2
    C = float(config.SL_BANDWIDTH_HZ) * math.log2(1.0 + snr_d)
    return tx_payload_bits() / C


def inet_tx_time_s() -> float:
    """Internet relay transmission time in seconds for one compressed payload."""
    snr = _snr_linear(float(config.INET_SNR_DB))
    C = float(config.INET_BANDWIDTH_HZ) * math.log2(1.0 + snr)
    return 2.0 * tx_payload_bits() / C


def sl_tx_cost_norm(dist_m: float) -> float:
    """Normalised sidelink TX cost in (0, 1] for feature vectors."""
    v2x_range = float(config.COMM_RANGE)
    snr_0 = _snr_linear(float(config.SL_SNR_AT_MAX_RANGE_DB))
    snr_d = snr_0 * (v2x_range / max(float(dist_m), 1.0)) ** 2
    cap_ref = math.log2(1.0 + snr_0)
    cap_d = math.log2(1.0 + snr_d)
    return cap_ref / cap_d
