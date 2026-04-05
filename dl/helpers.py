"""
dl/helpers.py — Utility functions for the DPL subsystem.

Adapted from v2x_sim/helpers.py.
"""

import itertools
import math

import numpy as np
import torch
import torch.nn as nn

import config
from dl.models import build_model


def eval_weight_snapshots(weight_snapshots: list[dict], test_loader) -> tuple:
    """Evaluate model-weight snapshots on the global test set."""
    criterion = nn.CrossEntropyLoss()
    totals = [[0.0, 0, 0] for _ in weight_snapshots]
    models = []

    for weights in weight_snapshots:
        model = build_model(config.DATASET, config.MODEL_ARCH)
        model.load_state_dict(weights)
        model.eval()
        models.append(model)

    with torch.no_grad():
        for images, labels in test_loader:
            n = len(labels)
            for idx, model in enumerate(models):
                logits = model(images)
                loss = criterion(logits, labels)
                totals[idx][0] += loss.item() * n
                totals[idx][1] += int((logits.argmax(1) == labels).sum())
                totals[idx][2] += n

    per_loss = [total[0] / max(total[2], 1) for total in totals]
    per_acc = [total[1] / max(total[2], 1) for total in totals]
    return float(np.mean(per_loss)), float(np.mean(per_acc))


def eval_vehicles(vehicles, test_loader) -> tuple:
    """Evaluate every vehicle's model on the global test set.

    Returns:
        (avg_loss, avg_acc) — mean across all vehicles.
    """
    return eval_weight_snapshots(
        [v.get_shared_weights() for v in vehicles],
        test_loader,
    )


def clone_state_dict(state_dict: dict) -> dict:
    """Clone a model state_dict using tensor.clone() instead of deepcopy."""
    return {k: v.clone() for k, v in state_dict.items()}


def _inf_loader(loader):
    """Wrap a DataLoader in an infinite cycle iterator."""
    return itertools.cycle(loader)


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


def sl_tx_energy_j(dist_m: float) -> float:
    """Sidelink (PC5) TX energy in Joules for one model-parameter exchange.

    Formula: E = p_k × T,  T = γ·S / C_{k,j}
    where C_{k,j} = B·log2(1+ρ) and γ is the compression ratio.
    """
    v2x_range = float(config.COMM_RANGE)
    snr_0 = _snr_linear(float(config.SL_SNR_AT_MAX_RANGE_DB))
    snr_d = snr_0 * (v2x_range / max(float(dist_m), 1.0)) ** 2
    C = float(config.SL_BANDWIDTH_HZ) * math.log2(1.0 + snr_d)
    gamma = float(config.COMPRESSION_RATIO)
    T = gamma * _get_model_size_bits() / C
    return float(config.SL_TX_POWER_W) * T


def inet_tx_energy_j() -> float:
    """Internet (5G Uu relay) TX energy in Joules for one model-parameter exchange.

    Formula: E = 2 × p_k × T,  T = γ·S / C  (×2 for uplink + downlink relay legs)
    """
    snr = _snr_linear(float(config.INET_SNR_DB))
    C = float(config.INET_BANDWIDTH_HZ) * math.log2(1.0 + snr)
    gamma = float(config.COMPRESSION_RATIO)
    T = gamma * _get_model_size_bits() / C
    return 2.0 * float(config.INET_TX_POWER_W) * T


def sl_tx_cost_norm(dist_m: float) -> float:
    """Normalised sidelink TX cost in (0, 1] for feature vectors."""
    v2x_range = float(config.COMM_RANGE)
    snr_0 = _snr_linear(float(config.SL_SNR_AT_MAX_RANGE_DB))
    snr_d = snr_0 * (v2x_range / max(float(dist_m), 1.0)) ** 2
    cap_ref = math.log2(1.0 + snr_0)
    cap_d = math.log2(1.0 + snr_d)
    return cap_ref / cap_d
