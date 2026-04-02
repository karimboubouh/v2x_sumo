"""
dl/helpers.py — Utility functions for the DPL subsystem.

Adapted from v2x_sim/helpers.py.
"""

import itertools
import math

import numpy as np
import torch
import torch.nn as nn

from config import DL_CFG as CFG
from dl.models import build_model


def eval_vehicles(vehicles, test_loader) -> tuple:
    """Evaluate every vehicle's model on the global test set.

    Returns:
        (avg_loss, avg_acc) — mean across all vehicles.
    """
    criterion = nn.CrossEntropyLoss()
    totals = {v.id: [0.0, 0, 0] for v in vehicles}
    models = []

    for v in vehicles:
        model = build_model(CFG["DATASET"], CFG["MODEL_ARCH"])
        model.load_state_dict(v.get_shared_weights())
        model.eval()
        models.append((v.id, model))

    with torch.no_grad():
        for images, labels in test_loader:
            n = len(labels)
            for vid, model in models:
                logits = model(images)
                loss = criterion(logits, labels)
                totals[vid][0] += loss.item() * n
                totals[vid][1] += int((logits.argmax(1) == labels).sum())
                totals[vid][2] += n

    per_loss = [totals[v.id][0] / max(totals[v.id][2], 1) for v in vehicles]
    per_acc = [totals[v.id][1] / max(totals[v.id][2], 1) for v in vehicles]
    return float(np.mean(per_loss)), float(np.mean(per_acc))


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
        model = build_model(CFG["DATASET"], CFG["MODEL_ARCH"])
        n_params = sum(p.numel() for p in model.parameters())
        _MODEL_SIZE_BITS = float(n_params * 32)
    return _MODEL_SIZE_BITS


def sl_tx_energy_j(dist_m: float) -> float:
    """Sidelink (PC5) TX energy in Joules for one model-parameter exchange.

    Formula: E = p_k × T,  T = γ·S / C_{k,j}
    where C_{k,j} = B·log2(1+ρ) and γ is the compression ratio.
    """
    v2x_range = float(CFG["V2X_RANGE"])
    snr_0 = _snr_linear(float(CFG["SL_SNR_AT_MAX_RANGE_DB"]))
    snr_d = snr_0 * (v2x_range / max(float(dist_m), 1.0)) ** 2
    C = float(CFG["SL_BANDWIDTH_HZ"]) * math.log2(1.0 + snr_d)
    gamma = float(CFG.get("COMPRESSION_RATIO", 1.0))
    T = gamma * _get_model_size_bits() / C
    return float(CFG["SL_TX_POWER_W"]) * T


def inet_tx_energy_j() -> float:
    """Internet (5G Uu relay) TX energy in Joules for one model-parameter exchange.

    Formula: E = 2 × p_k × T,  T = γ·S / C  (×2 for uplink + downlink relay legs)
    """
    snr = _snr_linear(float(CFG["INET_SNR_DB"]))
    C = float(CFG["INET_BANDWIDTH_HZ"]) * math.log2(1.0 + snr)
    gamma = float(CFG.get("COMPRESSION_RATIO", 1.0))
    T = gamma * _get_model_size_bits() / C
    return 2.0 * float(CFG["INET_TX_POWER_W"]) * T


def sl_tx_cost_norm(dist_m: float) -> float:
    """Normalised sidelink TX cost in (0, 1] for feature vectors."""
    v2x_range = float(CFG["V2X_RANGE"])
    snr_0 = _snr_linear(float(CFG["SL_SNR_AT_MAX_RANGE_DB"]))
    snr_d = snr_0 * (v2x_range / max(float(dist_m), 1.0)) ** 2
    cap_ref = math.log2(1.0 + snr_0)
    cap_d = math.log2(1.0 + snr_d)
    return cap_ref / cap_d
