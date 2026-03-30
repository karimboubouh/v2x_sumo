"""FL-specific configuration constants.

Adapted from v2x_sim/config.py. Used by all fl/ and algorithms/ modules
via `from dl.config import DL_CFG as CFG`.
"""

DL_CFG = {
    # ── Algorithm ────────────────────────────────────────────
    "ALGORITHM": "FedAvg",  # FedAvg | D-PSGD | DPFL

    # ── Termination ──────────────────────────────────────────
    "MAX_TR_ROUNDS": 100,
    "TARGET_ACCURACY": 1.01,  # disable early stopping by default

    # ── Decentralized Learning ───────────────────────────────
    "DATASET": "MNIST",  # MNIST | CIFAR10 | CIFAR100
    "MODEL_ARCH": "DNN",  # DNN | CNN | LSTM | Transformer | ResNet
    "LOCAL_LR": 1e-3,  # Adam learning rate
    "BATCH_SIZE": 32,
    "BATCHES_PER_ROUND": 2,  # mini-batches processed per FL round
    "DATA_ALPHA": 0.3,  # Dirichlet alpha for non-IID (0.1=very non-IID, 10.0~IID)
    "SELF_WEIGHT": 0.5,  # personalized aggregation weight

    # ── DPFL ─────────────────────────────────────────────────
    "DPFL_UPDATE_EVERY": 10,  # Greedy Graph Construction frequency

    # ── V2X Network ──────────────────────────────────────────
    "V2X_RANGE": 250.0,  # sidelink range (m)
    "MAX_NEIGHBORS": 10,
    "INTERNET_RANGE": 2000.0,
    "MAX_INTERNET_NEIGHBORS": 3,
    "INTERNET_QUALITY_THRESHOLD": 0.45,

    # ── Shannon Channel Model (TX energy) ─────────────────────
    "SL_BANDWIDTH_HZ": 10e6,  # 10 MHz
    "SL_TX_POWER_W": 0.020,  # 20 mW
    "SL_SNR_AT_MAX_RANGE_DB": 10.0,
    "INET_BANDWIDTH_HZ": 20e6,  # 20 MHz
    "INET_TX_POWER_W": 0.200,  # 200 mW
    "INET_SNR_DB": 20.0,

    # ── Threading ────────────────────────────────────────────
    "N_TRAIN_WORKERS": 10,

    # ── Feature dimensions ───────────────────────────────────
    "OWN_DIM": 6,
    "NBR_DIM": 6,
}
