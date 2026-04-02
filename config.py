"""Central configuration for the SUMO V2V Communication Dashboard."""

import os
import sys

# ---------------------------------------------------------------------------
# SUMO paths
# ---------------------------------------------------------------------------
# Default: macOS framework installation
_FRAMEWORK_SUMO = "/Library/Frameworks/EclipseSUMO.framework/Versions/Current/EclipseSUMO/share/sumo"

SUMO_HOME = os.environ.get("SUMO_HOME", _FRAMEWORK_SUMO)

SUMO_TOOLS = os.path.join(SUMO_HOME, "tools")
SUMO_BIN = os.path.join(SUMO_HOME, "bin", "sumo")

# Add SUMO tools to Python path so we can import traci / sumolib
if SUMO_TOOLS not in sys.path:
    sys.path.insert(0, SUMO_TOOLS)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENARIOS_DIR = os.path.join(PROJECT_DIR, "scenarios")
OUT_DIR = os.path.join(PROJECT_DIR, "out")

# ---------------------------------------------------------------------------
# Available scenarios (UAE locations)
# ---------------------------------------------------------------------------
SCENARIOS = {
    "dubai_marina": {
        "name": "Dubai Marina",
        "description": "Dense urban grid with many intersections",
        "bbox": (55.130, 25.075, 55.145, 25.085),  # (west, south, east, north)
    },
    "sheikh_zayed_road": {
        "name": "Sheikh Zayed Road",
        "description": "Highway corridor, high-speed traffic",
        "bbox": (55.265, 25.195, 55.285, 25.210),
    },
    "abu_dhabi_corniche": {
        "name": "Abu Dhabi Corniche",
        "description": "Coastal road, moderate traffic",
        "bbox": (54.350, 24.455, 54.370, 24.465),
    },
    "sharjah_university": {
        "name": "Sharjah University City",
        "description": "Campus area with mixed road types",
        "bbox": (55.450, 25.290, 55.465, 25.300),
    },
    "yas_island": {
        "name": "Yas Island",
        "description": "Circuit-style roads, interesting topology",
        "bbox": (54.595, 24.480, 54.615, 24.495),
    },
}

DEFAULT_SCENARIO = "dubai_marina"

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
SIM_STEP_LENGTH = 1.0  # SUMO seconds per step
SIM_END_TIME = 3600  # 1 hour of simulation
NUM_VEHICLES = 50  # Target number of vehicles in the scenario
VEHICLE_FORCE_SPEED = None  # Force all vehicles to this speed in km/h (e.g. 50, 120, 280); None = SUMO default
TIME_TO_TELEPORT = 10  # Seconds before a stuck vehicle is teleported (-1 = disabled, 300 = default SUMO)

# ---------------------------------------------------------------------------
# Communication parameters (V2V)
# ---------------------------------------------------------------------------
COMM_RANGE = 500.0  # Maximum communication range in meters
MAX_NEIGHBORS = 5  # Maximum V2V connections per vehicle (top-N by signal quality)
COMM_POWER_DBM = -30.0  # Transmit power (reference at 1m)
PATH_LOSS_EXPONENT = 2.5  # Urban environment
NOISE_FLOOR_DBM = -90.0  # Receiver noise floor
SNR_THRESHOLD_DB = 10.0  # Minimum SNR for 50% delivery probability
BEACON_INTERVAL = 5.0  # Seconds between hello beacons
MESSAGE_TTL = 10  # Max retry steps for undelivered messages

# ---------------------------------------------------------------------------
# Dashboard / UI parameters
# ---------------------------------------------------------------------------
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
MAP_PANEL_HEIGHT = 500
LOG_PANEL_HEIGHT = 300
FPS = 120  # Target render frames per second (independent of simulation rate)
FONT_SIZE = 14
LOG_MAX_LINES = None  # Keep the full message log from the start of the simulation
STATUS_BAR_HEIGHT = 56  # Bottom status bar height (pixels)

# ---------------------------------------------------------------------------
# Decentralized Personalized Learning (DPL)
# ---------------------------------------------------------------------------
DL_CFG = {
    # ── Algorithm ────────────────────────────────────────────
    "ALGORITHM": "FedAvg",  # name of any algorithm in algorithms/<name>/algorithm.py

    # ── Termination ──────────────────────────────────────────
    "MAX_TR_ROUNDS": 100,
    "TARGET_ACCURACY": 0.97,  # disable early stopping by default using 1.01
    "EVAL_ROUNDS": 5,  # evaluate global test metrics every N shared rounds

    # ── Decentralized Learning ───────────────────────────────
    "DATASET": "MNIST",  # MNIST | FEMNIST | CIFAR10 | CIFAR100
    "MODEL_ARCH": "DNN",  # DNN | CNN | LSTM | Transformer | ResNet
    "LOCAL_LR": 1e-3,  # Adam learning rate
    "BATCH_SIZE": 32,
    "BATCHES_PER_ROUND": 20,   # mini-batches processed per DPL training round (10×32=320 samples/round)
    "DATA_ALPHA": 0.5,  # Dirichlet alpha for non-IID (0.1=very non-IID, 10.0~IID)
    "SELF_WEIGHT": 0.3,  # personalized aggregation weight (applied in FedAvg and DPFL)

    # ── Computation energy model (DVFS) ──────────────────────
    "KAPPA": 1e-28,  # κ — effective switched capacitance (F·cycle⁻²)
    #   mobile phone (Snapdragon/Apple A-series): ~1e-27
    #   Raspberry Pi (ARM Cortex-A53/A72):        ~5e-27
    #   vehicle compute (Tesla FSD chip):          ~1e-27
    #   laptop CPU (Intel Core / AMD Ryzen):       ~1e-26
    #   server CPU (Intel Xeon / AMD EPYC):        ~5e-26
    "CPU_FREQ_HZ": 1e9,  # f_k — local CPU frequency in Hz (1 GHz)
    "CPU_CYCLES_PER_SAMPLE": 1e5,  # L_k — CPU cycles per training sample
    #   MNIST   28×28×1  DNN:             ~1e5
    #   FEMNIST 28×28×1  DNN:             ~1e5
    #   CIFAR-10  32×32×3  CNN:           ~1e7
    #   CIFAR-100 32×32×3  ResNet:        ~5e7

    # ── Transmission sparsification ──────────────────────────
    "COMPRESSION_RATIO": 1.0,  # γ — fraction of model params transmitted (1.0 = full model)

    # ── V2X Network ──────────────────────────────────────────
    "V2X_RANGE": 250.0,  # sidelink range (m)
    "MAX_NEIGHBORS": 10,
    "INTERNET_RANGE": 2000.0,
    "MAX_INTERNET_NEIGHBORS": 3,
    "INTERNET_QUALITY_THRESHOLD": 0.45,

    # ── Shannon Channel Model (TX energy) ────────────────────
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

# Theme: "dark", "light", or "system" (auto-detect from macOS appearance)
THEME_MODE = "system"

# Console logging: minimum level to display (debug | info | success | result | warning | error)
LOG_LEVEL = "info"

# Show SUMO/TraCI internal error and warning messages in the console
TRACI_LOGS = False

# Colors (R, G, B) — legacy defaults (dark theme); prefer dashboard.theme.color()
COLOR_BG = (30, 30, 30)
COLOR_ROAD = (80, 80, 80)
COLOR_ROAD_HIGHLIGHT = (120, 120, 120)
COLOR_VEHICLE = (0, 180, 255)
COLOR_VEHICLE_OUTLINE = (255, 255, 255)
COLOR_LINK_STRONG = (0, 200, 80)
COLOR_LINK_WEAK = (200, 60, 60)
COLOR_TEXT = (220, 220, 220)
COLOR_LOG_BG = (20, 20, 25)
COLOR_LOG_HELLO = (100, 200, 100)
COLOR_LOG_DATA = (100, 150, 255)
COLOR_LOG_FL = (255, 180, 50)
COLOR_SEPARATOR = (60, 60, 60)
