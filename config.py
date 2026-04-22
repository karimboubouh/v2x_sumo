"""Central configuration for the SUMO V2V Communication Dashboard."""

import os
import sys

# ═══════════════════════════════════════════════════════════════════════════
# 1. SYSTEM PATHS
# ═══════════════════════════════════════════════════════════════════════════

# ── SUMO installation ────────────────────────────────────
# Default: macOS framework installation
_FRAMEWORK_SUMO = (
    "/Library/Frameworks/EclipseSUMO.framework/Versions/Current/EclipseSUMO/share/sumo"
)

SUMO_HOME = os.environ.get("SUMO_HOME", _FRAMEWORK_SUMO)
SUMO_TOOLS = os.path.join(SUMO_HOME, "tools")
SUMO_BIN = os.path.join(SUMO_HOME, "bin", "sumo")

# Add SUMO tools to Python path so we can import traci / sumolib
if SUMO_TOOLS not in sys.path:
    sys.path.insert(0, SUMO_TOOLS)

# ── Project layout ───────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENARIOS_DIR = os.path.join(PROJECT_DIR, "scenarios")
OUT_DIR = os.path.join(PROJECT_DIR, "out")


# ═══════════════════════════════════════════════════════════════════════════
# 2. SCENARIOS (UAE locations)
# ═══════════════════════════════════════════════════════════════════════════

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
    "khalifa_university": {
        "name": "Khalifa University",
        "description": "Shakhbout Bin Sultan St, Al Etihad, Abu Dhabi Island",
        "bbox": (54.385, 24.440, 54.408, 24.457),
    },
}

DEFAULT_SCENARIO = "khalifa_university"


# ═══════════════════════════════════════════════════════════════════════════
# 3. SIMULATION
# ═══════════════════════════════════════════════════════════════════════════

SIM_STEP_LENGTH = 1.0  # SUMO seconds per step
SIM_END_TIME = 3600  # 1 hour of simulation
NUM_VEHICLES = 50  # Target number of vehicles in the scenario
VEHICLE_FORCE_SPEED = None  # Force all vehicles to this speed in km/h (e.g. 50, 120, 280); None = SUMO default
TIME_TO_TELEPORT = 10  # Seconds before a stuck vehicle is teleported (-1 = disabled, 300 = default SUMO)


# ═══════════════════════════════════════════════════════════════════════════
# 4. V2V COMMUNICATION
# ═══════════════════════════════════════════════════════════════════════════

# ── Range & link filtering ───────────────────────────────
COMM_RANGE = 350.0  # 5G sidelink maximum communication range in meters (default 250.0)
INTERNET_RANGE = 1000.0  # Internet maximum communication range in meters
INTERNET_QUALITY_THRESHOLD = (
    0.05  # Min model-based score for internet candidates (default 0.05)
)
COMM_MAX_SIDELINK_LINKS = 10  # Max rendered/managed V2V sidelink links per vehicle

# ── Radio / channel ──────────────────────────────────────
COMM_POWER_DBM = -30.0  # Transmit power (reference at 1m)
PATH_LOSS_EXPONENT = 2.5  # Urban environment
NOISE_FLOOR_DBM = -90.0  # Receiver noise floor
SNR_THRESHOLD_DB = 10.0  # Minimum SNR for 50% delivery probability

# ── Protocol / messaging ─────────────────────────────────
BEACON_INTERVAL = 5.0  # Seconds between hello beacons
MESSAGE_TTL = 10  # Max retry steps for undelivered messages


# ═══════════════════════════════════════════════════════════════════════════
# 5. ENERGY MODEL
# ═══════════════════════════════════════════════════════════════════════════

# ── Shannon channel (TX energy) ──────────────────────────
SL_BANDWIDTH_HZ = 10e6  # 10 MHz
SL_TX_POWER_W = 0.020  # 20 mW
SL_SNR_AT_MAX_RANGE_DB = 10.0
INET_BANDWIDTH_HZ = 20e6  # 20 MHz
INET_TX_POWER_W = 0.200  # 200 mW
INET_SNR_DB = 20.0

# ── Computation (DVFS) ───────────────────────────────────
KAPPA = 1e-28  # κ — effective switched capacitance (F·cycle⁻²)
#   mobile phone (Snapdragon/Apple A-series): ~1e-27
#   Raspberry Pi (ARM Cortex-A53/A72):        ~5e-27
#   vehicle compute (Tesla FSD chip):          ~1e-27
#   laptop CPU (Intel Core / AMD Ryzen):       ~1e-26
#   server CPU (Intel Xeon / AMD EPYC):        ~5e-26
CPU_FREQ_HZ = 1e9  # f_k — local CPU frequency in Hz (1 GHz)
CPU_CYCLES_PER_SAMPLE = 1e5  # L_k — CPU cycles per training sample
#   MNIST   28×28×1  DNN:             ~1e5
#   FEMNIST 28×28×1  DNN:             ~1e5
#   CIFAR-10  32×32×3  CNN:           ~1e7
#   CIFAR-100 32×32×3  ResNet:        ~5e7


# ═══════════════════════════════════════════════════════════════════════════
# 6. DECENTRALIZED PERSONALIZED LEARNING (DPL)
# ═══════════════════════════════════════════════════════════════════════════

# ── Algorithm ────────────────────────────────────────────
ALGORITHM = "DANTE"  # name of any algorithm in algorithms/<name>/algorithm.py

# ── Dataset & model ──────────────────────────────────────
DATASET = "MNIST"  # MNIST | FEMNIST | CIFAR10 | CIFAR100
MODEL_ARCH = "DNN"  # DNN | CNN | LSTM | Transformer | ResNet
DATA_ALPHA = 0.5  # Dirichlet alpha for non-IID (0.1=very non-IID, 10.0~IID)
VALIDATION_FRACTION = 0.1  # fraction of each client's shard held out for validation

# ── Training ─────────────────────────────────────────────
LOCAL_LR = 1e-3  # Adam learning rate
BATCH_SIZE = 32
BATCHES_PER_ROUND = 10  # mini-batches per DPL training round (N×BATCH_SIZE samples/round); 0 or None = full epoch
COMPRESSION_RATIO = 1.0  # γ — fraction of model params transmitted (1.0 = full model)

# ── Evaluation & termination ─────────────────────────────
EVAL_ROUNDS = 10  # evaluate global metrics every N shared rounds
EVAL_SPLIT = "test"  # default evaluation split if an algorithm does not override it
EVAL_BATCHES_PER_ROUND = 20  # None/0 = full eval loader; positive int = cap eval to first N batches per loader
ASYNC_EVAL = (
    True  # True = background eval worker; False = run eval inline during the DPL step
)
TARGET_ACCURACY = (
    1.01  # accuracy threshold for early stopping; set ≥ 1.0 to use rounds mode instead
)
MAX_TR_ROUNDS = 200

# ── Byzantine robustness ─────────────────────────────────
# Fraction of vehicles that behave as Byzantine adversaries.
# Byzantine vehicles broadcast Gaussian-noise weights instead of their real
# model, poisoning the aggregation of any neighbor that selects them.
# Set to 0.0 to disable (default). Example: 0.2 = 20% of vehicles.
BYZANTINE_FRACTION: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 7. RUNTIME
# ═══════════════════════════════════════════════════════════════════════════

# ── Concurrency ──────────────────────────────────────────
N_TRAIN_WORKERS = 10

# ── Logging ──────────────────────────────────────────────
LOG_LEVEL = "info"  # Console logging: minimum level (debug | info | success | result | warning | error)
SAVE_LOGS = (
    False  # Save plain logger output into out/<experiment_id>/run.log during DPL runs
)
TRACI_LOGS = False  # Show SUMO/TraCI internal error and warning messages in the console


# ═══════════════════════════════════════════════════════════════════════════
# 8. DASHBOARD / UI
# ═══════════════════════════════════════════════════════════════════════════

# ── Window & layout ──────────────────────────────────────
HEADLESS = False  # Disable dashboard/graphics; console output only
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 400
MAP_PANEL_HEIGHT = 540
LOG_PANEL_HEIGHT = 220
STATUS_BAR_HEIGHT = 56  # Bottom status bar height (pixels)

# ── Rendering ────────────────────────────────────────────
FPS = 60  # Main-loop render cap in frames per second
DPI_SCALE = 1.0  # Manual DPI hint (1.0 = auto; 2.0 forces HiDPI scaling tweaks)
FL_LABEL_MIN_ZOOM = 3.0  # Minimum zoom multiple at which α labels appear on FL links

# ── Fonts & logs ─────────────────────────────────────────
FONT_SIZE_LOG = 12  # Base font size for the log panel (pt)
FONT_SIZE_MAP = (
    11  # Base font size for map UI elements (HUD, legend, zoom controls, etc.)
)
LOG_MAX_LINES = None  # Keep the full message log from the start of the simulation

# ── Theme ────────────────────────────────────────────────
THEME_MODE = (
    "system"  # "dark", "light", or "system" (auto-detect from macOS appearance)
)
