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
LOG_MAX_LINES = 50  # Max lines in the message log
STATUS_BAR_HEIGHT = 24  # Bottom status bar height (pixels)

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
