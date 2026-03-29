#!/bin/bash
# =============================================================================
# SUMO V2V Communication Dashboard - Build Script (macOS)
# =============================================================================
# This script:
#   1. Verifies SUMO is installed
#   2. Installs Python dependencies in a virtual environment
#   3. Downloads OpenStreetMap data for UAE scenarios
#   4. Generates SUMO network and route files
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ---- Step 1: Check prerequisites ----
log_info "Checking prerequisites..."

if ! command -v python3 &> /dev/null; then
    log_error "Python 3 not found."
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
log_info "Python version: $PYTHON_VERSION"

# ---- Step 2: Find SUMO installation ----
# Check common SUMO locations
FRAMEWORK_SUMO="/Library/Frameworks/EclipseSUMO.framework/Versions/Current/EclipseSUMO/share/sumo"

if [ -n "$SUMO_HOME" ] && [ -d "$SUMO_HOME" ]; then
    log_info "Using SUMO_HOME from environment: $SUMO_HOME"
elif [ -d "$FRAMEWORK_SUMO" ]; then
    export SUMO_HOME="$FRAMEWORK_SUMO"
    log_info "Found SUMO framework installation: $SUMO_HOME"
elif command -v sumo &> /dev/null; then
    SUMO_PATH=$(dirname "$(dirname "$(which sumo)")")
    export SUMO_HOME="$SUMO_PATH/share/sumo"
    log_info "Found SUMO in PATH: $SUMO_HOME"
else
    log_error "SUMO not found. Install SUMO:"
    echo "  brew install --cask sumo"
    echo "  # or download from https://www.eclipse.org/sumo/"
    exit 1
fi

# Verify SUMO binary
SUMO_BIN="$SUMO_HOME/bin/sumo"
if [ -f "$SUMO_BIN" ]; then
    SUMO_VER=$("$SUMO_BIN" --version 2>&1 | head -1)
    log_info "SUMO version: $SUMO_VER"
else
    log_error "SUMO binary not found at $SUMO_BIN"
    exit 1
fi

# Verify netconvert
NETCONVERT="$SUMO_HOME/bin/netconvert"
if [ ! -f "$NETCONVERT" ]; then
    log_error "netconvert not found at $NETCONVERT"
    exit 1
fi
log_info "netconvert found"

# ---- Step 3: Set up Python virtual environment ----
log_info "Setting up Python virtual environment..."

cd "$SCRIPT_DIR"

if [ ! -d "venv" ]; then
    python3 -m venv venv
    log_info "Virtual environment created at venv/"
fi

source venv/bin/activate
pip install --upgrade pip > /dev/null
pip install -r requirements.txt

log_info "Python dependencies installed"

# ---- Step 4: Download OSM maps and generate scenarios ----
log_info "Downloading OpenStreetMap data for UAE scenarios..."
python3 scenarios/download_maps.py

log_info "Generating SUMO networks and routes..."
python3 scenarios/generate_scenarios.py

# ---- Done ----
echo ""
echo "=============================================="
log_info "Build complete!"
echo "=============================================="
echo ""
echo "To run the dashboard:"
echo ""
echo "  source venv/bin/activate"
echo "  export SUMO_HOME=$SUMO_HOME"
echo "  python main.py --scenario dubai_marina"
echo ""
echo "Available scenarios:"
echo "  dubai_marina, sheikh_zayed_road, abu_dhabi_corniche,"
echo "  sharjah_university, yas_island"
echo ""
echo "Options:"
echo "  --scenario/-s    Choose scenario (default: dubai_marina)"
echo "  --comm-range/-r  Communication range in meters (default: 200)"
echo "  --fl-demo        Enable FL weight exchange demo"
echo ""
