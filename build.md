# Build Guide - SUMO V2V Communication Dashboard (macOS)

This guide walks through manual installation steps on macOS. For automated setup, run `bash build.sh` instead.

## Prerequisites

- macOS 12+ (Monterey or later)
- Python 3.9+

## Step 1: Install SUMO

Install the latest SUMO via Homebrew cask:

```bash
brew install --cask sumo
```

This installs SUMO to `/Library/Frameworks/EclipseSUMO.framework/`.

Verify:
```bash
/Library/Frameworks/EclipseSUMO.framework/Versions/Current/EclipseSUMO/share/sumo/bin/sumo --version
```

Set the environment variable (add to `~/.zshrc` for persistence):
```bash
export SUMO_HOME=/Library/Frameworks/EclipseSUMO.framework/Versions/Current/EclipseSUMO/share/sumo
```

## Step 2: Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 3: Download Maps and Generate Scenarios

```bash
# Download OpenStreetMap data for 5 UAE locations
python scenarios/download_maps.py

# Convert to SUMO networks and generate vehicle routes
python scenarios/generate_scenarios.py
```

## Step 4: Run the Dashboard

```bash
python main.py --scenario dubai_marina
```

## Troubleshooting

### SUMO binary not found
Make sure `SUMO_HOME` is set correctly:
```bash
echo $SUMO_HOME
ls $SUMO_HOME/bin/sumo
```

### netconvert fails
Ensure SUMO was installed with all components via `brew install --cask sumo`.

### Pygame window doesn't appear
On macOS, Pygame may need accessibility permissions. Check System Preferences > Privacy & Security > Accessibility.

### OSM download fails
The Overpass API may be rate-limited. Wait a few minutes and try again, or download the OSM files manually from https://www.openstreetmap.org/export.
