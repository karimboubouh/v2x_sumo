# SUMO V2V Communication Dashboard

A real-time dashboard for simulating and visualizing Vehicle-to-Vehicle (V2V) communication in urban traffic scenarios using SUMO (Simulation of Urban Mobility) and Python.

## Overview

This project combines SUMO traffic simulation with a distance-based V2V communication model to simulate vehicles moving on real-world road networks while exchanging messages in a peer-to-peer fashion. The dashboard provides real-time visualization of vehicle movements and message exchanges.

The communication layer is designed to support Federated Learning (FL) model weight exchange between vehicles.

## Features

- **Real-world road networks** from 5 UAE locations via OpenStreetMap
- **Distance-based V2V communication** using log-distance path loss model
- **Real-time Pygame dashboard** with split-screen layout:
  - Top: animated map with vehicles and communication links
  - Bottom: scrolling message log with color-coded message types
- **P2P message exchange** with probabilistic delivery based on link quality
- **FL-ready architecture** with stub for model weight serialization
- **TraCI integration** for real-time SUMO simulation control

## Scenarios

| Scenario | Location | Road Type |
|----------|----------|-----------|
| `dubai_marina` | Dubai Marina | Dense urban grid |
| `sheikh_zayed_road` | Sheikh Zayed Road | Highway corridor |
| `abu_dhabi_corniche` | Abu Dhabi Corniche | Coastal road |
| `sharjah_university` | Sharjah University City | Campus/mixed roads |
| `yas_island` | Yas Island, Abu Dhabi | Circuit-style roads |

## Architecture

```
main.py (orchestrator loop)
    |
    |-- SumoManager          Steps SUMO via TraCI, reads vehicle positions/speeds
    |       |
    |       v
    |-- CommManager           Computes V2V links, delivers messages probabilistically
    |       |
    |       v
    |-- DashboardApp          Renders map + message log via Pygame
```

**Data flow per step:**
1. `SumoManager.step()` advances SUMO and returns vehicle states (x, y, speed, angle)
2. `CommManager.update()` computes neighbor links using Euclidean distance, applies path loss model, delivers queued messages
3. `DashboardApp.render()` draws the road network, vehicles, V2V links, and message log

## Communication Model

Uses a log-distance path loss model for realistic V2V link quality:

```
RSSI(d) = P_tx - 10 * n * log10(d)
SNR = RSSI - noise_floor
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| P_tx | -30 dBm | Transmit power (ref. 1m) |
| n | 2.5 | Path loss exponent (urban) |
| Noise floor | -90 dBm | Receiver noise floor |
| Range | 200 m | Maximum communication range |

Link quality degrades with distance: reliable (~1.0) at 10m, moderate (~0.5) at 100m, weak (~0.15) at 200m.

## Quick Start

### Automated Setup (macOS)

```bash
bash build.sh
```

This installs SUMO 1.8.0 from source, sets up a Python virtual environment, downloads map data, and generates all scenarios.

### Run

```bash
source .venv/bin/activate
export SUMO_HOME=$HOME/.local/sumo-1.8.0/share/sumo
python main.py --scenario dubai_marina
```

### Options

```
--scenario/-s    Choose scenario (default: dubai_marina)
--comm-range/-r  Communication range in meters (default: 200)
--num-vehicles   Target vehicle count (default: 20)
--fl-demo        Enable FL weight exchange demo messages
```

### Controls

- **ESC** or **Q**: Quit the dashboard
- Close window to exit

## Manual Setup

See [build.md](build.md) for step-by-step manual installation instructions.

## Project Structure

```
sumo/
├── main.py                  # Entry point
├── config.py                # Central configuration
├── scenarios/               # UAE map data and SUMO scenario files
│   ├── download_maps.py     # Download OSM data via Overpass API
│   ├── generate_scenarios.py# Convert OSM to SUMO networks
│   └── dubai_marina/        # (and 4 other scenario directories)
├── simulation/
│   └── sumo_manager.py      # SUMO TraCI interface
├── communication/
│   ├── v2v_link.py          # Link quality model
│   ├── message.py           # Message data structures
│   └── comm_manager.py      # Communication manager
├── dashboard/
│   ├── app.py               # Pygame dashboard
│   ├── map_view.py          # Map panel renderer
│   └── log_view.py          # Message log panel
└── fl_interface/
    └── fl_payload.py        # FL weight serialization stub
```

## Dependencies

- **SUMO 1.8.0** (built from source)
- **Python 3.9+**
- **pygame** (real-time visualization)
- **requests** (OSM data download)
- **traci / sumolib** (bundled with SUMO)

## FL Integration

The `fl_interface/fl_payload.py` module provides a stub for Federated Learning weight exchange. To integrate with a real FL framework:

1. Replace `FLPayload.dummy_weights()` with actual model weight extraction
2. Use `comm_manager.send_message()` with `msg_type="fl_weights"` to send weights to neighbors
3. Handle received weights in the main loop by checking delivered messages

## References

- [SUMO Documentation](https://www.eclipse.org/sumo/)
- [TraCI Python API](https://sumo.dlr.de/docs/TraCI/Interfacing_TraCI_from_Python.html)
- [TraCI Vehicle Value Retrieval](https://sumo.dlr.de/docs/TraCI/Vehicle_Value_Retrieval.html)
- [OpenStreetMap](https://www.openstreetmap.org/)
