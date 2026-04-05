# SUMO V2V Communication Dashboard with Decentralized Personalized Learning

A real-time **Vehicle-to-Vehicle (V2V) communication simulator** built on top of [SUMO](https://eclipse.dev/sumo/) and [TraCI](https://sumo.dlr.de/docs/TraCI.html), with a Pygame-based dashboard and an integrated **Decentralized Personalized Learning (DPL)** framework. Each vehicle trains a personalized model tailored to its own local data distribution and objectives, collaborating with neighbors over simulated wireless links — no central server required.

---

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Project Structure](#project-structure)
- [Scenarios](#scenarios)
- [Decentralized Personalized Learning System](#decentralized-personalized-learning-system)
  - [How It Works](#how-it-works)
  - [Link Types](#link-types)
  - [Algorithms](#algorithms)
  - [Models and Datasets](#models-and-datasets)
- [Implementing a New Algorithm](#implementing-a-new-algorithm)
  - [Step 1 — Create the algorithm file](#step-1--create-the-algorithm-file)
  - [Step 2 — Implement the required methods](#step-2--implement-the-required-methods)
  - [Step 3 — Register the algorithm](#step-3--register-the-algorithm)
  - [Step 4 — Expose via CLI](#step-4--expose-via-cli)
  - [Full example: Ring-Gossip algorithm](#full-example-ring-gossip-algorithm)
- [Configuration Reference](#configuration-reference)
- [Dashboard Controls](#dashboard-controls)
- [Logging](#logging)

---

## Features

- **Real-time SUMO integration** via TraCI — live vehicle positions, speeds, and road topology
- **Pygame dashboard** with map view, scrollable message log, system status bar, and dark/light themes
- **Physics-based V2V link model** — log-distance path loss, RSSI, SNR, and delivery probability
- **Decentralized Personalized Learning** — each vehicle trains a personalized PyTorch model on its own non-IID data partition, without sharing raw data
- **Three DPL algorithms** — FedAvg (static graph baseline), D-PSGD (Metropolis-Hastings gossip), DPFL (Greedy Graph Construction for personalization)
- **Two wireless link types** — 5G PC5 sidelink (D2D, short range) and Internet relay (long range, quality-filtered)
- **Background training** — training runs in a ThreadPoolExecutor, never blocking the render loop
- **Five UAE road scenarios** — marina, highway, coastal, campus, circuit
- **Fully gated behind `--dl`** — without the flag the app is a pure V2V simulator with zero PyTorch overhead (the `--dl` flag name is kept for brevity in the CLI)

---

## Architecture Overview

```
main.py
  │
  ├── SumoManager          (simulation/sumo_manager.py)
  │     └── TraCI → VehicleState dict every step
  │
  ├── CommManager          (communication/comm_manager.py)
  │     └── V2V link computation, message queuing, beacon delivery
  │
  ├── DLEnvironment        (dl/env.py)          [only when --dl]
  │     ├── Vehicle × N   (dl/vehicle.py)       personalized model per vehicle
  │     │     ├── nn.Module  (dl/models.py)
  │     │     └── DataLoader (dl/data.py)       local non-IID data partition
  │     └── DLAlgorithm   (algorithms/)        personalization strategy
  │           ├── select_neighbors()  → collaboration topology
  │           └── aggregate()         → personalized model update
  │
  └── DashboardApp         (dashboard/app.py)
        ├── MapView         road network + vehicles + links
        ├── LogView         scrolling V2V message log
        ├── MenuBar         scenario name + controls
        └── StatusBar       CPU / RAM / GPU / runtime
```

**Simulation loop (per frame):**

```
sumo.step()          → vehicle_states          (SUMO TraCI)
comm.update()        → new V2V messages         (CommManager)
dl_env.step()        → DPL metrics              (DLEnvironment, if --dl)
dashboard.render()   → frame to screen          (Pygame)
```

---

## Installation

### Prerequisites

- Python 3.10+
- [SUMO](https://eclipse.dev/sumo/docs/Downloads.html) >= 1.18 (with `sumo-gui` optional)
- `SUMO_HOME` environment variable set

```bash
# macOS (Homebrew)
brew install sumo
export SUMO_HOME=/opt/homebrew/opt/sumo/share/sumo

# Ubuntu / Debian
sudo apt install sumo sumo-tools
export SUMO_HOME=/usr/share/sumo
```

### Python dependencies

```bash
git clone <this-repo>
cd sumo
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt`:
```
pygame>=2.5.0
psutil>=5.9.0
requests>=2.28.0
torch>=2.0.0
torchvision>=0.15.0
```

> **Note:** `torch` and `torchvision` are only *used* when `--dl` is passed. If you don't need the DPL component, the app runs fine without them installed.

---

## Quick Start

```bash
# Basic V2V simulation (no DPL)
python main.py --scenario dubai_marina --num-vehicles 30

# With Decentralized Personalized Learning (FedAvg baseline + MNIST + DNN, 20 vehicles)
python main.py --scenario dubai_marina --dl -n 20

# DPFL algorithm, CIFAR-10, CNN backbone
python main.py --scenario sheikh_zayed_road --dl \
    --dl-algorithm DPFL --dl-dataset CIFAR10 --dl-model CNN

# Fast simulation (2x speed, 100 vehicles)
python main.py --scenario yas_island -n 100 --speed 2.0

# Unlimited speed + debug logs
python main.py --scenario abu_dhabi_corniche --speed 0 -v debug
```

---

## CLI Reference

| Flag | Short | Default | Description |
|---|---|---|---|
| `--scenario` | `-s` | `dubai_marina` | Road scenario to run |
| `--num-vehicles` | `-n` | `50` | Target number of vehicles |
| `--comm-range` | `-r` | `500` | V2V communication range (meters) |
| `--speed` | `-x` | `1.0` | Sim speed multiplier (`0` = unlimited) |
| `--force-speed` | | off | Force all vehicles to N km/h |
| `--verbose` | `-v` | `info` | Log level: `debug` `info` `success` `result` `warning` `error` |
| `--dl` | | off | Enable decentralized personalized learning |
| `--dl-algorithm` | | `FedAvg` | DPL algorithm: `FedAvg` `D-PSGD` `DPFL` |
| `--dl-dataset` | | `MNIST` | Training dataset: `MNIST` `FEMNIST` `CIFAR10` `CIFAR100` |
| `--dl-model` | | `DNN` | Model architecture: `DNN` `CNN` `LSTM` `Transformer` `ResNet` |
| `--dl-demo` | | off | Periodic dummy weight exchange demo over CommManager |

---

## Project Structure

```
sumo/
├── main.py                     # Entry point + simulation loop
├── parser.py                   # CLI argument definitions
├── config.py                   # Global SUMO / dashboard constants
├── logger.py                   # Colored console logging
├── requirements.txt
│
├── simulation/
│   └── sumo_manager.py         # TraCI wrapper, VehicleState dataclass
│
├── communication/
│   ├── comm_manager.py         # V2V link management, message queuing
│   ├── v2v_link.py             # Link quality model (path loss, SNR)
│   └── message.py              # V2VMessage dataclass
│
├── dl/                         # Decentralized Personalized Learning package
│   ├── __init__.py
│   ├── config.py               # DPL constants
│   ├── vehicle.py              # Vehicle class (personalized model + SUMO integration)
│   ├── env.py                  # DLEnvironment orchestrator
│   ├── models.py               # DNN, CNN, LSTM, Transformer, ResNet
│   ├── data.py                 # Dirichlet non-IID dataset partitioning
│   └── helpers.py              # clone_state_dict, TX energy helpers
│
├── algorithms/                 # DPL algorithm implementations
│   ├── __init__.py             # build_algorithm() factory
│   ├── base.py                 # DLAlgorithm ABC, link constants
│   ├── fedavg/
│   │   └── algorithm.py        # Static graph, equal-weight averaging
│   ├── dsgd/
│   │   └── algorithm.py        # Metropolis-Hastings gossip
│   └── dpfl/
│       └── algorithm.py        # Greedy Graph Construction
│
├── dashboard/
│   ├── app.py                  # DashboardApp — main GUI controller
│   ├── map_view.py             # Road network + vehicle rendering
│   ├── log_view.py             # Scrollable V2V message log
│   ├── menu.py                 # Top menu bar
│   ├── status_bar.py           # Bottom CPU/RAM/GPU/time bar
│   └── theme.py                # Dark / light theme definitions
│
├── fl_interface/
│   └── fl_payload.py           # DL weight serialization (demo stub)
│
└── scenarios/
    ├── dubai_marina/
    ├── sheikh_zayed_road/
    ├── abu_dhabi_corniche/
    ├── sharjah_university/
    ├── yas_island/
    ├── khalifa_university/
    ├── download_maps.py        # fetch OSM data via Overpass API
    └── generate_scenarios.py   # build SUMO networks + routes
```

---

## Scenarios

Six pre-generated SUMO networks based on real UAE roads (OpenStreetMap):

| Key | Name | Characteristics |
|---|---|---|
| `dubai_marina` | Dubai Marina | Dense urban grid, tight intersections |
| `sheikh_zayed_road` | Sheikh Zayed Road | High-speed highway corridor |
| `abu_dhabi_corniche` | Abu Dhabi Corniche | Coastal boulevard |
| `sharjah_university` | Sharjah University City | Campus-style road network |
| `yas_island` | Yas Island | Circuit-style loop roads |
| `khalifa_university` | Khalifa University | University campus on Airport Road, Abu Dhabi |

Each scenario folder contains:
- `network.net.xml` — Road topology
- `routes.rou.xml` — Pre-computed vehicle routes
- `simulation.sumocfg` — SUMO configuration

### Generating a New Scenario

Adding a new scenario requires three steps: register it in `config.py`, download the OSM map, and build the SUMO network.

**Step 1 — Register the scenario in `config.py`**

Add an entry to the `SCENARIOS` dict. The `bbox` is `(west, south, east, north)` in decimal degrees. A ~2–3 km² area works well; larger areas slow down netconvert and produce sparser vehicle density.

```python
# config.py
SCENARIOS = {
    ...
    "my_location": {
        "name": "My Location",
        "description": "Short description of the road type",
        "bbox": (lon_west, lat_south, lon_east, lat_north),
    },
}
```

To find the bounding box: open [OpenStreetMap](https://www.openstreetmap.org), navigate to your area, click **Export** → **Manually select a different area**, draw the box, and read off the four coordinates.

**Step 2 — Download the OSM map**

```bash
python scenarios/download_maps.py
```

This fetches road data from the Overpass API for every scenario in `SCENARIOS` that does not yet have a `map.osm` file. Existing files are skipped.

**Step 3 — Build the SUMO network and routes**

```bash
python scenarios/generate_scenarios.py
```

This runs `netconvert` (OSM → SUMO network) and `randomTrips.py` (generate vehicle routes) for every scenario that does not yet have a `network.net.xml`. Existing networks are skipped; routes are always regenerated to match the current `NUM_VEHICLES` in `config.py`.

**Run your new scenario:**

```bash
python main.py --scenario my_location -n 20 --speed 0
```

---

## Decentralized Personalized Learning System

### How It Works

When `--dl` is passed:

1. **Startup**: The dataset (MNIST / FEMNIST / CIFAR-10 / CIFAR-100) is partitioned among vehicles using **Dirichlet non-IID splitting** (controlled by `DATA_ALPHA` in `config.py`). Lower alpha means more heterogeneous data — mimicking vehicles with distinct sensing environments and objectives.

2. **Each vehicle** owns a personal `nn.Module`, an optimizer, and its own data partition. Vehicles do **not** share raw data — only model weights. The goal is for each vehicle to converge to a model that best fits *its own* data distribution, not a single global model.

3. **Every simulation step** (`DLEnvironment.step()`):
   - Vehicle positions are updated from SUMO TraCI
   - Neighbor topology is discovered (sidelink + internet links)
   - The active algorithm selects collaborators and performs a personalized model update
   - Idle vehicles are submitted to a `ThreadPoolExecutor` for background local training

4. **Console output** whenever a training round completes:
   ```
   [RESULT ] DPL Round 5 | Loss: 1.2345 | Acc: 56.78%
   ```

### Link Types

| Type | Constant | Range | Use case |
|---|---|---|---|
| 5G PC5 Sidelink | `LINK_SIDELINK = 0.0` | <= 250 m | Direct D2D, low latency |
| Internet relay | `LINK_INTERNET = 1.0` | <= 2000 m | Cloud-routed, quality-filtered |

Internet links are only established when their quality score (cosine similarity of model parameters x neighbor accuracy) exceeds `INTERNET_QUALITY_THRESHOLD` (default `0.45`).

### Algorithms

#### FedAvg

Used here as a **DPL baseline**. Classic equal-weight model averaging on a **static random graph** built at initialization. Each vehicle gets up to `MAX_NEIGHBORS` random peers assigned once. The graph does not change as vehicles move. Since all vehicles average toward a global consensus, this algorithm does *not* personalize — it serves as the lower bound for comparison.

```
theta_v  <-  (theta_v + sum_j theta_j) / (1 + |neighbors|)
```

Best for: baseline comparisons against personalized methods.

#### D-PSGD (Decentralized SGD with gossip)

**Dynamic topology** — uses all current internet neighbors. Aggregation weights follow the **Metropolis-Hastings** rule, ensuring a doubly-stochastic mixing matrix:

```
W_ij = 1 / (max(deg_i, deg_j) + 1)
W_ii = 1 - sum_{j != i} W_ij
```

Gossip averaging allows partial personalization through topology sparsity: vehicles only mix with physically nearby peers, so local data distributions naturally shape the model. Best for: dynamic topologies where proximity-driven collaboration is sufficient.

#### DPFL (Decentralized Personalized Learning)

The primary **personalization algorithm**. Uses a **dynamic topology** with intelligent peer selection via **Greedy Graph Construction (GGC)**. Every `DPFL_UPDATE_EVERY` training rounds, each vehicle greedily selects the exact set of collaborators whose models, when aggregated with its own, maximally reduce its personal validation loss. Between rebuilds the cached set is reused, pruned to currently reachable peers.

This ensures each vehicle ends up with a model specialized for its own data distribution and task, while still benefiting from knowledge shared by compatible neighbors.

Best for: heterogeneous non-IID settings where each vehicle has distinct data and objectives.

### Models and Datasets

| Dataset | Input shape | Classes |
|---|---|---|
| MNIST | (1, 28, 28) | 10 |
| FEMNIST | (1, 28, 28) | 62 |
| CIFAR-10 | (3, 32, 32) | 10 |
| CIFAR-100 | (3, 32, 32) | 100 |

`FEMNIST` is provided through `torchvision.datasets.EMNIST(split="byclass")`, which is the closest built-in equivalent in this codebase's current PyTorch / torchvision stack.

| Architecture | Description |
|---|---|
| DNN | 2-layer MLP (64 hidden units) — fast, lightweight |
| CNN | Single conv block + classifier |
| LSTM | Treats image rows as a time sequence |
| Transformer | Patch-based vision transformer |
| ResNet | Residual network |

---

## Implementing a New Algorithm

All DPL algorithms live in `algorithms/` and inherit from `DLAlgorithm` (defined in `algorithms/base.py`). The base class is named `DLAlgorithm` — standing for *Decentralized Learning Algorithm* — and every algorithm must implement four methods that control how a vehicle selects collaborators and updates its personal model. Adding a new algorithm requires four steps.

### Step 1 — Create the algorithm file

Create a subdirectory and file:

```
algorithms/
└── my_algo/
    ├── __init__.py      (empty)
    └── algorithm.py
```

### Step 2 — Implement the required methods

```python
# algorithms/my_algo/algorithm.py

from algorithms.base import DLAlgorithm, LINK_SIDELINK, LINK_INTERNET
import config
from dl.helpers import clone_state_dict


class MyAlgorithm(DLAlgorithm):
    """One-line description of your algorithm."""

    name = "MyAlgo"

    # Set True  → DLEnvironment calls neighbors_of() each step and passes
    #              physical (Vehicle, dist, link_type) candidates to you.
    # Set False → you manage your own graph; candidates list will be empty.
    needs_dynamic_neighbors = True

    def setup(self, vehicles: list) -> None:
        """
        Called ONCE after all Vehicle objects are created.
        Use this to initialise any per-vehicle state your algorithm needs.

        Args:
            vehicles: list of Vehicle, indexed by vehicle.id
        """
        for v in vehicles:
            v._my_state = {}    # attach custom per-vehicle attributes here

    def select_neighbors(self, v, candidates: list, env) -> tuple:
        """
        Decide which vehicles vehicle `v` will aggregate with this step.

        Args:
            v         : the Vehicle making the selection
            candidates: list of (Vehicle, distance_m, link_type) tuples.
                        Only populated when needs_dynamic_neighbors=True.
            env       : DLEnvironment — access env.vehicles for the full list.

        Returns:
            connections : set[int]        accepted neighbor IDs
            alphas      : dict[int,float]  aggregation weight per neighbor
            link_types  : dict[int,float]  LINK_SIDELINK or LINK_INTERNET per neighbor
            transition  : any | None       pass-through data for post_step (RL use)
        """
        connections = set()
        alphas = {}
        link_types = {}

        for nbr, dist, lt in candidates:
            connections.add(nbr.id)
            link_types[nbr.id] = lt

        # Equal weighting including self
        w = 1.0 / (len(connections) + 1) if connections else 1.0
        alphas = {nid: w for nid in connections}

        return connections, alphas, link_types, None

    def aggregate(self, v, vehicles: list) -> None:
        """
        Merge accepted neighbors' models into vehicle v's local model.

        This method is called for EVERY vehicle every step, right after
        select_neighbors(). Use v.connections (populated by select_neighbors)
        to know which neighbors were accepted.

        Args:
            v        : the Vehicle whose model is being updated
            vehicles : full list of Vehicle objects, indexed by .id

        Thread safety: always acquire v._lock before writing to v.model.
        Use nbr.get_shared_weights() for a thread-safe copy of each neighbor.
        """
        accepted = [vehicles[nid] for nid in v.connections if nid < len(vehicles)]
        if not accepted:
            return

        nbr_sds = [nbr.get_shared_weights() for nbr in accepted]
        n = len(nbr_sds) + 1   # count self

        with v._lock:
            own_sd = v.model.state_dict()
            new_sd = clone_state_dict(own_sd)
            for key in new_sd:
                if not new_sd[key].is_floating_point():
                    continue
                agg = own_sd[key].float()
                for sd in nbr_sds:
                    agg = agg + sd[key].float()
                new_sd[key] = agg / n
            v.model.load_state_dict(new_sd)

    # ── Optional overrides ────────────────────────────────────────────────

    def extra_loss(self, v) -> "torch.Tensor | None":
        """
        Return an additional loss term to be added during local training.
        Called inside Vehicle.train_local() on every mini-batch.

        Example use: FedProx proximal term  mu/2 * ||w - w_ref||^2
        Return None (default) for no extra regularisation.
        """
        return None

    def post_step(self, vehicles: list, transitions: dict, step_n: int) -> dict:
        """
        Called once per DPL step after all vehicles have aggregated.
        Intended for RL-based reward bookkeeping; return {} otherwise.
        """
        return {}
```

### Step 3 — Register the algorithm

Open `algorithms/__init__.py` and add an import and a dispatch branch:

```python
# algorithms/__init__.py  (relevant excerpt)

from algorithms.my_algo.algorithm import MyAlgorithm   # add this import


def build_algorithm(cfg: dict) -> DLAlgorithm:
    name = cfg.get("ALGORITHM", "FedAvg")
    if name == "FedAvg":
        return FedAvgAlgorithm()
    if name == "D-PSGD":
        return DSGDAlgorithm()
    if name == "DPFL":
        return DPFLAlgorithm()
    if name == "MyAlgo":            # add this branch
        return MyAlgorithm()
    raise ValueError(f"Unknown algorithm: {name}")
```

### Step 4 — Expose via CLI

Open `parser.py` and add `"MyAlgo"` to the choices list:

```python
parser.add_argument(
    "--dl-algorithm",
    default="FedAvg",
    choices=["FedAvg", "D-PSGD", "DPFL", "MyAlgo"],   # add here
    dest="dl_algorithm",
    help="DPL algorithm (default: FedAvg)",
)
```

Run your new algorithm:

```bash
python main.py --scenario dubai_marina --dl --dl-algorithm MyAlgo
```

---

### Full example: Ring-Gossip algorithm

A minimal complete example — each vehicle only exchanges weights with its two logical ring-neighbors (sorted by integer ID), regardless of physical proximity:

```python
# algorithms/ring_gossip/algorithm.py

from algorithms.base import DLAlgorithm, LINK_INTERNET
from dl.helpers import clone_state_dict


class RingGossipAlgorithm(DLAlgorithm):
    """Each vehicle aggregates with its two ring-neighbors (by ID)."""

    name = "RingGossip"
    needs_dynamic_neighbors = False   # topology is logical, not physical

    def setup(self, vehicles):
        n = len(vehicles)
        for v in vehicles:
            # Assign left and right neighbors on a ring
            v._ring_neighbors = [(v.id - 1) % n, (v.id + 1) % n]

    def select_neighbors(self, v, candidates, env):
        connections = set(v._ring_neighbors)
        link_types  = {nid: LINK_INTERNET for nid in connections}
        w           = 1.0 / (len(connections) + 1)
        alphas      = {nid: w for nid in connections}
        return connections, alphas, link_types, None

    def aggregate(self, v, vehicles):
        accepted = [vehicles[nid] for nid in v.connections if nid < len(vehicles)]
        if not accepted:
            return
        nbr_sds = [nbr.get_shared_weights() for nbr in accepted]
        n = len(nbr_sds) + 1
        with v._lock:
            own_sd = v.model.state_dict()
            new_sd = clone_state_dict(own_sd)
            for key in new_sd:
                if not new_sd[key].is_floating_point():
                    continue
                agg = own_sd[key].float()
                for sd in nbr_sds:
                    agg = agg + sd[key].float()
                new_sd[key] = agg / n
            v.model.load_state_dict(new_sd)
```

Then register it in `algorithms/__init__.py` and add `"RingGossip"` to `parser.py`:

```bash
python main.py --scenario dubai_marina --dl --dl-algorithm RingGossip
```

---

## Configuration Reference

### `config.py` — Global settings

| Constant | Default | Description |
|---|---|---|
| `NUM_VEHICLES` | `50` | Default vehicle count |
| `COMM_RANGE` | `500` | V2V communication range in meters |
| `MAX_NEIGHBORS` | `5` | Max V2V connections per vehicle (top-N by signal quality) |
| `SIM_STEP_LENGTH` | `1.0` | SUMO step duration (seconds) |
| `TIME_TO_TELEPORT` | `10` | Seconds before a stuck vehicle is teleported (`-1` to disable) |
| `VEHICLE_FORCE_SPEED` | `None` | Force speed in km/h (`None` = SUMO car-following model) |
| `TRACI_LOGS` | `False` | Show or hide SUMO's own stderr output |
| `FPS` | `120` | Dashboard frame rate cap |
| `STATUS_BAR_HEIGHT` | `72` | Height of the bottom status bar (pixels) |
| `THEME_MODE` | `"system"` | `"dark"`, `"light"`, or `"system"` |

### `config.py` — DPL settings

| Constant | Default | Description |
|---|---|---|
| `V2X_RANGE` | `250` | Sidelink D2D range in meters |
| `MAX_NEIGHBORS` | `10` | Max sidelink collaborators |
| `INTERNET_RANGE` | `2000` | Max distance for internet relay links |
| `MAX_INTERNET_NEIGHBORS` | `3` | Max internet collaborators |
| `INTERNET_QUALITY_THRESHOLD` | `0.45` | Minimum quality score to accept an internet peer |
| `LOCAL_LR` | `1e-3` | Adam learning rate |
| `BATCH_SIZE` | `32` | Mini-batch size |
| `BATCHES_PER_ROUND` | `2` | Mini-batches processed per DPL training round |
| `DATA_ALPHA` | `0.3` | Dirichlet alpha (lower = more non-IID data) |
| `MAX_TR_ROUNDS` | `100` | Hard stop after this many training rounds |
| `TARGET_ACCURACY` | `1.01` | Stop when accuracy exceeds this (`> 1.0` = disabled) |
| `N_TRAIN_WORKERS` | `10` | ThreadPoolExecutor worker threads |
| `DPFL_UPDATE_EVERY` | `10` | Rounds between DPFL graph rebuilds |

---

## Dashboard Controls

| Key / Action | Effect |
|---|---|
| `Space` | Pause / resume simulation |
| `ESC` or `Q` | Quit |
| `T` | Toggle dark / light theme |
| Click + drag divider | Resize map / log panels |
| Scroll (log panel) | Scroll message history |

---

## Logging

The logger writes colored output to stdout. Control the minimum level with `--verbose` / `-v`:

| Level | Color | Typical use |
|---|---|---|
| `debug` | dim white | TraCI calls, internal state |
| `info` | cyan | Startup messages, configuration |
| `success` | green | Dashboard ready, DPL initialized |
| `result` | blue | DPL round metrics, periodic step stats |
| `warning` | yellow | Shutdown signals, non-fatal issues |
| `error` | red | Fatal errors |

Example console output:

```
[INFO    ] Starting SUMO V2V Dashboard
[INFO    ] Scenario: Dubai Marina (dubai_marina)
[INFO    ] Vehicles: 20 | Comm range: 500m | Speed: 1.0x (real-time)
[INFO    ] DPL: FedAvg | MNIST | DNN
[SUCCESS ] Dashboard ready. Press ESC or Q to quit.
[INFO    ] Partitioning MNIST for 20 vehicles...
[SUCCESS ] DPL ready: FedAvg | MNIST/DNN | 20 vehicles
[RESULT  ] DPL Round 1 | Loss: 2.1034 | Acc: 18.75%
[RESULT  ] DPL Round 2 | Loss: 1.8762 | Acc: 31.20%
[RESULT  ] Step 100 | Time: 100s | Vehicles: 20 | Links: 47 | Msgs sent: 412 delivered: 398
```
