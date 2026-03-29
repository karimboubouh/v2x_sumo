"""SUMO simulation manager using TraCI."""

import os
import random
import sys
from dataclasses import dataclass

# Ensure config is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sumolib
import traci

import config  # noqa: E402 - sets up sys.path for traci/sumolib


@dataclass
class VehicleState:
    """State of a single vehicle at a simulation step."""

    vehicle_id: str
    x: float
    y: float
    speed: float
    angle: float
    edge_id: str


class SumoManager:
    """Manages a headless SUMO simulation via TraCI."""

    def __init__(self, scenario_key, num_vehicles=None, force_speed=None):
        self.scenario_key = scenario_key
        self._num_vehicles = num_vehicles if num_vehicles is not None else config.NUM_VEHICLES
        # None = use SUMO's default car-following model; set to km/h to force a fixed speed
        # Stored in km/h; converted to m/s (/3.6) when passed to SUMO TraCI
        _fs = force_speed if force_speed is not None else config.VEHICLE_FORCE_SPEED
        self._force_speed_kmh = float(_fs) if _fs is not None else None
        scenario_dir = os.path.join(config.SCENARIOS_DIR, scenario_key)
        self.sumocfg = os.path.join(scenario_dir, "simulation.sumocfg")
        self.net_file = os.path.join(scenario_dir, "network.net.xml")
        self._net = None
        self._net_bounds = None
        self._edge_shapes = None
        self._sim_time = 0.0
        self._running = False
        # Managed vehicle pool: IDs we own and keep alive permanently
        self._managed_ids = [f"mv_{i}" for i in range(self._num_vehicles)]
        # Last known state per vehicle — used to keep them visible during respawn gap
        self._last_states = {}
        # Track which vehicles have had force_speed applied this life-cycle
        self._speed_applied: set = set()

    def start(self):
        """Start the SUMO simulation."""
        if not os.path.exists(self.sumocfg):
            raise FileNotFoundError(
                f"SUMO config not found: {self.sumocfg}\nRun: python scenarios/generate_scenarios.py"
            )

        sumo_binary = config.SUMO_BIN
        if not os.path.isfile(sumo_binary):
            from shutil import which
            found = which("sumo")
            if found:
                sumo_binary = found
            else:
                raise FileNotFoundError(f"SUMO binary not found at {sumo_binary}.\nSet SUMO_HOME or install SUMO.")

        sumo_cmd = [
            sumo_binary,
            "-c",
            self.sumocfg,
            "--step-length",
            str(config.SIM_STEP_LENGTH),
            "--time-to-teleport",
            str(config.TIME_TO_TELEPORT),
            "--no-warnings",
            "true",
            "--no-step-log",
            "true",
        ]

        if not config.TRACI_LOGS:
            # Redirect fd 2 to /dev/null before traci.start() forks SUMO.
            # The child process inherits the silent stderr; we restore ours right after.
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            saved_stderr_fd = os.dup(2)
            os.dup2(devnull_fd, 2)
            os.close(devnull_fd)

        try:
            traci.start(sumo_cmd)
        finally:
            if not config.TRACI_LOGS:
                os.dup2(saved_stderr_fd, 2)
                os.close(saved_stderr_fd)
        self._running = True

        # Load network for geometry and route finding
        if os.path.exists(self.net_file):
            self._net = sumolib.net.readNet(self.net_file)

        # Pre-add all managed vehicles before the first simulationStep
        self._initialize_managed_vehicles()

    def _initialize_managed_vehicles(self):
        """Add all managed vehicles to the simulation at startup."""
        if not self._net:
            return
        for veh_id in self._managed_ids:
            self._add_vehicle(veh_id)

    def _add_vehicle(self, veh_id):
        """Add a single vehicle with a random cross-map route. Returns True on success."""
        edges = self._get_driveable_edges()
        if len(edges) < 2:
            return False

        for _ in range(50):
            origin = random.choice(edges)
            # Pick destination from a different quadrant for map coverage
            origin_shape = origin.getShape()
            ox, oy = origin_shape[len(origin_shape) // 2]
            dest_obj = self._pick_distant_edge(ox, oy)
            if dest_obj is None or dest_obj is origin:
                continue
            route = self._net.getShortestPath(origin, dest_obj)
            if route and route[0]:
                edge_ids = [e.getID() for e in route[0]]
                route_id = f"r_{veh_id}_{random.randint(0, 999999)}"
                try:
                    traci.route.add(route_id, edge_ids)
                    traci.vehicle.add(veh_id, route_id)
                    return True
                except traci.TraCIException:
                    continue
        return False

    def _maybe_reroute(self, veh_id):
        """Assign a new far-away destination when vehicle is on its last route edge."""
        try:
            route = traci.vehicle.getRoute(veh_id)
            idx = traci.vehicle.getRouteIndex(veh_id)
            if idx < len(route) - 2:
                return  # plenty of route remaining
            x, y = traci.vehicle.getPosition(veh_id)
            dest = self._pick_distant_edge(x, y)
            if dest:
                traci.vehicle.changeTarget(veh_id, dest.getID())
        except traci.TraCIException:
            pass

    def _pick_distant_edge(self, from_x, from_y):
        """Return a driveable edge from a different map quadrant than (from_x, from_y)."""
        edges = self._get_driveable_edges()
        if not edges:
            return None

        bounds = self.get_network_bounds()
        cx = (bounds[0] + bounds[2]) / 2
        cy = (bounds[1] + bounds[3]) / 2
        cur_q = (from_x > cx, from_y > cy)

        def edge_quadrant(e):
            shape = e.getShape()
            mid = shape[len(shape) // 2]
            return (mid[0] > cx, mid[1] > cy)

        other = [e for e in edges if edge_quadrant(e) != cur_q]
        return random.choice(other) if other else random.choice(edges)

    def step(self):
        """Advance simulation by one step. Returns dict of managed vehicle states."""
        if not self._running:
            return {}

        traci.simulationStep()
        self._sim_time = traci.simulation.getTime()

        active_ids = set(traci.vehicle.getIDList())

        # Pre-emptively reroute vehicles near the end of their current route
        # so they never reach the destination and disappear
        for veh_id in self._managed_ids:
            if veh_id in active_ids:
                self._maybe_reroute(veh_id)

        # Re-add any managed vehicle that has already been removed
        for veh_id in self._managed_ids:
            if veh_id not in active_ids:
                self._add_vehicle(veh_id)

        # --force-speed: override SUMO's car-following model completely.
        # Without it, vehicles use SUMO's default model (natural speed variation).
        if self._force_speed_kmh and self._force_speed_kmh > 0:
            target_ms = self._force_speed_kmh / 3.6
            for veh_id in self._managed_ids:
                if veh_id in active_ids:
                    try:
                        if veh_id not in self._speed_applied:
                            # First step after spawn: raise the max-speed ceiling and
                            # disable ALL speed safety checks so setSpeed is obeyed
                            # unconditionally (no safe-distance, no right-of-way,
                            # no red-light override).
                            traci.vehicle.setMaxSpeed(veh_id, target_ms)
                            traci.vehicle.setSpeedMode(veh_id, 0)
                            self._speed_applied.add(veh_id)
                        # Force exact speed EVERY step — this is the only
                        # guaranteed way to maintain a constant speed in SUMO.
                        traci.vehicle.setSpeed(veh_id, target_ms)
                    except traci.TraCIException:
                        pass
                else:
                    # Vehicle disappeared; reset so settings are re-applied after respawn
                    self._speed_applied.discard(veh_id)

        # Collect states; fall back to last known position during respawn gap
        vehicles = {}
        for veh_id in self._managed_ids:
            if veh_id not in active_ids:
                if veh_id in self._last_states:
                    vehicles[veh_id] = self._last_states[veh_id]
                continue
            try:
                x, y = traci.vehicle.getPosition(veh_id)
                speed = traci.vehicle.getSpeed(veh_id)
                angle = traci.vehicle.getAngle(veh_id)
                edge = traci.vehicle.getRoadID(veh_id)
                state = VehicleState(
                    vehicle_id=veh_id,
                    x=x, y=y,
                    speed=speed,
                    angle=angle,
                    edge_id=edge,
                )
                vehicles[veh_id] = state
                self._last_states[veh_id] = state
            except traci.TraCIException:
                if veh_id in self._last_states:
                    vehicles[veh_id] = self._last_states[veh_id]

        return vehicles

    def _get_driveable_edges(self):
        """Return cached list of edges that allow passenger vehicles."""
        if not hasattr(self, '_driveable_edges'):
            if self._net is None:
                return []
            self._driveable_edges = [
                e for e in self._net.getEdges(withInternal=False)
                if any(lane.allows('passenger') for lane in e.getLanes())
            ]
        return self._driveable_edges

    def get_sim_time(self):
        """Return current simulation time in seconds."""
        return self._sim_time

    def get_network_bounds(self):
        """Return network bounding box: (x_min, y_min, x_max, y_max)."""
        if self._net_bounds is None and self._running:
            try:
                boundary = traci.simulation.getNetBoundary()
                self._net_bounds = (
                    boundary[0][0],
                    boundary[0][1],
                    boundary[1][0],
                    boundary[1][1],
                )
            except traci.TraCIException:
                self._net_bounds = (0, 0, 1000, 1000)
        return self._net_bounds or (0, 0, 1000, 1000)

    def get_edge_shapes(self):
        """Return list of edge shapes for rendering the road network."""
        if self._edge_shapes is None and self._net:
            self._edge_shapes = []
            for edge in self._net.getEdges(withInternal=False):
                shape = edge.getShape()
                if shape and len(shape) >= 2:
                    self._edge_shapes.append(shape)
        return self._edge_shapes or []

    def get_vehicle_count(self):
        """Return current number of active managed vehicles."""
        if self._running:
            active = set(traci.vehicle.getIDList())
            return sum(1 for v in self._managed_ids if v in active)
        return 0

    def is_running(self):
        """Check if simulation should continue."""
        return self._running

    def stop(self):
        """Stop the SUMO simulation and clean up."""
        if self._running:
            try:
                traci.close()
            except Exception:
                pass
            self._running = False
