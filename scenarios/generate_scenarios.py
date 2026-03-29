#!/usr/bin/env python3
"""Convert downloaded OSM maps into SUMO networks and generate vehicle routes."""

import os
import sys
import subprocess

# Add parent directory to path so we can import config and logger
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import logger

NETCONVERT = os.path.join(config.SUMO_HOME, "bin", "netconvert")
RANDOM_TRIPS = os.path.join(config.SUMO_TOOLS, "randomTrips.py")


def find_binary(name):
    """Find a SUMO binary, checking SUMO_HOME/bin first, then PATH."""
    sumo_bin = os.path.join(config.SUMO_HOME, "bin", name)
    if os.path.isfile(sumo_bin):
        return sumo_bin
    # Try system PATH
    from shutil import which
    found = which(name)
    if found:
        return found
    return sumo_bin  # Return default even if missing; will error later


def generate_network(scenario_dir):
    """Convert OSM to SUMO network using netconvert."""
    osm_path = os.path.join(scenario_dir, "map.osm")
    net_path = os.path.join(scenario_dir, "network.net.xml")

    if not os.path.exists(osm_path):
        logger.log(f"{osm_path} not found. Run download_maps.py first.", "error")
        return False

    if os.path.exists(net_path):
        logger.log(f"{net_path} already exists, skipping", "debug")
        return True

    netconvert = find_binary("netconvert")
    cmd = [
        netconvert,
        "--osm-files", osm_path,
        "--output-file", net_path,
        "--geometry.remove",
        "--ramps.guess",
        "--junctions.join",
        "--tls.guess-signals",
        "--tls.discard-simple",
        "--tls.join",
        "--no-turnarounds",
        "--offset.disable-normalization",
    ]

    logger.log("Running netconvert...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            logger.log(f"netconvert failed:\n{result.stderr[:500]}", "error")
            return False
    except FileNotFoundError:
        logger.log(f"netconvert not found at {netconvert}", "error")
        return False

    logger.log(f"Network saved to {net_path}", "success")
    return True


def generate_routes(scenario_dir):
    """Generate random vehicle routes using randomTrips.py."""
    net_path = os.path.join(scenario_dir, "network.net.xml")
    route_path = os.path.join(scenario_dir, "routes.rou.xml")
    trips_path = os.path.join(scenario_dir, "trips.xml")

    if not os.path.exists(net_path):
        logger.log(f"{net_path} not found. Generate network first.", "error")
        return False

    # Always regenerate routes so they reflect current NUM_VEHICLES config
    for stale in (trips_path, route_path):
        if os.path.exists(stale):
            os.remove(stale)

    random_trips = RANDOM_TRIPS
    if not os.path.exists(random_trips):
        # Try to find it
        alt = os.path.join(config.SUMO_TOOLS, "randomTrips.py")
        if os.path.exists(alt):
            random_trips = alt

    # Use period=1 and end=NUM_VEHICLES so all vehicles depart in the first
    # NUM_VEHICLES seconds → every vehicle is on the road almost immediately.
    cmd = [
        sys.executable, random_trips,
        "-n", net_path,
        "-o", trips_path,
        "-r", route_path,
        "-b", "0",
        "-e", str(config.NUM_VEHICLES),
        "-p", "1",
        "--seed", "42",
        "--validate",
    ]

    logger.log("Running randomTrips.py...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            logger.log(f"randomTrips exit code {result.returncode}", "warning")
            if result.stderr:
                logger.log(f"{result.stderr[:300]}")
    except FileNotFoundError:
        logger.log(f"randomTrips.py not found at {random_trips}", "error")
        return False

    if os.path.exists(route_path):
        logger.log(f"Routes saved to {route_path}", "success")
        return True
    else:
        logger.log("Route file was not created", "error")
        return False


def generate_sumocfg(scenario_dir, scenario_key):
    """Generate a SUMO configuration file."""
    cfg_path = os.path.join(scenario_dir, "simulation.sumocfg")

    content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
               xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="network.net.xml"/>
        <route-files value="routes.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{config.SIM_END_TIME}"/>
        <step-length value="{config.SIM_STEP_LENGTH}"/>
    </time>
    <processing>
        <time-to-teleport value="{config.TIME_TO_TELEPORT}"/>
    </processing>
</configuration>
"""
    with open(cfg_path, "w") as f:
        f.write(content)

    logger.log(f"Config saved to {cfg_path}", "success")
    return True


def main():
    logger.log("Generating SUMO scenarios from OSM data", "info")

    success = 0
    for key, info in config.SCENARIOS.items():
        logger.log(f"{info['name']}", "info")
        scenario_dir = os.path.join(config.SCENARIOS_DIR, key)

        ok = True
        ok = generate_network(scenario_dir) and ok
        ok = generate_routes(scenario_dir) and ok
        ok = generate_sumocfg(scenario_dir, key) and ok

        if ok:
            success += 1

    logger.log(f"Done: {success}/{len(config.SCENARIOS)} scenarios generated.", "success")
    return 0 if success == len(config.SCENARIOS) else 1


if __name__ == "__main__":
    sys.exit(main())
