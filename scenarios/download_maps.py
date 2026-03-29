#!/usr/bin/env python3
"""Download OpenStreetMap data for all UAE scenarios via the Overpass API."""

import os
import sys
import time
import requests

# Add parent directory to path so we can import config and logger
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import logger

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def download_osm(scenario_key, scenario_info, output_dir):
    """Download OSM data for a single scenario."""
    osm_path = os.path.join(output_dir, "map.osm")

    if os.path.exists(osm_path):
        logger.log(f"{osm_path} already exists, skipping", "debug")
        return True

    west, south, east, north = scenario_info["bbox"]

    # Overpass QL query: get all roads within the bounding box
    query = f"""
    [out:xml][timeout:60];
    (
      way["highway"~"motorway|trunk|primary|secondary|tertiary|residential|unclassified|living_street|service"]
        ({south},{west},{north},{east});
    );
    (._;>;);
    out body;
    """

    logger.log(f"Downloading OSM data for {scenario_info['name']}...")
    try:
        resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.log(f"Failed to download: {e}", "error")
        return False

    os.makedirs(output_dir, exist_ok=True)
    with open(osm_path, "w", encoding="utf-8") as f:
        f.write(resp.text)

    size_kb = len(resp.text) / 1024
    logger.log(f"Saved {osm_path} ({size_kb:.1f} KB)", "success")
    return True


def main():
    logger.log("Downloading OpenStreetMap data for UAE scenarios", "info")

    success = 0
    for key, info in config.SCENARIOS.items():
        logger.log(f"{info['name']} — {info['description']}", "info")
        output_dir = os.path.join(config.SCENARIOS_DIR, key)
        if download_osm(key, info, output_dir):
            success += 1
        # Be polite to the Overpass API
        time.sleep(2)

    logger.log(f"Done: {success}/{len(config.SCENARIOS)} scenarios downloaded.", "success")
    return 0 if success == len(config.SCENARIOS) else 1


if __name__ == "__main__":
    sys.exit(main())
