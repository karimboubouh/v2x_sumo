"""
algorithms/ — Decentralized Personalized Learning algorithms.

To add a new algorithm, create algorithms/<folder>/algorithm.py containing
a concrete DLAlgorithm subclass with a unique `name` class attribute, plus
an algorithms/<folder>/config.py for any algorithm-specific parameters.

No files outside algorithms/ need to change — the registry is built
automatically at import time by scanning all subfolders.

Usage:
    from algorithms import build_algorithm, get_available_algorithms
    from config import DL_CFG
    algo = build_algorithm(DL_CFG)
    algo.setup(vehicles)
"""

import importlib
import inspect
import os

from algorithms.base import DLAlgorithm, LINK_SIDELINK, LINK_INTERNET

# ── Auto-discovery registry ───────────────────────────────────────────────────

_REGISTRY: dict[str, type[DLAlgorithm]] = {}


def _discover_algorithms() -> None:
    """Scan algorithms/ subdirectories and register every concrete DLAlgorithm.

    Convention: each subfolder must contain an `algorithm.py` file with exactly
    one non-abstract DLAlgorithm subclass.  The class's `name` attribute is
    used as the registry key (must be unique across all algorithms).
    """
    algorithms_dir = os.path.dirname(__file__)
    for entry in sorted(os.scandir(algorithms_dir), key=lambda e: e.name):
        if not entry.is_dir() or entry.name.startswith("_"):
            continue
        algo_file = os.path.join(entry.path, "algorithm.py")
        if not os.path.isfile(algo_file):
            continue
        module_name = f"algorithms.{entry.name}.algorithm"
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, DLAlgorithm)
                and obj is not DLAlgorithm
                and not inspect.isabstract(obj)
                and obj.name != "base"
            ):
                _REGISTRY[obj.name] = obj


_discover_algorithms()


# ── Public API ────────────────────────────────────────────────────────────────

def get_available_algorithms() -> list[str]:
    """Return sorted list of all auto-discovered algorithm names."""
    return sorted(_REGISTRY.keys())


def build_algorithm(cfg: dict) -> DLAlgorithm:
    """Instantiate and return the algorithm specified by cfg["ALGORITHM"].

    Each algorithm class is called with no arguments; algorithm-specific
    parameters are read from the algorithm's own config.py at construction time.

    Raises:
        ValueError: if cfg["ALGORITHM"] does not match any discovered algorithm.
    """
    name = cfg["ALGORITHM"]
    if name not in _REGISTRY:
        available = ", ".join(get_available_algorithms())
        raise ValueError(
            f"Unknown algorithm {name!r}. Available: {available}"
        )
    return _REGISTRY[name]()


__all__ = [
    "DLAlgorithm",
    "LINK_SIDELINK",
    "LINK_INTERNET",
    "build_algorithm",
    "get_available_algorithms",
]
