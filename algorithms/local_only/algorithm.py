"""
LocalOnly — personalized local training without any collaboration links.
"""

from algorithms.base import DLAlgorithm


class LocalOnlyAlgorithm(DLAlgorithm):
    """Dedicated no-collaboration baseline for personalized local training."""

    name = "LocalOnly"
    needs_dynamic_neighbors = False
    evaluation_mode = "personalized"
    max_sidelink_neighbors = 0
    max_internet_neighbors = 0
    max_collab_neighbors = 0

    def select_neighbors(self, v, candidates: list, env) -> tuple:
        del v, candidates, env
        return set(), {}, {}, None

    def aggregate(self, v, vehicles: list) -> None:
        del v, vehicles
