"""
base.py — Abstract base class for all DPL algorithms.

Every algorithm implements hooks called by the DPL environment:
  setup(vehicles)           — one-time init
  select_neighbors(v, ...)  — decide which neighbors to collaborate with
  aggregate(v, vehicles)    — merge neighbor models into v's model
  extra_loss(v)             — optional regularization term
  post_step(...)            — post-step processing (rewards for RL)

Adapted from v2x_sim/algorithms/base.py.
"""

from abc import ABC, abstractmethod

import torch

# ── Link type constants ───────────────────────────────────────────────────────
LINK_SIDELINK = 0.0   # 5G-NR PC5  — direct D2D sidelink  (low cost)
LINK_INTERNET = 1.0   # 5G Uu      — uplink -> cloud -> downlink  (higher cost)


class DLAlgorithm(ABC):
    """Abstract interface that every algorithm must implement."""

    name: str = "base"
    needs_dynamic_neighbors: bool = True
    evaluation_mode: str = "global"  # "global" or "personalized"
    max_sidelink_neighbors: int = 0
    max_internet_neighbors: int = 0
    max_collab_neighbors: int = 1
    own_feature_dim: int = 6
    neighbor_feature_dim: int = 8

    def setup(self, vehicles: list) -> None:
        """One-time initialization called after all Vehicle objects are created."""

    def build_candidates(self, v, env) -> list:
        """Return the candidate neighbor list for vehicle ``v``."""
        return env.neighbors_of(v)

    @abstractmethod
    def select_neighbors(self, v, candidates: list, env) -> tuple:
        """
        Decide which candidate neighbors to collaborate with this step.

        Returns:
            (connections, alphas, link_types, transition)
        """

    @abstractmethod
    def aggregate(self, v, vehicles: list) -> None:
        """Merge accepted neighbor models into vehicle v's local model."""

    def extra_loss(self, v) -> torch.Tensor | None:
        """Optional extra regularization term added to the local training loss."""
        return None

    def post_step(self, vehicles: list, transitions: dict, step_n: int) -> dict:
        """Called once per simulation step after aggregation and training."""
        return {v.id: 0.0 for v in vehicles}

    def consume_debug_logs(self) -> list[str]:
        """Return algorithm-specific debug lines accumulated since the last call."""
        return []

    def get_baseline_gain(self, v) -> float:
        """Return the learned local-progress baseline used for state features."""
        del v
        return 0.0

    def get_retention_value(self, v, neighbor_id: int) -> float:
        """Return the learned retention value for one neighbor."""
        del v, neighbor_id
        return 0.0

    def __repr__(self) -> str:
        return f"{type(self).__name__}[{self.name}]"

    def __str__(self) -> str:
        return self.name
