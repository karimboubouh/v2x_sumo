"""DPFL algorithm configuration.

Algorithm-specific parameters live here so that DPFL is fully self-contained.
To tune DPFL behaviour, edit this file — no other file needs to change.
"""

# Personalized aggregation weight retained from the local model during DPFL.
SELF_WEIGHT: float = 0.3

# How often (in training rounds) the Greedy Graph Construction is re-run.
# Lower values → more frequent topology updates but higher compute cost.
DPFL_UPDATE_EVERY: int = 10
