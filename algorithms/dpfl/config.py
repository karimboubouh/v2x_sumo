"""DPFL algorithm configuration.

Algorithm-specific parameters live here so that DPFL is fully self-contained.
To tune DPFL behavior, edit this file.
"""

# Fixed self-retention in [0, 1], or None to treat the local model as one
# peer-sized contributor with no special retention mass.
SELF_WEIGHT: float | None = 0.5

# Candidate and collaboration caps.
MAX_SIDELINK_NEIGHBORS: int = 0
MAX_INTERNET_NEIGHBORS: int = 10
MAX_COLLAB_NEIGHBORS: int = 10

# Evaluation split used for runtime metrics and DPFL collaborator scoring.
EVAL_SPLIT: str = "test"

# How often (in training rounds) the Greedy Graph Construction is re-run.
# Lower values → more frequent topology updates but higher compute cost.
DPFL_UPDATE_EVERY: int = 4
