"""FedAvg algorithm configuration."""

# None = classic FedAvg (equal weighting across self + neighbors).
# Numeric value in [0, 1] = personalized FedAvg with fixed self-retention.
SELF_WEIGHT: float | None = None  # 0.3

# Candidate and collaboration caps.
MAX_SIDELINK_NEIGHBORS: int = 0
MAX_INTERNET_NEIGHBORS: int = 10
MAX_COLLAB_NEIGHBORS: int = 10

# Evaluation split used for runtime metrics and saved plots.
EVAL_SPLIT: str = "test"
