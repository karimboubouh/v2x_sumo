"""CHOCO-SGD algorithm configuration."""

# Candidate and collaboration caps.
MAX_SIDELINK_NEIGHBORS: int = 0
MAX_INTERNET_NEIGHBORS: int = 10
MAX_COLLAB_NEIGHBORS: int = 10

# Evaluation split used for runtime metrics and saved plots.
EVAL_SPLIT: str = "test"

# Consensus step size γ used in the modified gossip step.
CONSENSUS_STEPSIZE: float = 0.4

# Compression operator used for model-difference exchange.
# Supported: "topk", "sign", "identity"
COMPRESSION: str = "topk"

# Fraction of entries retained by the top-k compressor.
COMPRESSION_RATIO: float = 0.1
