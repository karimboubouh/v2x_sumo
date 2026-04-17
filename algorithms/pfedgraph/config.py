"""pFedGraph algorithm configuration."""

# Candidate and collaboration caps.
MAX_SIDELINK_NEIGHBORS: int = 0
MAX_INTERNET_NEIGHBORS: int = 10
MAX_COLLAB_NEIGHBORS: int = 10

# Evaluation split used for runtime metrics and saved plots.
EVAL_SPLIT: str = "test"

# Paper guidance: alpha should scale with the client count K.
GRAPH_ALPHA_SCALE: float = 0.08

# Local cosine-regularization coefficient λ.
REG_LAMBDA: float = 0.01

# Similarities above this threshold are treated as fully shareable.
SIMILARITY_CAP: float = 0.9
