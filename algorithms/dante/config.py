"""DANTE algorithm configuration.

The remaining settings are intentionally limited to paper-native quantities,
simulator budget controls, and standard PPO optimization parameters.
"""

# Feature dimensions consumed by the DANTE encoder.
OWN_DIM: int = 7
NBR_DIM: int = 9

# Personalized aggregation weight retained from the local model.
# ``None`` disables fixed self-retention and gives the local model one
# peer-sized share in the final aggregation.
SELF_WEIGHT: float | None = 0.5

# Candidate and collaboration caps.
MAX_SIDELINK_NEIGHBORS: int = 10
MAX_INTERNET_NEIGHBORS: int = 10
MAX_COLLAB_NEIGHBORS: int = 10

# Evaluation split used for runtime metrics and saved plots.
EVAL_SPLIT: str = "test"

# Shared hidden width used by the GAT encoder, actor, and critic.
GAT_HIDDEN_DIM: int = 64

# Trust recursion coefficient from the paper.
TRUST_SMOOTHING: float = 0.2

# Per-round admissibility budgets.
# ``None`` means "derive a simulator-grounded default" inside the algorithm.
ROUND_ENERGY_BUDGET_J: float = 0.04
ROUND_BANDWIDTH_BUDGET_BITS: float | None = None
ROUND_LATENCY_BUDGET_S: float | None = None

# PPO optimizer settings.
PPO_LR: float = 3e-4
PPO_UPDATE_EVERY: int = 8
PPO_EPOCHS: int = 4
PPO_CLIP_EPS: float = 0.2
PPO_GAMMA: float = 0.99
PPO_GAE_LAMBDA: float = 0.95
PPO_VALUE_COEF: float = 0.5
PPO_ENTROPY_COEF: float = 0.05
PPO_MAX_GRAD_NORM: float = 1.0
