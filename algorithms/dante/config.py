"""DANTE algorithm configuration."""

# Feature dimensions consumed by the GAT encoder.
OWN_DIM: int = 6
NBR_DIM: int = 6

# Personalized aggregation weight retained from the local model.
SELF_WEIGHT: float = 0.5

# Candidate and collaboration caps.
MAX_SIDELINK_NEIGHBORS: int = 10
MAX_INTERNET_NEIGHBORS: int = 10
MAX_COLLAB_NEIGHBORS: int = 10

# Evaluation split used for runtime metrics and saved plots.
EVAL_SPLIT: str = "test"

# Shared hidden width used by the GAT encoder, actor, and critic.
GAT_HIDDEN_DIM: int = 64

# PPO optimiser settings.
PPO_LR: float = 3e-4
PPO_REWARD_SOURCE: str = "validation"  # "validation" or "training"
PPO_UPDATE_EVERY: int = 4
PPO_EPOCHS: int = 4
PPO_CLIP_EPS: float = 0.2
PPO_GAMMA: float = 0.99
PPO_GAE_LAMBDA: float = 0.95
PPO_VALUE_COEF: float = 0.5
PPO_ENTROPY_COEF: float = 0.01
PPO_MAX_GRAD_NORM: float = 1.0
