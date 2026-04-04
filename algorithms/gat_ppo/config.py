"""GAT_PPO algorithm configuration."""

# Feature dimensions consumed by the GAT encoder.
OWN_DIM: int = 6
NBR_DIM: int = 6

# Personalized aggregation weight retained from the local model.
SELF_WEIGHT: float = 0.3

# Shared hidden width used by the GAT encoder, actor, and critic.
GAT_HIDDEN_DIM: int = 64

# PPO optimiser settings.
PPO_LR: float = 3e-4
PPO_UPDATE_EVERY: int = 8
PPO_EPOCHS: int = 4
PPO_CLIP_EPS: float = 0.2
PPO_GAMMA: float = 0.99
PPO_GAE_LAMBDA: float = 0.95
PPO_VALUE_COEF: float = 0.5
PPO_ENTROPY_COEF: float = 0.01
PPO_MAX_GRAD_NORM: float = 1.0
