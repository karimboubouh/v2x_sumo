"""IPPO algorithm configuration.

Hyperparameters are kept identical to GAT_PPO so that any performance
difference is attributable solely to the removal of the graph attention
encoder and the switch to uniform aggregation weights.
"""

# Feature dimensions consumed by the IPPO encoder.
OWN_DIM: int = 6
NBR_DIM: int = 6

# Personalized aggregation: fraction of the local model retained each round.
SELF_WEIGHT: float = 0.3

# MLP hidden width (replaces GAT_HIDDEN_DIM — no attention layer).
MLP_HIDDEN_DIM: int = 64

# PPO optimiser — identical to GAT_PPO for a fair comparison.
PPO_LR: float = 3e-4
PPO_UPDATE_EVERY: int = 8
PPO_EPOCHS: int = 4
PPO_CLIP_EPS: float = 0.2
PPO_GAMMA: float = 0.99
PPO_GAE_LAMBDA: float = 0.95
PPO_VALUE_COEF: float = 0.5
PPO_ENTROPY_COEF: float = 0.01
PPO_MAX_GRAD_NORM: float = 1.0
