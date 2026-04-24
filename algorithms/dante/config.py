"""DANTE algorithm configuration.

The remaining settings are intentionally limited to paper-native quantities,
simulator budget controls, and standard PPO optimization parameters.
"""

# Feature dimensions consumed by the DANTE encoder.
#
# Own state keeps only the signals needed to estimate local learning status and
# remaining round budgets:
#   [val_loss, val_acc, energy_ratio, bandwidth_ratio, latency_slack]
#
# Neighbor state keeps only signals directly tied to expected collaboration
# value and cost:
#   [update_align, energy_cost, latency_cost, link_type, trust, robust_score]
OWN_DIM: int = 5
NBR_DIM: int = 6

# Personalized aggregation weight retained from the local model.
# ``SELF_WEIGHT`` keeps a fixed paper-style self-retention weight. Setting it to
# ``None`` activates the explicit self-weight schedule below.
SELF_WEIGHT: float | None = None
SELF_WEIGHT_START: float = 0.85
SELF_WEIGHT_END: float = 0.60

# Paper-style per-round payoff weights on normalized energy, bandwidth, and
# latency costs. They are kept explicit so the simulator can calibrate
# communication cost without changing the observable validation-drop term.
REWARD_LAMBDA_E: float = 0.03
REWARD_LAMBDA_B: float = 0.02
REWARD_LAMBDA_T: float = 0.03

# Slack in the paper's trust signal:
# F_val(w_i^{t+1}) <= F_val(w_i^t) + tau_val.
VALIDATION_LOSS_SLACK: float = 1e-3

# Candidate and collaboration caps.
MAX_SIDELINK_NEIGHBORS: int = 10
MAX_INTERNET_NEIGHBORS: int = 2
MAX_COLLAB_NEIGHBORS: int = 4
MAX_INTERNET_EXPLORATION_NEIGHBORS: int = 1

# PPO needs a small amount of early collaboration data before it can learn the
# marginal value of peer updates.  Warmup probes are still restricted by the
# simulator energy/bandwidth/latency budgets and by safe aggregation.
EXPLORATION_WARMUP_ROUNDS: int = 20
MIN_EXPLORATION_LINKS: int = 2
EXPLORATION_ALIGNMENT_FLOOR: float = 0.05

# Evaluation split used for runtime metrics and saved plots.
EVAL_SPLIT: str = "test"

# Shared hidden width used by the GAT encoder, actor, and critic.
GAT_HIDDEN_DIM: int = 64

# Trust recursion coefficient from the paper.
TRUST_SMOOTHING: float = 0.2

# Per-round admissibility budgets.
# ``None`` means "derive a simulator-grounded default" inside the algorithm.
ROUND_ENERGY_BUDGET_J: float | None = None
ROUND_BANDWIDTH_BUDGET_BITS: float | None = None
ROUND_LATENCY_BUDGET_S: float | None = None

# Validation-gated aggregation. Fractions scale the maximum non-self mass
# (1 - alpha_ii); 0.0 makes local-only fallback an explicit candidate.
AGGREGATION_MU_FRACTIONS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
SAFE_AGGREGATION_SLACK: float = 2e-2
AGGREGATION_DOMAIN: str = "model"  # model | delta

# With no Byzantine attackers configured, robustness should down-weight unusual
# honest non-IID peers but should not hard-reject them.
DISABLE_ROBUST_REJECTION_WITHOUT_BYZANTINE: bool = True

# Geometry-only quality gates for the efficient reward/trust surrogate.
QUALITY_TRUST_THRESHOLD: float = 0.01

# PPO optimizer settings.
PPO_LR: float = 3e-4
PPO_UPDATE_EVERY: int = 4
PPO_EPOCHS: int = 4
PPO_CLIP_EPS: float = 0.2
PPO_GAMMA: float = 0.99
PPO_GAE_LAMBDA: float = 0.95
PPO_VALUE_COEF: float = 0.5
PPO_ENTROPY_COEF: float = 0.01
PPO_MAX_GRAD_NORM: float = 1.0
