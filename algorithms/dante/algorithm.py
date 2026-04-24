"""
DANTE — paper-faithful GAT + PPO neighbor selection with SL-first retention.

Each vehicle owns an independent policy that:
  1. discovers low-cost peers over sidelink and sparse Uu exploration,
  2. retains previously helpful peers for later Uu reuse,
  3. samples per-candidate Bernoulli actions over the full feasible set,
  4. enforces only hard admissibility after sampling, and
  5. aggregates accepted updates with fused attention, trust, and robustness.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

import config
from algorithms.base import DLAlgorithm, LINK_INTERNET, LINK_SIDELINK
from algorithms.dante.config import (
    AGGREGATION_DOMAIN,
    DISABLE_ROBUST_REJECTION_WITHOUT_BYZANTINE,
    EXPLORATION_ALIGNMENT_FLOOR,
    EXPLORATION_WARMUP_ROUNDS,
    GAT_HIDDEN_DIM,
    MAX_COLLAB_NEIGHBORS,
    MAX_INTERNET_EXPLORATION_NEIGHBORS,
    MAX_INTERNET_NEIGHBORS,
    MAX_SIDELINK_NEIGHBORS,
    MIN_EXPLORATION_LINKS,
    NBR_DIM,
    OWN_DIM,
    PPO_CLIP_EPS,
    PPO_ENTROPY_COEF,
    PPO_EPOCHS,
    PPO_GAE_LAMBDA,
    PPO_GAMMA,
    PPO_LR,
    PPO_MAX_GRAD_NORM,
    PPO_UPDATE_EVERY,
    PPO_VALUE_COEF,
    QUALITY_TRUST_THRESHOLD,
    REWARD_LAMBDA_B,
    REWARD_LAMBDA_E,
    REWARD_LAMBDA_T,
    ROUND_BANDWIDTH_BUDGET_BITS,
    ROUND_ENERGY_BUDGET_J,
    ROUND_LATENCY_BUDGET_S,
    SELF_WEIGHT,
    SELF_WEIGHT_END,
    SELF_WEIGHT_START,
    TRUST_SMOOTHING,
    VALIDATION_LOSS_SLACK,
)
from dl.helpers import (
    clone_state_dict,
    inet_tx_energy_j,
    inet_tx_time_s,
    sl_tx_energy_j,
    sl_tx_time_s,
    tx_payload_bits,
)


ALIGN_IDX = 0
ENERGY_COST_IDX = 1
LATENCY_COST_IDX = 2
LINK_TYPE_IDX = 3
TRUST_IDX = 4
ROBUST_HISTORY_IDX = 5

_EPS = 1e-8
_ROBUST_PASS_THRESHOLD = math.exp(-1.0)


class _GATLayer(nn.Module):
    """Single-head graph attention over one ego node and N candidate neighbors."""

    def __init__(self, own_dim: int, nbr_dim: int, hidden_dim: int):
        super().__init__()
        self.self_proj = nn.Linear(own_dim, hidden_dim, bias=False)
        self.nbr_proj = nn.Linear(nbr_dim, hidden_dim, bias=False)
        self.attn = nn.Linear(hidden_dim * 2, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(
        self,
        own_state: torch.Tensor,
        nbr_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h_i = self.self_proj(own_state)
        if nbr_features.numel() == 0:
            empty_nbr = nbr_features.new_zeros((0, self.nbr_proj.out_features))
            empty_attn = nbr_features.new_zeros((0,))
            return h_i, empty_nbr.new_zeros((self.nbr_proj.out_features,)), empty_nbr, empty_attn

        v_j = self.nbr_proj(nbr_features)
        h_rep = h_i.unsqueeze(0).expand(v_j.shape[0], -1)
        e_ij = self.attn(self.leaky_relu(torch.cat([h_rep, v_j], dim=-1))).squeeze(-1)
        beta_ij = torch.softmax(e_ij, dim=0)
        context = torch.sum(beta_ij.unsqueeze(-1) * v_j, dim=0)
        return h_i, context, v_j, beta_ij


class _GATActorCritic(nn.Module):
    """Shared GAT encoder with Bernoulli actor and scalar critic."""

    def __init__(self, own_dim: int, nbr_dim: int, hidden_dim: int):
        super().__init__()
        self.gat = _GATLayer(own_dim, nbr_dim, hidden_dim)
        self.selector = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, own_state: torch.Tensor, nbr_features: torch.Tensor) -> dict:
        h_i, context, nbr_emb, attention = self.gat(own_state, nbr_features)
        fused = torch.cat([h_i, context], dim=-1)
        value = self.critic(fused).squeeze(-1)

        if nbr_emb.numel() == 0:
            selector_logits = nbr_features.new_zeros((0,))
        else:
            h_rep = h_i.unsqueeze(0).expand(nbr_emb.shape[0], -1)
            context_rep = context.unsqueeze(0).expand(nbr_emb.shape[0], -1)
            selector_in = torch.cat([h_rep, context_rep, nbr_emb], dim=-1)
            selector_logits = self.selector(selector_in).squeeze(-1)

        return {
            "value": value,
            "selector_logits": selector_logits,
            "attention": attention,
        }

    def evaluate_actions(
        self,
        own_state: torch.Tensor,
        nbr_features: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.forward(own_state, nbr_features)
        logits = out["selector_logits"]
        if logits.numel() == 0:
            log_prob = out["value"].new_zeros(())
            entropy = out["value"].new_zeros(())
            return log_prob, entropy, out["value"]
        dist = Bernoulli(logits=logits)
        action_t = actions.to(dtype=torch.float32, device=logits.device)
        log_prob = dist.log_prob(action_t).sum()
        entropy = dist.entropy().sum()
        return log_prob, entropy, out["value"]


class _VehiclePPOAgent:
    """Independent PPO learner attached to a single vehicle."""

    def __init__(self, own_dim: int, nbr_dim: int, hidden_dim: int):
        self.policy = _GATActorCritic(own_dim, nbr_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=PPO_LR)
        self.pending_transition: dict | None = None
        self.pending_round: int | None = None
        self.rollout: list[dict] = []

    def act(
        self,
        own_state: np.ndarray,
        nbr_features: np.ndarray,
    ) -> dict:
        own_t = torch.as_tensor(own_state, dtype=torch.float32)
        nbr_t = torch.as_tensor(nbr_features, dtype=torch.float32)

        self.policy.eval()
        with torch.no_grad():
            out = self.policy(own_t, nbr_t)
            selector_prob = torch.sigmoid(out["selector_logits"])
            if out["selector_logits"].numel() == 0:
                action = np.zeros((0,), dtype=np.float32)
                log_prob = 0.0
            else:
                dist = Bernoulli(logits=out["selector_logits"])
                sampled = dist.sample()
                action = sampled.cpu().numpy().astype(np.float32, copy=True)
                log_prob = float(dist.log_prob(sampled).sum().item())

        return {
            "own_state": own_state.astype(np.float32, copy=True),
            "nbr_features": nbr_features.astype(np.float32, copy=True),
            "action": action,
            "value": float(out["value"].item()),
            "selector_logits": out["selector_logits"].cpu().numpy().astype(np.float32, copy=True),
            "selector_prob": selector_prob.cpu().numpy().astype(np.float32, copy=True),
            "log_prob": float(log_prob),
            "attention": out["attention"].cpu().numpy().astype(np.float32, copy=True),
        }

    def store_pending(self, transition: dict, target_round: int) -> None:
        self.pending_transition = dict(transition)
        self.pending_round = int(target_round)

    def finalize_pending(self, reward: float, next_value: float, done: bool) -> None:
        if self.pending_transition is None:
            return

        transition = dict(self.pending_transition)
        transition.update({
            "reward": float(reward),
            "next_value": float(next_value),
            "done": bool(done),
        })
        self.rollout.append(transition)
        self.pending_transition = None
        self.pending_round = None

    def should_update(self, force: bool = False) -> bool:
        if not self.rollout:
            return False
        return force or len(self.rollout) >= PPO_UPDATE_EVERY

    def update(self) -> None:
        if not self.rollout:
            return

        rewards = np.array([t["reward"] for t in self.rollout], dtype=np.float32)
        values = np.array([t["value"] for t in self.rollout], dtype=np.float32)
        next_values = np.array([t["next_value"] for t in self.rollout], dtype=np.float32)
        dones = np.array([t["done"] for t in self.rollout], dtype=np.float32)

        advantages = np.zeros_like(rewards)
        gae = 0.0
        for idx in reversed(range(len(self.rollout))):
            mask = 1.0 - dones[idx]
            delta = rewards[idx] + PPO_GAMMA * next_values[idx] * mask - values[idx]
            gae = delta + PPO_GAMMA * PPO_GAE_LAMBDA * mask * gae
            advantages[idx] = gae
        returns = advantages + values

        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        indices = np.arange(len(self.rollout))
        self.policy.train()

        for _ in range(PPO_EPOCHS):
            np.random.shuffle(indices)
            for idx in indices:
                sample = self.rollout[idx]
                own_t = torch.as_tensor(sample["own_state"], dtype=torch.float32)
                nbr_t = torch.as_tensor(sample["nbr_features"], dtype=torch.float32)
                act_t = torch.as_tensor(sample["action"], dtype=torch.float32)

                log_prob, entropy, value = self.policy.evaluate_actions(
                    own_t,
                    nbr_t,
                    act_t,
                )
                old_log_prob = torch.tensor(sample["log_prob"], dtype=torch.float32)
                advantage = torch.tensor(advantages[idx], dtype=torch.float32)
                return_t = torch.tensor(returns[idx], dtype=torch.float32)

                ratio = torch.exp(log_prob - old_log_prob)
                surr_1 = ratio * advantage
                surr_2 = torch.clamp(ratio, 1.0 - PPO_CLIP_EPS, 1.0 + PPO_CLIP_EPS) * advantage
                policy_loss = -torch.min(surr_1, surr_2)
                value_loss = F.mse_loss(value, return_t)
                loss = (
                    policy_loss
                    + PPO_VALUE_COEF * value_loss
                    - PPO_ENTROPY_COEF * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), PPO_MAX_GRAD_NORM)
                self.optimizer.step()

        self.rollout.clear()


class DANTEAlgorithm(DLAlgorithm):
    """Per-vehicle paper-grounded DANTE with SL-first retention."""

    name = "DANTE"
    needs_dynamic_neighbors = True
    evaluation_mode = "personalized"
    finalize_rewards_before_selection = True
    own_feature_dim = OWN_DIM
    neighbor_feature_dim = NBR_DIM

    def __init__(self):
        self._agents: dict[int, _VehiclePPOAgent] = {}
        self._retained_neighbors: dict[int, dict[int, dict]] = {}
        self._trust_scores: dict[int, dict[int, float]] = {}
        self._peer_robustness: dict[int, dict[int, dict[str, float | bool]]] = {}
        self._baseline_gain: dict[int, float] = {}
        self._budget_state: dict[int, dict[str, float]] = {}
        self._round_feedback: dict[int, dict] = {}
        self._reward_records: list[dict] = []
        self._debug_lines: list[str] = []
        self._last_selected_neighbors: dict[int, set[int]] = {}
        self._diagnostic_totals = {
            "completed_rounds": 0,
            "retained_offered": 0,
            "retained_selected": 0,
            "retained_skipped": 0,
            "selected_total": 0,
            "selected_useful": 0,
            "selected_internet": 0,
            "retained_positive_reused": 0,
            "fallback_with_selection": 0,
            "positive_dval_negative_reward": 0,
            "selection_overlap_sum": 0.0,
            "retention_survival_sum": 0.0,
            "retention_survival_count": 0,
            "usefulness_sum": 0.0,
            "usefulness_count": 0,
            "baseline_reward_sum": 0.0,
            "normalized_gain_sum": 0.0,
            "normalized_comm_cost_sum": 0.0,
            "reward_positive_count": 0,
            "proposal_pruned_after_budget": 0,
            "proposal_rejected_pretransfer": 0,
            "collaboration_advantage_sum": 0.0,
            "late_round_internet_links": 0,
        }

        self.reward_source = "training"
        self.max_sidelink_neighbors = int(MAX_SIDELINK_NEIGHBORS)
        self.max_internet_neighbors = int(MAX_INTERNET_NEIGHBORS)
        self.max_internet_exploration_neighbors = int(MAX_INTERNET_EXPLORATION_NEIGHBORS)
        self.max_collab_neighbors = int(MAX_COLLAB_NEIGHBORS)
        self.exploration_warmup_rounds = max(int(EXPLORATION_WARMUP_ROUNDS), 0)
        self.min_exploration_links = max(int(MIN_EXPLORATION_LINKS), 0)
        self.exploration_alignment_floor = max(float(EXPLORATION_ALIGNMENT_FLOOR), 0.0)
        self.trust_smoothing = float(np.clip(TRUST_SMOOTHING, 0.0, 1.0))
        self._disable_robust_rejection_without_byzantine = bool(
            DISABLE_ROBUST_REJECTION_WITHOUT_BYZANTINE
        )
        self.validation_loss_slack = max(float(VALIDATION_LOSS_SLACK), 0.0)
        self.quality_trust_threshold = max(float(QUALITY_TRUST_THRESHOLD), 0.0)
        self.aggregation_domain = str(AGGREGATION_DOMAIN).strip().lower()
        if self.aggregation_domain not in {"model", "delta"}:
            raise ValueError(
                f"DANTE AGGREGATION_DOMAIN must be 'model' or 'delta', got {AGGREGATION_DOMAIN!r}"
            )
        self._round_compute_energy_j = self._default_compute_energy_j()
        self.round_energy_budget_j = max(
            (
                float(ROUND_ENERGY_BUDGET_J)
                if ROUND_ENERGY_BUDGET_J is not None
                else self._default_energy_budget_j()
            ),
            _EPS,
        )
        self.payload_bits = float(tx_payload_bits())
        self.round_bandwidth_budget_bits = (
            float(ROUND_BANDWIDTH_BUDGET_BITS)
            if ROUND_BANDWIDTH_BUDGET_BITS is not None
            else self.payload_bits * max(self.max_collab_neighbors, 1)
        )
        self.round_bandwidth_budget_bits = max(self.round_bandwidth_budget_bits, self.payload_bits)
        self.round_latency_budget_s = (
            float(ROUND_LATENCY_BUDGET_S)
            if ROUND_LATENCY_BUDGET_S is not None
            else self._default_latency_budget_s()
        )
        self.round_latency_budget_s = max(self.round_latency_budget_s, _EPS)
        self._max_rounds = max(int(config.MAX_TR_ROUNDS), 1)
        self._total_energy_budget_j = self.round_energy_budget_j * self._max_rounds
        self._total_bandwidth_budget_bits = (
            self.round_bandwidth_budget_bits * self._max_rounds
        )
        self._late_round_start = max(1, int(math.ceil(0.8 * self._max_rounds)))

    def _round_train_samples(self) -> int:
        batches = config.BATCHES_PER_ROUND if config.BATCHES_PER_ROUND else 1
        return max(int(batches), 1) * max(int(config.BATCH_SIZE), 1)

    def _default_compute_energy_j(self) -> float:
        return float(
            float(config.KAPPA)
            * float(self._round_train_samples())
            * float(config.CPU_CYCLES_PER_SAMPLE)
            * (float(config.CPU_FREQ_HZ) ** 2)
        )

    def _default_energy_budget_j(self) -> float:
        # The paper constrains computation plus communication. Use a simulator
        # grounded default that admits the intended small collaboration set while
        # still making early overspending reduce future capacity.
        sl_budget = float(sl_tx_energy_j(config.COMM_RANGE)) * max(self.max_collab_neighbors, 1)
        inet_budget = (
            float(inet_tx_energy_j())
            * max(min(self.max_internet_neighbors, self.max_collab_neighbors), 0)
        )
        return self._default_compute_energy_j() + max(sl_budget, inet_budget, _EPS)

    def _default_latency_budget_s(self) -> float:
        compute_s = (
            float(self._round_train_samples()) * float(config.CPU_CYCLES_PER_SAMPLE)
        ) / max(float(config.CPU_FREQ_HZ), 1.0)
        return compute_s + max(float(inet_tx_time_s()), float(sl_tx_time_s(config.COMM_RANGE)))

    def setup(self, vehicles: list) -> None:
        for v in vehicles:
            self._agents[v.id] = _VehiclePPOAgent(OWN_DIM, NBR_DIM, GAT_HIDDEN_DIM)
            self._retained_neighbors[v.id] = {}
            self._trust_scores[v.id] = {}
            self._peer_robustness[v.id] = {}
            self._last_selected_neighbors[v.id] = set()
            self._baseline_gain[v.id] = 0.0
            self._budget_state[v.id] = {
                "remaining_energy_j": self._total_energy_budget_j,
                "remaining_bandwidth_bits": self._total_bandwidth_budget_bits,
                "last_latency_slack_ratio": 1.0,
            }
            self._round_feedback[v.id] = {"neighbors": {}, "fallback": False}

    def _active_energy_budget(self, vehicle_id: int) -> float:
        remaining = self._budget_state[int(vehicle_id)]["remaining_energy_j"]
        return max(min(self.round_energy_budget_j, remaining), _EPS)

    def _active_comm_energy_budget(self, vehicle_id: int) -> float:
        # Reserve one local training round's compute energy so the paper's
        # E_cmp + E_tx <= E_bar admissibility constraint is respected.
        return max(self._active_energy_budget(vehicle_id) - self._round_compute_energy_j, _EPS)

    def _active_bandwidth_budget(self, vehicle_id: int) -> float:
        remaining = self._budget_state[int(vehicle_id)]["remaining_bandwidth_bits"]
        return max(min(self.round_bandwidth_budget_bits, remaining), _EPS)

    def get_budget_features(self, v) -> tuple[float, float, float]:
        state = self._budget_state[int(v.id if hasattr(v, "id") else v)]
        energy_ratio = float(np.clip(
            state["remaining_energy_j"] / max(self._total_energy_budget_j, _EPS),
            0.0,
            1.0,
        ))
        bandwidth_ratio = float(np.clip(
            state["remaining_bandwidth_bits"] / max(self._total_bandwidth_budget_bits, _EPS),
            0.0,
            1.0,
        ))
        latency_ratio = float(np.clip(state.get("last_latency_slack_ratio", 1.0), 0.0, 1.0))
        return energy_ratio, bandwidth_ratio, latency_ratio

    def _ensure_trust_entry(self, vehicle_id: int, neighbor_id: int) -> float:
        trust = self._trust_scores.setdefault(int(vehicle_id), {})
        if int(neighbor_id) not in trust:
            trust[int(neighbor_id)] = 1.0
        return float(trust[int(neighbor_id)])

    def get_trust_score(self, v, neighbor_id: int) -> float:
        vehicle_id = int(v.id if hasattr(v, "id") else v)
        return self._ensure_trust_entry(vehicle_id, int(neighbor_id))

    def get_retention_value(self, v, neighbor_id: int) -> float:
        return self.get_trust_score(v, neighbor_id)

    def get_baseline_gain(self, v) -> float:
        vehicle_id = int(v.id if hasattr(v, "id") else v)
        return float(self._baseline_gain.get(vehicle_id, 0.0))

    def _set_trust_score(self, vehicle_id: int, neighbor_id: int, value: float) -> None:
        self._trust_scores.setdefault(int(vehicle_id), {})[int(neighbor_id)] = float(
            np.clip(value, 0.0, 1.0)
        )

    def _set_baseline_gain(self, vehicle_id: int, value: float) -> None:
        self._baseline_gain[int(vehicle_id)] = float(np.clip(value, -1.0, 1.0))

    def _ensure_robustness_entry(self, vehicle_id: int, neighbor_id: int) -> dict[str, float | bool]:
        memory = self._peer_robustness.setdefault(int(vehicle_id), {})
        if int(neighbor_id) not in memory:
            memory[int(neighbor_id)] = {
                "last_robust_score": 1.0,
                "last_robust_pass": True,
            }
        return memory[int(neighbor_id)]

    def get_last_robust_score(self, vehicle_id: int, neighbor_id: int) -> float:
        entry = self._ensure_robustness_entry(int(vehicle_id), int(neighbor_id))
        return float(np.clip(entry.get("last_robust_score", 1.0), 0.0, 1.0))

    def get_last_robust_pass(self, vehicle_id: int, neighbor_id: int) -> bool:
        entry = self._ensure_robustness_entry(int(vehicle_id), int(neighbor_id))
        return bool(entry.get("last_robust_pass", True))

    def _update_robustness_memory(
        self,
        vehicle_id: int,
        neighbor_id: int,
        robust_score: float,
        robust_pass: bool,
        alpha: float,
    ) -> None:
        entry = self._ensure_robustness_entry(int(vehicle_id), int(neighbor_id))
        raw_score = float(np.clip(robust_score, 0.0, 1.0))
        effective_pass = bool(robust_pass) and float(alpha) > _EPS
        if effective_pass:
            stored_score = raw_score
        else:
            stored_score = float(np.clip(min(raw_score, _ROBUST_PASS_THRESHOLD), 0.0, 1.0))
        entry["last_robust_score"] = stored_score
        entry["last_robust_pass"] = bool(effective_pass)

    def build_candidates(self, v, env) -> list:
        """Discover new peers over sidelink, plus sparse Uu exploration/reuse."""
        sidelink = env.sidelink_neighbors_of(v)
        visible = {nbr.id for nbr, _, _ in sidelink}
        retained = self._retained_neighbors.get(v.id, {})
        injected = []
        retained_items = sorted(
            retained.items(),
            key=lambda item: (
                -(
                    self.get_trust_score(v.id, item[0])
                    * self.get_last_robust_score(v.id, item[0])
                    * max(
                        self._current_update_align(v, env.vehicles[item[0]])
                        if 0 <= int(item[0]) < len(env.vehicles)
                        else 0.0,
                        0.0,
                    )
                ),
                -self.get_trust_score(v.id, item[0]),
                -self.get_last_robust_score(v.id, item[0]),
                -int(item[1].get("last_good_round", -1)),
                item[0],
            ),
        )
        for nid, meta in retained_items:
            del meta
            if len(injected) >= self.max_internet_neighbors:
                break
            if nid == v.id or nid >= len(env.vehicles):
                self._drop_retained_neighbor(v.id, nid)
                continue
            if nid in visible:
                continue
            nbr = env.vehicles[nid]
            usefulness = self.get_trust_score(v.id, nid)
            history_robust = self.get_last_robust_score(v.id, nid)
            history_pass = self.get_last_robust_pass(v.id, nid)
            # Offer retained peers over Uu only when trust still indicates they are
            # more likely useful than harmful.
            if usefulness <= 0.5 or history_robust <= _EPS or not history_pass:
                continue
            injected.append((nbr, 0.0, LINK_INTERNET))

        exploration_budget = max(
            min(self.max_internet_exploration_neighbors, self.max_internet_neighbors - len(injected)),
            0,
        )
        if exploration_budget > 0 and hasattr(env, "neighbors_of"):
            known = visible | {nbr.id for nbr, _, _ in injected}
            internet_candidates = [
                (nbr, dist, link_type)
                for nbr, dist, link_type in env.neighbors_of(v)
                if float(link_type) == LINK_INTERNET
                and nbr.id not in known
                and nbr.id not in retained
            ]
            internet_candidates.sort(
                key=lambda item: (
                    -max(self._current_update_align(v, item[0]), 0.0),
                    float(item[1]),
                    int(item[0].id),
                )
            )
            injected.extend(internet_candidates[:exploration_budget])

        return sidelink + injected

    def _retain_neighbor(self, vehicle_id: int, neighbor_id: int, round_no: int) -> None:
        retained = self._retained_neighbors.setdefault(int(vehicle_id), {})
        meta = retained.setdefault(int(neighbor_id), {
            "first_retained_round": int(round_no),
            "last_good_round": int(round_no),
            "last_selected_round": int(round_no),
            "selected_count": 0,
            "reuse_count": 0,
        })
        meta["last_good_round"] = int(round_no)

    def _mark_selected_neighbor(
        self,
        vehicle_id: int,
        neighbor_id: int,
        round_no: int,
        reused: bool,
    ) -> None:
        retained = self._retained_neighbors.setdefault(int(vehicle_id), {})
        meta = retained.get(int(neighbor_id))
        if meta is None:
            return
        meta["last_selected_round"] = int(round_no)
        meta["selected_count"] = int(meta.get("selected_count", 0)) + 1
        if reused:
            meta["reuse_count"] = int(meta.get("reuse_count", 0)) + 1

    def _drop_retained_neighbor(self, vehicle_id: int, neighbor_id: int) -> bool:
        retained = self._retained_neighbors.setdefault(int(vehicle_id), {})
        return retained.pop(int(neighbor_id), None) is not None

    def _neighbor_energy_j(self, link_type: float, dist: float) -> float:
        if link_type == LINK_SIDELINK:
            return float(sl_tx_energy_j(dist))
        return float(inet_tx_energy_j())

    def _neighbor_latency_s(self, link_type: float, dist: float) -> float:
        if link_type == LINK_SIDELINK:
            return float(sl_tx_time_s(dist))
        return float(inet_tx_time_s())

    def _neighbor_bandwidth_bits(self, link_type: float, dist: float) -> float:
        del link_type, dist
        return self.payload_bits

    def feature_energy_cost(self, v, link_type: float, dist: float) -> float:
        return float(np.clip(
            self._neighbor_energy_j(link_type, dist) / self._active_comm_energy_budget(v.id),
            0.0,
            4.0,
        ))

    def feature_bandwidth_cost(self, v, link_type: float, dist: float) -> float:
        return float(np.clip(
            self._neighbor_bandwidth_bits(link_type, dist) / self._active_bandwidth_budget(v.id),
            0.0,
            4.0,
        ))

    def feature_latency_cost(self, v, link_type: float, dist: float) -> float:
        return float(np.clip(
            self._neighbor_latency_s(link_type, dist) / self.round_latency_budget_s,
            0.0,
            4.0,
        ))

    def _current_update_align(self, v, nbr) -> float:
        g_v = np.asarray(v.get_update_vec(), dtype=np.float32)
        g_n = np.asarray(nbr.get_update_vec(), dtype=np.float32)
        return float(np.clip(
            np.dot(g_v, g_n) / (np.linalg.norm(g_v) * np.linalg.norm(g_n) + 1e-8),
            -1.0,
            1.0,
        ))

    def _normalized_comm_cost(
        self,
        comm_energy_j: float,
        bandwidth_bits: float,
        latency_s: float,
        energy_budget_j: float | None = None,
        bandwidth_budget_bits: float | None = None,
        latency_budget_s: float | None = None,
    ) -> float:
        energy_budget = max(
            float(self.round_energy_budget_j if energy_budget_j is None else energy_budget_j),
            _EPS,
        )
        bandwidth_budget = max(
            float(
                self.round_bandwidth_budget_bits
                if bandwidth_budget_bits is None
                else bandwidth_budget_bits
            ),
            _EPS,
        )
        latency_budget = max(
            float(self.round_latency_budget_s if latency_budget_s is None else latency_budget_s),
            _EPS,
        )
        return float((
            float(comm_energy_j) / energy_budget
            + float(bandwidth_bits) / bandwidth_budget
            + float(latency_s) / latency_budget
        ) / 3.0)

    def _normalized_cost_terms(
        self,
        comm_energy_j: float,
        bandwidth_bits: float,
        latency_s: float,
        energy_budget_j: float | None = None,
        bandwidth_budget_bits: float | None = None,
        latency_budget_s: float | None = None,
    ) -> tuple[float, float, float]:
        energy_budget = max(
            float(self.round_energy_budget_j if energy_budget_j is None else energy_budget_j),
            _EPS,
        )
        bandwidth_budget = max(
            float(
                self.round_bandwidth_budget_bits
                if bandwidth_budget_bits is None
                else bandwidth_budget_bits
            ),
            _EPS,
        )
        latency_budget = max(
            float(self.round_latency_budget_s if latency_budget_s is None else latency_budget_s),
            _EPS,
        )
        return (
            float(comm_energy_j) / energy_budget,
            float(bandwidth_bits) / bandwidth_budget,
            float(latency_s) / latency_budget,
        )

    def _proposal_score(
        self,
        *,
        source: str,
        attention: float,
        align: float,
        trust: float,
        robust_history: float,
        comm_cost: float,
        align_floor: float = 0.0,
    ) -> float:
        positive_align = max(float(align), float(align_floor), 0.0)
        if positive_align <= _EPS:
            return 0.0
        benefit = (
            float(np.clip(attention, 0.0, 1.0))
            * float(np.clip(trust, 0.0, 1.0))
            * positive_align
        )
        if str(source) == "retained":
            benefit *= float(np.clip(robust_history, 0.0, 1.0))
        return float(benefit / max(float(comm_cost), _EPS))

    def _link_label(self, link_type: float) -> str:
        return "SL" if float(link_type) == LINK_SIDELINK else "IN"

    def _queue_debug_line(self, text: str) -> None:
        self._debug_lines.append(str(text))

    def _format_selected_debug(self, selected_neighbors: list[dict], feedback: dict, trust_updates: dict) -> str:
        if not selected_neighbors:
            return "none"

        parts = []
        for item in selected_neighbors:
            nid = int(item["nid"])
            link_label = self._link_label(item["link_type"])
            source = "ret" if str(item.get("source", "explore")) == "retained" else "new"
            info = trust_updates.get(nid, {})
            peer_feedback = feedback["neighbors"].get(nid, {})
            parts.append(
                f"{nid}:{link_label}/{source}"
                f"(q={info.get('prev_trust', 0.0):.2f}->{info.get('new_trust', info.get('prev_trust', 0.0)):.2f},"
                f"phi={int(info.get('phi', 0.0))},"
                f"a={peer_feedback.get('alpha', 0.0):.2f},"
                f"r={peer_feedback.get('robust_score', 0.0):.2f})"
            )
        return ", ".join(parts)

    def _retained_support_ids(self, vehicle_id: int) -> list[int]:
        retained = self._retained_neighbors.get(int(vehicle_id), {})
        return sorted(
            int(nid)
            for nid in retained
            if self.get_trust_score(vehicle_id, nid) > 0.5
            and self.get_last_robust_pass(vehicle_id, nid)
            and self.get_last_robust_score(vehicle_id, nid) > _EPS
        )

    def _format_retained_debug(self, vehicle_id: int) -> str:
        retained = self._retained_neighbors.get(int(vehicle_id), {})
        support_ids = self._retained_support_ids(vehicle_id)
        if not support_ids:
            return "none"

        parts = []
        for nid in sorted(
            support_ids,
            key=lambda neighbor_id: (
                -(
                    self.get_trust_score(vehicle_id, neighbor_id)
                    * self.get_last_robust_score(vehicle_id, neighbor_id)
                ),
                neighbor_id,
            ),
        ):
            meta = retained[int(nid)]
            parts.append(
                f"{nid}(q={self.get_trust_score(vehicle_id, nid):.2f},"
                f"last={int(meta.get('last_good_round', -1))})"
            )
        return ", ".join(parts)

    def _resolve_self_weight(self, round_no: int, n_neighbors: int) -> float:
        if n_neighbors <= 0:
            return 1.0
        if SELF_WEIGHT is None:
            start = float(SELF_WEIGHT_START)
            end = float(SELF_WEIGHT_END)
            if not (0.0 <= start <= 1.0 and 0.0 <= end <= 1.0):
                raise ValueError(
                    "DANTE SELF_WEIGHT_START/END must be in [0, 1], "
                    f"got {SELF_WEIGHT_START!r} and {SELF_WEIGHT_END!r}"
                )
            if self._max_rounds <= 1:
                return end
            progress = float(np.clip(round_no / max(self._max_rounds - 1, 1), 0.0, 1.0))
            return float(start + (end - start) * progress)
        self_w = float(SELF_WEIGHT)
        if not 0.0 <= self_w <= 1.0:
            raise ValueError(
                f"DANTE SELF_WEIGHT must be None or in [0, 1], got {SELF_WEIGHT!r}"
            )
        return self_w

    def _coordinate_trimmed_mean_array(self, vectors: list[np.ndarray], trim_count: int) -> np.ndarray:
        stacked = np.stack(vectors, axis=0).astype(np.float32, copy=False)
        if trim_count > 0 and stacked.shape[0] > 2 * trim_count:
            sorted_vals = np.sort(stacked, axis=0)
            core = sorted_vals[trim_count:stacked.shape[0] - trim_count]
        else:
            core = stacked
        return core.mean(axis=0)

    def _gradient_robustness_scores(self, grad_vectors: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        if not grad_vectors:
            return (
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=bool),
            )
        if len(grad_vectors) == 1:
            return (
                np.ones((1,), dtype=np.float32),
                np.ones((1,), dtype=bool),
            )

        trim_count = self._trim_count(len(grad_vectors))
        center = self._coordinate_trimmed_mean_array(grad_vectors, trim_count)
        distances = np.asarray(
            [float(np.linalg.norm(vec - center)) for vec in grad_vectors],
            dtype=np.float32,
        )
        positive = distances[distances > _EPS]
        sigma = float(np.median(positive)) if positive.size else float(np.max(distances))
        sigma = max(sigma, _EPS)
        scores = np.exp(-distances / sigma).astype(np.float32, copy=False)
        passes = scores >= _ROBUST_PASS_THRESHOLD
        return scores, passes

    def select_neighbors(self, v, candidates: list, env) -> tuple:
        if not v.training_done.is_set():
            available = {nbr.id: link_type for nbr, _, link_type in candidates}
            connections = {nid for nid in v.connections if nid in available}
            alphas = {nid: float(v.alphas.get(nid, 0.0)) for nid in connections}
            link_types = {nid: available[nid] for nid in connections}
            return connections, alphas, link_types, None

        agent = self._agents[v.id]
        own_state = v.own_features()
        nbr_features = env.neighbor_features(v, candidates)
        retained = self._retained_neighbors.get(v.id, {})
        eligible_retained_ids = {nbr.id for nbr, _, _ in candidates if nbr.id in retained}
        decision = agent.act(own_state, nbr_features)
        sampled_action = np.asarray(decision.get("action", []), dtype=np.float32)

        available_energy = self._active_comm_energy_budget(v.id)
        available_bw = self._active_bandwidth_budget(v.id)
        max_count = int(self.max_collab_neighbors)
        if self.payload_bits > 0.0:
            max_count = min(max_count, int(available_bw // self.payload_bits))

        proposals = []
        pretransfer_rejected = 0
        for idx, (nbr, dist, link_type) in enumerate(candidates):
            if idx >= sampled_action.size or float(sampled_action[idx]) < 0.5:
                continue

            energy_j = self._neighbor_energy_j(link_type, dist)
            bandwidth_bits = self._neighbor_bandwidth_bits(link_type, dist)
            latency_s = self._neighbor_latency_s(link_type, dist)
            if (
                energy_j > available_energy + _EPS
                or bandwidth_bits > available_bw + _EPS
                or latency_s > self.round_latency_budget_s + _EPS
            ):
                pretransfer_rejected += 1
                continue

            attention = (
                float(max(decision["attention"][idx], 0.0))
                if idx < len(decision["attention"])
                else 0.0
            )
            trust = float(nbr_features[idx][TRUST_IDX]) if idx < len(nbr_features) else 1.0
            align = float(nbr_features[idx][ALIGN_IDX]) if idx < len(nbr_features) else 0.0
            robust_history = (
                float(nbr_features[idx][ROBUST_HISTORY_IDX])
                if idx < len(nbr_features)
                else 1.0
            )
            source = "retained" if nbr.id in retained else "explore"
            comm_cost = self._normalized_comm_cost(
                energy_j,
                bandwidth_bits,
                latency_s,
                energy_budget_j=available_energy,
                bandwidth_budget_bits=available_bw,
                latency_budget_s=self.round_latency_budget_s,
            )
            score = self._proposal_score(
                source=source,
                attention=attention,
                align=align,
                trust=trust,
                robust_history=robust_history,
                comm_cost=comm_cost,
            )
            if score <= _EPS:
                pretransfer_rejected += 1
                continue
            proposals.append({
                "idx": int(idx),
                "nid": int(nbr.id),
                "link_type": float(link_type),
                "energy_j": float(energy_j),
                "bandwidth_bits": float(bandwidth_bits),
                "latency_s": float(latency_s),
                "comm_cost": float(comm_cost),
                "beta": float(attention),
                "score": float(score),
                "source": source,
                "selector_prob": float(decision["selector_prob"][idx]) if idx < len(decision["selector_prob"]) else 0.0,
            })

        proposals.sort(
            key=lambda item: (
                float(item["score"]),
                float(item["beta"]),
                float(item["selector_prob"]),
            ),
            reverse=True,
        )

        selected: list[dict] = []
        total_energy = 0.0
        total_bw = 0.0
        for item in proposals:
            if len(selected) >= max_count:
                break
            next_energy = total_energy + float(item["energy_j"])
            next_bw = total_bw + float(item["bandwidth_bits"])
            if next_energy > available_energy + _EPS:
                continue
            if next_bw > available_bw + _EPS:
                continue
            selected.append(item)
            total_energy = next_energy
            total_bw = next_bw

        warmup_active = int(v.tr_rounds) < self.exploration_warmup_rounds
        min_required = min(
            self.min_exploration_links if warmup_active else 0,
            max_count,
        )
        should_probe = len(selected) < min_required or (not selected and not eligible_retained_ids)
        if should_probe and len(selected) < max_count:
            probe_candidates = []
            selected_ids = {int(item["nid"]) for item in selected}
            used_indices = {int(item["idx"]) for item in selected}
            probe_align_floor = self.exploration_alignment_floor if warmup_active else 0.0
            for idx, (nbr, dist, link_type) in enumerate(candidates):
                if int(idx) in used_indices or int(nbr.id) in selected_ids:
                    continue
                if not warmup_active and (float(link_type) != LINK_SIDELINK or nbr.id in retained):
                    continue
                energy_j = self._neighbor_energy_j(link_type, dist)
                bandwidth_bits = self._neighbor_bandwidth_bits(link_type, dist)
                latency_s = self._neighbor_latency_s(link_type, dist)
                if (
                    total_energy + energy_j > available_energy + _EPS
                    or total_bw + bandwidth_bits > available_bw + _EPS
                    or latency_s > self.round_latency_budget_s + _EPS
                ):
                    continue
                attention = (
                    float(max(decision["attention"][idx], 0.0))
                    if idx < len(decision["attention"])
                    else 0.0
                )
                trust = float(nbr_features[idx][TRUST_IDX]) if idx < len(nbr_features) else 1.0
                align = float(nbr_features[idx][ALIGN_IDX]) if idx < len(nbr_features) else 0.0
                if align < 0.0:
                    continue
                robust_history = (
                    float(nbr_features[idx][ROBUST_HISTORY_IDX])
                    if idx < len(nbr_features)
                    else 1.0
                )
                comm_cost = self._normalized_comm_cost(
                    energy_j,
                    bandwidth_bits,
                    latency_s,
                    energy_budget_j=available_energy,
                    bandwidth_budget_bits=available_bw,
                    latency_budget_s=self.round_latency_budget_s,
                )
                score = self._proposal_score(
                    source="explore",
                    attention=attention,
                    align=align,
                    trust=trust,
                    robust_history=robust_history,
                    comm_cost=comm_cost,
                    align_floor=probe_align_floor,
                )
                if score <= _EPS:
                    continue
                probe_candidates.append({
                    "idx": int(idx),
                    "nid": int(nbr.id),
                    "link_type": float(link_type),
                    "energy_j": float(energy_j),
                    "bandwidth_bits": float(bandwidth_bits),
                    "latency_s": float(latency_s),
                    "comm_cost": float(comm_cost),
                    "beta": float(attention),
                    "score": float(score),
                    "source": "retained" if nbr.id in retained else "explore",
                    "selector_prob": float(decision["selector_prob"][idx]) if idx < len(decision["selector_prob"]) else 0.0,
                    "forced_probe": True,
                })
            probe_candidates.sort(
                key=lambda item: (
                    1 if float(item["link_type"]) == LINK_SIDELINK else 0,
                    float(item["score"]),
                    float(item["beta"]),
                    float(item["selector_prob"]),
                ),
                reverse=True,
            )
            for item in probe_candidates:
                if len(selected) >= max_count:
                    break
                if len(selected) >= min_required and not (not selected and not eligible_retained_ids):
                    break
                next_energy = total_energy + float(item["energy_j"])
                next_bw = total_bw + float(item["bandwidth_bits"])
                if next_energy > available_energy + _EPS or next_bw > available_bw + _EPS:
                    continue
                selected.append(item)
                total_energy = next_energy
                total_bw = next_bw

        proposal_pruned_after_budget = max(len(proposals) - len(selected), 0)
        connections = {item["nid"] for item in selected}
        link_types = {item["nid"]: item["link_type"] for item in selected}
        alphas = {}
        if selected:
            beta = np.array([max(item["beta"], 0.0) for item in selected], dtype=np.float32)
            total_beta = float(beta.sum())
            if total_beta <= _EPS:
                beta = np.full(len(selected), 1.0 / len(selected), dtype=np.float32)
            else:
                beta = beta / total_beta
            for item, weight in zip(selected, beta):
                alphas[item["nid"]] = float(weight)

        if not v.training_done.is_set() or env._vehicle_is_done(v):
            return connections, alphas, link_types, None

        executed_mask = np.zeros((len(candidates),), dtype=np.float32)
        for item in selected:
            executed_mask[int(item["idx"])] = 1.0
        selector_logits_t = torch.as_tensor(
            decision["selector_logits"],
            dtype=torch.float32,
        )
        executed_mask_t = torch.as_tensor(executed_mask, dtype=torch.float32)
        if selector_logits_t.numel() == 0:
            log_prob = 0.0
        else:
            log_prob = float(Bernoulli(logits=selector_logits_t).log_prob(executed_mask_t).sum().item())
        transition = {
            "own_state": decision["own_state"],
            "nbr_features": decision["nbr_features"],
            "action": executed_mask.astype(np.float32, copy=True),
            "log_prob": float(log_prob),
            "value": decision["value"],
            "comm_energy_j": float(sum(item["energy_j"] for item in selected)),
            "bandwidth_bits": float(sum(item["bandwidth_bits"] for item in selected)),
            "latency_s": float(max((item["latency_s"] for item in selected), default=0.0)),
            "energy_budget_j": self._active_energy_budget(v.id),
            "comm_energy_budget_j": float(available_energy),
            "bandwidth_budget_bits": self._active_bandwidth_budget(v.id),
            "latency_budget_s": self.round_latency_budget_s,
            "eligible_retained_ids": sorted(int(nid) for nid in eligible_retained_ids),
            "candidate_counts": {
                "sl": int(sum(1 for _, _, lt in candidates if float(lt) == LINK_SIDELINK)),
                "in": int(sum(1 for _, _, lt in candidates if float(lt) == LINK_INTERNET)),
                "total": int(len(candidates)),
            },
            "proposal_rejected_pretransfer": int(pretransfer_rejected),
            "proposal_pruned_after_budget": int(proposal_pruned_after_budget),
            "selected_neighbors": [
                {
                    "nid": int(item["nid"]),
                    "link_type": float(item["link_type"]),
                    "source": str(item.get("source", "explore")),
                }
                for item in selected
            ],
            "target_round": int(v.tr_rounds + 1),
        }
        return connections, alphas, link_types, transition

    def _trim_count(self, n_neighbors: int) -> int:
        if n_neighbors <= 1:
            return 0
        byz_frac = max(float(getattr(config, "BYZANTINE_FRACTION", 0.0)), 0.0)
        trim = int(math.floor(byz_frac * n_neighbors))
        return max(min(trim, (n_neighbors - 1) // 2), 0)

    def _coordinate_trimmed_mean(self, nbr_sds: list[dict], trim_count: int) -> dict:
        template = nbr_sds[0]
        trimmed = {}
        for key, tensor in template.items():
            if not tensor.is_floating_point():
                trimmed[key] = tensor.clone()
                continue
            stacked = torch.stack([sd[key].float() for sd in nbr_sds], dim=0)
            if trim_count > 0 and stacked.shape[0] > 2 * trim_count:
                sorted_vals, _ = torch.sort(stacked, dim=0)
                core = sorted_vals[trim_count:stacked.shape[0] - trim_count]
            else:
                core = stacked
            trimmed[key] = core.mean(dim=0)
        return trimmed

    def _robustness_scores(self, nbr_sds: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not nbr_sds:
            return (
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=bool),
            )
        if len(nbr_sds) == 1:
            return (
                np.zeros((1,), dtype=np.float32),
                np.ones((1,), dtype=np.float32),
                np.ones((1,), dtype=bool),
            )

        trim_count = self._trim_count(len(nbr_sds))
        trimmed_mean = self._coordinate_trimmed_mean(nbr_sds, trim_count)
        distances = []
        for sd in nbr_sds:
            sq_norm = 0.0
            for key, center in trimmed_mean.items():
                tensor = sd[key]
                if not tensor.is_floating_point():
                    continue
                diff = tensor.float() - center
                sq_norm += float(torch.sum(diff * diff).item())
            distances.append(math.sqrt(max(sq_norm, 0.0)))
        distances = np.asarray(distances, dtype=np.float32)
        positive = distances[distances > _EPS]
        sigma = float(np.median(positive)) if positive.size else float(np.max(distances))
        sigma = max(sigma, _EPS)
        scores = np.exp(-distances / sigma).astype(np.float32, copy=False)
        passes = scores >= _ROBUST_PASS_THRESHOLD
        if (
            self._disable_robust_rejection_without_byzantine
            and float(getattr(config, "BYZANTINE_FRACTION", 0.0)) <= 0.0
        ):
            passes = np.ones_like(passes, dtype=bool)
        return distances, scores, passes

    def _state_delta(self, current_state: dict, reference_state: dict) -> dict:
        delta = {}
        for key, tensor in current_state.items():
            if not tensor.is_floating_point():
                delta[key] = tensor.clone()
                continue
            ref_tensor = reference_state.get(key, tensor)
            delta[key] = tensor.detach().clone().float() - ref_tensor.detach().clone().float()
        return delta

    def _state_vec(self, state_dict: dict) -> np.ndarray:
        parts = []
        for tensor in state_dict.values():
            if not tensor.is_floating_point():
                continue
            parts.append(tensor.detach().cpu().numpy().ravel().astype(np.float32, copy=False))
        if not parts:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(parts).astype(np.float32, copy=False)

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.clip(
            float(np.dot(a, b)) / (float(np.linalg.norm(a)) * float(np.linalg.norm(b)) + _EPS),
            -1.0,
            1.0,
        ))

    def _sign_alignment(self, a: np.ndarray, b: np.ndarray) -> float:
        mask = (np.abs(a) > _EPS) | (np.abs(b) > _EPS)
        if not np.any(mask):
            return 0.0
        return float(np.mean(np.sign(a[mask]) == np.sign(b[mask])))

    def _peer_update_state(self, nbr) -> dict:
        get_update = getattr(nbr, "get_shared_update", None)
        if callable(get_update):
            return get_update()

        shared = nbr.get_shared_weights()
        reference = getattr(nbr, "_ref_weights", None)
        if reference is None:
            return clone_state_dict(shared)
        return self._state_delta(shared, reference)

    def aggregate(self, v, vehicles: list) -> None:
        if not v.training_done.is_set():
            return

        accepted = [vehicles[nid] for nid in v.connections if nid < len(vehicles)]
        if not accepted:
            self._round_feedback[v.id] = {"neighbors": {}, "fallback": True}
            return

        base_weights = np.array(
            [max(float(v.alphas.get(nbr.id, 0.0)), 0.0) for nbr in accepted],
            dtype=np.float32,
        )
        total_base = float(base_weights.sum())
        if total_base <= _EPS:
            base_weights = np.full(len(accepted), 1.0 / len(accepted), dtype=np.float32)
        else:
            base_weights = base_weights / total_base

        trusts = np.array(
            [self.get_trust_score(v.id, nbr.id) for nbr in accepted],
            dtype=np.float32,
        )
        peer_updates = [self._peer_update_state(nbr) for nbr in accepted]
        peer_model_states = (
            [nbr.get_shared_weights() for nbr in accepted]
            if self.aggregation_domain == "model"
            else []
        )
        distances, robust_scores, robust_passes = self._robustness_scores(peer_updates)

        with v._lock:
            own_sd = clone_state_dict(v.model.state_dict())
            ref_sd = clone_state_dict(getattr(v, "_ref_weights", own_sd))

        local_update = self._state_delta(own_sd, ref_sd)
        local_vec = self._state_vec(local_update)
        local_norm = float(np.linalg.norm(local_vec))
        peer_vecs = [self._state_vec(update) for update in peer_updates]
        peer_norms = np.asarray(
            [float(np.linalg.norm(vec)) for vec in peer_vecs],
            dtype=np.float32,
        )
        alignments = np.asarray(
            [self._cosine(local_vec, vec) if local_vec.size and vec.size else 0.0 for vec in peer_vecs],
            dtype=np.float32,
        )
        norm_ratios = np.asarray(
            [
                min(
                    float(peer_norm) / max(local_norm, _EPS),
                    local_norm / max(float(peer_norm), _EPS),
                )
                for peer_norm in peer_norms
            ],
            dtype=np.float32,
        )
        norm_ratios = np.clip(norm_ratios, 0.0, 1.0)
        sign_alignments = np.asarray(
            [self._sign_alignment(local_vec, vec) if local_vec.size and vec.size else 0.0 for vec in peer_vecs],
            dtype=np.float32,
        )
        sign_factors = 0.5 + 0.5 * np.clip(sign_alignments, 0.0, 1.0)
        warmup_floor = (
            self.exploration_alignment_floor
            if int(v.tr_rounds) < self.exploration_warmup_rounds
            else 0.0
        )
        positive_alignment = np.clip(alignments, 0.0, 1.0)
        direction_scores = np.maximum(positive_alignment, warmup_floor)
        geometry_quality = np.clip(
            (
                0.60 * positive_alignment
                + 0.20 * norm_ratios
                + 0.20 * np.clip(sign_alignments, 0.0, 1.0)
            )
            * robust_scores,
            0.0,
            1.0,
        )
        selection_quality = np.maximum(geometry_quality, warmup_floor * robust_scores)
        effective_scores = (
            base_weights
            * trusts
            * robust_passes.astype(np.float32)
            * selection_quality
        )
        total_effective = float(effective_scores.sum())

        round_feedback = {
            "neighbors": {},
            "fallback": total_effective <= _EPS,
            "surrogate_gain": 0.0,
            "policy_gain": 0.0,
            "update_alignment": 0.0,
            "agg_delta_norm": 0.0,
            "local_update_norm": float(local_norm),
        }
        for nbr, base_beta, trust, robust_score, passed, distance, align, norm_ratio, sign_alignment, quality in zip(
            accepted,
            base_weights,
            trusts,
            robust_scores,
            robust_passes,
            distances,
            alignments,
            norm_ratios,
            sign_alignments,
            geometry_quality,
        ):
            round_feedback["neighbors"][nbr.id] = {
                "base_beta": float(base_beta),
                "trust": float(trust),
                "robust_score": float(robust_score),
                "robust_pass": bool(passed),
                "distance": float(distance),
                "alignment": float(align),
                "norm_ratio": float(norm_ratio),
                "sign_alignment": float(sign_alignment),
                "quality": float(quality),
                "alpha": 0.0,
            }

        if total_effective <= _EPS:
            v.alphas = {}
            self._round_feedback[v.id] = round_feedback
            return

        nbr_weights = effective_scores / total_effective
        self_w = self._resolve_self_weight(v.tr_rounds, len(accepted))
        peer_mass_cap = max(1.0 - self_w, 0.0)

        if peer_mass_cap <= _EPS:
            with v._lock:
                v.model.load_state_dict(own_sd)
                v._param_vec = None
            v.alphas = {}
            v.connections = set()
            round_feedback["fallback"] = True
            self._round_feedback[v.id] = round_feedback
            return

        best_state = clone_state_dict(own_sd if self.aggregation_domain == "model" else ref_sd)
        weighted_peer_vec = np.zeros_like(local_vec, dtype=np.float32)
        for weight, peer_vec in zip(nbr_weights, peer_vecs):
            if peer_vec.shape == weighted_peer_vec.shape:
                weighted_peer_vec += float(peer_mass_cap) * float(weight) * peer_vec

        for key in best_state:
            if not best_state[key].is_floating_point():
                continue
            aggregate_update = torch.zeros_like(best_state[key].float())
            if self.aggregation_domain == "model":
                for weight, state in zip(nbr_weights, peer_model_states):
                    peer_tensor = state.get(key)
                    if peer_tensor is None or not peer_tensor.is_floating_point():
                        continue
                    aggregate_update = aggregate_update + float(weight) * peer_tensor.float()
                best_state[key] = (
                    float(self_w) * own_sd[key].float()
                    + float(peer_mass_cap) * aggregate_update
                )
            else:
                local_tensor = local_update.get(key, torch.zeros_like(best_state[key]))
                for weight, update in zip(nbr_weights, peer_updates):
                    peer_tensor = update.get(key)
                    if peer_tensor is None or not peer_tensor.is_floating_point():
                        continue
                    aggregate_update = aggregate_update + float(weight) * peer_tensor.float()
                best_state[key] = (
                    ref_sd[key].float()
                    + float(self_w) * local_tensor.float()
                    + float(peer_mass_cap) * aggregate_update
                )

        if local_vec.size and weighted_peer_vec.size:
            surrogate_gain = float(np.dot(local_vec, weighted_peer_vec) / (local_norm * local_norm + _EPS))
            update_alignment = self._cosine(local_vec, weighted_peer_vec)
        else:
            surrogate_gain = 0.0
            update_alignment = 0.0
        round_feedback["surrogate_gain"] = float(np.clip(surrogate_gain, -1.0, 1.0))
        round_feedback["policy_gain"] = float(
            np.clip(peer_mass_cap * float(np.dot(nbr_weights, geometry_quality)), 0.0, 1.0)
        )
        round_feedback["update_alignment"] = float(update_alignment)
        round_feedback["agg_delta_norm"] = float(np.linalg.norm(weighted_peer_vec))

        final_alphas = {}
        with v._lock:
            v.model.load_state_dict(best_state)
            v._param_vec = None

        for nbr, weight in zip(accepted, nbr_weights):
            alpha = peer_mass_cap * float(weight)
            if alpha > 0.0:
                final_alphas[nbr.id] = alpha
            round_feedback["neighbors"][nbr.id]["alpha"] = alpha

        v.alphas = final_alphas
        v.connections = set(final_alphas)
        self._round_feedback[v.id] = round_feedback

    def _compute_reward(
        self,
        learning_term: float,
        comm_energy_j: float,
        bandwidth_bits: float,
        latency_s: float,
        energy_budget_j: float,
        bandwidth_budget_bits: float,
        latency_budget_s: float,
    ) -> float:
        energy_cost, bandwidth_cost, latency_cost = self._normalized_cost_terms(
            comm_energy_j,
            bandwidth_bits,
            latency_s,
            energy_budget_j=energy_budget_j,
            bandwidth_budget_bits=bandwidth_budget_bits,
            latency_budget_s=latency_budget_s,
        )
        weighted_cost = (
            float(REWARD_LAMBDA_E) * energy_cost
            + float(REWARD_LAMBDA_B) * bandwidth_cost
            + float(REWARD_LAMBDA_T) * latency_cost
        )
        return float(learning_term - weighted_cost)

    def _consume_round_feedback(self, vehicle_id: int) -> dict:
        feedback = self._round_feedback.get(int(vehicle_id), {"neighbors": {}, "fallback": True})
        self._round_feedback[int(vehicle_id)] = {"neighbors": {}, "fallback": True}
        return feedback

    def post_step(self, vehicles: list, transitions: dict, step_n: int) -> dict:
        del step_n
        rewards = {}

        for v in vehicles:
            agent = self._agents[v.id]
            next_transition = transitions.get(v.id)

            if (
                agent.pending_transition is not None
                and agent.pending_round is not None
                and v.tr_rounds >= agent.pending_round
            ):
                pending = dict(agent.pending_transition)
                feedback = self._consume_round_feedback(v.id)

                with v._lock:
                    prev_reward_loss = float(
                        getattr(v, "_prev_reward_loss", getattr(v, "_prev_val_loss", 0.0))
                    )
                    current_reward_loss = float(
                        getattr(v, "current_reward_loss", getattr(v, "current_val_loss", 0.0))
                    )
                    prev_val_loss = float(getattr(v, "_prev_val_loss", prev_reward_loss))
                    current_val_loss = float(getattr(v, "current_val_loss", current_reward_loss))

                reward_loss_delta = prev_reward_loss - current_reward_loss
                val_loss_delta = prev_val_loss - current_val_loss
                if abs(reward_loss_delta) > _EPS or abs(val_loss_delta) <= _EPS:
                    prev_loss_for_metric = prev_reward_loss
                    loss_delta_metric = reward_loss_delta
                else:
                    prev_loss_for_metric = prev_val_loss
                    loss_delta_metric = val_loss_delta

                metric_gain = float(loss_delta_metric) / max(float(prev_loss_for_metric), _EPS)
                baseline_reward_before = self.get_baseline_gain(v.id)
                bandwidth_bits = float(pending.get("bandwidth_bits", 0.0))
                latency_s = float(pending.get("latency_s", 0.0))
                latency_budget_s = float(pending.get("latency_budget_s", self.round_latency_budget_s))
                comm_energy_j = float(pending.get("comm_energy_j", 0.0))
                selected_neighbors = pending.get("selected_neighbors", [])
                local_like_round = (not selected_neighbors) or bool(feedback.get("fallback", False))
                if local_like_round:
                    comm_energy_j = 0.0
                    bandwidth_bits = 0.0
                    latency_s = 0.0

                comp_hist = getattr(v, "computation_energy_hist", [])
                comp_energy_delta = float(comp_hist[-1]) if comp_hist else 0.0
                total_energy_j = float(comm_energy_j) + max(comp_energy_delta, 0.0)
                energy_budget_j = float(pending.get("energy_budget_j", self.round_energy_budget_j))
                comm_energy_budget_j = float(
                    pending.get("comm_energy_budget_j", max(energy_budget_j - self._round_compute_energy_j, _EPS))
                )
                surrogate_gain = float(
                    feedback.get("policy_gain", feedback.get("surrogate_gain", metric_gain))
                )
                learning_gain = 0.0 if local_like_round else float(np.clip(surrogate_gain, -1.0, 1.0))
                normalized_gain = learning_gain
                normalized_comm_cost = self._normalized_comm_cost(
                    comm_energy_j,
                    bandwidth_bits,
                    latency_s,
                    energy_budget_j=comm_energy_budget_j,
                    bandwidth_budget_bits=self.round_bandwidth_budget_bits,
                    latency_budget_s=self.round_latency_budget_s,
                )

                reward = self._compute_reward(
                    learning_term=learning_gain,
                    comm_energy_j=comm_energy_j,
                    bandwidth_bits=bandwidth_bits,
                    latency_s=latency_s,
                    energy_budget_j=comm_energy_budget_j,
                    bandwidth_budget_bits=self.round_bandwidth_budget_bits,
                    latency_budget_s=self.round_latency_budget_s,
                )
                collaboration_advantage = float(reward - baseline_reward_before)
                updated_baseline = baseline_reward_before
                if local_like_round:
                    updated_baseline = (
                        (1.0 - self.trust_smoothing) * baseline_reward_before
                        + self.trust_smoothing * reward
                    )
                    self._set_baseline_gain(v.id, updated_baseline)

                budget_state = self._budget_state[v.id]
                budget_state["remaining_energy_j"] = max(
                    budget_state["remaining_energy_j"] - total_energy_j,
                    0.0,
                )
                budget_state["remaining_bandwidth_bits"] = max(
                    budget_state["remaining_bandwidth_bits"] - bandwidth_bits,
                    0.0,
                )
                budget_state["last_latency_slack_ratio"] = float(np.clip(
                    (latency_budget_s - latency_s) / max(latency_budget_s, _EPS),
                    0.0,
                    1.0,
                ))

                round_no = int(agent.pending_round)
                trust_updates = {}

                selected_ids = {int(item["nid"]) for item in selected_neighbors}
                eligible_retained_ids = {
                    int(nid) for nid in pending.get("eligible_retained_ids", [])
                }
                useful_selected_ids: set[int] = set()
                retained_positive_reused_ids: set[int] = set()

                for item in selected_neighbors:
                    nid = int(item["nid"])
                    source = str(item.get("source", "explore"))
                    peer_feedback = feedback["neighbors"].get(nid, {})
                    robust_pass = bool(peer_feedback.get("robust_pass", False))
                    alpha = float(peer_feedback.get("alpha", 0.0))
                    robust_score = float(peer_feedback.get("robust_score", 0.0))
                    self._update_robustness_memory(
                        v.id,
                        nid,
                        robust_score=robust_score,
                        robust_pass=robust_pass,
                        alpha=alpha,
                    )
                    quality = float(peer_feedback.get("quality", max(learning_gain, 0.0)))
                    phi = float(np.clip(quality, 0.0, 1.0)) if (
                        robust_pass
                        and alpha > _EPS
                        and quality >= self.quality_trust_threshold
                    ) else 0.0
                    prev_trust = self.get_trust_score(v.id, nid)
                    updated_trust = (
                        (1.0 - self.trust_smoothing) * prev_trust
                        + self.trust_smoothing * phi
                    )
                    self._set_trust_score(v.id, nid, updated_trust)
                    trust_updates[nid] = {
                        "prev_trust": float(prev_trust),
                        "new_trust": float(updated_trust),
                        "phi": float(phi),
                    }
                    reused = source == "retained"

                    if phi > 0.0:
                        useful_selected_ids.add(nid)
                        if reused:
                            retained_positive_reused_ids.add(nid)
                        self._retain_neighbor(v.id, nid, round_no)
                        self._mark_selected_neighbor(v.id, nid, round_no, reused=reused)
                    elif reused:
                        self._mark_selected_neighbor(v.id, nid, round_no, reused=True)
                    elif updated_trust <= _EPS and not self.get_last_robust_pass(v.id, nid):
                        self._drop_retained_neighbor(v.id, nid)

                next_value = float(next_transition["value"]) if next_transition is not None else 0.0
                done = next_transition is None
                agent.finalize_pending(reward, next_value, done)
                v.reward_hist.append(reward)
                rewards[v.id] = reward

                selected_sl = int(sum(
                    1 for item in selected_neighbors
                    if float(item["link_type"]) == LINK_SIDELINK
                ))
                selected_in = int(sum(
                    1 for item in selected_neighbors
                    if float(item["link_type"]) == LINK_INTERNET
                ))
                candidate_counts = pending.get("candidate_counts", {})
                retained_selected_ids = sorted(selected_ids & eligible_retained_ids)
                retained_skipped_ids = sorted(eligible_retained_ids - selected_ids)
                prev_selected_ids = self._last_selected_neighbors.get(v.id, set())
                selection_union = prev_selected_ids | selected_ids
                selection_overlap = (
                    len(prev_selected_ids & selected_ids) / len(selection_union)
                    if selection_union
                    else 1.0
                )
                dropped_prev_selected = sorted(prev_selected_ids - selected_ids)
                self._last_selected_neighbors[v.id] = set(selected_ids)
                self._diagnostic_totals["completed_rounds"] += 1
                self._diagnostic_totals["retained_offered"] += len(eligible_retained_ids)
                self._diagnostic_totals["retained_selected"] += len(retained_selected_ids)
                self._diagnostic_totals["retained_skipped"] += len(retained_skipped_ids)
                self._diagnostic_totals["selected_total"] += len(selected_neighbors)
                self._diagnostic_totals["selected_useful"] += len(useful_selected_ids)
                self._diagnostic_totals["selected_internet"] += selected_in
                self._diagnostic_totals["selection_overlap_sum"] += float(selection_overlap)
                self._diagnostic_totals["retained_positive_reused"] += len(retained_positive_reused_ids)
                self._diagnostic_totals["baseline_reward_sum"] += float(updated_baseline)
                self._diagnostic_totals["normalized_gain_sum"] += float(normalized_gain)
                self._diagnostic_totals["normalized_comm_cost_sum"] += float(normalized_comm_cost)
                self._diagnostic_totals["reward_positive_count"] += int(float(reward) > 0.0)
                self._diagnostic_totals["proposal_rejected_pretransfer"] += int(
                    pending.get("proposal_rejected_pretransfer", 0)
                )
                self._diagnostic_totals["proposal_pruned_after_budget"] += int(
                    pending.get("proposal_pruned_after_budget", 0)
                )
                self._diagnostic_totals["collaboration_advantage_sum"] += float(collaboration_advantage)
                if selected_neighbors and bool(feedback.get("fallback", False)):
                    self._diagnostic_totals["fallback_with_selection"] += 1
                if float(loss_delta_metric) > 0.0 and float(reward) < 0.0:
                    self._diagnostic_totals["positive_dval_negative_reward"] += 1
                if round_no >= self._late_round_start:
                    self._diagnostic_totals["late_round_internet_links"] += int(selected_in)
                for nid in eligible_retained_ids:
                    meta = self._retained_neighbors.get(v.id, {}).get(nid)
                    if meta is not None:
                        first_round = int(meta.get("first_retained_round", round_no))
                        self._diagnostic_totals["retention_survival_sum"] += float(
                            max(round_no - first_round + 1, 0)
                        )
                        self._diagnostic_totals["retention_survival_count"] += 1
                usefulness_values = [
                    self.get_trust_score(v.id, nid)
                    for nid in self._retained_neighbors.get(v.id, {})
                ]
                self._diagnostic_totals["usefulness_sum"] += float(sum(usefulness_values))
                self._diagnostic_totals["usefulness_count"] += int(len(usefulness_values))
                self._reward_records.append({
                    "vehicle_id": int(v.id),
                    "round": int(round_no),
                    "reward": float(reward),
                    "normalized_gain": float(normalized_gain),
                    "normalized_comm_cost": float(normalized_comm_cost),
                    "collaboration_advantage": float(collaboration_advantage),
                })

                energy_ratio, bandwidth_ratio, latency_ratio = self.get_budget_features(v)
                self._queue_debug_line(
                    f"DANTE V{v.id} R{round_no} | cand SL/IN {int(candidate_counts.get('sl', 0))}/{int(candidate_counts.get('in', 0))}"
                    f" | sel SL/IN {selected_sl}/{selected_in}"
                    f" | retained pool/offered/sel/skip {len(self._retained_support_ids(v.id))}/{len(eligible_retained_ids)}/{len(retained_selected_ids)}/{len(retained_skipped_ids)}"
                    f" | reward {reward:+.4f} | dLoss {float(loss_delta_metric):+.4f}"
                    f" | gain/base_reward/adv {normalized_gain:+.4f}/{baseline_reward_before:+.4f}/{float(collaboration_advantage):+.4f}"
                    f" | comm_cost {normalized_comm_cost:.4f} | useful {len(useful_selected_ids)}/{len(selected_neighbors)}"
                    f" | energy comm/comp/tot {comm_energy_j:.4f}/{max(comp_energy_delta, 0.0):.4f}/{total_energy_j:.4f} J"
                    f" | bw {bandwidth_bits:.0f} b | lat {latency_s:.4f}/{latency_budget_s:.4f} s"
                    f" | budget E/B/L {energy_ratio:.2f}/{bandwidth_ratio:.2f}/{latency_ratio:.2f}"
                    f" | fallback {int(bool(feedback.get('fallback', False)))}"
                    f" | overlap {selection_overlap:.2f}"
                    f" | reject/prune {int(pending.get('proposal_rejected_pretransfer', 0))}/{int(pending.get('proposal_pruned_after_budget', 0))}"
                )
                self._queue_debug_line(
                    f"DANTE V{v.id} peers | selected [{self._format_selected_debug(selected_neighbors, feedback, trust_updates)}]"
                    f" | retained [{self._format_retained_debug(v.id)}]"
                    f" | skipped_retained {retained_skipped_ids if retained_skipped_ids else '[]'}"
                    f" | dropped_prev {dropped_prev_selected if dropped_prev_selected else '[]'}"
                )

            if next_transition is not None:
                next_feedback = self._round_feedback.get(v.id, {"neighbors": {}, "fallback": False})
                if bool(next_feedback.get("fallback", False)):
                    next_transition = dict(next_transition)
                    next_transition["selected_neighbors"] = []
                    next_transition["comm_energy_j"] = 0.0
                    next_transition["bandwidth_bits"] = 0.0
                    next_transition["latency_s"] = 0.0
                agent.store_pending(next_transition, next_transition["target_round"])

            force_update = next_transition is None and agent.pending_transition is None
            if agent.should_update(force=force_update):
                agent.update()

        return rewards

    def _reward_window_stats(self) -> dict[str, dict[str, float]]:
        records = list(self._reward_records)
        if not records:
            empty = {
                "mean_reward": 0.0,
                "positive_reward_rate": 0.0,
                "mean_collaboration_advantage": 0.0,
            }
            return {"early": dict(empty), "mid": dict(empty), "late": dict(empty)}

        rewards = np.asarray([float(item["reward"]) for item in records], dtype=np.float32)
        collaboration_advantages = np.asarray(
            [float(item.get("collaboration_advantage", 0.0)) for item in records],
            dtype=np.float32,
        )
        index_groups = np.array_split(np.arange(len(records), dtype=np.int64), 3)
        labels = ("early", "mid", "late")
        stats = {}
        for label, indices in zip(labels, index_groups):
            if indices.size <= 0:
                stats[label] = {
                    "mean_reward": 0.0,
                    "positive_reward_rate": 0.0,
                    "mean_collaboration_advantage": 0.0,
                }
                continue
            values = rewards[indices]
            stats[label] = {
                "mean_reward": float(np.mean(values)),
                "positive_reward_rate": float(np.mean(values > 0.0)),
                "mean_collaboration_advantage": float(np.mean(collaboration_advantages[indices])),
            }
        return stats

    def export_diagnostics(self) -> dict:
        retained_active = {
            int(vehicle_id): self._retained_support_ids(vehicle_id)
            for vehicle_id in self._retained_neighbors
        }
        completed_rounds = int(self._diagnostic_totals["completed_rounds"])
        selection_overlap = (
            float(self._diagnostic_totals["selection_overlap_sum"]) / completed_rounds
            if completed_rounds > 0
            else 1.0
        )
        retained_reuse_rate = (
            float(self._diagnostic_totals["retained_selected"]) / max(
                int(self._diagnostic_totals["retained_offered"]),
                1,
            )
        )
        retained_positive_reuse_rate = (
            float(self._diagnostic_totals["retained_positive_reused"]) / max(
                int(self._diagnostic_totals["retained_selected"]),
                1,
            )
        )
        retention_survival_rounds = (
            float(self._diagnostic_totals["retention_survival_sum"]) / max(
                int(self._diagnostic_totals["retention_survival_count"]),
                1,
            )
        )
        usefulness_mean = (
            float(self._diagnostic_totals["usefulness_sum"]) / max(
                int(self._diagnostic_totals["usefulness_count"]),
                1,
            )
        )
        baseline_reward_mean = (
            float(self._diagnostic_totals["baseline_reward_sum"]) / max(completed_rounds, 1)
        )
        normalized_gain_mean = (
            float(self._diagnostic_totals["normalized_gain_sum"]) / max(completed_rounds, 1)
        )
        mean_normalized_comm_cost = (
            float(self._diagnostic_totals["normalized_comm_cost_sum"]) / max(completed_rounds, 1)
        )
        collaboration_advantage_mean = (
            float(self._diagnostic_totals["collaboration_advantage_sum"]) / max(completed_rounds, 1)
        )
        reward_positive_rate = (
            float(self._diagnostic_totals["reward_positive_count"]) / max(completed_rounds, 1)
        )
        useful_selection_rate = (
            float(self._diagnostic_totals["selected_useful"]) / max(
                int(self._diagnostic_totals["selected_total"]),
                1,
            )
        )
        fallback_with_selection_rate = (
            float(self._diagnostic_totals["fallback_with_selection"]) / max(completed_rounds, 1)
        )
        positive_dval_negative_reward_rate = (
            float(self._diagnostic_totals["positive_dval_negative_reward"]) / max(
                completed_rounds,
                1,
            )
        )
        reward_window_stats = self._reward_window_stats()
        reward_values = np.asarray(
            [float(item["reward"]) for item in self._reward_records],
            dtype=np.float32,
        )
        reward_slope = 0.0
        if reward_values.size >= 2:
            xs = np.arange(reward_values.size, dtype=np.float32)
            reward_slope = float(np.polyfit(xs, reward_values, 1)[0])
        return {
            "completed_rounds": completed_rounds,
            "retained_offered": int(self._diagnostic_totals["retained_offered"]),
            "retained_selected": int(self._diagnostic_totals["retained_selected"]),
            "retained_skipped": int(self._diagnostic_totals["retained_skipped"]),
            "selected_total_peers": int(self._diagnostic_totals["selected_total"]),
            "selected_useful_peers": int(self._diagnostic_totals["selected_useful"]),
            "selected_internet": int(self._diagnostic_totals["selected_internet"]),
            "fallback_with_selection": int(self._diagnostic_totals["fallback_with_selection"]),
            "fallback_with_selection_rate": float(fallback_with_selection_rate),
            "positive_dval_negative_reward": int(self._diagnostic_totals["positive_dval_negative_reward"]),
            "positive_dval_negative_reward_rate": float(positive_dval_negative_reward_rate),
            "retained_reuse_rate": float(retained_reuse_rate),
            "retained_positive_reuse_rate": float(retained_positive_reuse_rate),
            "useful_selection_rate": float(useful_selection_rate),
            "retention_survival_rounds": float(retention_survival_rounds),
            "usefulness_mean": float(usefulness_mean),
            "baseline_reward_mean": float(baseline_reward_mean),
            "normalized_gain_mean": float(normalized_gain_mean),
            "mean_normalized_comm_cost": float(mean_normalized_comm_cost),
            "collaboration_advantage_mean": float(collaboration_advantage_mean),
            "reward_positive_rate": float(reward_positive_rate),
            "reward_slope": float(reward_slope),
            "reward_window_stats": reward_window_stats,
            "late_round_mean_reward": float(reward_window_stats["late"]["mean_reward"]),
            "late_round_collaboration_advantage": float(
                reward_window_stats["late"]["mean_collaboration_advantage"]
            ),
            "proposal_pruned_after_budget": int(self._diagnostic_totals["proposal_pruned_after_budget"]),
            "proposal_rejected_pretransfer": int(self._diagnostic_totals["proposal_rejected_pretransfer"]),
            "late_round_internet_links": int(self._diagnostic_totals["late_round_internet_links"]),
            "selection_overlap": float(selection_overlap),
            "selection_overlap_count": completed_rounds,
            "retained_active": retained_active,
        }

    def consume_debug_logs(self) -> list[str]:
        lines = list(self._debug_lines)
        self._debug_lines.clear()
        return lines
