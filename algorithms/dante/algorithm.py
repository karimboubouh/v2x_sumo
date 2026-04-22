"""
DANTE — paper-faithful GAT + PPO neighbor selection with SL-first retention.

Each vehicle owns an independent policy that:
  1. discovers new peers over sidelink only,
  2. promotes good sidelink peers to retained Internet candidates,
  3. samples keep/drop decisions with a local PPO actor, and
  4. aggregates accepted updates with fused attention, trust, and robustness.
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
    GAT_HIDDEN_DIM,
    MAX_COLLAB_NEIGHBORS,
    MAX_INTERNET_NEIGHBORS,
    MAX_SIDELINK_NEIGHBORS,
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
    ROUND_BANDWIDTH_BUDGET_BITS,
    ROUND_ENERGY_BUDGET_J,
    ROUND_LATENCY_BUDGET_S,
    SELF_WEIGHT,
    TRUST_SMOOTHING,
)
from dl.helpers import (
    clone_state_dict,
    inet_tx_energy_j,
    inet_tx_time_s,
    sl_tx_energy_j,
    sl_tx_time_s,
    tx_payload_bits,
)


GRAD_ALIGN_IDX = 0
DISTANCE_IDX = 1
ENERGY_COST_IDX = 2
BANDWIDTH_COST_IDX = 3
LATENCY_COST_IDX = 4
REL_MOBILITY_IDX = 5
LINK_TYPE_IDX = 6
TRUST_IDX = 7
RETENTION_IDX = 8

_EPS = 1e-8
_VALIDATION_LOSS_SLACK = 0.0
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
    """Shared GAT encoder with a Bernoulli actor and scalar critic."""

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
        actor_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.forward(own_state, nbr_features)
        logits = out["selector_logits"]
        if actor_indices is not None and actor_indices.numel() > 0:
            logits = logits.index_select(0, actor_indices.long())
        elif actor_indices is not None:
            logits = logits.new_zeros((0,))
        if logits.numel() == 0:
            log_prob = out["value"].new_zeros(())
            entropy = out["value"].new_zeros(())
        else:
            dist = Bernoulli(logits=logits)
            log_prob = dist.log_prob(actions).sum()
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
        actor_indices: np.ndarray | list[int] | None = None,
    ) -> dict:
        own_t = torch.as_tensor(own_state, dtype=torch.float32)
        nbr_t = torch.as_tensor(nbr_features, dtype=torch.float32)
        actor_idx = np.asarray(actor_indices if actor_indices is not None else [], dtype=np.int64)

        self.policy.eval()
        with torch.no_grad():
            out = self.policy(own_t, nbr_t)
            selector_prob = torch.sigmoid(out["selector_logits"])
            if actor_idx.size <= 0:
                action = out["selector_logits"].new_zeros((0,))
            else:
                actor_logits = out["selector_logits"].index_select(
                    0,
                    torch.as_tensor(actor_idx, dtype=torch.long),
                )
                dist = Bernoulli(logits=actor_logits)
                action = dist.sample()

        return {
            "own_state": own_state.astype(np.float32, copy=True),
            "nbr_features": nbr_features.astype(np.float32, copy=True),
            "action": action.cpu().numpy().astype(np.float32, copy=True),
            "value": float(out["value"].item()),
            "actor_indices": actor_idx.astype(np.int64, copy=True),
            "selector_logits": out["selector_logits"].cpu().numpy().astype(np.float32, copy=True),
            "selector_prob": selector_prob.cpu().numpy().astype(np.float32, copy=True),
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
                actor_idx_t = torch.as_tensor(sample["actor_indices"], dtype=torch.long)

                log_prob, entropy, value = self.policy.evaluate_actions(
                    own_t,
                    nbr_t,
                    act_t,
                    actor_indices=actor_idx_t,
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
    own_feature_dim = OWN_DIM
    neighbor_feature_dim = NBR_DIM

    def __init__(self):
        self._agents: dict[int, _VehiclePPOAgent] = {}
        self._retained_neighbors: dict[int, dict[int, dict]] = {}
        self._trust_scores: dict[int, dict[int, float]] = {}
        self._retention_values: dict[int, dict[int, float]] = {}
        self._peer_robustness: dict[int, dict[int, dict[str, float | bool]]] = {}
        self._baseline_gain: dict[int, float] = {}
        self._budget_state: dict[int, dict[str, float]] = {}
        self._round_feedback: dict[int, dict] = {}
        self._promotion_events: list[dict] = []
        self._debug_lines: list[str] = []
        self._last_selected_neighbors: dict[int, set[int]] = {}
        self._last_actor_selected_neighbors: dict[int, set[int]] = {}
        self._diagnostic_totals = {
            "completed_rounds": 0,
            "retained_offered": 0,
            "retained_selected": 0,
            "retained_skipped": 0,
            "selected_internet": 0,
            "fallback_with_selection": 0,
            "positive_dval_negative_reward": 0,
            "selection_overlap_sum": 0.0,
            "retained_reused": 0,
            "retention_survival_sum": 0.0,
            "retention_survival_count": 0,
            "retained_value_sum": 0.0,
            "retained_value_count": 0,
            "baseline_gain_sum": 0.0,
            "excess_gain_sum": 0.0,
            "proposal_pruned_after_budget": 0,
            "proposal_rejected_pretransfer": 0,
            "executed_actor_overlap_sum": 0.0,
            "late_round_internet_links": 0,
        }

        self.reward_source = "validation"
        self.max_sidelink_neighbors = int(MAX_SIDELINK_NEIGHBORS)
        self.max_internet_neighbors = int(MAX_INTERNET_NEIGHBORS)
        self.max_collab_neighbors = int(MAX_COLLAB_NEIGHBORS)
        self.trust_smoothing = float(np.clip(TRUST_SMOOTHING, 0.0, 1.0))
        self.round_energy_budget_j = max(float(ROUND_ENERGY_BUDGET_J), _EPS)
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
        self._total_energy_budget_j = self.round_energy_budget_j * max(int(config.MAX_TR_ROUNDS), 1)
        self._total_bandwidth_budget_bits = (
            self.round_bandwidth_budget_bits * max(int(config.MAX_TR_ROUNDS), 1)
        )
        max_rounds = max(int(config.MAX_TR_ROUNDS), 1)
        self._late_round_start = max(1, int(math.ceil(0.8 * max_rounds)))

    def _default_latency_budget_s(self) -> float:
        batches = config.BATCHES_PER_ROUND if config.BATCHES_PER_ROUND else 1
        batch_size = max(int(config.BATCH_SIZE), 1)
        compute_s = (
            float(batches) * batch_size * float(config.CPU_CYCLES_PER_SAMPLE)
        ) / max(float(config.CPU_FREQ_HZ), 1.0)
        return compute_s + max(float(inet_tx_time_s()), float(sl_tx_time_s(config.COMM_RANGE)))

    def setup(self, vehicles: list) -> None:
        for v in vehicles:
            self._agents[v.id] = _VehiclePPOAgent(OWN_DIM, NBR_DIM, GAT_HIDDEN_DIM)
            self._retained_neighbors[v.id] = {}
            self._trust_scores[v.id] = {}
            self._retention_values[v.id] = {}
            self._peer_robustness[v.id] = {}
            self._last_selected_neighbors[v.id] = set()
            self._last_actor_selected_neighbors[v.id] = set()
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

    def _ensure_retention_entry(self, vehicle_id: int, neighbor_id: int) -> float:
        values = self._retention_values.setdefault(int(vehicle_id), {})
        if int(neighbor_id) not in values:
            values[int(neighbor_id)] = 0.0
        return float(values[int(neighbor_id)])

    def get_trust_score(self, v, neighbor_id: int) -> float:
        vehicle_id = int(v.id if hasattr(v, "id") else v)
        return self._ensure_trust_entry(vehicle_id, int(neighbor_id))

    def get_retention_value(self, v, neighbor_id: int) -> float:
        vehicle_id = int(v.id if hasattr(v, "id") else v)
        return self._ensure_retention_entry(vehicle_id, int(neighbor_id))

    def get_baseline_gain(self, v) -> float:
        vehicle_id = int(v.id if hasattr(v, "id") else v)
        return float(self._baseline_gain.get(vehicle_id, 0.0))

    def _set_trust_score(self, vehicle_id: int, neighbor_id: int, value: float) -> None:
        self._trust_scores.setdefault(int(vehicle_id), {})[int(neighbor_id)] = float(
            np.clip(value, 0.0, 1.0)
        )

    def _set_retention_value(self, vehicle_id: int, neighbor_id: int, value: float) -> None:
        self._retention_values.setdefault(int(vehicle_id), {})[int(neighbor_id)] = float(
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
        """Discover new peers over sidelink only; inject retained peers over Internet."""
        sidelink = env.sidelink_neighbors_of(v)
        visible = {nbr.id for nbr, _, _ in sidelink}
        retained = self._retained_neighbors.get(v.id, {})
        injected = []
        stale_retained_ids = []

        retained_items = sorted(
            retained.items(),
            key=lambda item: (
                -(
                    self.get_retention_value(v.id, item[0])
                    * self.get_trust_score(v.id, item[0])
                    * self.get_last_robust_score(v.id, item[0])
                ),
                -self.get_retention_value(v.id, item[0]),
                -self.get_last_robust_score(v.id, item[0]),
                -self.get_trust_score(v.id, item[0]),
                -int(item[1].get("last_good_round", -1)),
                item[0],
            ),
        )
        for nid, meta in retained_items:
            del meta
            if len(injected) >= self.max_internet_neighbors:
                break
            if nid == v.id or nid >= len(env.vehicles):
                stale_retained_ids.append(int(nid))
                continue
            if nid in visible:
                continue
            nbr = env.vehicles[nid]
            dist = float(np.linalg.norm(v.pos - nbr.pos))
            if dist > float(config.INTERNET_RANGE):
                continue
            injected.append((nbr, dist, LINK_INTERNET))

        for nid in stale_retained_ids:
            self._drop_retained_neighbor(v.id, nid)

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
        meta = retained.setdefault(int(neighbor_id), {
            "first_retained_round": int(round_no),
            "last_good_round": int(round_no),
            "last_selected_round": int(round_no),
            "selected_count": 0,
            "reuse_count": 0,
        })
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
            self._neighbor_energy_j(link_type, dist) / self._active_energy_budget(v.id),
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

    def _proposal_benefit(
        self,
        selector_prob: float,
        trust: float,
        grad_align: float,
        robust_score: float,
    ) -> float:
        return (
            float(selector_prob)
            * float(np.clip(trust, 0.0, 1.0))
            * max(float(grad_align), 0.0)
            * float(np.clip(robust_score, 0.0, 1.0))
        )

    def _retained_benefit(
        self,
        retention_value: float,
        trust: float,
        robust_score: float,
    ) -> float:
        return (
            float(np.clip(retention_value, 0.0, 1.0))
            * float(np.clip(trust, 0.0, 1.0))
            * float(np.clip(robust_score, 0.0, 1.0))
        )

    def _action_log_prob(
        self,
        selector_logits: np.ndarray,
        actor_indices: np.ndarray,
        executed_action: np.ndarray,
    ) -> float:
        if actor_indices.size <= 0 or executed_action.size <= 0:
            return 0.0
        actor_logits = torch.as_tensor(
            np.asarray(selector_logits, dtype=np.float32)[actor_indices],
            dtype=torch.float32,
        )
        dist = Bernoulli(logits=actor_logits)
        action_t = torch.as_tensor(executed_action, dtype=torch.float32)
        return float(dist.log_prob(action_t).sum().item())

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
                f"m={info.get('prev_retention', 0.0):.2f}->{info.get('new_retention', info.get('prev_retention', 0.0)):.2f},"
                f"phi={int(info.get('phi', 0.0))},"
                f"a={peer_feedback.get('alpha', 0.0):.2f},"
                f"r={peer_feedback.get('robust_score', 0.0):.2f},"
                f"c={info.get('peer_credit', 0.0):.2f})"
            )
        return ", ".join(parts)

    def _format_retained_debug(self, vehicle_id: int) -> str:
        retained = self._retained_neighbors.get(int(vehicle_id), {})
        if not retained:
            return "none"

        parts = []
        for nid, meta in sorted(
            retained.items(),
            key=lambda item: (
                -(
                    self.get_retention_value(vehicle_id, item[0])
                    * self.get_trust_score(vehicle_id, item[0])
                    * self.get_last_robust_score(vehicle_id, item[0])
                ),
                item[0],
            ),
        ):
            parts.append(
                f"{nid}(q={self.get_trust_score(vehicle_id, nid):.2f},"
                f"m={self.get_retention_value(vehicle_id, nid):.2f},"
                f"last={int(meta.get('last_good_round', -1))})"
            )
        return ", ".join(parts)

    def _resolve_self_weight(self, n_neighbors: int) -> float:
        if n_neighbors <= 0:
            return 1.0
        if SELF_WEIGHT is None:
            return 1.0 / (n_neighbors + 1.0)
        self_w = float(SELF_WEIGHT)
        if not 0.0 <= self_w <= 1.0:
            raise ValueError(
                f"DANTE SELF_WEIGHT must be None or in [0, 1], got {SELF_WEIGHT!r}"
            )
        return self_w

    def _select_budgeted_subset(self, vehicle_id: int, proposals: list[dict]) -> list[dict]:
        """Choose the highest-benefit feasible subset under the current budgets."""
        if not proposals:
            return []

        available_energy = self._active_energy_budget(vehicle_id)
        available_bw = self._active_bandwidth_budget(vehicle_id)
        filtered = []
        for item in proposals:
            if item["latency_s"] > self.round_latency_budget_s:
                continue
            if item["bandwidth_bits"] > available_bw:
                continue
            if item["energy_j"] > available_energy:
                continue
            filtered.append(item)

        if not filtered:
            return []

        max_count = int(self.max_collab_neighbors)
        if self.payload_bits > 0.0:
            bw_count_cap = int(available_bw // self.payload_bits)
            max_count = min(max_count, bw_count_cap)
        if max_count <= 0:
            return []

        energy_unit = max(float(available_energy) / 200.0, 1e-4)
        capacity = max(int(np.floor(available_energy / energy_unit + 1e-9)), 0)
        if capacity <= 0:
            return []

        states: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {(0, 0): (0.0, ())}
        for idx, item in enumerate(filtered):
            cost_steps = max(int(np.ceil(item["energy_j"] / energy_unit - 1e-12)), 0)
            if cost_steps > capacity:
                continue
            updates = dict(states)
            for (count, used_steps), (benefit, picks) in states.items():
                if count >= max_count:
                    continue
                next_steps = used_steps + cost_steps
                if next_steps > capacity:
                    continue
                next_key = (count + 1, next_steps)
                next_benefit = benefit + float(item["benefit"])
                current = updates.get(next_key)
                if current is None or next_benefit > current[0]:
                    updates[next_key] = (next_benefit, picks + (idx,))
            states = updates

        best_benefit, best_indices = max(states.values(), key=lambda entry: entry[0])
        if best_benefit <= _EPS or not best_indices:
            return []
        chosen = [filtered[idx] for idx in best_indices]
        chosen.sort(key=lambda item: item["benefit"], reverse=True)
        return chosen

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
        actor_indices = np.asarray(
            [idx for idx, (nbr, _, _) in enumerate(candidates) if nbr.id not in retained],
            dtype=np.int64,
        )
        decision = agent.act(own_state, nbr_features, actor_indices=actor_indices)
        actor_index_positions = {
            int(candidate_idx): pos
            for pos, candidate_idx in enumerate(actor_indices.tolist())
        }
        grad_vectors = [nbr.get_grad_vec() for nbr, _, _ in candidates]
        grad_robust_scores, grad_robust_passes = self._gradient_robustness_scores(grad_vectors)

        proposals = []
        pretransfer_rejected = 0
        for idx, (nbr, dist, link_type) in enumerate(candidates):
            is_retained = nbr.id in retained
            actor_pos = actor_index_positions.get(idx)
            selected_by_actor = (
                actor_pos is not None
                and actor_pos < len(decision["action"])
                and float(decision["action"][actor_pos]) >= 0.5
            )
            if not is_retained and not selected_by_actor:
                continue

            energy_j = self._neighbor_energy_j(link_type, dist)
            bandwidth_bits = self._neighbor_bandwidth_bits(link_type, dist)
            latency_s = self._neighbor_latency_s(link_type, dist)
            grad_align = float(nbr_features[idx][GRAD_ALIGN_IDX])
            trust = float(nbr_features[idx][TRUST_IDX])
            retention_value = float(nbr_features[idx][RETENTION_IDX])
            selector_prob = float(decision["selector_prob"][idx]) if idx < len(decision["selector_prob"]) else 0.0
            robust_score = self.get_last_robust_score(v.id, nbr.id)
            grad_robust_score = (
                float(grad_robust_scores[idx]) if idx < len(grad_robust_scores) else 1.0
            )
            grad_robust_pass = (
                bool(grad_robust_passes[idx]) if idx < len(grad_robust_passes) else True
            )
            beta = float(decision["attention"][idx]) if idx < len(decision["attention"]) else 0.0
            if grad_align <= 0.0 or not grad_robust_pass:
                pretransfer_rejected += 1
                continue
            if is_retained:
                benefit = self._retained_benefit(retention_value, trust, robust_score)
            else:
                benefit = self._proposal_benefit(selector_prob, trust, grad_align, robust_score)
            if benefit <= _EPS:
                pretransfer_rejected += 1
                continue
            proposals.append({
                "nid": int(nbr.id),
                "link_type": float(link_type),
                "energy_j": float(energy_j),
                "bandwidth_bits": float(bandwidth_bits),
                "latency_s": float(latency_s),
                "benefit": float(benefit),
                "beta": max(beta, 0.0),
                "source": "retained" if is_retained else "explore",
                "grad_robust_score": grad_robust_score,
            })

        selected = self._select_budgeted_subset(v.id, proposals)
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

        selected_ids = {int(item["nid"]) for item in selected}
        actor_action = np.asarray(
            [
                1.0 if int(candidates[candidate_idx][0].id) in selected_ids else 0.0
                for candidate_idx in actor_indices.tolist()
            ],
            dtype=np.float32,
        )
        transition = {
            "own_state": decision["own_state"],
            "nbr_features": decision["nbr_features"],
            "action": actor_action,
            "actor_indices": decision["actor_indices"],
            "log_prob": self._action_log_prob(
                decision["selector_logits"],
                decision["actor_indices"],
                actor_action,
            ),
            "value": decision["value"],
            "comm_energy_j": float(sum(item["energy_j"] for item in selected)),
            "bandwidth_bits": float(sum(item["bandwidth_bits"] for item in selected)),
            "latency_s": float(max((item["latency_s"] for item in selected), default=0.0)),
            "energy_budget_j": self._active_energy_budget(v.id),
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
                    "grad_robust_score": float(item.get("grad_robust_score", 1.0)),
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
        return distances, scores, passes

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
        nbr_sds = [nbr.get_shared_weights() for nbr in accepted]
        distances, robust_scores, robust_passes = self._robustness_scores(nbr_sds)
        effective_scores = base_weights * trusts * robust_scores * robust_passes.astype(np.float32)
        total_effective = float(effective_scores.sum())

        round_feedback = {
            "neighbors": {},
            "fallback": total_effective <= _EPS,
        }
        for nbr, base_beta, trust, robust_score, passed, distance in zip(
            accepted,
            base_weights,
            trusts,
            robust_scores,
            robust_passes,
            distances,
        ):
            round_feedback["neighbors"][nbr.id] = {
                "base_beta": float(base_beta),
                "trust": float(trust),
                "robust_score": float(robust_score),
                "robust_pass": bool(passed),
                "distance": float(distance),
                "alpha": 0.0,
            }

        if total_effective <= _EPS:
            v.alphas = {}
            self._round_feedback[v.id] = round_feedback
            return

        nbr_weights = effective_scores / total_effective
        self_w = self._resolve_self_weight(len(accepted))

        with v._lock:
            own_sd = v.model.state_dict()
            new_sd = clone_state_dict(own_sd)
            for key in new_sd:
                if not new_sd[key].is_floating_point():
                    continue
                agg = self_w * own_sd[key].float()
                for weight, sd in zip(nbr_weights, nbr_sds):
                    nbr_tensor = sd.get(key, own_sd[key])
                    agg = agg + (1.0 - self_w) * float(weight) * nbr_tensor.float()
                new_sd[key] = agg
            v.model.load_state_dict(new_sd)
            v._param_vec = None

        final_alphas = {}
        for nbr, weight in zip(accepted, nbr_weights):
            alpha = (1.0 - self_w) * float(weight)
            if alpha > 0.0:
                final_alphas[nbr.id] = alpha
            round_feedback["neighbors"][nbr.id]["alpha"] = alpha

        v.alphas = final_alphas
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
        cost = (
            float(comm_energy_j) / max(float(energy_budget_j), _EPS)
            + float(bandwidth_bits) / max(float(bandwidth_budget_bits), _EPS)
            + float(latency_s) / max(float(latency_budget_s), _EPS)
        ) / 3.0
        return float(learning_term - cost)

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
                    prev_val_loss = float(v._prev_val_loss)
                    val_loss_delta = v._prev_val_loss - v.current_val_loss

                normalized_gain = float(val_loss_delta) / max(float(prev_val_loss), _EPS)
                baseline_before = self.get_baseline_gain(v.id)
                bandwidth_bits = float(pending.get("bandwidth_bits", 0.0))
                latency_s = float(pending.get("latency_s", 0.0))
                latency_budget_s = float(pending.get("latency_budget_s", self.round_latency_budget_s))
                comm_energy_j = float(pending.get("comm_energy_j", 0.0))
                selected_neighbors = pending.get("selected_neighbors", [])
                local_like_round = (not selected_neighbors) or bool(feedback.get("fallback", False))
                excess_gain = float(normalized_gain - baseline_before)
                updated_baseline = baseline_before
                if local_like_round:
                    updated_baseline = (
                        (1.0 - self.trust_smoothing) * baseline_before
                        + self.trust_smoothing * normalized_gain
                    )
                    self._set_baseline_gain(v.id, updated_baseline)

                reward = self._compute_reward(
                    learning_term=excess_gain,
                    comm_energy_j=comm_energy_j,
                    bandwidth_bits=bandwidth_bits,
                    latency_s=latency_s,
                    energy_budget_j=self.round_energy_budget_j,
                    bandwidth_budget_bits=self.round_bandwidth_budget_bits,
                    latency_budget_s=self.round_latency_budget_s,
                )

                comp_hist = getattr(v, "computation_energy_hist", [])
                comp_energy_delta = float(comp_hist[-1]) if comp_hist else 0.0
                total_energy_j = float(comm_energy_j) + max(comp_energy_delta, 0.0)

                budget_state = self._budget_state[v.id]
                budget_state["remaining_energy_j"] = max(
                    budget_state["remaining_energy_j"] - comm_energy_j,
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
                validation_safe = v.current_val_loss <= (v._prev_val_loss + _VALIDATION_LOSS_SLACK)
                promoted_count = 0
                demoted_policy_count = 0
                demoted_bad_effect_count = 0
                trust_updates = {}

                selected_ids = {int(item["nid"]) for item in selected_neighbors}
                eligible_retained_ids = {
                    int(nid) for nid in pending.get("eligible_retained_ids", [])
                }

                for item in selected_neighbors:
                    nid = int(item["nid"])
                    link_type = float(item["link_type"])
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
                    peer_credit = float(np.clip(
                        max(excess_gain, 0.0) * alpha * robust_score,
                        0.0,
                        1.0,
                    ))
                    phi = 1.0 if (robust_pass and alpha > _EPS and excess_gain > 0.0) else 0.0
                    prev_trust = self.get_trust_score(v.id, nid)
                    prev_retention = self.get_retention_value(v.id, nid)
                    updated_trust = (1.0 - self.trust_smoothing) * prev_trust + self.trust_smoothing * phi
                    updated_retention = (
                        (1.0 - self.trust_smoothing) * prev_retention
                        + self.trust_smoothing * peer_credit
                    )
                    self._set_trust_score(v.id, nid, updated_trust)
                    self._set_retention_value(v.id, nid, updated_retention)
                    trust_updates[nid] = {
                        "prev_trust": float(prev_trust),
                        "new_trust": float(updated_trust),
                        "prev_retention": float(prev_retention),
                        "new_retention": float(updated_retention),
                        "phi": float(phi),
                        "peer_credit": float(peer_credit),
                    }
                    reused = source == "retained"
                    if reused or peer_credit > 0.0:
                        self._mark_selected_neighbor(v.id, nid, round_no, reused=reused)

                    if peer_credit > 0.0:
                        if link_type == LINK_SIDELINK and nid not in self._retained_neighbors.get(v.id, {}):
                            promoted_count += 1
                        self._retain_neighbor(v.id, nid, round_no)

                self._promotion_events.append({
                    "vehicle_id": int(v.id),
                    "round": round_no,
                    "promoted": int(promoted_count),
                    "demoted_policy": int(demoted_policy_count),
                    "demoted_bad_effect": int(demoted_bad_effect_count),
                    "retained_active": int(len(self._retained_neighbors.get(v.id, {}))),
                    "validation_safe": bool(validation_safe),
                })

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
                actor_selected_ids = {
                    int(item["nid"])
                    for item in selected_neighbors
                    if str(item.get("source", "explore")) != "retained"
                }
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
                prev_actor_selected_ids = self._last_actor_selected_neighbors.get(v.id, set())
                actor_union = prev_actor_selected_ids | actor_selected_ids
                executed_actor_overlap = (
                    len(prev_actor_selected_ids & actor_selected_ids) / len(actor_union)
                    if actor_union
                    else 1.0
                )
                self._last_actor_selected_neighbors[v.id] = set(actor_selected_ids)
                self._diagnostic_totals["completed_rounds"] += 1
                self._diagnostic_totals["retained_offered"] += len(eligible_retained_ids)
                self._diagnostic_totals["retained_selected"] += len(retained_selected_ids)
                self._diagnostic_totals["retained_skipped"] += len(retained_skipped_ids)
                self._diagnostic_totals["selected_internet"] += selected_in
                self._diagnostic_totals["selection_overlap_sum"] += float(selection_overlap)
                self._diagnostic_totals["retained_reused"] += len(retained_selected_ids)
                self._diagnostic_totals["baseline_gain_sum"] += float(updated_baseline)
                self._diagnostic_totals["excess_gain_sum"] += float(excess_gain)
                self._diagnostic_totals["proposal_rejected_pretransfer"] += int(
                    pending.get("proposal_rejected_pretransfer", 0)
                )
                self._diagnostic_totals["proposal_pruned_after_budget"] += int(
                    pending.get("proposal_pruned_after_budget", 0)
                )
                self._diagnostic_totals["executed_actor_overlap_sum"] += float(executed_actor_overlap)
                if selected_neighbors and bool(feedback.get("fallback", False)):
                    self._diagnostic_totals["fallback_with_selection"] += 1
                if float(val_loss_delta) > 0.0 and float(reward) < 0.0:
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
                retained_values = [
                    self.get_retention_value(v.id, nid)
                    for nid in self._retained_neighbors.get(v.id, {})
                ]
                self._diagnostic_totals["retained_value_sum"] += float(sum(retained_values))
                self._diagnostic_totals["retained_value_count"] += int(len(retained_values))

                energy_ratio, bandwidth_ratio, latency_ratio = self.get_budget_features(v)
                self._queue_debug_line(
                    f"DANTE V{v.id} R{round_no} | cand SL/IN {int(candidate_counts.get('sl', 0))}/{int(candidate_counts.get('in', 0))}"
                    f" | sel SL/IN {selected_sl}/{selected_in}"
                    f" | retained pool/offered/sel/skip {len(self._retained_neighbors.get(v.id, {}))}/{len(eligible_retained_ids)}/{len(retained_selected_ids)}/{len(retained_skipped_ids)}"
                    f" | reward {reward:+.4f} | dVal {float(val_loss_delta):+.4f}"
                    f" | base/excess {updated_baseline:+.4f}/{float(excess_gain):+.4f}"
                    f" | energy comm/comp/tot {float(pending.get('comm_energy_j', 0.0)):.4f}/{max(comp_energy_delta, 0.0):.4f}/{total_energy_j:.4f} J"
                    f" | bw {bandwidth_bits:.0f} b | lat {latency_s:.4f}/{latency_budget_s:.4f} s"
                    f" | budget E/B/L {energy_ratio:.2f}/{bandwidth_ratio:.2f}/{latency_ratio:.2f}"
                    f" | fallback {int(bool(feedback.get('fallback', False)))}"
                    f" | overlap {selection_overlap:.2f}/{executed_actor_overlap:.2f}"
                    f" | reject/prune {int(pending.get('proposal_rejected_pretransfer', 0))}/{int(pending.get('proposal_pruned_after_budget', 0))}"
                    f" | promote/demote {promoted_count}/{demoted_policy_count}/{demoted_bad_effect_count}"
                )
                self._queue_debug_line(
                    f"DANTE V{v.id} peers | selected [{self._format_selected_debug(selected_neighbors, feedback, trust_updates)}]"
                    f" | retained [{self._format_retained_debug(v.id)}]"
                    f" | skipped_retained {retained_skipped_ids if retained_skipped_ids else '[]'}"
                    f" | dropped_prev {dropped_prev_selected if dropped_prev_selected else '[]'}"
                )

            if next_transition is not None:
                agent.store_pending(next_transition, next_transition["target_round"])

            force_update = next_transition is None and agent.pending_transition is None
            if agent.should_update(force=force_update):
                agent.update()

        return rewards

    def export_diagnostics(self) -> dict:
        retained_active = {
            int(vehicle_id): sorted(int(nid) for nid in retained.keys())
            for vehicle_id, retained in self._retained_neighbors.items()
        }
        completed_rounds = int(self._diagnostic_totals["completed_rounds"])
        selection_overlap = (
            float(self._diagnostic_totals["selection_overlap_sum"]) / completed_rounds
            if completed_rounds > 0
            else 1.0
        )
        executed_actor_overlap = (
            float(self._diagnostic_totals["executed_actor_overlap_sum"]) / completed_rounds
            if completed_rounds > 0
            else 1.0
        )
        retained_reuse_rate = (
            float(self._diagnostic_totals["retained_reused"]) / max(
                int(self._diagnostic_totals["retained_offered"]),
                1,
            )
        )
        retention_survival_rounds = (
            float(self._diagnostic_totals["retention_survival_sum"]) / max(
                int(self._diagnostic_totals["retention_survival_count"]),
                1,
            )
        )
        retained_value_mean = (
            float(self._diagnostic_totals["retained_value_sum"]) / max(
                int(self._diagnostic_totals["retained_value_count"]),
                1,
            )
        )
        baseline_gain_mean = (
            float(self._diagnostic_totals["baseline_gain_sum"]) / max(completed_rounds, 1)
        )
        excess_gain_mean = (
            float(self._diagnostic_totals["excess_gain_sum"]) / max(completed_rounds, 1)
        )
        return {
            "promotion_events": list(self._promotion_events),
            "promotion_totals": {
                "promoted": int(sum(event["promoted"] for event in self._promotion_events)),
                "demoted_policy": int(sum(event["demoted_policy"] for event in self._promotion_events)),
                "demoted_bad_effect": int(sum(event["demoted_bad_effect"] for event in self._promotion_events)),
            },
            "retained_offered": int(self._diagnostic_totals["retained_offered"]),
            "retained_selected": int(self._diagnostic_totals["retained_selected"]),
            "retained_skipped": int(self._diagnostic_totals["retained_skipped"]),
            "selected_internet": int(self._diagnostic_totals["selected_internet"]),
            "fallback_with_selection": int(self._diagnostic_totals["fallback_with_selection"]),
            "positive_dval_negative_reward": int(self._diagnostic_totals["positive_dval_negative_reward"]),
            "retained_reuse_rate": float(retained_reuse_rate),
            "retention_survival_rounds": float(retention_survival_rounds),
            "retained_value_mean": float(retained_value_mean),
            "baseline_gain_mean": float(baseline_gain_mean),
            "excess_gain_mean": float(excess_gain_mean),
            "proposal_pruned_after_budget": int(self._diagnostic_totals["proposal_pruned_after_budget"]),
            "proposal_rejected_pretransfer": int(self._diagnostic_totals["proposal_rejected_pretransfer"]),
            "executed_actor_overlap": float(executed_actor_overlap),
            "late_round_internet_links": int(self._diagnostic_totals["late_round_internet_links"]),
            "selection_overlap": float(selection_overlap),
            "selection_overlap_count": completed_rounds,
            "retained_active": retained_active,
        }

    def consume_debug_logs(self) -> list[str]:
        lines = list(self._debug_lines)
        self._debug_lines.clear()
        return lines
