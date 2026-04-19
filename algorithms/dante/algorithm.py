"""
DANTE — Graph Attention Network + PPO neighbor selection.

Each vehicle owns an independent policy that:
  1. encodes its candidate neighborhood with a shared GAT backbone,
  2. samples keep/drop decisions with a PPO selector head, and
  3. assigns aggregation weights with a separate mixer head trained on
     gradient-alignment targets.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

from algorithms.base import DLAlgorithm, LINK_INTERNET, LINK_SIDELINK
from algorithms.dante.config import (
    COST_LAMBDA,
    GAT_HIDDEN_DIM,
    MAX_COLLAB_NEIGHBORS,
    MAX_INTERNET_NEIGHBORS,
    MAX_SIDELINK_NEIGHBORS,
    MIXER_LOSS_COEF,
    MIXER_TAU,
    NBR_DIM,
    OWN_DIM,
    PPO_CLIP_EPS,
    PPO_ENTROPY_COEF,
    PPO_EPOCHS,
    PPO_GAE_LAMBDA,
    PPO_GAMMA,
    PPO_LR,
    PPO_MAX_GRAD_NORM,
    PPO_REWARD_SOURCE,
    PPO_UPDATE_EVERY,
    PPO_VALUE_COEF,
    REWARD_ACC_WEIGHT,
    ROUND_BANDWIDTH_BUDGET_BITS,
    ROUND_ENERGY_BUDGET_J,
    ROUND_LATENCY_BUDGET_S,
    SELECTOR_INIT_BIAS,
    SELF_WEIGHT,
    TYPICAL_ROUND_ENERGY_J,
)
from dl.helpers import (
    clone_state_dict,
    inet_tx_energy_j,
    inet_tx_time_s,
    sl_tx_energy_j,
    sl_tx_time_s,
    tx_payload_bits,
)


COS_SIM_IDX = 0
GRAD_ALIGN_IDX = 1
TX_COST_IDX = 3
NBR_ACC_IDX = 5
RETAINED_SCORE_IDX = 7


def _normalize_reward_source(value: str) -> str:
    value = str(value).strip().lower()
    if value not in {"training", "validation"}:
        raise ValueError(
            f"Unsupported DANTE PPO_REWARD_SOURCE {value!r}. "
            "Expected 'training' or 'validation'."
        )
    return value


def _softmax_numpy(logits: np.ndarray, tau: float = 1.0) -> np.ndarray:
    if logits.size == 0:
        return np.zeros((0,), dtype=np.float32)
    scaled = np.asarray(logits, dtype=np.float32) / max(float(tau), 1e-6)
    scaled = scaled - float(np.max(scaled))
    probs = np.exp(scaled)
    total = float(probs.sum())
    if total <= 1e-12:
        return np.full(probs.shape, 1.0 / len(probs), dtype=np.float32)
    return (probs / total).astype(np.float32, copy=False)


def _mixer_target_from_features(nbr_features: np.ndarray) -> np.ndarray:
    """Cheap teacher for the mixer: statistically relevant neighbors win."""
    if nbr_features.size == 0:
        return np.zeros((0,), dtype=np.float32)

    cos_sim = np.clip(nbr_features[:, COS_SIM_IDX], -1.0, 1.0)
    grad_align = np.clip(nbr_features[:, GRAD_ALIGN_IDX], -1.0, 1.0)
    nbr_acc = np.clip(nbr_features[:, NBR_ACC_IDX], 0.0, 1.0)

    teacher_logits = 0.5 * cos_sim + 2.5 * grad_align + 0.5 * nbr_acc
    return _softmax_numpy(teacher_logits, tau=MIXER_TAU)


def _link_label(link_type: float) -> str:
    return "IN" if link_type == LINK_INTERNET else "SL"


def _format_energy(energy_j: float) -> str:
    if energy_j >= 1.0:
        return f"{energy_j:.2f}J"
    return f"{energy_j * 1000.0:.1f}mJ"


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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if nbr_features.numel() == 0:
            empty_nbr = nbr_features.new_zeros((0, self.nbr_proj.out_features))
            empty_emb = nbr_features.new_zeros((self.nbr_proj.out_features,))
            return empty_emb, empty_nbr

        h_i = self.self_proj(own_state)
        v_j = self.nbr_proj(nbr_features)
        h_rep = h_i.unsqueeze(0).expand(v_j.shape[0], -1)
        e_ij = self.attn(self.leaky_relu(torch.cat([h_rep, v_j], dim=-1))).squeeze(-1)
        attn = torch.softmax(e_ij, dim=0)
        emb = torch.sum(attn.unsqueeze(-1) * v_j, dim=0)
        return emb, v_j


class _GATActorCritic(nn.Module):
    """Shared GAT encoder with decoupled selector, mixer, and critic heads."""

    def __init__(self, own_dim: int, nbr_dim: int, hidden_dim: int):
        super().__init__()
        self.gat = _GATLayer(own_dim, nbr_dim, hidden_dim)
        self.own_encoder = nn.Sequential(
            nn.Linear(own_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        selector_out = nn.Linear(hidden_dim, 1)
        nn.init.constant_(selector_out.bias, float(SELECTOR_INIT_BIAS))
        self.selector = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            selector_out,
        )
        self.mixer = nn.Sequential(
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
        own_enc = self.own_encoder(own_state)
        emb, nbr_emb = self.gat(own_state, nbr_features)
        fused = torch.cat([own_enc, emb], dim=-1)
        value = self.critic(fused).squeeze(-1)

        if nbr_emb.numel() == 0:
            selector_logits = nbr_features.new_zeros((0,))
            mixer_logits = nbr_features.new_zeros((0,))
        else:
            fused_rep = fused.unsqueeze(0).expand(nbr_emb.shape[0], -1)
            head_in = torch.cat([fused_rep, nbr_emb], dim=-1)
            selector_logits = self.selector(head_in).squeeze(-1)
            mixer_logits = self.mixer(head_in).squeeze(-1)

        return {
            "value": value,
            "selector_logits": selector_logits,
            "mixer_logits": mixer_logits,
        }

    def evaluate_actions(
        self,
        own_state: torch.Tensor,
        nbr_features: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.forward(own_state, nbr_features)
        if out["selector_logits"].numel() == 0:
            log_prob = out["value"].new_zeros(())
            entropy = out["value"].new_zeros(())
        else:
            dist = Bernoulli(logits=out["selector_logits"])
            log_prob = dist.log_prob(actions).mean()
            entropy = dist.entropy().mean()
        return log_prob, entropy, out["value"], out["mixer_logits"]


class _VehiclePPOAgent:
    """Independent PPO learner attached to a single vehicle."""

    def __init__(self, own_dim: int, nbr_dim: int, hidden_dim: int):
        self.policy = _GATActorCritic(own_dim, nbr_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=PPO_LR)
        self.pending_transition: dict | None = None
        self.pending_round: int | None = None
        self.rollout: list[dict] = []

    def act(self, own_state: np.ndarray, nbr_features: np.ndarray) -> dict:
        own_t = torch.as_tensor(own_state, dtype=torch.float32)
        nbr_t = torch.as_tensor(nbr_features, dtype=torch.float32)

        self.policy.eval()
        with torch.no_grad():
            out = self.policy(own_t, nbr_t)
            if out["selector_logits"].numel() == 0:
                action = out["selector_logits"]
                log_prob = 0.0
                selector_prob = out["selector_logits"]
                mixer_prob = out["mixer_logits"]
            else:
                dist = Bernoulli(logits=out["selector_logits"])
                action = dist.sample()
                log_prob = float(dist.log_prob(action).mean().item())
                selector_prob = torch.sigmoid(out["selector_logits"])
                mixer_prob = torch.softmax(out["mixer_logits"] / MIXER_TAU, dim=0)

        return {
            "own_state": own_state.astype(np.float32, copy=True),
            "nbr_features": nbr_features.astype(np.float32, copy=True),
            "action": action.cpu().numpy().astype(np.float32, copy=True),
            "log_prob": log_prob,
            "value": float(out["value"].item()),
            "selector_prob": selector_prob.cpu().numpy().astype(np.float32, copy=True),
            "mixer_prob": mixer_prob.cpu().numpy().astype(np.float32, copy=True),
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

        if len(advantages) >= PPO_UPDATE_EVERY:
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

                log_prob, entropy, value, mixer_logits = self.policy.evaluate_actions(own_t, nbr_t, act_t)
                old_log_prob = torch.tensor(sample["log_prob"], dtype=torch.float32)
                advantage = torch.tensor(advantages[idx], dtype=torch.float32)
                return_t = torch.tensor(returns[idx], dtype=torch.float32)

                ratio = torch.exp(log_prob - old_log_prob)
                surr_1 = ratio * advantage
                surr_2 = torch.clamp(ratio, 1.0 - PPO_CLIP_EPS, 1.0 + PPO_CLIP_EPS) * advantage
                policy_loss = -torch.min(surr_1, surr_2)
                value_loss = F.mse_loss(value, return_t)

                mixer_loss = value.new_zeros(())
                if mixer_logits.numel() > 0:
                    target_probs = torch.as_tensor(
                        _mixer_target_from_features(sample["nbr_features"]),
                        dtype=torch.float32,
                    )
                    mixer_log_probs = F.log_softmax(mixer_logits / MIXER_TAU, dim=0)
                    mixer_loss = -(target_probs * mixer_log_probs).sum()

                loss = (
                    policy_loss
                    + PPO_VALUE_COEF * value_loss
                    + MIXER_LOSS_COEF * mixer_loss
                    - PPO_ENTROPY_COEF * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), PPO_MAX_GRAD_NORM)
                self.optimizer.step()

        self.rollout.clear()


class DANTEAlgorithm(DLAlgorithm):
    """Per-vehicle GAT + PPO selector with a decoupled learned mixer."""

    name = "DANTE"
    needs_dynamic_neighbors = True
    evaluation_mode = "personalized"

    def __init__(self):
        self._agents: dict[int, _VehiclePPOAgent] = {}
        self._retained_neighbors: dict[int, dict[int, dict]] = {}
        self._selection_state: dict[int, dict] = {}
        self._pending_debug_events: list[dict] = []
        self._pending_debug_index: dict[tuple[int, int], dict] = {}
        self.reward_source = _normalize_reward_source(PPO_REWARD_SOURCE)
        self.max_sidelink_neighbors = int(MAX_SIDELINK_NEIGHBORS)
        self.max_internet_neighbors = int(MAX_INTERNET_NEIGHBORS)
        self.max_collab_neighbors = int(MAX_COLLAB_NEIGHBORS)
        self.typical_round_energy_j = max(float(TYPICAL_ROUND_ENERGY_J), 1e-8)
        self.round_energy_budget_j = max(float(ROUND_ENERGY_BUDGET_J), 1e-8)
        self.round_bandwidth_budget_bits = (
            None
            if ROUND_BANDWIDTH_BUDGET_BITS is None
            else max(float(ROUND_BANDWIDTH_BUDGET_BITS), 0.0)
        )
        self.round_latency_budget_s = (
            None
            if ROUND_LATENCY_BUDGET_S is None
            else max(float(ROUND_LATENCY_BUDGET_S), 0.0)
        )
        self.payload_bits = float(tx_payload_bits())

    def setup(self, vehicles: list) -> None:
        own_dim = int(OWN_DIM)
        nbr_dim = int(NBR_DIM)

        for v in vehicles:
            agent = _VehiclePPOAgent(own_dim, nbr_dim, GAT_HIDDEN_DIM)
            self._agents[v.id] = agent
            self._retained_neighbors[v.id] = {}
            self._selection_state[v.id] = {
                "selected": set(),
                "link_types": {},
                "streaks": {},
            }

    def build_candidates(self, v, env) -> list:
        """Use SL-only discovery, then re-inject promoted retained peers as IN."""
        sidelink = env.sidelink_neighbors_of(v)
        visible = {nbr.id for nbr, _, _ in sidelink}
        retained = self._retained_neighbors.get(v.id, {})
        injected = []

        for nid, meta in sorted(
            retained.items(),
            key=lambda item: (-float(item[1].get("retained_score", 0.0)), item[0]),
        ):
            if nid == v.id or nid in visible or nid >= len(env.vehicles):
                continue
            nbr = env.vehicles[nid]
            dist = float(np.linalg.norm(v.pos - nbr.pos))
            injected.append((nbr, dist, LINK_INTERNET))

        return sidelink + injected

    def get_retained_score(self, v, neighbor_id: int) -> float:
        vehicle_id = int(v.id if hasattr(v, "id") else v)
        meta = self._retained_neighbors.get(vehicle_id, {}).get(int(neighbor_id))
        if meta is None:
            return 0.0
        return float(np.clip(meta.get("retained_score", 0.0), 0.0, 1.0))

    def _retained_score_from_grad_align(self, grad_align: float) -> float:
        align = float(np.clip(grad_align, 0.0, 1.0))
        return float(np.clip(0.5 + 0.5 * align, 0.0, 1.0))

    def _upsert_retained_neighbor(
        self,
        vehicle_id: int,
        neighbor_id: int,
        round_no: int,
        grad_align: float,
    ) -> bool:
        retained = self._retained_neighbors.setdefault(int(vehicle_id), {})
        is_new = int(neighbor_id) not in retained
        retained[int(neighbor_id)] = {
            "retained_score": self._retained_score_from_grad_align(grad_align),
            "last_good_round": int(round_no),
            "last_grad_align": float(grad_align),
        }
        return is_new

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

    def _shape_reward(self, progress: float, energy_j: float) -> float:
        cost_norm = float(max(energy_j, 0.0)) / self.typical_round_energy_j
        if progress >= 0.0:
            reward = progress / (1.0 + COST_LAMBDA * cost_norm)
        else:
            reward = progress * (1.0 + COST_LAMBDA * cost_norm)
        return float(np.clip(reward, -1.0, 1.0))

    def _proposal_benefit(self, selector_prob: float, mixer_prob: float, nbr_feature: np.ndarray) -> float:
        """Predict how useful a neighbor is, separate from resource constraints."""
        grad_align = max(float(nbr_feature[GRAD_ALIGN_IDX]), 0.0)
        cos_sim = max(float(nbr_feature[COS_SIM_IDX]), 0.0)
        nbr_acc = float(np.clip(nbr_feature[NBR_ACC_IDX], 0.0, 1.0))
        retained_score = float(np.clip(nbr_feature[RETAINED_SCORE_IDX], 0.0, 1.0))
        return float(selector_prob) * (
            1.5 * float(mixer_prob)
            + 1.0 * grad_align
            + 0.25 * cos_sim
            + 0.25 * nbr_acc
            + 0.25 * retained_score
        )

    def _select_budgeted_subset(self, proposals: list[dict]) -> list[dict]:
        """Choose the highest-benefit feasible subset under round budgets."""
        if not proposals:
            return []

        filtered = []
        for item in proposals:
            if (
                self.round_latency_budget_s is not None
                and item["latency_s"] > self.round_latency_budget_s
            ):
                continue
            if (
                self.round_bandwidth_budget_bits is not None
                and item["bandwidth_bits"] > self.round_bandwidth_budget_bits
            ):
                continue
            filtered.append(item)

        if not filtered:
            return []

        max_count = int(self.max_collab_neighbors)
        if self.round_bandwidth_budget_bits is not None and self.payload_bits > 0.0:
            bw_count_cap = int(self.round_bandwidth_budget_bits // self.payload_bits)
            max_count = min(max_count, bw_count_cap)
        if max_count <= 0:
            return []

        budget_j = max(float(self.round_energy_budget_j), 0.0)
        if budget_j <= 0.0:
            return []

        energy_unit = max(budget_j / 200.0, 1e-4)
        capacity = max(int(np.floor(budget_j / energy_unit + 1e-9)), 0)

        states: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {
            (0, 0): (0.0, ())
        }
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

        _, best_indices = max(states.values(), key=lambda entry: entry[0])
        chosen = [filtered[idx] for idx in best_indices]
        chosen.sort(key=lambda item: item["benefit"], reverse=True)
        return chosen

    def _record_selection_debug(
        self,
        v,
        candidates: list,
        selected: list[dict],
        target_round: int,
    ) -> None:
        state = self._selection_state.setdefault(
            v.id,
            {"selected": set(), "link_types": {}, "streaks": {}},
        )
        prev_selected = set(state["selected"])
        prev_link_types = dict(state["link_types"])
        prev_streaks = dict(state["streaks"])
        retained = self._retained_neighbors.get(v.id, {})

        available_now = {nbr.id for nbr, _, _ in candidates}
        current_selected = {item["nid"] for item in selected}
        current_link_types = {item["nid"]: item["link_type"] for item in selected}
        retained_injected = {
            nbr.id
            for nbr, _, link_type in candidates
            if nbr.id in retained and link_type == LINK_INTERNET
        }

        prev_available = prev_selected & available_now
        kept = prev_available & current_selected
        new_neighbors = current_selected - prev_selected
        dropped_available = prev_available - current_selected
        dropped_unavailable = prev_selected - available_now

        prev_in_selected = {
            nid for nid, link_type in prev_link_types.items() if link_type == LINK_INTERNET
        }
        current_in = {
            item["nid"] for item in selected if item["link_type"] == LINK_INTERNET
        }
        prev_in_available = prev_in_selected & available_now
        kept_prev_in = prev_in_available & current_selected
        kept_prev_in_as_in = kept_prev_in & current_in
        kept_prev_in_as_sl = kept_prev_in - current_in

        current_streaks = {}
        for item in selected:
            nid = item["nid"]
            current_streaks[nid] = (
                prev_streaks.get(nid, 0) + 1 if nid in prev_selected else 1
            )

        selected_neighbors = []
        for item in sorted(
            selected,
            key=lambda row: (-current_streaks[row["nid"]], -float(row["benefit"]), row["nid"]),
        ):
            selected_neighbors.append({
                "nid": int(item["nid"]),
                "link": _link_label(float(item["link_type"])),
                "streak": int(current_streaks[item["nid"]]),
                "benefit": float(item["benefit"]),
                "energy_j": float(item["energy_j"]),
            })

        def _sorted_prev(ids: set[int]) -> list[dict]:
            return [
                {
                    "nid": int(nid),
                    "link": _link_label(float(prev_link_types.get(nid, LINK_SIDELINK))),
                    "streak": int(prev_streaks.get(nid, 1)),
                }
                for nid in sorted(ids, key=lambda nid: (-prev_streaks.get(nid, 1), nid))
            ]

        event = {
            "vehicle_id": int(v.id),
            "target_round": int(target_round),
            "candidate_count": int(len(candidates)),
            "selected_count": int(len(current_selected)),
            "selected_sl_count": int(sum(item["link"] == "SL" for item in selected_neighbors)),
            "selected_in_count": int(sum(item["link"] == "IN" for item in selected_neighbors)),
            "retained_injected_count": int(len(retained_injected)),
            "promoted_count": 0,
            "demoted_policy_count": 0,
            "demoted_bad_effect_count": 0,
            "prev_available_count": int(len(prev_available)),
            "kept_count": int(len(kept)),
            "new_count": int(len(new_neighbors)),
            "dropped_available_count": int(len(dropped_available)),
            "dropped_unavailable_count": int(len(dropped_unavailable)),
            "prev_in_available_count": int(len(prev_in_available)),
            "kept_prev_in_count": int(len(kept_prev_in)),
            "kept_prev_in_as_in_count": int(len(kept_prev_in_as_in)),
            "kept_prev_in_as_sl_count": int(len(kept_prev_in_as_sl)),
            "selected_neighbors": selected_neighbors,
            "new_neighbors": [
                row for row in selected_neighbors if row["nid"] in new_neighbors
            ],
            "dropped_neighbors": _sorted_prev(dropped_available),
            "lost_neighbors": _sorted_prev(dropped_unavailable),
        }
        self._pending_debug_events.append(event)
        self._pending_debug_index[(int(v.id), int(target_round))] = event

        state["selected"] = current_selected
        state["link_types"] = current_link_types
        state["streaks"] = current_streaks

    def _record_retention_debug_outcome(
        self,
        vehicle_id: int,
        target_round: int,
        promoted_count: int,
        demoted_policy_count: int,
        demoted_bad_effect_count: int,
    ) -> None:
        event = self._pending_debug_index.get((int(vehicle_id), int(target_round)))
        if event is None:
            return
        event["promoted_count"] += int(promoted_count)
        event["demoted_policy_count"] += int(demoted_policy_count)
        event["demoted_bad_effect_count"] += int(demoted_bad_effect_count)

    def consume_debug_logs(self) -> list[str]:
        events = list(self._pending_debug_events)
        self._pending_debug_events.clear()
        self._pending_debug_index.clear()
        if not events:
            return []

        total_prev_available = sum(event["prev_available_count"] for event in events)
        total_kept = sum(event["kept_count"] for event in events)
        total_prev_in_available = sum(event["prev_in_available_count"] for event in events)
        total_kept_prev_in = sum(event["kept_prev_in_count"] for event in events)
        total_kept_prev_in_as_in = sum(
            event["kept_prev_in_as_in_count"] for event in events
        )
        avg_selected = float(np.mean([event["selected_count"] for event in events]))
        avg_new = float(np.mean([event["new_count"] for event in events]))
        avg_dropped_available = float(
            np.mean([event["dropped_available_count"] for event in events])
        )
        avg_lost = float(np.mean([event["dropped_unavailable_count"] for event in events]))
        total_retained_injected = sum(event["retained_injected_count"] for event in events)
        total_promoted = sum(event["promoted_count"] for event in events)
        total_demoted_policy = sum(event["demoted_policy_count"] for event in events)
        total_demoted_bad = sum(event["demoted_bad_effect_count"] for event in events)

        lines = [
            (
                "DANTE persistence | "
                f"updates={len(events)} | "
                f"avg sel={avg_selected:.2f} | "
                f"retIN={total_retained_injected} | "
                f"promoted={total_promoted} | "
                f"demoted(policy)={total_demoted_policy} | "
                f"demoted(bad)={total_demoted_bad} | "
                f"keep(avail)={total_kept}/{total_prev_available}"
                + (
                    f" ({100.0 * total_kept / total_prev_available:.1f}%)"
                    if total_prev_available
                    else ""
                )
                + " | "
                f"keep prevIN(avail)={total_kept_prev_in}/{total_prev_in_available}"
                + (
                    f" ({100.0 * total_kept_prev_in / total_prev_in_available:.1f}%)"
                    if total_prev_in_available
                    else ""
                )
                + " | "
                f"keep prevIN as IN={total_kept_prev_in_as_in}/{total_prev_in_available}"
                + (
                    f" ({100.0 * total_kept_prev_in_as_in / total_prev_in_available:.1f}%)"
                    if total_prev_in_available
                    else ""
                )
                + " | "
                f"avg new={avg_new:.2f} | "
                f"avg drop(avail)={avg_dropped_available:.2f} | "
                f"avg lost(unavail)={avg_lost:.2f}"
            )
        ]

        def _format_neighbors(items: list[dict], limit: int, include_metrics: bool) -> str:
            if not items:
                return "-"
            head = items[:limit]
            parts = []
            for item in head:
                text = f"{item['nid']}:{item['link']}:s{item['streak']}"
                if include_metrics:
                    text += (
                        f":b{item['benefit']:.2f}"
                        f":e{_format_energy(item['energy_j'])}"
                    )
                parts.append(text)
            if len(items) > limit:
                parts.append(f"+{len(items) - limit} more")
            return ", ".join(parts)

        for event in sorted(events, key=lambda row: row["vehicle_id"]):
            keep_text = (
                f"{event['kept_count']}/{event['prev_available_count']}"
                if event["prev_available_count"] > 0
                else "n/a"
            )
            prev_in_text = (
                f"{event['kept_prev_in_count']}/{event['prev_in_available_count']}"
                if event["prev_in_available_count"] > 0
                else "n/a"
            )
            prev_in_same_mode_text = (
                f"{event['kept_prev_in_as_in_count']}/{event['prev_in_available_count']}"
                if event["prev_in_available_count"] > 0
                else "n/a"
            )
            lines.append(
                f"DANTE v{event['vehicle_id']:02d} -> r{event['target_round']} | "
                f"cand={event['candidate_count']} | "
                f"sel={event['selected_count']} "
                f"(SL={event['selected_sl_count']} IN={event['selected_in_count']}) | "
                f"retIN={event['retained_injected_count']} | "
                f"prom={event['promoted_count']} | "
                f"demote(policy/bad)={event['demoted_policy_count']}/{event['demoted_bad_effect_count']} | "
                f"keep={keep_text} avail | "
                f"prevIN keep={prev_in_text} | "
                f"prevIN sameIN={prev_in_same_mode_text} | "
                f"now=[{_format_neighbors(event['selected_neighbors'], limit=5, include_metrics=True)}] | "
                f"new=[{_format_neighbors(event['new_neighbors'], limit=4, include_metrics=False)}] | "
                f"drop=[{_format_neighbors(event['dropped_neighbors'], limit=4, include_metrics=False)}] | "
                f"lost=[{_format_neighbors(event['lost_neighbors'], limit=4, include_metrics=False)}]"
            )

        return lines

    def select_neighbors(self, v, candidates: list, env) -> tuple:
        if not v.training_done.is_set():
            available = {nbr.id: link_type for nbr, _, link_type in candidates}
            connections = {nid for nid in v.connections if nid in available}
            alphas = {nid: float(v.alphas.get(nid, 0.0)) for nid in connections}
            link_types = {nid: available[nid] for nid in connections}
            return connections, alphas, link_types, None

        agent = self._agents[v.id]
        retained = self._retained_neighbors.get(v.id, {})
        own_state = v.own_features()
        nbr_features = env.neighbor_features(v, candidates)
        decision = agent.act(own_state, nbr_features)
        eligible_retained_ids = {
            nbr.id for nbr, _, _ in candidates if nbr.id in retained
        }

        selected = []
        for idx, (nbr, dist, link_type) in enumerate(candidates):
            if idx >= len(decision["action"]) or decision["action"][idx] < 0.5:
                continue

            energy_j = self._neighbor_energy_j(link_type, dist)
            latency_s = self._neighbor_latency_s(link_type, dist)
            bandwidth_bits = self._neighbor_bandwidth_bits(link_type, dist)
            cost_norm = energy_j / self.typical_round_energy_j
            selector_prob = float(decision["selector_prob"][idx])
            mixer_prob = float(decision["mixer_prob"][idx])
            benefit = self._proposal_benefit(selector_prob, mixer_prob, nbr_features[idx])
            utility = benefit - 0.1 * cost_norm
            selected.append({
                "idx": idx,
                "nid": nbr.id,
                "link_type": link_type,
                "energy_j": energy_j,
                "latency_s": latency_s,
                "bandwidth_bits": bandwidth_bits,
                "mixer_prob": mixer_prob,
                "benefit": benefit,
                "utility": utility,
                "grad_align": float(nbr_features[idx][GRAD_ALIGN_IDX]),
                "was_retained": bool(nbr.id in retained),
            })

        selected = self._select_budgeted_subset(selected)
        target_round = int(v.tr_rounds + (0 if env._vehicle_is_done(v) else 1))
        self._record_selection_debug(v, candidates, selected, target_round)

        connections = {item["nid"] for item in selected}
        link_types = {item["nid"]: item["link_type"] for item in selected}
        alphas = {}

        if selected:
            kept_mix = np.array([max(item["mixer_prob"], 0.0) for item in selected], dtype=np.float32)
            mix_total = float(kept_mix.sum())
            if mix_total <= 1e-8:
                kept_mix = np.full(len(selected), 1.0 / len(selected), dtype=np.float32)
            else:
                kept_mix = kept_mix / mix_total
            for item, alpha in zip(selected, kept_mix):
                alphas[item["nid"]] = float(alpha)

        round_energy_j = float(sum(item["energy_j"] for item in selected))

        transition = None
        if v.training_done.is_set() and not env._vehicle_is_done(v):
            transition = {
                "own_state": decision["own_state"],
                "nbr_features": decision["nbr_features"],
                "action": decision["action"],
                "log_prob": decision["log_prob"],
                "value": decision["value"],
                "energy_j": round_energy_j,
                "eligible_retained_ids": sorted(eligible_retained_ids),
                "selected_neighbors": [
                    {
                        "nid": int(item["nid"]),
                        "link_type": float(item["link_type"]),
                        "grad_align": float(item["grad_align"]),
                        "was_retained": bool(item["was_retained"]),
                    }
                    for item in selected
                ],
                "target_round": int(v.tr_rounds + 1),
            }

        return connections, alphas, link_types, transition

    def aggregate(self, v, vehicles: list) -> None:
        if not v.training_done.is_set():
            return

        accepted = [vehicles[nid] for nid in v.connections if nid < len(vehicles)]
        if not accepted:
            return

        raw_weights = np.array(
            [max(float(v.alphas.get(nbr.id, 0.0)), 0.0) for nbr in accepted],
            dtype=np.float32,
        )
        total = float(raw_weights.sum())
        if total <= 1e-8:
            nbr_weights = np.repeat(1.0 / len(accepted), len(accepted))
        else:
            nbr_weights = raw_weights / total

        nbr_sds = [nbr.get_shared_weights() for nbr in accepted]
        self_w = float(SELF_WEIGHT)

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

    def post_step(self, vehicles: list, transitions: dict, step_n: int) -> dict:
        rewards = {}

        for v in vehicles:
            agent = self._agents[v.id]
            next_transition = transitions.get(v.id)

            if (
                agent.pending_transition is not None
                and agent.pending_round is not None
                and v.tr_rounds >= agent.pending_round
            ):
                pending_transition = dict(agent.pending_transition)
                with v._lock:
                    train_loss_delta = v._prev_loss - v.current_loss
                    reward_loss_delta = v._prev_reward_loss - v.current_reward_loss
                    reward_acc_delta = v.current_reward_acc - v._prev_reward_acc

                if self.reward_source == "validation":
                    progress = 0.5 * train_loss_delta + 0.5 * reward_loss_delta
                else:
                    progress = reward_loss_delta
                progress += REWARD_ACC_WEIGHT * reward_acc_delta

                round_no = int(agent.pending_round)
                positive_progress = progress > 0.0
                promoted_count = 0
                demoted_policy_count = 0
                demoted_bad_effect_count = 0
                selected_neighbors = pending_transition.get("selected_neighbors", [])
                eligible_retained_ids = {
                    int(nid) for nid in pending_transition.get("eligible_retained_ids", [])
                }
                selected_ids = {int(item["nid"]) for item in selected_neighbors}

                for nid in sorted(eligible_retained_ids - selected_ids):
                    if self._drop_retained_neighbor(v.id, nid):
                        demoted_policy_count += 1

                for item in selected_neighbors:
                    nid = int(item["nid"])
                    link_type = float(item["link_type"])
                    grad_align = float(item["grad_align"])
                    was_retained = bool(item["was_retained"])
                    is_good = positive_progress and grad_align > 0.0

                    if is_good and (was_retained or link_type == LINK_SIDELINK):
                        is_new = self._upsert_retained_neighbor(
                            v.id,
                            nid,
                            round_no,
                            grad_align,
                        )
                        if link_type == LINK_SIDELINK and not was_retained and is_new:
                            promoted_count += 1
                    elif was_retained and self._drop_retained_neighbor(v.id, nid):
                        demoted_bad_effect_count += 1

                self._record_retention_debug_outcome(
                    v.id,
                    round_no,
                    promoted_count=promoted_count,
                    demoted_policy_count=demoted_policy_count,
                    demoted_bad_effect_count=demoted_bad_effect_count,
                )

                energy_j = float(pending_transition.get("energy_j", 0.0))
                reward = self._shape_reward(progress, energy_j)
                next_value = float(next_transition["value"]) if next_transition is not None else 0.0
                done = next_transition is None
                agent.finalize_pending(reward, next_value, done)
                v.reward_hist.append(reward)
                rewards[v.id] = reward

            if next_transition is not None:
                agent.store_pending(next_transition, next_transition["target_round"])

            force_update = next_transition is None and agent.pending_transition is None
            if agent.should_update(force=force_update):
                agent.update()

        return rewards
