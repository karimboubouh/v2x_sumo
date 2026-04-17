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
    SELECTOR_INIT_BIAS,
    SELF_WEIGHT,
    TYPICAL_ROUND_ENERGY_J,
)
from dl.helpers import clone_state_dict, inet_tx_energy_j, sl_tx_energy_j


COS_SIM_IDX = 0
GRAD_ALIGN_IDX = 1
TX_COST_IDX = 3
NBR_ACC_IDX = 5


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
    """Cheap teacher for the mixer: aligned, accurate, low-cost neighbors win."""
    if nbr_features.size == 0:
        return np.zeros((0,), dtype=np.float32)

    grad_align = np.clip(nbr_features[:, GRAD_ALIGN_IDX], -1.0, 1.0)
    nbr_acc = np.clip(nbr_features[:, NBR_ACC_IDX], 0.0, 1.0)
    tx_cost = np.clip(nbr_features[:, TX_COST_IDX], 0.0, None)

    teacher_logits = 2.5 * grad_align + 0.5 * nbr_acc - np.log1p(tx_cost)
    return _softmax_numpy(teacher_logits, tau=MIXER_TAU)


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
        self.reward_source = _normalize_reward_source(PPO_REWARD_SOURCE)
        self.max_sidelink_neighbors = int(MAX_SIDELINK_NEIGHBORS)
        self.max_internet_neighbors = int(MAX_INTERNET_NEIGHBORS)
        self.max_collab_neighbors = int(MAX_COLLAB_NEIGHBORS)
        self.typical_round_energy_j = max(float(TYPICAL_ROUND_ENERGY_J), 1e-8)

    def setup(self, vehicles: list) -> None:
        own_dim = int(OWN_DIM)
        nbr_dim = int(NBR_DIM)

        for v in vehicles:
            agent = _VehiclePPOAgent(own_dim, nbr_dim, GAT_HIDDEN_DIM)
            self._agents[v.id] = agent

    def _neighbor_energy_j(self, link_type: float, dist: float) -> float:
        if link_type == LINK_SIDELINK:
            return float(sl_tx_energy_j(dist))
        return float(inet_tx_energy_j())

    def _shape_reward(self, progress: float, energy_j: float) -> float:
        cost_norm = float(max(energy_j, 0.0)) / self.typical_round_energy_j
        if progress >= 0.0:
            reward = progress / (1.0 + COST_LAMBDA * cost_norm)
        else:
            reward = progress * (1.0 + COST_LAMBDA * cost_norm)
        return float(np.clip(reward, -1.0, 1.0))

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
        decision = agent.act(own_state, nbr_features)

        selected = []
        for idx, (nbr, dist, link_type) in enumerate(candidates):
            if idx >= len(decision["action"]) or decision["action"][idx] < 0.5:
                continue

            energy_j = self._neighbor_energy_j(link_type, dist)
            cost_norm = energy_j / self.typical_round_energy_j
            selector_prob = float(decision["selector_prob"][idx])
            mixer_prob = float(decision["mixer_prob"][idx])
            utility = selector_prob * mixer_prob / (1.0 + cost_norm)
            selected.append({
                "idx": idx,
                "nid": nbr.id,
                "link_type": link_type,
                "energy_j": energy_j,
                "mixer_prob": mixer_prob,
                "utility": utility,
            })

        max_k = int(self.max_collab_neighbors)
        if len(selected) > max_k:
            selected.sort(key=lambda item: item["utility"], reverse=True)
            selected = selected[:max_k]

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
                with v._lock:
                    train_loss_delta = v._prev_loss - v.current_loss
                    reward_loss_delta = v._prev_reward_loss - v.current_reward_loss
                    reward_acc_delta = v.current_reward_acc - v._prev_reward_acc

                if self.reward_source == "validation":
                    progress = 0.5 * train_loss_delta + 0.5 * reward_loss_delta
                else:
                    progress = reward_loss_delta
                progress += REWARD_ACC_WEIGHT * reward_acc_delta

                energy_j = float(agent.pending_transition.get("energy_j", 0.0))
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
