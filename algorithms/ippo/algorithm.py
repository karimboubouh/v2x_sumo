"""
IPPO — Independent PPO ablation baseline for GAT_PPO.

Removes the graph attention encoder entirely.  Neighborhood context is built
by mean-pooling projected neighbor features (uniform 1/N weights).  FL
aggregation also uses uniform weights — no learned α at any stage.

Key differences from GAT_PPO
────────────────────────────
  Encoding   : mean pooling  vs  attention-weighted sum
  Aggregation: uniform 1/N   vs  re-normalised α_ij
  Alphas     : uniform 1/|C| vs  softmax attention scores

Everything else — PPO update, GAE, reward shaping, per-candidate
Bernoulli decisions, simulation hooks — is identical to GAT_PPO.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

from algorithms.base import DLAlgorithm, LINK_INTERNET, LINK_SIDELINK
from algorithms.ippo.config import (
    MLP_HIDDEN_DIM,
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
    SELF_WEIGHT,
)
from dl.helpers import clone_state_dict, sl_tx_cost_norm
import config as global_cfg


# ── Network ───────────────────────────────────────────────────────────────────

class _MLPActorCritic(nn.Module):
    """
    Flat MLP encoder with mean-pooled neighborhood context.

    Architecture
    ────────────
    own_enc  = MLP(own_dim → H → H)            shape (H,)
    nbr_proj = Linear(nbr_dim → H, bias=False) shared per-neighbor projection
    mean_v   = (1/N) Σ_j nbr_proj(nbr_j)      shape (H,)   ← NO ATTENTION
    fused    = cat(own_enc, mean_v)            shape (2H,)

    actor:  for each candidate j
              logit_j = MLP(fused ‖ nbr_proj(nbr_j)) → scalar
              Bernoulli(σ(logit_j)) → keep/drop

    critic: MLP(fused) → scalar state value
    """

    def __init__(self, own_dim: int, nbr_dim: int, hidden_dim: int) -> None:
        super().__init__()
        H = hidden_dim

        self.own_encoder = nn.Sequential(
            nn.Linear(own_dim, H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.ReLU(),
        )
        # Shared linear projection for neighbor features (no bias — translation
        # invariant; mirrors the GAT's nbr_proj for architectural symmetry).
        self.nbr_proj = nn.Linear(nbr_dim, H, bias=False)

        self.actor = nn.Sequential(
            nn.Linear(2 * H + H, H),
            nn.ReLU(),
            nn.Linear(H, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(2 * H, H),
            nn.ReLU(),
            nn.Linear(H, 1),
        )

    def _encode(
        self,
        own_state: torch.Tensor,
        nbr_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (fused, logits, nbr_emb).

        fused    : (2H,) — context vector for critic and actor expansion
        logits   : (N,)  — un-normalised Bernoulli logits per candidate
                           empty tensor when N == 0
        nbr_emb  : (N,H) — projected neighbor features (for evaluate_actions)
        """
        own_enc = self.own_encoder(own_state)                  # (H,)

        if nbr_features.numel() == 0:
            # No candidates: mean vector is zero
            H = own_enc.shape[-1]
            mean_v   = own_enc.new_zeros((H,))
            fused    = torch.cat([own_enc, mean_v], dim=-1)    # (2H,)
            logits   = nbr_features.new_zeros((0,))
            nbr_emb  = nbr_features.new_zeros((0, H))
            return fused, logits, nbr_emb

        nbr_emb = self.nbr_proj(nbr_features)                  # (N, H)
        mean_v  = nbr_emb.mean(dim=0)                          # (H,)  ← MEAN POOL
        fused   = torch.cat([own_enc, mean_v], dim=-1)         # (2H,)

        fused_rep = fused.unsqueeze(0).expand(nbr_emb.shape[0], -1)  # (N, 2H)
        logits = self.actor(
            torch.cat([fused_rep, nbr_emb], dim=-1)            # (N, 3H)
        ).squeeze(-1)                                           # (N,)

        return fused, logits, nbr_emb

    def forward(
        self,
        own_state: torch.Tensor,
        nbr_features: torch.Tensor,
    ) -> dict:
        fused, logits, _ = self._encode(own_state, nbr_features)
        value = self.critic(fused).squeeze(-1)
        return {"value": value, "logits": logits}

    def evaluate_actions(
        self,
        own_state: torch.Tensor,
        nbr_features: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fused, logits, _ = self._encode(own_state, nbr_features)
        value = self.critic(fused).squeeze(-1)
        if logits.numel() == 0:
            log_prob = value.new_zeros(())
            entropy  = value.new_zeros(())
        else:
            dist     = Bernoulli(logits=logits)
            log_prob = dist.log_prob(actions).sum()
            entropy  = dist.entropy().sum()
        return log_prob, entropy, value


# ── Per-vehicle PPO agent ─────────────────────────────────────────────────────

class _VehiclePPOAgent:
    """Independent PPO learner — identical update logic to GAT-PPO."""

    def __init__(self, own_dim: int, nbr_dim: int, hidden_dim: int) -> None:
        self.policy    = _MLPActorCritic(own_dim, nbr_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=PPO_LR)
        self.pending_transition: dict | None = None
        self.pending_round:      int  | None = None
        self.rollout: list[dict] = []

    # ── Inference ─────────────────────────────────────────────────────────────

    def act(self, own_state: np.ndarray, nbr_features: np.ndarray) -> dict:
        own_t = torch.as_tensor(own_state,    dtype=torch.float32)
        nbr_t = torch.as_tensor(nbr_features, dtype=torch.float32)

        self.policy.eval()
        with torch.no_grad():
            out = self.policy(own_t, nbr_t)
            if out["logits"].numel() == 0:
                action   = out["logits"]   # empty tensor
                log_prob = 0.0
            else:
                dist     = Bernoulli(logits=out["logits"])
                action   = dist.sample()
                log_prob = float(dist.log_prob(action).sum().item())

        return {
            "own_state":    own_state.astype(np.float32,    copy=True),
            "nbr_features": nbr_features.astype(np.float32, copy=True),
            "action":       action.cpu().numpy().astype(np.float32, copy=True),
            "log_prob":     log_prob,
            "value":        float(out["value"].item()),
        }

    # ── Rollout management ────────────────────────────────────────────────────

    def store_pending(self, transition: dict, target_round: int) -> None:
        self.pending_transition = dict(transition)
        self.pending_round      = int(target_round)

    def finalize_pending(self, reward: float, next_value: float, done: bool) -> None:
        if self.pending_transition is None:
            return
        transition = dict(self.pending_transition)
        transition.update({
            "reward":     float(reward),
            "next_value": float(next_value),
            "done":       bool(done),
        })
        self.rollout.append(transition)
        self.pending_transition = None
        self.pending_round      = None

    def should_update(self, force: bool = False) -> bool:
        if not self.rollout:
            return False
        return force or len(self.rollout) >= PPO_UPDATE_EVERY

    # ── PPO update (GAE + clipped surrogate) — identical to GAT-PPO ──────────

    def update(self) -> None:
        if not self.rollout:
            return

        rewards     = np.array([t["reward"]     for t in self.rollout], dtype=np.float32)
        values      = np.array([t["value"]      for t in self.rollout], dtype=np.float32)
        next_values = np.array([t["next_value"] for t in self.rollout], dtype=np.float32)
        dones       = np.array([t["done"]       for t in self.rollout], dtype=np.float32)

        # GAE advantages
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for idx in reversed(range(len(self.rollout))):
            mask   = 1.0 - dones[idx]
            delta  = rewards[idx] + PPO_GAMMA * next_values[idx] * mask - values[idx]
            gae    = delta + PPO_GAMMA * PPO_GAE_LAMBDA * mask * gae
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
                own_t  = torch.as_tensor(sample["own_state"],    dtype=torch.float32)
                nbr_t  = torch.as_tensor(sample["nbr_features"], dtype=torch.float32)
                act_t  = torch.as_tensor(sample["action"],       dtype=torch.float32)

                log_prob, entropy, value = self.policy.evaluate_actions(own_t, nbr_t, act_t)
                old_log_prob = torch.tensor(sample["log_prob"], dtype=torch.float32)
                advantage    = torch.tensor(advantages[idx],    dtype=torch.float32)
                return_t     = torch.tensor(returns[idx],       dtype=torch.float32)

                ratio    = torch.exp(log_prob - old_log_prob)
                surr_1   = ratio * advantage
                surr_2   = torch.clamp(ratio, 1.0 - PPO_CLIP_EPS, 1.0 + PPO_CLIP_EPS) * advantage
                pol_loss = -torch.min(surr_1, surr_2)
                val_loss = F.mse_loss(value, return_t)
                loss     = pol_loss + PPO_VALUE_COEF * val_loss - PPO_ENTROPY_COEF * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), PPO_MAX_GRAD_NORM)
                self.optimizer.step()

        self.rollout.clear()


# ── Algorithm class ───────────────────────────────────────────────────────────

class IPPOAlgorithm(DLAlgorithm):
    """
    IPPO — Independent PPO without graph attention (ablation baseline).

    Neighbor selection: independent Bernoulli PPO per vehicle.
    Neighborhood encoding: mean pooling of projected neighbor features.
    FL aggregation: uniform 1/|C_i| weights (no attention scores).
    """

    name = "IPPO"
    needs_dynamic_neighbors = True

    def __init__(self) -> None:
        self._agents: dict[int, _VehiclePPOAgent] = {}

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self, vehicles: list) -> None:
        for v in vehicles:
            agent = _VehiclePPOAgent(OWN_DIM, NBR_DIM, MLP_HIDDEN_DIM)
            self._agents[v.id] = agent
            v._ippo_agent = agent

    # ── Neighbor selection ────────────────────────────────────────────────────

    def select_neighbors(self, v, candidates: list, env) -> tuple:
        # While training is in progress keep the existing neighbor set alive.
        if not v.training_done.is_set():
            available   = {nbr.id: link_type for nbr, _, link_type in candidates}
            connections = {nid for nid in v.connections if nid in available}
            alphas      = {nid: float(v.alphas.get(nid, 0.0)) for nid in connections}
            link_types  = {nid: available[nid] for nid in connections}
            return connections, alphas, link_types, None

        agent        = self._agents[v.id]
        own_state    = v.own_features()
        nbr_features = env.neighbor_features(v, candidates)
        decision     = agent.act(own_state, nbr_features)

        connections: set  = set()
        link_types:  dict = {}
        tx_cost:     float = 0.0

        for idx, (nbr, dist, link_type) in enumerate(candidates):
            if idx >= len(decision["action"]) or decision["action"][idx] < 0.5:
                continue
            connections.add(nbr.id)
            link_types[nbr.id] = link_type
            if link_type == LINK_SIDELINK:
                tx_cost += float(sl_tx_cost_norm(dist)) * 0.02
            else:
                tx_cost += 0.05

        # Cap to MAX_NEIGHBORS — no attention score to rank by, so keep
        # candidates in the order the environment presented them (FIFO).
        max_k = int(global_cfg.DL_CFG["MAX_NEIGHBORS"])
        if len(connections) > max_k:
            overflow = list(connections)[max_k:]
            for nid in overflow:
                connections.discard(nid)
                link_types.pop(nid, None)

        # Uniform alpha: 1 / |C_i|  (visualised as equal-width links)
        uniform_a = 1.0 / max(len(connections), 1)
        alphas    = {nid: uniform_a for nid in connections}

        transition = None
        if v.training_done.is_set() and not env._vehicle_is_done(v):
            transition = {
                "own_state":    decision["own_state"],
                "nbr_features": decision["nbr_features"],
                "action":       decision["action"],
                "log_prob":     decision["log_prob"],
                "value":        decision["value"],
                "cost":         float(tx_cost),
                "target_round": int(v.tr_rounds + 1),
            }

        return connections, alphas, link_types, transition

    # ── Aggregation — uniform FedAvg ──────────────────────────────────────────

    def aggregate(self, v, vehicles: list) -> None:
        if not v.training_done.is_set():
            return

        accepted = [vehicles[nid] for nid in v.connections if nid < len(vehicles)]
        if not accepted:
            return

        nbr_w  = 1.0 / len(accepted)          # uniform weight per neighbor
        self_w = float(SELF_WEIGHT)
        nbr_sds = [nbr.get_shared_weights() for nbr in accepted]

        with v._lock:
            own_sd = v.model.state_dict()
            new_sd = clone_state_dict(own_sd)
            for key in new_sd:
                if not new_sd[key].is_floating_point():
                    continue
                agg = self_w * own_sd[key].float()
                for sd in nbr_sds:
                    agg = agg + (1.0 - self_w) * nbr_w * sd[key].float()
                new_sd[key] = agg
            v.model.load_state_dict(new_sd)
            v._param_vec = None

    # ── Post-step: reward assignment + PPO update trigger ─────────────────────
    # Identical to GAT_PPO — the reward signal and update scheduling do not
    # depend on attention weights.

    def post_step(self, vehicles: list, transitions: dict, step_n: int) -> dict:
        rewards: dict = {}

        for v in vehicles:
            agent           = self._agents[v.id]
            next_transition = transitions.get(v.id)

            if (
                agent.pending_transition is not None
                and agent.pending_round is not None
                and v.tr_rounds >= agent.pending_round
            ):
                reward = max(float(v._prev_loss - v.current_loss), 0.0)
                reward -= float(agent.pending_transition.get("cost", 0.0))
                next_value = (
                    float(next_transition["value"])
                    if next_transition is not None
                    else 0.0
                )
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
