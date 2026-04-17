# DANTE — Execution walkthrough from vehicle $i$'s perspective

---

## 0. Initialization (`setup`)

When the experiment starts, every vehicle gets its own **independent PPO agent** — a `_VehiclePPOAgent`. That agent owns a single `_GATActorCritic` neural network with three sub-modules:

```
_GATActorCritic
├── own_encoder   (MLP: 6 → 64 → 64)    encodes v_i's own state
├── gat           (_GATLayer)            attends over candidates
├── actor         (MLP: 192 → 64 → 1)   decides keep/drop per candidate
└── critic        (MLP: 128 → 64 → 1)   estimates state value V(s)
```

Nobody shares weights with anyone. Vehicle $i$'s policy is entirely its own.

---

## Step 1 — Observe the state

At the start of each collaboration step, `select_neighbors` is called. Vehicle $i$ observes two things:

**Own state** — `v.own_features()` → shape `(6,)`:

| index | meaning |
|-------|---------|
| 0 | `current_loss / 5` |
| 1 | `current_acc` |
| 2 | `\|connections\| / MAX_COLLAB_NEIGHBORS` |
| 3 | `pos_x / network_size` |
| 4 | `pos_y / network_size` |
| 5 | Byzantine flag (0 or 1) |

**Neighbor features** — `env.neighbor_features(v, candidates)` → shape `(N, 6)`:

| index | meaning |
|-------|---------|
| 0 | cosine similarity of model parameters with $j$ |
| 1 | normalized distance to $j$ |
| 2 | tx_cost = `norm_dist²` |
| 3 | relative heading speed (direction divergence) |
| 4 | neighbor accuracy |
| 5 | link type (sidelink=0, internet=1) |

---

## Step 2 — Forward pass through the GAT+PPO policy

`agent.act(own_state, nbr_features)`

**2a. GAT layer:**

```
h_i  = self_proj(own_state)               # shape (64,)
v_j  = nbr_proj(nbr_features)             # shape (N, 64)
e_ij = attn(LeakyReLU([h_i_rep ; v_j]))   # shape (N,)
α    = softmax(e_ij)                       # attention weights, sum to 1
emb  = Σ α_j * v_j                        # shape (64,)  ← aggregated neighborhood
```

The attention $\alpha_j$ tells the model how "relevant" neighbor $j$ is, before any keep/drop decision is made.

**2b. Own encoder:** `own_enc = MLP(own_state)` → shape `(64,)`

**2c. Fused context:** `fused = concat(own_enc, emb)` → shape `(128,)`

**2d. Critic:** `V(s) = critic(fused)` → scalar (used later for PPO)

**2e. Actor — per-neighbor decision:**

```
for each neighbor j:
    logit_j = actor(concat(fused_rep, v_j))
    action_j ~ Bernoulli(sigmoid(logit_j))   # 1 = keep, 0 = drop
```

---

## Step 3 — Filter and cap the neighborhood

Vehicle $i$ iterates its candidates:
- keeps neighbor $j$ only if `action[j] == 1`
- records `alphas[j.id] = attention[j]` and accumulates `tx_cost`

If more than `MAX_COLLAB_NEIGHBORS` were kept, the lowest-attention ones are dropped:

```python
# keep only top-k by attention weight
for nid in sorted(connections, key=lambda n: alphas[n])[:-max_k]:
    connections.discard(nid)
```

A **pending transition** is saved:
```
{own_state, nbr_features, action, log_prob, value, tx_cost, target_round = tr_rounds + 1}
```

This is held until the round completes and a reward can be assigned.

---

## Step 4 — Local training

Vehicle $i$ trains locally for `BATCHES_PER_ROUND` mini-batches in a background thread:
- `_prev_loss` is set to the old `current_loss`
- model is trained, `current_loss` and `current_acc` are updated
- `tr_rounds += 1`, `training_done` event is set

---

## Step 5 — Attention-weighted aggregation (`aggregate`)

Vehicle $i$ merges its model with the accepted neighbors', using the **GAT attention weights** as aggregation coefficients:

```
w_j     = α_j / Σ α_k                              # normalize over kept neighbors
new_θ_i = 0.3 * θ_i + 0.7 * Σ w_j * θ_j
```

The 0.3 / 0.7 split is `SELF_WEIGHT = 0.3`. The same $\alpha_j$ values that the GAT computed to decide *who to keep* are reused to decide *how much to trust* each keeper. This is the key coupling in DANTE.

---

## Step 6 — Reward signal and PPO update (`post_step`)

Once `v.tr_rounds >= target_round` (the pending round has completed):

**Reward:**
```
r = max(prev_loss - current_loss, 0)   # loss improvement
  - tx_cost                            # communication penalty
```

Positive if the round reduced loss, penalized by how expensive the chosen links were.

The pending transition is **finalized** with `(reward, next_value, done)` and pushed into the rollout buffer.

**PPO update** triggers when `len(rollout) >= PPO_UPDATE_EVERY` (= 8 transitions) or at episode end:

1. Compute **GAE advantages** over the rollout (γ = 0.99, λ = 0.95)
2. For `PPO_EPOCHS = 4` epochs, shuffle and iterate:
   - Re-evaluate log-probs and value under the current policy
   - Compute clipped surrogate loss: `min(r·A, clip(r, 0.8, 1.2)·A)`
   - Add value loss (MSE) and entropy bonus
   - Gradient step with `max_grad_norm = 1.0` clipping
3. Rollout buffer is cleared

---

## Summary

```
Round t:
  [observe own_state + nbr_features]
           ↓
  [GAT attention α_j + Bernoulli actions]
           ↓
  [filter neighbors → connections, store transition]
           ↓
  [train locally → loss ↓, tr_rounds++]
           ↓
  [aggregate: own*0.3 + Σ α_j*nbr_j*0.7]
           ↓
  [reward = Δloss - tx_cost]
           ↓
  [PPO update every 8 rounds]
           ↓
Round t+1  (better policy)
```

The core novelty: the **same attention weights that select neighbors also control aggregation weights**, and the whole selection policy is learned end-to-end via PPO with a communication-aware reward.
