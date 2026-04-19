# DANTE — Execution walkthrough from vehicle $i$'s perspective

---

## 0. Initialization (`setup`)

Every vehicle gets its own independent PPO agent with one shared encoder and three heads:

```
_GATActorCritic
├── own_encoder   (MLP: 6 → 64 → 64)    encodes v_i's own state
├── gat           (_GATLayer)            builds shared neighborhood context
├── selector      (MLP: 192 → 64 → 1)   PPO Bernoulli keep/drop logit
├── mixer         (MLP: 192 → 64 → 1)   aggregation score per neighbor
└── critic        (MLP: 128 → 64 → 1)   state value V(s)
```

The selector and mixer are now decoupled:
- selector decides **who to keep**
- mixer decides **how much weight each kept neighbor gets**

---

## Step 1 — Observe the state

`select_neighbors()` consumes:

**Own state** — `v.own_features()` → shape `(6,)`

**Neighbor features** — `env.neighbor_features(v, candidates)` → shape `(N, 8)`

| index | meaning |
|-------|---------|
| 0 | cosine similarity of model parameters with $j$ |
| 1 | gradient alignment with $j$ |
| 2 | normalized distance to $j$ |
| 3 | normalized transfer energy cost |
| 4 | relative heading divergence |
| 5 | neighbor accuracy |
| 6 | link type (sidelink=0, internet=1) |
| 7 | retained-neighbor score |

The new signals are **gradient alignment** and the **retained-neighbor score** used by SL-first DANTE when a previously good sidelink peer is re-injected as an internet candidate.

---

## Step 2 — Shared encoder + decoupled heads

`agent.act(own_state, nbr_features)`

### 2a. Shared GAT context

```
h_i  = self_proj(own_state)
v_j  = nbr_proj(nbr_features)
e_ij = attn(LeakyReLU([h_i_rep ; v_j]))
a_ij = softmax(e_ij)
emb  = Σ a_ij * v_j
```

This attention is now only an internal context mechanism. It is **not** reused as the aggregation weight.

### 2b. Heads

```
own_enc = own_encoder(own_state)
fused   = concat(own_enc, emb)
value   = critic(fused)

selector_logit_j = selector(concat(fused_rep, v_j))
mixer_logit_j    = mixer(concat(fused_rep, v_j))
```

The selector samples:

```
action_j ~ Bernoulli(sigmoid(selector_logit_j))
```

The mixer produces a separate soft ranking:

```
mixer_prob = softmax(mixer_logits / tau)
```

The mixer is trained to approximate statistical relevance, not transmission cost. Cost is handled later by the admissibility layer.

---

## Step 3 — Keep only useful neighbors

Vehicle $i$ only keeps neighbors with `action_j = 1`.

Each selected neighbor gets a predicted benefit score from the selector confidence, mixer relevance, and alignment features. Cost is tracked separately.

```
benefit_j ≈ selector_prob_j * relevance_j
```

From the PPO-accepted neighbors, DANTE then builds the final collaboration set under the paper-style admissibility constraints:

```
|C_i| <= K_i
sum_j energy_j <= round_energy_budget
sum_j bandwidth_j <= round_bandwidth_budget     # optional
max_j latency_j <= round_latency_budget         # optional
```

Among all feasible subsets, DANTE keeps the one with the largest total predicted benefit. This means a more expensive but highly relevant neighbor can still be preferred over several cheap but weak candidates, as long as the resulting set stays within budget.

Final aggregation weights are then recomputed from the mixer probabilities over the kept subset only.

The pending PPO transition stores:

```
{own_state, nbr_features, action, log_prob, value, energy_j, target_round}
```

---

## Step 4 — Local training + gradient capture

During `train_local()`:
- the vehicle trains for `BATCHES_PER_ROUND` mini-batches
- the last mini-batch gradient of the first two parameter tensors is captured
- `last_grad_vec` is saved for future neighbor scoring

This gives the next round a direct proxy for whether a neighbor's update direction is aligned with mine.

---

## Step 5 — Aggregation uses the mixer, not the selector

After local training, accepted neighbors are aggregated with:

```
w_j     = mixer_prob_j / Σ mixer_prob_k     # over kept neighbors only
new_θ_i = SELF_WEIGHT * θ_i + (1-SELF_WEIGHT) * Σ w_j * θ_j
```

The mixer head controls aggregation. The selector head no longer leaks into the aggregation rule.

---

## Step 6 — Reward and PPO update

When the pending round finishes, DANTE computes a blended progress signal:

```
progress = 0.5 * Δtrain_loss + 0.5 * Δreward_loss + c_acc * Δreward_acc
```

Communication cost uses actual transmission energy:

```
cost_norm = round_energy_j / TYPICAL_ROUND_ENERGY_J
```

Reward is shaped to prefer helpful, sparse neighborhoods:

```
if progress >= 0:
    reward = progress / (1 + λ * cost_norm)
else:
    reward = progress * (1 + λ * cost_norm)
```

This means:
- helpful, cheap neighborhoods keep most of their reward
- expensive neighborhoods get discounted
- expensive bad neighborhoods are punished harder

The PPO update still uses GAE and a clipped surrogate, but now:
- updates happen every `8` transitions instead of `4`
- advantage normalization is skipped on tiny rollouts
- the mixer head gets an auxiliary supervised loss toward a gradient-alignment target

---

## Summary

```
Round t:
  [observe own state + 7-dim neighbor features]
           ↓
  [shared GAT context]
           ↓
  [selector PPO head decides keep/drop]
           ↓
  [mixer head scores kept neighbors]
           ↓
  [budget-feasible subset selection]
           ↓
  [local training + gradient capture]
           ↓
  [aggregate with mixer weights]
           ↓
  [reward = progress shaped by actual energy]
           ↓
  [PPO update + mixer auxiliary update]
```

The key change is that DANTE now learns **selection** and **aggregation** with separate mechanisms while exposing a direct gradient-alignment signal and a reward that makes excessive internet/sidelink usage costly.
