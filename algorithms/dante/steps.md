# DANTE - Execution walkthrough from vehicle i's perspective

## 0. Initialization

Each vehicle gets an independent PPO agent with:

```text
_GATActorCritic
|- _GATLayer   (single-head attention over ego + candidate neighbors)
|- selector    (Bernoulli actor over feasible peers)
`- critic      (state-value head)
```

The algorithm also keeps three local state tables per vehicle:

- trust scores `q_ij`
- retained neighbors eligible for Internet reinjection
- residual energy/bandwidth budgets plus last-round latency slack

## 1. Observe the local state

`v.own_features()` now exposes:

1. validation loss
2. validation accuracy
3. remaining energy ratio
4. remaining bandwidth ratio
5. previous-round latency slack

`env.neighbor_features(v, candidates)` now exposes:

1. gradient alignment
2. normalized energy cost
3. normalized latency cost
4. link type (`SL` or `IN`)
5. trust score `q_ij`
6. last robust score

## 2. Candidate discovery

DANTE discovers new neighbors over sidelink only.

- PC5 neighbors come directly from physical discovery.
- Previously helpful sidelink peers are retained locally.
- Retained peers may be re-injected as Internet candidates even after they leave PC5 range.

This keeps the paper's decentralized logic while matching the intended "PC5 first, then keep good peers over Uu" behavior.

## 3. GAT encoding and PPO action

For the current candidate set:

```text
h_i      = W_s x_i
v_ij     = W_n x_ij
beta_ij  = softmax(attention([h_i ; v_ij]))
c_i      = sum_j beta_ij * v_ij
```

The actor outputs one Bernoulli per feasible peer:

```text
pi(a_ij = 1 | o_i) = sigmoid(f([h_i ; c_i ; v_ij]))
```

The critic estimates:

```text
V(o_i) = g([h_i ; c_i])
```

PPO samples a per-candidate binary action over the full feasible set
`new sidelink peers + retained Internet peers`.

## 4. Admissibility and subset selection

After PPO samples the feasible set, DANTE keeps only hard admissibility checks.

- hard cap on the number of collaborators
- hard per-round latency constraint
- current residual energy budget
- current residual bandwidth budget

If the sampled set exceeds the active budget or cap, DANTE greedily keeps peers by:

```text
score_ij = beta_ij * max(q_ij * s_ij, 0) / (eps + comm_cost_ij)
```

with actor probability used only as a tie-break. There is no second
validation-based set acceptance gate after PPO.

## 5. Robust aggregation

Accepted neighbor models are aggregated with the paper-style fused weight:

```text
alpha_ij proportional to beta_ij * q_ij * r_ij
```

where:

- `beta_ij` is the GAT attention
- `q_ij` is trust
- `r_ij` is the robust score from the adaptive trimmed-mean filter

The local self-weight follows the paper's scheduled `alpha_ii^(t)` idea rather
than staying fixed:

```text
alpha_ii^(t) = linear_schedule(0.80 -> 0.50)
```

If every accepted peer fails the robust filter, DANTE falls back to pure local training.

## 6. Local training and trust update

After aggregation, the vehicle trains locally and evaluates the updated model on its validation split.

For each selected neighbor:

```text
phi_ij = 1{robust_pass_ij} * 1{alpha_ij > 0} * 1{val_loss_delta > 0}
q_ij <- (1 - rho) q_ij + rho phi_ij
```

Retention behavior:

- a helpful sidelink peer (`phi_ij = 1`) is promoted to the retained set
- retained peers remain in memory while they stay addressable or useful
- retained peers reappear over Internet when they are active, not visible on PC5, trusted, and robust

## 7. Reward

DANTE reports a weighted normalized form of the paper stage payoff:

```text
reported_reward =
    normalized_validation_gain
    - (
        lambda_E * communication_energy / round_energy_budget
        + lambda_B * bandwidth / round_bandwidth_budget
        + lambda_T * latency / round_latency_budget
      )

where lambda_E = lambda_B = lambda_T = 0.10

ppo_reward = reported_reward - ema(no_collab_reward)
```

Computation energy is still reported in experiment totals, but the reported
reward keeps only the communication-side stage payoff. PPO is trained on
collaboration advantage relative to the running no-collaboration baseline so the
policy is not rewarded for local learning progress it did not cause.

## 8. PPO update

Once enough local transitions are collected, the vehicle runs PPO with:

- GAE advantages
- clipped surrogate loss
- entropy regularization
- local trajectories only

No central critic, no shared parameters, and no global state are introduced.
