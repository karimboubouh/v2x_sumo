# DANTE - Execution walkthrough from vehicle i's perspective

## 0. Initialization

Each vehicle gets an independent PPO agent with:

```text
_GATActorCritic
|- _GATLayer   (single-head attention over ego + candidate neighbors)
|- selector    (Bernoulli actor over candidate edges)
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
6. accepted-neighbor ratio

`env.neighbor_features(v, candidates)` now exposes:

1. gradient alignment
2. normalized distance
3. normalized energy cost
4. normalized bandwidth cost
5. normalized latency cost
6. relative mobility
7. link type (`SL` or `IN`)
8. trust score `q_ij`

## 2. Candidate discovery

DANTE discovers new neighbors over sidelink only.

- PC5 neighbors come directly from physical discovery.
- Previously helpful sidelink peers are retained locally.
- Retained peers may be re-injected as Internet candidates if they stay within Internet range.

This keeps the paper's decentralized logic while matching the intended "PC5 first, then keep good peers over Uu" behavior.

## 3. GAT encoding and PPO action

For the current candidate set:

```text
h_i      = W_s x_i
v_ij     = W_n x_ij
beta_ij  = softmax(attention([h_i ; v_ij]))
c_i      = sum_j beta_ij * v_ij
```

The actor samples one Bernoulli decision per candidate:

```text
a_ij ~ Bernoulli(sigmoid(f([h_i ; c_i ; v_ij])))
```

The critic estimates:

```text
V(o_i) = g([h_i ; c_i])
```

PPO stores the joint Bernoulli log-probability as a sum across candidates.

## 4. Admissibility and subset selection

After sampling, DANTE keeps only a budget-feasible subset.

- hard cap on the number of collaborators
- hard per-round latency constraint
- current residual energy budget
- current residual bandwidth budget

Among feasible subsets, DANTE keeps the one with the largest total predicted benefit:

```text
benefit_ij = p_select_ij * max(q_ij * s_ij, 0)
```

where `s_ij` is gradient alignment.

## 5. Robust aggregation

Accepted neighbor models are aggregated with the paper-style fused weight:

```text
alpha_ij proportional to beta_ij * q_ij * r_ij
```

where:

- `beta_ij` is the GAT attention
- `q_ij` is trust
- `r_ij` is the robust score from the adaptive trimmed-mean filter

If every accepted peer fails the robust filter, DANTE falls back to pure local training.

## 6. Local training and trust update

After aggregation, the vehicle trains locally and evaluates the updated model on its validation split.

For each selected neighbor:

```text
phi_ij = 1{robust_pass_ij} * 1{validation loss did not worsen}
q_ij <- (1 - rho) q_ij + rho phi_ij
```

Retention behavior:

- a helpful sidelink peer (`phi_ij = 1`) is promoted to the retained set
- a retained Internet peer is dropped if it is offered but not selected
- a retained peer is also dropped if it is selected and later gets `phi_ij = 0`

## 7. Reward

DANTE now uses a validation-only stage payoff:

```text
reward =
    validation_loss_drop
    - total_energy / active_energy_budget
    - bandwidth / active_bandwidth_budget
    - latency / active_latency_budget
```

`total_energy` includes both communication energy induced by the chosen links and the local computation energy spent in the round.

## 8. PPO update

Once enough local transitions are collected, the vehicle runs PPO with:

- GAE advantages
- clipped surrogate loss
- entropy regularization
- local trajectories only

No central critic, no shared parameters, and no global state are introduced.
