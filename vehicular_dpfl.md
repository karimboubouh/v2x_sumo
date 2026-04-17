# Dynamic Network Formation for Robust Decentralized Personalized Learning in Vehicular Networks

> This proposal refines and specializes the earlier generic P2P DPFL draft to the vehicular networks setting.

---

`````

`````



## 0. $Notes$

- [ ] Datasets for realistic road traffic
- [ ] SUMO simulator
- [ ] Running algorithm in the SUMO  simulator



```python
Algorithm choices
-----------------
  DANTE    Proposed: Graph Attention Network + Proximal Policy Optimization.
           Each vehicle uses a GAT to encode its neighborhood and a PPO agent
           to decide which neighbors to collaborate with. Aggregation is
           personalized and attention-weighted.

  IPPO     Ablation: Independent PPO without graph attention. Each vehicle
           uses a flat MLP actor-critic that receives its own state plus
           mean-pooled neighbor features. Useful to isolate the contribution
           of the GAT component versus pure RL selection.

  FedAvg   Baseline: fixed initial-neighbor graph, equal-weight FedAvg,
           no RL. Benchmarks the selection mechanism.
    
    
- DANTE (our algorithm)
    Each vehicle's independent GAT+PPO agent decides which in-range
    neighbors to collaborate with each simulation step.
    Aggregation is personalized and attention-weighted:
        new_θ_i = SELF_WEIGHT·θ_i + (1−SELF_WEIGHT)·Σ_j ᾱ_j·θ_j
    where ᾱ_j = α_j / Σ_k α_k are re-normalized GAT attention weights.

- IPPO (ablation)
    Same RL loop as DANTE but without graph attention.  A flat MLP
    actor-critic with mean-pooled neighborhood features replaces the
    GATActorCritic network.  Aggregation uses uniform weights (α = 1/N).
    Comparing IPPO vs DANTE isolates the benefit of attention.

  FedAvg (baseline)
    At init, each vehicle records the set of vehicles currently in V2X
    range (static graph, never updated).  Every FL round it aggregates
    with equal weights from that fixed neighbor set.  No RL agent.
```

```python
Link modes
------------------------------------
  Sidelink Direct 5G-NR V2X sidelink (PC5 interface). Available when
            distance ≤ V2X_RANGE. Low latency, low cost.

  Internet Indirect 5G uplink → cloud → downlink path. Available when
            V2X_RANGE < distance ≤ INTERNET_RANGE and the neighbor's
            quality score exceeds INTERNET_QUALITY_THRESHOLD. Higher
            transmission cost but keeps high-quality distant neighbors.
        

Two types of V2X links are modelled:

  Sidelink (link_type = 0)
    5G-NR PC5 direct D2D communication. Available when d ≤ V2X_RANGE.
    TX cost via Shannon capacity: E(d) = P_sl·S / (B_sl·log₂(1+SNR_0·(R/d)²))
    Closer vehicles → higher SNR → higher throughput → less energy.

  Internet (link_type = 1)
    5G uplink → cloud → downlink path (Uu interface).  Available when
    V2X_RANGE < d ≤ INTERNET_RANGE  AND  the neighbor's quality score
    (cosine_sim × accuracy) ≥ INTERNET_QUALITY_THRESHOLD.
    TX cost is fixed (vehicle-to-vehicle distance is irrelevant — the link
    goes UE → nearest BS → cloud → BS → UE).
    Capped at MAX_INTERNET_NEIGHBORS per vehicle per step.
```





## 1. Introduction

Connected and autonomous vehicles generate rich, high-dimensional sensory data that can be exploited to learn predictive models for collision risk, trajectory forecasting, and cooperative perception. Centralizing this data is often infeasible due to bandwidth constraints, latency requirements, privacy, and reliability concerns. Federated learning (FL) enables collaborative training without sharing raw data, but classical FL assumes a central server, which introduces a single point of failure and a communication bottleneck that is especially problematic in dense urban vehicular networks.

Decentralized federated learning (DFL) and decentralized personalized FL (DPFL) remove the server and perform training in a peer-to-peer manner. Each client maintains a personalized model and exchanges updates with selected neighbors over a communication graph. Recent work shows that carefully constructed collaboration graphs can significantly improve personalization under communication budgets, e.g., DPFL’s bi-level graph construction and directed DPFL variants [^1].

However, existing approaches:

- Typically use greedy or hand-designed collaboration graphs, rather than learned adaptive policies;
- Treat the graph as static or quasi-static, ignoring strong coupling between mobility, link cost, and learning benefit in vehicular networks;
- Handle Byzantine robustness separately via fixed robust aggregators, not as part of the collaboration-formation logic [^2].

In this proposal, we target `fully decentralized, robust, personalized learning in vehicular networks`:

- No central server or coordinator: both model training and collaboration-structure optimization are decentralized.
- Each vehicle trains a `personalized model` on its own data.
- Vehicles dynamically form a `directed collaboration graph` by choosing which neighbors’ updates to use, over which `V2X mode (D2D / 5G path)`, and how much `resource (energy, bandwidth, latency budget) to allocate`.
- Some neighbors can be `Byzantine`, possibly *all* current neighbors for a vehicle.

We formulate this as a `dynamic network formation game` over the collaboration graph, rather than as a hedonic partition game, which is structurally mismatched to the V2X setting. The game’s payoffs capture personalized learning gain, resource costs, and trust-weighted robustness. We then solve the induced decentralized control problem via a `fully decentralized multi-agent actor–critic` algorithm (F2A2-style), rather than Independent PPO or centralized critics, to respect the “no central entity” constraint and mitigate non-stationarity [^3].

**Contributions:**

1. **Vehicular DPFL as a dynamic network formation game.** We model V2X collaboration as a directed network formation game whose players are vehicles, whose actions are link-formation decisions, and whose payoffs combine personalized loss improvement, communication/computation cost, and trust-weighted robustness.
2. **Fully decentralized MARL for link formation.** We instantiate a **flexible fully decentralized approximate actor–critic (F2A2)** for vehicular DPFL, learning per-vehicle link-formation and resource-allocation policies without any central learner, consistent with V2X constraints [^3]. 
3. **Integrated robustness to Byzantine neighbors.** We incorporate robust update aggregation and local trust estimation directly into the game’s utilities and MARL state, so vehicles can behave safely even when all currently reachable neighbors are Byzantine.
4. **Realistic V2X evaluation.** We propose experiments on nuScenes, INTERACTION, Argoverse, HighD, and NGSIM for cooperative risk prediction and trajectory forecasting, under realistic mobility and V2X link models.

---

## 2. System Model and Problem Definition

### 2.1 Vehicular Network and Communication

Time is slotted: $t=0,1,2,\dots$. At time $t$, the set of active vehicles is $\mathcal{V}_t$.

Vehicles communicate over a `hybrid V2X network`:

- **Sidelink (D2D)** (e.g., 3GPP PC5): short-range, low-latency, subject to pathloss and interference;
- **5G network path** (uplink to BS / edge, then downlink): longer-range, higher latency and cost.

We define a `physical connectivity graph` $\mathcal{G}_t^{\mathrm{phys}}=(\mathcal{V}_t,\mathcal{E}_t^{\mathrm{phys}})$, where $(i,j)\in\mathcal{E}_t^{\mathrm{phys}}$ if there exists at least one feasible V2X channel (D2D or 5G path) between vehicles $i$ and $j$ at time $t$, according to their positions, channel states, and scheduling.

For each vehicle $i\in\mathcal{V}_t$, its `feasible neighbor set` is
$$
\mathcal{F}_i(t) = \{ j\in\mathcal{V}_t\setminus\{i\} : (i,j)\in\mathcal{E}_t^{\mathrm{phys}}\}.
$$

A `directed collaboration edge` $(j\to i)$ at time $t$ means that vehicle $i$ uses the update sent by vehicle $j$ in its model update at round $t$. The resulting `collaboration graph` is
$$
\mathcal{G}_t^{\mathrm{coll}} = (\mathcal{V}_t,\mathcal{E}_t^{\mathrm{coll}}),\quad
\mathcal{E}_t^{\mathrm{coll}}\subseteq \mathcal{V}_t\times\mathcal{V}_t.
$$

For vehicle $i$, its `collaboration neighborhood` at time $t$ is
$$
C_i(t) = \{ j\in \mathcal{F}_i(t): (j\to i)\in\mathcal{E}_t^{\mathrm{coll}}\}.
$$

> **Note.** Unlike classical hedonic games, we do not require coalitions to be **disjoint** or form a **partition**. Instead, we have overlapping, asymmetric neighborhoods, which more accurately reflect V2X information exchange. This aligns with strategic network formation frameworks, where links, not partitions, are the primary objects.

### 2.2 Data and Personalized Learning

Each vehicle $i$ has a private local dataset $\mathcal{D}_i$ drawn from a distribution $P_i$; distributions are heterogeneous (non-IID) across vehicles.

Each vehicle maintains a personalized model $w_i\in\mathbb{R}^d$ and aims to minimize its **own** expected loss:
$$
F_i(w) = \mathbb{E}_{(x,y)\sim P_i}[\ell(w;x,y)],
$$
for a task-specific loss $\ell(\cdot)$ (e.g., cross-entropy for collision risk, NLL for trajectory prediction).

Because $P_i\neq P_j$ in general, enforcing global consensus $w_i\equiv w_j$ can be highly sub-optimal; personalized objectives are the right target in DPFL [^1].

### 2.3 Resource Model and Adversary Model

Each vehicle has per-round **resource costs**:

- $E_i(t)$: energy consumed (communication + computation).
- $B_i(t)$: bandwidth usage.
- $L_i(t)$: end-to-end round latency.

We track a per-vehicle resource state $r_i^{\mathrm{res}}(t)$ encoding remaining energy, bandwidth budgets, and latency slack.

`Byzantine adversaries`. At each time $t$, a subset $\mathcal{B}_t \subseteq \mathcal{V}_t$ may behave arbitrarily (Byzantine): sending arbitrary model updates, possibly adversarial. We allow worst case for a vehicle: it may have $C_i(t)\subseteq \mathcal{B}_t$ (all its current collaborators Byzantine).

Each honest vehicle $i$ maintains a `local trust score `$q_{ij}(t)\in[0,1]$ for each neighbor $j\in\mathcal{F}_i(t)$, updated based on:

- Historical update alignment (cosine similarity, bounded distance);
- Robust multi-neighbor statistics (e.g., clustering and trimming, as in robust DFL).

These trust scores are `entirely local` and do not rely on any global reputation or server.

---

## 3. Notation

| Symbol / Term                   | Meaning                                                      |
| ------------------------------- | ------------------------------------------------------------ |
| $\mathcal{V}_t$                 | Set of active vehicles at round $t$                          |
| $\mathcal{G}_t^{\mathrm{phys}}$ | Physical connectivity graph at time $t$ (V2X feasibility)    |
| $\mathcal{G}_t^{\mathrm{coll}}$ | Directed collaboration graph at time $t$                     |
| $\mathcal{F}_i(t)$              | Feasible neighbors of vehicle $i$ at time $t$                |
| $C_i(t)$                        | Collaboration neighborhood of vehicle $i$ at time $t$        |
| $\mathcal{D}_i$, $P_i$          | Local dataset and data distribution for vehicle $i$          |
| $w_i(t)$                        | Personalized model of vehicle $i$ at round $t$               |
| $g_i(t)$                        | Local gradient estimate at $i$ at round $t$                  |
| $\Delta w_i(t)$                 | Model update applied by $i$ at round $t$                     |
| $q_{ij}(t)$                     | Trust score assigned by $i$ to neighbor $j$ at time $t$      |
| $E_i(t),B_i(t),L_i(t)$          | Energy, bandwidth, latency costs for $i$ at round $t$        |
| $r_i^{\mathrm{res}}(t)$         | Resource state (remaining budgets) of vehicle $i$            |
| $\pi_i$                         | Policy of vehicle $i$ for link formation and resource allocation |
| $a_i(t)$                        | Action of vehicle $i$ at time $t$ (link choices, modes, resources) |
| $r_i(t)$                        | Learning reward at time $t$ for vehicle $i$ (loss reduction minus costs) |
| $\gamma$                        | Discount factor for long-term returns                        |

---



## 4. Dynamic Network Formation Game

We now formalize link formation as a `dynamic strategic network formation game` over the collaboration graph.

### 4.1 Game Definition

**Definition 1 (Dynamic network formation game).**  
A dynamic network formation game for vehicular DPFL consists of:

- A (time-varying) player set $\mathcal{V}_t$ (the vehicles);

- A state space $\mathcal{S}$ (vehicles’ positions, resource states, models, trust scores, etc.);

- For each $i\in\mathcal{V}_t$, a set of feasible actions $\mathcal{A}_i(s)$ at state $s$;

- A collaboration graph construction rule that maps actions to edges:
  $$
  (j\to i)\in\mathcal{E}_t^{\mathrm{coll}} \quad \Longleftrightarrow \quad a_{ij}(t)\neq \text{off};
  $$

- Utility functions $u_i(s,a)$ representing one-step payoffs (learning gain minus costs);

- State transitions $s(t+1)\sim P(\cdot\mid s(t),a(t))$, induced by mobility and learning.

At each round $t$, in state $s(t)$, each vehicle $i$ chooses an action $a_i(t)\in\mathcal{A}_i(s(t))$ (link decisions, modes, resources), resulting in a collaboration graph $\mathcal{G}_t^{\mathrm{coll}}$, updated models $w_i(t+1)$, and a next state $s(t+1)$.

> **Note.** This follows the general class of strategic network formation models, where agents decide which links to form or sever and utilities depend on the resulting network. Pairwise stability and related concepts are defined at the level of graphs, not partitions, which matches our directed collaboration graph.

### 4.2 Per-Round Payoff

The one-step utility of vehicle $i$ at time $t$ is

$$
u_i\big(s(t),a(t)\big)
\;\triangleq\;
\underbrace{\Delta F_i(t)}_{\text{loss reduction}}
\;-\;\alpha_i E_i(t) - \beta_i B_i(t) - \delta_i L_i(t),
$$

where $\Delta F_i(t) = F_i(w_i(t)) - F_i(w_i(t+1))$ is estimated on a small local validation set.

> **Note.** Using **loss reduction** instead of loss level aligns the per-step reward with the marginal benefit of collaboration at that round. This is standard in FL client-selection and scheduling formulations that trade off accuracy improvement against resource usage. ([arxiv.org](https://arxiv.org/abs/2406.06520))

For intuition and analysis, we also consider a **trust-weighted gradient alignment proxy** for loss reduction:
$$
G_i\big(C_i(t), s(t)\big)
= \sum_{j\in C_i(t)} q_{ij}(t)\,
  \sigma\!\left(
  \frac{g_i(t)^\top g_j(t)}{\|g_i(t)\|\,\|g_j(t)\| + \epsilon}
  \right),
$$
where $\sigma$ is a monotone squashing function (e.g., identity or ReLU). This provides a differentiable surrogate for the expected benefit of including neighbor $j$’s update in $i$’s model step and is rooted in the standard first-order transfer approximation for personalized FL. ([arxiv.org](https://arxiv.org/html/2406.06520?utm_source=chatgpt.com))  

In theoretical analysis, we use $G_i$ to study structure; in implementation, the MARL reward uses $\Delta F_i$.

### 4.3 Long-Term Objective

Each vehicle $i$ aims to maximize its **discounted cumulative utility**:

$$
J_i(\pi) = 
\mathbb{E}\Big[\sum_{t=0}^\infty \gamma^t\, u_i\big(s(t),a(t)\big)\Big],
$$

where actions are generated by a decentralized policy profile $\pi=(\pi_i)_{i\in\mathcal{V}}$, and $a_i(t)\sim\pi_i(\cdot\mid o_i(t))$ given local observation $o_i(t)$.

This is a **stochastic game** over the evolving collaboration graph and learning process.

### 4.4 Static Benchmark: Pairwise Stable Networks

For a **frozen** state $s$ and static utilities $u_i(\cdot)$, we can define a static network formation game and pairwise stability.

**Definition 2 (Pairwise stable network, informal).**
A directed network $\mathcal{G}^{\mathrm{coll}}$ is *pairwise stable* if:

- No vehicle $i$ strictly benefits from unilaterally deleting any of its outgoing or incoming links (given the formation rule we adopt); and
- No pair of vehicles $i,j$ both strictly benefit from adding a missing link between them.

> **Note.** Pairwise stability is the canonical equilibrium concept in strategic network formation with unilateral/bilateral link formation. We use it as a *static benchmark* for the collaboration graph: MARL policies should tend to produce networks that are approximately pairwise stable in slowly varying regimes, even though we do not solve for static equilibria explicitly. ([cse.cuhk.edu.hk](https://www.cse.cuhk.edu.hk/~cslui/CMSC5734/jackson-wolinsky-1996.pdf?utm_source=chatgpt.com))

---

## 5. Robust Decentralized Personalized Learning Layer

Given a collaboration graph $\mathcal{G}_t^{\mathrm{coll}}$ and local trust scores, we now specify the personalized learning dynamics.

### 5.1 Local Gradient and Update

At round $t$, vehicle $i$ samples a mini-batch $\mathcal{B}_i(t)\subset\mathcal{D}_i$ and computes

$$
g_i(t) = \frac{1}{|\mathcal{B}_i(t)|}\sum_{(x,y)\in\mathcal{B}_i(t)} \nabla_w \ell(w_i(t);x,y),
\quad
\Delta w_i^{\mathrm{loc}}(t) = -\eta_i(t)\,g_i(t).
$$

Vehicle $i$ receives updates $\{\widetilde{\Delta w}_j(t): j\in C_i(t)\}$ from selected neighbors.

### 5.2 Robust Aggregation and Weights

To handle Byzantine neighbors, each vehicle applies a **local robust aggregation** scheme:

1. Compute distances $\|\widetilde{\Delta w}_j(t) - \Delta w_i^{\mathrm{loc}}(t)\|$ and simple per-coordinate statistics across neighbors.
2. Apply a robust rule, e.g., coordinatewise trimmed mean or clipping (ClippedGossip / BRIDGE-style), to derive **robustness scores** $r_{ij}(t)\in[0,1]$. ([jmlr.org](https://jmlr.org/papers/volume24/20-700/20-700.pdf?utm_source=chatgpt.com))  

Combine trust and robustness:

$$
\omega_{ij}(t) =
\frac{q_{ij}(t)\, r_{ij}(t)}{\sum_{k\in C_i(t)} q_{ik}(t)\, r_{ik}(t) + \epsilon}.
$$

Final update:

$$
\Delta w_i(t) = 
\lambda_{\mathrm{self}}\,\Delta w_i^{\mathrm{loc}}(t)
+ (1-\lambda_{\mathrm{self}})\sum_{j\in C_i(t)} \omega_{ij}(t)\,\widetilde{\Delta w}_j(t),
\quad
w_i(t+1) = w_i(t) + \Delta w_i(t),
$$

with $\lambda_{\mathrm{self}}\in[0,1]$ controlling personalization vs collaboration, analogous to pFedMe/Ditto-style formulations. ([arxiv.org](https://arxiv.org/abs/2406.06520?utm_source=chatgpt.com))  

If all neighbors in $C_i(t)$ are Byzantine, their updates will repeatedly fail robustness and validation checks, driving $r_{ij}(t)$ and/or $q_{ij}(t)$ toward zero; thus $\omega_{ij}(t)\approx 0$, and the update defaults to (almost) pure local SGD.

### 5.3 Trust Update

We propose an exponentially weighted trust update:

$$
q_{ij}(t+1) = (1-\rho)\,q_{ij}(t) + \rho\,\phi_{ij}(t),\quad \rho\in(0,1],
$$

where $\phi_{ij}(t)\in[0,1]$ reflects:

- Whether $j$’s update passed the robust filter;
- The sign and magnitude of validation-loss change when including $j$’s update.

> **Note.** This local trust-update mechanism is conceptually similar to robust DFL schemes (e.g., dual-domain clustering + trust bootstrapping) but is tailored to our fully decentralized and dynamic V2X setting. ([arxiv.org](https://arxiv.org/html/2406.06520?utm_source=chatgpt.com)).

---

## 6. MARL Formulation and Algorithm

We cast the dynamic network formation game as a **cooperative Dec-POMDP** and solve it with a **fully decentralized actor–critic** (F2A2-style). ([jmlr.org](https://jmlr.org/papers/volume24/20-700/20-700.pdf?utm_source=chatgpt.com))  

### 6.1 Dec-POMDP View

**Definition 3 (Decentralized partially observable MDP, sketch).**  
A Dec-POMDP is a tuple $(\mathcal{S},\{\mathcal{A}_i\},P,R,\{\mathcal{O}_i\},O,\gamma)$ where:

- $\mathcal{S}$ is the global state space;
- $\mathcal{A}_i$ is the action space of agent $i$;
- $P$ is the state transition kernel;
- $R$ defines per-agent rewards $r_i(s,a)$;
- $\mathcal{O}_i$ is the observation space of agent $i$; $O$ is the observation kernel;
- $\gamma\in(0,1)$ is the discount factor.

Each agent chooses actions based only on its local observation history.

We map our setting to a Dec-POMDP by:

- State $s(t)$: vehicle positions, channel states, resource states, models, trust scores;

- Observation $o_i(t)$: local model and gradient, resource state, neighbor features/fading, local trust scores:
  $$
  o_i(t) = \Big(w_i(t),g_i(t),r_i^{\mathrm{res}}(t),\{f_{ij}(t),q_{ij}(t)\}_{j\in\mathcal{F}_i(t)}\Big);
  $$

- Actions $a_i(t)$: per-neighbor link/mode decisions and resource allocations (defined below);

- Reward $r_i(t)=u_i(s(t),a(t))$ as in Section 4.2;

- Transition: induced by mobility, resource depletion, trust evolution, and model updates.

> **Note.** Dec-POMDP is the canonical formalism for cooperative MARL under partial observability, and it fits our setting where each vehicle has only local observations and no global controller. ([jmlr.org](https://jmlr.org/papers/volume24/20-700/20-700.pdf?utm_source=chatgpt.com))  

### 6.2 Action Space and Policy Parameterization

To avoid combinatorial explosion, we factor the action across neighbors.

For each vehicle $i$, define a **candidate neighbor subset** $\mathcal{F}_i^{(k)}(t)\subseteq\mathcal{F}_i(t)$ of size at most $k$ (e.g., nearest neighbors or those with best link quality).

For each $j\in\mathcal{F}_i^{(k)}(t)$, the agent selects a discrete **link-mode action**:
$$
a_{ij}(t)\in\{\text{off},\,\text{D2D},\,\text{5G}\},
$$
and continuous **resource parameters** $\rho_i(t)$ (power, number of local SGD steps, etc.).

The collaboration set is:
$$
C_i(t) = \{ j\in\mathcal{F}_i^{(k)}(t) : a_{ij}(t)\neq\text{off} \}.
$$

The policy $\pi_i(a_i\mid o_i)$ is parameterized by a neural network with:

- A **set encoder** (e.g., DeepSets or attention) over neighbor features $(f_{ij}(t),q_{ij}(t))$ to handle variable-size neighborhoods;
- Outputs for per-neighbor logits (over $\{\text{off},\text{D2D},\text{5G}\}$) and continuous resource parameters.

### 6.3 Reward for MARL

We use the per-round reward
$$
r_i(t) = \Delta F_i(t) - \alpha_i E_i(t) - \beta_i B_i(t) - \delta_i L_i(t),
$$
where $\Delta F_i(t)$ is estimated using a small local validation set.

Resource budgets are modeled as **soft constraints**: remaining budgets are part of $r_i^{\mathrm{res}}(t)$, and consuming too much resource leads to future penalties (e.g., vehicles dropping out when energy is exhausted), which the policy learns to avoid.

### 6.4 Chosen MARL Algorithm: Fully Decentralized Actor–Critic (F2A2-type)

We adopt a **fully decentralized approximate actor–critic** loosely following F2A2: ([jmlr.org](https://jmlr.org/papers/volume24/20-700/20-700.pdf?utm_source=chatgpt.com))  

- Each vehicle $i$ maintains:
  - Actor parameters $\theta_i$ for local policy $\pi_{\theta_i}(a_i\mid o_i)$,
  - Critic parameters $\phi_i$ for local value function $V_{\phi_i}(o_i)$.
- Training alternates between:
  1. **Local rollout and updates:** Each agent collects trajectories, computes advantages via GAE, and updates its actor and critic via a primal–dual actor–critic objective.
  2. **Neighbor-only diffusion:** Agents exchange **parameter updates** (or compressed gradients) with direct neighbors and apply a diffusion/consensus step on shared parameters, using robust gossip to mitigate Byzantine influence.

> **Note.** F2A2 is specifically designed for **fully decentralized** MARL, unlike MAPPO (which uses a centralized critic) and many CTDE approaches. It achieves improved stability by jointly optimizing policy and value in a primal–dual fashion and has been demonstrated to scale to large cooperative environments. This makes it a natural match for our “no central entity” V2X DPFL setting. ([jmlr.org](https://jmlr.org/papers/volume24/20-700/20-700.pdf?utm_source=chatgpt.com))  

We contrast this with MAPPO and IPPO in Table 3 (see Section 8).

### 6.5 Algorithm (Pseudo-Code)

Below is LaTeX-style pseudo-code for the overall learning procedure.

```latex
\begin{algorithm}[t]
\caption{Robust Decentralized Network Formation for Vehicular DPFL}
\label{alg:vehicular-dpfl}
\begin{algorithmic}[1]
\Require Local datasets $\{\mathcal{D}_i\}$, initial models $\{w_i(0)\}$,
         initial trust $q_{ij}(0)=1$, actor--critic params $\{\theta_i,\phi_i\}$,
         cost weights $\alpha_i,\beta_i,\delta_i$, discount $\gamma$
\For{episode $e = 1,2,\dots,E$}
  \State Initialize environment state $s(0)$ (vehicle positions, channels, resources)
  \For{round $t = 0,\dots,T-1$}
    \For{each vehicle $i$ in parallel}
      \State Sample mini-batch from $\mathcal{D}_i$, compute $g_i(t)$ and $\Delta w_i^{\mathrm{loc}}(t)$
      \State Observe neighbor features $\{f_{ij}(t),q_{ij}(t)\}_{j\in\mathcal{F}_i(t)}$
      \State Form observation $o_i(t)$
      \State Sample action $a_i(t) \sim \pi_{\theta_i}(\cdot \mid o_i(t))$
      \State Infer collaboration set $C_i(t)$ and resource allocation from $a_i(t)$
      \State Exchange model updates with neighbors in $C_i(t)$ using chosen modes
      \State Perform robust aggregation to obtain $\Delta w_i(t)$ and update $w_i(t+1)$
      \State Evaluate validation loss to compute $\Delta F_i(t)$
      \State Measure $E_i(t), B_i(t), L_i(t)$ and compute reward 
             $r_i(t) = \Delta F_i(t) - \alpha_i E_i(t) - \beta_i B_i(t) - \delta_i L_i(t)$
      \State Update trust scores $q_{ij}(t+1)$ based on robustness and validation checks
      \State Store transition $(o_i(t), a_i(t), r_i(t), o_i(t+1))$ in local buffer
    \EndFor
    \State Environment transitions to $s(t+1)$ via mobility + resource dynamics
  \EndFor
  \For{each vehicle $i$ in parallel}
    \State Compute advantages $\hat{A}_i$ and returns from local buffer (e.g., GAE)
    \State Update critic params $\phi_i$ via gradient descent on value loss
    \State Update actor params $\theta_i$ via primal--dual PPO-style objective
    \State Exchange parameters with neighbors and apply robust diffusion/consensus
  \EndFor
\EndFor
\State \textbf{return} Learned policies $\{\pi_{\theta_i}\}$ and personalized models $\{w_i\}$
\end{algorithmic}
\end{algorithm}
```

---

## 7. Theoretical Analysis (Concise)

Here we outline target theoretical results and justify the concepts used. Full proofs would appear in the final paper.

### 7.1 Assumptions

We assume:

1. **Loss regularity.** Each $F_i$ is $L$-smooth and lower-bounded; stochastic gradients have bounded variance.

2. **Robust aggregation.** The local robust aggregator yields an **effective gradient estimate** $\widehat{g}_i(t)$ such that, even with arbitrary adversarial updates among neighbors, its bias is bounded:
   $$
   \|\mathbb{E}[\widehat{g}_i(t)] - \nabla F_i(w_i(t))\| \le \zeta,
   $$
   for some $\zeta\ge 0$ (as in robust FL analyses). ([arxiv.org](https://arxiv.org/html/2406.06520?utm_source=chatgpt.com))  

3. **Bounded rewards.** Per-step rewards $r_i(t)$ are uniformly bounded.

4. **Communication graph connectivity over time.** In non-adversarial regimes, the collaboration graph is jointly strongly connected over bounded time windows; under full adversarial neighborhoods, vehicles default to local SGD.

### 7.2 Convergence of Personalized Learning Layer (Fixed Policies)

**Proposition 1 (Robust local convergence, sketch).**  
Fix link-formation policies $\{\pi_i\}$. Under Assumptions 1–2 and a standard diminishing step-size schedule $\eta_i(t)$, the sequence of updates
$$
w_i(t+1) = w_i(t) - \eta_i(t)\,\widehat{g}_i(t)
$$
for each honest vehicle $i$ converges to a neighborhood of a stationary point of $F_i$, with radius $O(\zeta)$.

*Sketch.* This follows standard arguments for SGD with bounded gradient bias. Robust aggregation ensures that Byzantine neighbors cannot drive the gradient arbitrarily far from $\nabla F_i$, so convergence guarantees degrade gracefully with the bias bound $\zeta$. ([arxiv.org](https://arxiv.org/html/2406.06520?utm_source=chatgpt.com))  

> **Note.** This ensures that even if MARL policies occasionally choose “bad” neighbors, as long as trust and robust aggregation eventually reject consistently harmful updates, honest vehicles’ models still converge.

### 7.3 Policy Optimization Guarantees (F2A2)

**Theorem 1 (Asymptotic policy convergence, informal).**  
Under Assumptions 3–4 and standard technical conditions on function approximation (bounded parameterization, compatible gradient estimators), the fully decentralized actor–critic updates (F2A2-style) converge to a stationary point of the joint objective
$$
J(\pi) = \sum_{i} J_i(\pi),
$$
up to approximation error.

*Rationale.* F2A2 is designed as a primal–dual hybrid gradient method on the joint policy and value parameters, and the original work establishes convergence properties for cooperative MARL with diffusion-style communication. Our setting fits its assumptions (bounded rewards, diffusion over a time-varying communication graph), so their convergence results carry over with minor adaptations. ([jmlr.org](https://jmlr.org/papers/volume24/20-700/20-700.pdf?utm_source=chatgpt.com))  

> **Note.** This addresses one of the major concerns with purely independent learning (IPPO): lack of convergence guarantees in cooperative settings with coupled rewards and partial observability. ([arxiv.org](https://arxiv.org/abs/2011.09533?utm_source=chatgpt.com))  

### 7.4 Approximate Network Stability

In a static environment (fixed mobility and channel states), for fixed policies, the induced long-run **average collaboration graph** can be analyzed.

**Proposition 2 (Approximate pairwise stability, informal).**  
Suppose the environment is effectively static and F2A2 converges to stationary policies $\{\pi_i^\star\}$. Then, under mild regularity assumptions (smooth utilities, sufficiently expressive policies), the resulting induced collaboration graph distribution is concentrated on graphs that are *approximately pairwise stable* with respect to the one-step utilities $u_i$.

*Rationale.* At stationarity, no unilateral local policy change by an agent (corresponding to consistent link-addition or link-deletion deviations) yields sustained improvement in its long-term objective $J_i$. For static environments and Markovian policies, this implies approximate local optimality with respect to link changes, which is closely related to pairwise stability in network formation. ([cse.cuhk.edu.hk](https://www.cse.cuhk.edu.hk/~cslui/CMSC5734/jackson-wolinsky-1996.pdf?utm_source=chatgpt.com))  

We do not claim exact equilibria; instead we use this to justify talking about **empirically stable collaboration structures** induced by MARL policies.

---

## 8. Comparison of Game-Theoretic and MARL Choices

### 8.1 Game-Theoretic Model Comparison

**Table 2 – Game-theoretic abstractions for V2X collaboration**

| Model                                       | Coalition structure                                      | Strategic choice                                    | Typical stability concept                   | Pros                                                 | Cons in V2X context                                          | Suitability here |
| ------------------------------------------- | -------------------------------------------------------- | --------------------------------------------------- | ------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------ | ---------------- |
| Hedonic partition games                     | Partition of agents into disjoint coalitions             | Join/leave a coalition                              | Nash / Individual stability                 | Rich theory; clear partition structure               | Requires each agent in exactly one coalition; poorly matches overlapping, asymmetric collaborations | Poor             |
| Overlapping coalition formation (OCF) games | Agents can join multiple coalitions with resource shares | Join/leave multiple coalitions with resource splits | Core / stability for overlapping structures | Flexible modeling of multi-task participation        | Complex coalition structure; heavy overhead for fast-moving vehicles ([link.springer.com](https://link.springer.com/content/pdf/10.1007/978-3-540-24790-6_15.pdf?utm_source=chatgpt.com)) | Moderate         |
| **Strategic network formation (ours)**      | Directed/undirected graphs; overlapping neighborhoods    | Form/sever links                                    | Pairwise stability, Nash network            | Directly models who talks to whom; matches V2X links | Stability analysis more subtle with dynamics and directionality | **High**         |

> **Note.** We deliberately avoid hedonic partition models because vehicular collaborations are overlapping and asymmetric; network formation games, as formalized by Jackson & Wolinsky and follow-ups, are a better abstraction. ([cse.cuhk.edu.hk](https://www.cse.cuhk.edu.hk/~cslui/CMSC5734/jackson-wolinsky-1996.pdf?utm_source=chatgpt.com))  

### 8.2 MARL Algorithm Comparison

**Table 3 – MARL algorithm comparison**

| Algorithm                             | Training / Execution                                 | Critic type                                            | Communication pattern                  | Pros                                                         | Cons for this setting                                        | Use in our work  |
| ------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------ | -------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------- |
| **F2A2 (ours)**                       | Fully decentralized / decentralized                  | Local critics with diffusion / consensus (primal–dual) | Parameter exchange over neighbor graph | No central learner; handles large cooperative tasks; convergence analysis available ([jmlr.org](https://jmlr.org/papers/volume24/20-700/20-700.pdf?utm_source=chatgpt.com)) | Requires robust gossip; more implementation complexity       | **Primary**      |
| MAPPO                                 | Centralized training, decentralized execution (CTDE) | Centralized critic observing joint state / actions     | Critic parameters on central learner   | Strong empirical performance on SMAC/Hanabi; stable training ([arxiv.org](https://arxiv.org/abs/2103.01955?utm_source=chatgpt.com)) | Violates “no central entity”; centralized critic unrealistic in V2X DPFL | Baseline only    |
| IPPO                                  | Decentralized training & execution                   | Local critic per agent                                 | No coordination besides environment    | Simple; strong empirical performance in some benchmarks ([arxiv.org](https://arxiv.org/abs/2011.09533?utm_source=chatgpt.com)) | Non-stationarity; no convergence guarantees; no explicit coordination | Baseline only    |
| CTDE value-decomposition (e.g., QMIX) | CTDE                                                 | Central mixing network                                 | Centralized training infrastructure    | Excellent credit assignment in discrete control tasks        | Requires central learner; not naturally fully decentralized  | Baseline (maybe) |

> **Note.** We choose F2A2-style decentralized actor–critic specifically to align with the **absence of any central coordinating entity** in vehicular networks, while still leveraging multi-agent coordination and enjoying convergence guarantees unavailable to pure IPPO. ([jmlr.org](https://jmlr.org/papers/volume24/20-700/20-700.pdf?utm_source=chatgpt.com))  

---

## 9. Experimental Plan

### 9.1 Use Case 1: Cooperative Collision / Near-Miss Risk Prediction

**Task.** Predict the probability of a collision or near-miss event in the next $3\!-\!5$ seconds for each ego vehicle, using its own sensory history and (implicitly) the history encoded via learned collaboration.

**Datasets.**

- **nuScenes** – multimodal urban driving in Boston/Singapore. ([openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Decentralized_Directed_Collaboration_for_Personalized_Federated_Learning_CVPR_2024_paper.pdf?utm_source=chatgpt.com))  
- **INTERACTION** – multi-agent urban and highway driving with complex interactions. ([openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Decentralized_Directed_Collaboration_for_Personalized_Federated_Learning_CVPR_2024_paper.pdf?utm_source=chatgpt.com))  

**Setup.**

- Each vehicle trajectory in the dataset corresponds to a client.
- Build a simulation overlay that:
  - Uses recorded positions/velocities to derive physical V2X connectivity and pathloss;
  - Simulates D2D and 5G latency and bandwidth using simplified models from V2X literature.
- Local model: light-weight risk classifier (e.g., CNN/MLP on kinematic features and local context).
- Compare collaboration strategies:
  - **Local-only** training;
  - **Random fixed graph** DFL;
  - **DPFL** graph construction under fixed budgets; ([arxiv.org](https://arxiv.org/abs/2406.06520?utm_source=chatgpt.com))  
  - **DFedPGP** directed DPFL; ([arxiv.org](https://arxiv.org/abs/2405.17876?utm_source=chatgpt.com))  
  - **Robust DFL baselines** (e.g., BRIDGE/ClippedGossip) without MARL; ([jmlr.org](https://jmlr.org/papers/volume24/20-700/20-700.pdf?utm_source=chatgpt.com))  
  - **Our F2A2-based dynamic network formation**.

**Metrics.**

- Personalized AUC / F1 by vehicle.
- Energy per vehicle to reach a target AUC.
- Average and tail latency per training round.
- Collaboration graph statistics (average degree, churn).
- Robustness under:
  - random Byzantine noise,
  - sign-flipping attacks,
  - backdoor / model-replacement attacks. ([arxiv.org](https://arxiv.org/html/2406.06520?utm_source=chatgpt.com))  

### 9.2 Use Case 2: Cooperative Trajectory Prediction

**Task.** Predict multi-step future trajectories of ego vehicles given past motion and surroundings.

**Datasets.**

- **Argoverse** 1/2 – urban trajectories with map context.
- **HighD** and **NGSIM** – highway trajectories. ([ijcai.org](https://www.ijcai.org/proceedings/2025/161?utm_source=chatgpt.com))  

**Setup.**

- Assign vehicles to clients based on trajectory segments or driver IDs, yielding natural non-IID splits.
- Local model: sequence model (GRU/LSTM/Transformer) with personalized heads.
- Same collaboration baselines as above.

**Metrics.**

- ADE/FDE (Average / Final Displacement Error) per vehicle.
- Same resource and robustness metrics as in Use Case 1.

### 9.3 Implementation Tools

- **Deep learning / MARL:** PyTorch + custom F2A2 implementation; we may leverage MARL code collections (QMIX, IPPO, MAPPO, etc.) for baselines. ([github.com](https://github.com/LantaoYu/MARL-Papers?utm_source=chatgpt.com))  
- **Mobility and channel simulation:** extend existing V2X simulators or build on open-source traffic simulators (e.g., SUMO) with custom V2X models.
- **DFL/DPFL baselines:** official DPFL and DFedPGP code where available. ([proceedings.mlr.press](https://proceedings.mlr.press/v258/kharrat25a.html?utm_source=chatgpt.com))  

---

## 10. Summary

This proposal:

- Uses the **correct game-theoretic abstraction** (strategic network formation) for vehicular collaboration, instead of misaligned hedonic partition games.
- Selects a **fully decentralized actor–critic (F2A2-style)** as the MARL backbone, rather than IPPO or MAPPO, to honor the “no central entity” constraint while mitigating non-stationarity.
- Integrates **robustness and personalization** directly into utilities and MARL state.
- Targets **realistic V2X evaluation** with standard autonomous-driving datasets.

With a solid theoretical section (as outlined) and strong empirical comparisons against DPFL, DFedPGP, and robust DFL baselines, this should be competitive for NeurIPS-level venues in the decentralized FL / MARL / game-theoretic learning space.

[^D]: 



## 11. References

[^1]: https://arxiv.org/abs/2406.06520	"Decentralized Personalized Federated Learning"

[^2]: https://arxiv.org/abs/2405.17876 "Decentralized Directed Collaboration for Personalized Federated Learning"
[^3]: https://arxiv.org/abs/2004.11145 "F2A2: Flexible Fully-decentralized Approximate Actor-critic for Cooperative Multi-agent Reinforcement Learning"















































------



## 1. Why $\Delta F_i(t)$ is expensive and not necessary

Your original per‑round utility:

$$
u_i(s(t),a(t))
\triangleq
\Delta F_i(t)
-\alpha_i E_i(t)-\beta_i B_i(t)-\delta_i L_i(t),
$$

with
$$
\Delta F_i(t) = F_i(w_i(t)) - F_i\big(w_i(t+1)\big)
$$
assumes you *measure* loss before and after the update on a local validation set. That means **extra forward passes per round per vehicle**, purely for reward shaping.

For a vehicular / IoT device:

- You **must** already do a forward+backward pass on a mini‑batch to get the local gradient $g_i(t)$ for training.
- You **do not** want additional passes on a separate validation set just to estimate $\Delta F_i(t)$ — that’s exactly the overhead you’re worried about.

So we want a surrogate for “how good this update is” that:

1. Uses only *already available* quantities (like $g_i(t)$ and the received updates),
2. Involves only cheap vector ops (inner products / norms), not extra passes over data,
3. Still carries some principled link to loss reduction and robustness.

------

## 2. Cosine similarity as a proxy: how good is it?

Your idea: measure **gradient alignment** using cosine similarity, e.g.

$$
s_{ij}(t)

\frac{g_i(t)^\top g_j(t)}
{|g_i(t)|,|g_j(t)|+\varepsilon}
\in[-1,1].
$$

### 2.1. Why this is not crazy

There’s a *lot* of FL work using gradient *directions* (cosine similarity) as a proxy for “helpful vs harmful” updates:

- **FLTrust**: server compares each client update direction with a “root” gradient from a small trusted dataset, and uses cosine similarity to derive trust weights. ([arXiv](https://arxiv.org/abs/2012.13995?utm_source=chatgpt.com))
- **CosPer, DP‑FedSim, PNCS, etc.**: use cosine similarity between local gradients and a reference (global or prototype) to decide client weights or personalization. ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0020025524006741?utm_source=chatgpt.com))
- **FoolsGold and follow‑ups**: use gradient similarity (including cosine) *across clients* to detect sybil collusion. ([arXiv](https://arxiv.org/abs/1808.04866?utm_source=chatgpt.com))
- **DFL‑Dual** (Byzantine‑robust *decentralized* FL) uses model‑domain distances and clustering — strongly related to gradient geometry — to bootstrap trust. ([CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/papers/Sun_Byzantine-robust_Decentralized_Federated_Learning_via_Dual-domain_Clustering_and_Trust_Bootstrapping_CVPR_2024_paper.pdf?utm_source=chatgpt.com))
- Recent work on **gradient alignment in FL** (FedLAG, FedGA, implicit alignment) all argue that angle between gradients is a crucial signal for personalization, convergence and conflict detection. ([OpenReview](https://openreview.net/pdf?id=qH6pzxPZ0d&utm_source=chatgpt.com))

So: your instinct to use gradient/cosine alignment as a resource‑light proxy is very aligned with the literature.

### 2.2. But cosine similarity alone is *not* enough

The other side of the story:

- There are **backdoor and adaptive attacks** that explicitly *use gradient alignment* to remain stealthy while injecting poison. E.g., the ICCV 2025 “adaptive layer‑wise gradient alignment” backdoor attack shows you can align malicious gradients with the previous global update and still successfully backdoor the model. ([CVF Open Access](https://openaccess.thecvf.com/content/ICCV2025/papers/Yang_Stealthy_Backdoor_Attack_in_Federated_Learning_via_Adaptive_Layer-wise_Gradient_ICCV_2025_paper.pdf?utm_source=chatgpt.com))
- Several new defenses (e.g. **AlignIns**) combine cosine similarity with **sign alignment** and temporal consistency—cosine alone is not robust enough. ([CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_Detecting_Backdoor_Attacks_in_Federated_Learning_via_Direction_Alignment_Inspection_CVPR_2025_paper.pdf?utm_source=chatgpt.com))
- Classic Byzantine‑robust FL (trimmed mean, median, Krum, MultiKrum, Bulyan, etc.) rely on **pairwise distances + robust statistics** rather than cosine alone, exactly because directional criteria can be gamed by attackers. ([MDPI](https://www.mdpi.com/2079-9292/12/13/2926?utm_source=chatgpt.com))

So, cosine similarity is:

- **Good** as a **component**: cheap, principled first‑order proxy for “do we step in roughly the same descent direction?”.
- **Insufficient** alone for strong adversaries (backdoor, adaptive, sybil), especially without any validation data.

------

## 3. A better dataset‑free surrogate for $\Delta F_i(t)$

You don’t actually need the *exact* loss drop. First‑order Taylor expansion gives you a clean surrogate:

For small updates:

$$
F_i(w_i(t) + \Delta w_i(t))
\approx
F_i(w_i(t)) + \nabla F_i(w_i(t))^\top \Delta w_i(t).
$$
So the **predicted loss reduction** is:
$$
\Delta F_i(t) \approx - \nabla F_i(w_i(t))^\top \Delta w_i(t).
$$
If we identify $\nabla F_i(w_i(t))$ with your stochastic gradient $g_i(t)$, then define:
$$
\widehat{\Delta F}_i(t) \; \triangleq \; - g_i(t)^\top \Delta w_i(t),
$$
which is:

- **Zero extra dataset usage**: you already computed $g_i(t)$ once on your local mini‑batch to do SGD.
- Just a few **inner products and norms**, which are cheap relative to backprop.
- Exactly the same quantity that cosine similarity uses, but keeping **magnitude**:

$$
-g_i(t)^\top \Delta w_i(t) =
  |g_i(t)|,|\Delta w_i(t)|\cos\theta_i,
$$

where $\theta_i$ is the angle between $g_i(t)$ and $\Delta w_i(t)$.

So **replace** the loss‑based term in your per‑round utility by this surrogate:
$$
u_i(s(t),a(t)) \; \triangleq \; \widehat{\Delta F}_i(t) - \alpha_i E_i(t) - \beta_i B_i(t) - \delta_i L_i(t), \quad \widehat{\Delta F}_i(t) = - g_i(t)^\top \Delta w_i(t).
$$
This keeps a *principled* link to loss reduction (first‑order approximation) without any validation set.

> Intuitively: if the aggregated update $\Delta w_i(t)$ is roughly pointing *against* the gradient and has reasonable magnitude, then $-g_i^\top\Delta w_i > 0$ and we expect a loss decrease. If it points the wrong way (obtuse angle) or is extremely large, it will be penalized.

This is strictly stronger than pure cosine similarity because it also handles **degenerate updates** (very small norm but aligned, or very big norm pointing “good” but causing instability).

------

## 4. Scoring individual neighbors without data

Now, how to evaluate each neighbor $j$’s update *without* extra data, and in a way that’s both efficient and more robust than bare cosine?

You already have:

- Local gradient $g_i(t)$,
- Local candidate update $\Delta w_i^{\mathrm{loc}}(t)$,
- Neighbor updates $\widetilde{\Delta w}_j(t)$ for $j\in C_i(t)$.

Everything below uses only these vectors (plus history), no extra dataset.

### 4.1. Geometry‑only, per‑round features

For each neighbor $j$ at time $t$:

1. **Directional alignment** with local gradient:

$$
a_{ij}(t)

   \frac{g_i(t)^\top \widetilde{\Delta w}_j(t)}
   {|g_i(t)|,|\widetilde{\Delta w}_j(t)|+\varepsilon}.
$$

2. **Magnitude sanity**:

   Norm ratio:
   $$
   m_{ij}(t)=
   \min!\left(
   \frac{|\widetilde{\Delta w}_j(t)|}{|\Delta w_i^{\mathrm{loc}}(t)|+\varepsilon},
   \frac{|\Delta w_i^{\mathrm{loc}}(t)|}{|\widetilde{\Delta w}_j(t)|+\varepsilon}
   \right)\in(0,1],
   $$
   so extreme norms are penalized.

3. **Distance to a robust local centroid**:

Compute a local robust centroid, e.g. coordinatewise median:
$$
\widetilde{\Delta w}_{\mathrm{med}}(t) = \operatorname{median}\big(\{\Delta w_i^{\mathrm{loc}}(t)\}\cup\{\widetilde{\Delta w}\ k(t)\}_{k\in C_i(t)}\big),
$$
 then
$$
d_{ij}(t) = \frac{|\widetilde{\Delta w}*j(t) - \widetilde{\Delta w}*{\mathrm{med}}(t)|}
   {|\widetilde{\Delta w}_{\mathrm{med}}(t)|+\varepsilon}.
$$
This is exactly the kind of “model‑domain distance + clustering” signal used in DFL‑Dual and geometric robust aggregators. ([CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/papers/Sun_Byzantine-robust_Decentralized_Federated_Learning_via_Dual-domain_Clustering_and_Trust_Bootstrapping_CVPR_2024_paper.pdf?utm_source=chatgpt.com))

4. **Sign alignment ratio** (AlignIns‑style):

   - Compute the *principal sign* vector (e.g., sign of median or majority sign across neighbors).
   - For client $j$, define:

     $$
     \text{SAR}_{ij}(t) =
     \frac{1}{d}\sum_{k=1}^d
     \mathbf{1}{\operatorname{sign}(\widetilde{\Delta w}_j^{(k)}(t))
     = \operatorname{sign}(\widetilde{\Delta w}_{\mathrm{med}}^{(k)}(t))}.
     $$
     AlignIns showed that combining cosine with sign‑alignment captures more subtle deviations than cosine alone. ([CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_Detecting_Backdoor_Attacks_in_Federated_Learning_via_Direction_Alignment_Inspection_CVPR_2025_paper.pdf?utm_source=chatgpt.com))

All of these are **$O(|C_i(t)|\cdot d)$ vector ops**, not extra data passes.

### 4.2. Combine them into a trust / quality score

Your existing trust $q_{ij}(t)$ (history) is updated by a per‑round quality signal $\phi_{ij}(t)$:

$$
q_{ij}(t+1) = (1-\rho)q_{ij}(t) + \rho\phi_{ij}(t).
$$

Replace the loss‑based part in $\phi_{ij}(t)$ by a function of the geometry features, for example:

$$
\phi_{ij}(t) = \sigma!\Big(
\lambda_1 a_{ij}(t)

- \lambda_2 m_{ij}(t)
- \lambda_3 \text{SAR}_{ij}(t)

- \lambda_4 d_{ij}(t)
  \Big),
$$

where $\sigma$ is a squashing function such as a shifted/scaled sigmoid, and $\lambda_k\ge 0$ are hyper‑parameters.

Interpretation:

- High $a_{ij}$ (directional alignment with your gradient) is good,
- High $m_{ij}$ (reasonable norm ratio) is good,
- High sign alignment $\text{SAR}_{ij}$ is good,
- Large deviation from the robust centroid ($d_{ij}$) is bad.

This **mirrors the structure** of several modern defenses:

- FLTrust: alignment + norm‑based scaling. ([arXiv](https://arxiv.org/abs/2012.13995?utm_source=chatgpt.com))
- FoolsGold and sybil defenses: similarity patterns + history, penalizing outliers or colluding clients. ([arXiv](https://arxiv.org/abs/1808.04866?utm_source=chatgpt.com))
- AlignIns: cosine + sign alignment + temporal consistency. ([CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_Detecting_Backdoor_Attacks_in_Federated_Learning_via_Direction_Alignment_Inspection_CVPR_2025_paper.pdf?utm_source=chatgpt.com))
- DFL‑Dual / RFCL: clustering in model space and trust bootstrapping. ([CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/papers/Sun_Byzantine-robust_Decentralized_Federated_Learning_via_Dual-domain_Clustering_and_Trust_Bootstrapping_CVPR_2024_paper.pdf?utm_source=chatgpt.com))

Crucially, **none of these require you to evaluate the loss on extra data** at each round.

------

## 5. Plugging this back into your utility and MARL reward

### 5.1. Reward for MARL (replace $\Delta F_i$)

Instead of

$$
r_i(t) = \Delta F_i(t) - \alpha_i E_i(t) - \beta_i B_i(t) - \delta_i L_i(t),
$$

use the first‑order surrogate:

$$
r_i(t) =
\underbrace{- g_i(t)^\top \Delta w_i(t)}_{\widehat{\Delta F}_i(t)}

- \alpha_i E_i(t)
- \beta_i B_i(t)
- \delta_i L_i(t).
$$

This is **fully dataset‑free beyond the training gradient** and still tells MARL:

- “Good” actions (good link choices) are those that produce an update $\Delta w_i(t)$ well aligned with the descent direction,
- “Bad” actions either misalign with the gradient or blow up the norm.

### 5.2. Utility proxy for graph formation

If you want to keep an explicit “learning gain” term in the game‑theoretic utility (instead of using $-g_i^\top \Delta w_i$ directly), define:

$$
G_i\big(C_i(t)\big)

\sum_{j\in C_i(t)} q_{ij}(t),\underbrace{
\frac{g_i(t)^\top \widetilde{\Delta w}_j(t)}
{|g_i(t)|,|\widetilde{\Delta w}*j(t)|+\varepsilon}
}*{\text{cosine alignment}},
$$
and set

$$
u_i(s(t),a(t))

G_i\big(C_i(t)\big)

- \alpha_i E_i(t) - \beta_i B_i(t) - \delta_i L_i(t).
$$
This is exactly the gradient‑alignment version of the utility we sketched in the proposal; you can choose either $G_i$ or $\widehat{\Delta F}_i$ for MARL reward — they’re closely related.

------

## 6. How “resilient” can this be *without* any dataset?

Important reality check:

- **No dataset‑free geometric score can perfectly detect all Byzantine attacks.**
  Adaptive backdoors can align their gradient with what you think is “good” (global or local) and still inject targeted misbehavior. ([CVF Open Access](https://openaccess.thecvf.com/content/ICCV2025/papers/Yang_Stealthy_Backdoor_Attack_in_Federated_Learning_via_Adaptive_Layer-wise_Gradient_ICCV_2025_paper.pdf?utm_source=chatgpt.com))
- Geometry‑only robust aggregators (median, Krum, MultiKrum, trimmed mean) all require assumptions like “strict majority of honest clients” to guarantee robustness. ([MDPI](https://www.mdpi.com/2079-9292/12/13/2926?utm_source=chatgpt.com))

Given your constraints (no extra data, possible all‑Byzantine neighborhoods):

- The **best you can do** is:
  - Default to **pure local SGD** when all neighbors look geometrically suspicious (trust decays to 0);
  - Use **geometry+history+clustering** (as above) to make it hard for naive or non‑adaptive adversaries to hurt you;
  - Explicitly acknowledge in the paper that under very strong adaptive adversaries, and without any trusted data, perfect detection is impossible — your goal is *pragmatic robustness*, not worst‑case Byzantine optimality.

This is still entirely acceptable for a NeurIPS‑level paper as long as you’re clear about the threat model and the limitations.

------

## 7. Concrete recommendation for your proposal

To adapt your proposal in a clean, coherent way:

1. **Drop explicit loss evaluation** from the utility and the MARL reward.
   - Replace $\Delta F_i(t)$ by $\widehat{\Delta F}_i(t) = -g_i(t)^\top\Delta w_i(t)$.
2. **Use cosine similarity + norm + clustering + sign alignment** for trust updates and neighbor scoring:
   - Implemen combination of $a_{ij}(t)$, $m_{ij}(t)$, $d_{ij}(t)$, and $\text{SAR}_{ij}(t)$.
   - Keep the EWMA trust update:
     $$
     q_{ij}(t+1) = (1-\rho) q_{ij}(t) + \rho,\phi_{ij}(t).
     $$
3. **Keep everything dataset‑free** beyond the one gradient computation per round you need anyway for local training.
4. **Explicitly cite the connection** to FLTrust / FoolsGold / DFL‑Dual / AlignIns to show that:
   - Your choice is *not arbitrary*,
   - It matches the direction the robust FL literature has taken for resource‑constrained clients. ([arXiv](https://arxiv.org/abs/2012.13995?utm_source=chatgpt.com))

If you want, I can rewrite the relevant parts of your proposal (the per‑round payoff definition, G_i, and trust update section) in their final, gradient‑only form, consistent with this design.
