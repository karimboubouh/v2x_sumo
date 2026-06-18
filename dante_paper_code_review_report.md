# Review Report: DANTE Paper and Implementation

Repository: `/home/ku5001201/Workspace/sumo`  
Paper entry point: `/home/ku5001201/Workspace/sumo/paper/main.tex`  
Implementation focus: `/home/ku5001201/Workspace/sumo/algorithms/dante/algorithm.py` and associated experiment artifacts  
Review date: 2026-04-27

## Executive Assessment

The manuscript proposes a timely and potentially publishable system: a fully decentralized personalized learning protocol for hybrid 5G C-V2X vehicular networks, combining attention-based neighbor selection, trust-aware aggregation, communication-energy awareness, and Byzantine filtering. The direction is strong, and the codebase contains a substantial implementation with meaningful simulator integration, baselines, tests, and experiment artifacts.

However, the current paper overstates several claims relative to the implementation and the mathematical analysis. The most important issues are:

1. The paper describes validation-loss-based rewards and binary trust updates, but the implementation primarily uses training-loss-derived reward signals and continuous geometry/quality-based trust.
2. The convergence proof is conditional and incomplete. It does not rigorously handle aggregate-then-train dynamics, multiple local SGD steps, non-IID neighbor drift, or the logarithmic term induced by the diminishing step size.
3. Several experiment statements do not match the saved experiment summaries, especially CIFAR-10 accuracy values, DPFL behavior under Byzantine attack, and the claimed benefit of the hybrid interface.
4. The Byzantine guarantee is stated too strongly. The proof assumes a bounded aggregation bias rather than deriving Byzantine resilience from the proposed trust/robustness mechanism.
5. The code implements additional selection, filtering, probing, fallback, and aggregation logic that is not described in the paper. These mechanisms are not minor details; they materially determine performance and robustness.

The paper can be strengthened substantially by narrowing the formal claims, aligning the methodology with the code, correcting experimental numbers, and reframing the theory as a conditional convergence result under explicit robust-aggregation assumptions.

## Reference Correctness

### Bibliographic Integrity

The BibTeX file is internally consistent with the manuscript: all cited keys are present. The following references are present in the bibliography but unused:

- `blanchard2017krum`
- `chen2022graph`
- `dinh2020moreau`
- `fallah2020personalized`
- `lebars2023topology`
- `li2021ditto`
- `lim2020survey`

These should either be removed or integrated deliberately. In particular, `blanchard2017krum`, `dinh2020moreau`, `fallah2020personalized`, and `li2021ditto` are relevant to Byzantine aggregation and personalized federated learning; citing them would improve the related-work positioning.

### ](https://proceedings.mlr.press/v139/karimireddy21a.html)

### Reference Quality Assessment

| Citation key or group                                                                                                                                                       | Status             | Assessment                                                                | Recommended action                                                                                                                                                   |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------:| ------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `vanhaesebrouck2017decentralized`, `boubouh2020robust`, `zantedeschi2020fully`, `kharrat2025dpfl`, `ye2023pFedGraph`, `liu2024dfedpgp`, `zhang2024personalized`             | Certain            | Personalized/decentralized FL references are appropriate.                 | Improve synthesis: distinguish graph learning, collaboration weights, pairwise personalization, and directed collaboration.                                          |
| `fang2024balance`, `yin2018byzantine`, `karimireddy2021learning`                                                                                                            | Certain            | Byzantine references exist, but their guarantees are not interchangeable. | Do not use `karimireddy2021learning` as the primary support for coordinate-wise trimmed-mean tolerance. Cite robust aggregation work directly and state assumptions. |
| `basmadjian2022advantages`, `bala2019mobile`, `jackson2005survey`, `bottou2018optimization`, `borkar2009stochastic`, `dewitt2020deep`, `shani2020adaptive`, `liu2019neural` | Certain to likely  | Supporting references are broadly suitable.                               | Keep only where directly used; otherwise reduce bibliography noise.                                                                                                  |
| Unused keys listed above                                                                                                                                                    | Present but unused | Bibliography contains relevant but uncited references.                    | Remove or use intentionally in the related-work and Byzantine-resilience sections.                                                                                   |

## Mathematical and Logical Correctness

he $x$



 

### Problem Formulation

The objective in the problem formulation is
\[
$\min_{\{w_i\}_{i=1}^N} \sum_{i=1}^N \rho_i F_i(w_i),$
\]
which is separable across clients. As written, this optimization problem does not require collaboration, graph formation, communication constraints, or robust aggregation. Collaboration appears only in the algorithm, not in the stated objective.

This weakens the formal motivation. A stronger formulation should either:

1. Add a graph-regularized personalization term, e.g.,
   \[
   $\min_{\{w_i\}} \sum_i \rho_i F_i(w_i) +
   \lambda \sum_{t}\sum_{(j,i)\in\mathcal{E}^{(t)}} a_{ij}^{(t)} d(w_i,w_j),$
   \]
   subject to resource and latency constraints; or
2. State explicitly that the objective is personalized empirical risk minimization and that DANTE is a communication-constrained stochastic algorithm for improving the optimization path, rather than solving a coupled optimization problem.

### Communication and Constraint Model

The paper models both PC5 sidelink and Uu-assisted communication with rate, latency, and energy expressions. This is conceptually appropriate, but the implementation uses simplified helper functions with fixed SNR assumptions for Internet/Uu-style links and range-based sidelink SNR. It does not implement full NR sidelink scheduling, cellular resource allocation, queueing, interference, or gNB uplink/downlink modeling.

The paper should therefore describe the communication model as an abstraction calibrated to hybrid interfaces, not as a full 3GPP-compliant physical/MAC-layer model.

There is also a notation issue: graph edges are defined as $(j \to i)$, but some energy and latency terms use \(E_{ij}\), \(B_{ij}\), and \(T_{ij}\). The manuscript should standardize sender-receiver indexing. If \(j\) transmits to \(i\), then all link quantities should consistently use either \((j,i)\) or \((i,j)\), with one convention stated once.

### Neighbor-Selection Objective

The payoff \(U_i^{(t)}\) combines learning gain, energy, bandwidth, and latency. This is sensible, but the paper describes action selection as attention top-\(K\), whereas the code performs PPO sampling followed by feasibility pruning, budget rejection, alignment screening, trust/robustness gating, forced exploration probes, and proposal scoring. This is a materially different policy.

The manuscript should formalize the actual decision process:

\[
$\mathcal{C}_i^{(t)}
= \operatorname{PruneBudget}
\left(
\operatorname{Rank}_{K}
\left(
\pi_{\theta_i}(\cdot \mid s_i^{(t)}, \{s_{ij}^{(t)}\})
\right)
\right),$
\]
with explicit definitions of feasibility filters, proposal score, forced probes, and post-selection action masks.

### Trust and Robustness

The paper states that trust is updated using a binary indicator derived from validation-loss improvement:
\[
$q_{ij}^{(t+1)} = (1-\lambda_q)q_{ij}^{(t)} + \lambda_q \phi_{ij}^{(t)}, \quad
\phi_{ij}^{(t)} \in \{0,1\}.$
\]

The implementation does not follow this model. It updates trust using a continuous quality value derived from peer feedback, loss trend, model-alignment geometry, robust-pass indicators, and selection weights. `VALIDATION_LOSS_SLACK` is configured but not used in the DANTE trust update. Therefore, the paper must either revise the trust definition to match the code or the code must be changed to implement the paper's binary validation-based trust.

The robustness score \(r_{ij}^{(t)}\) is also more complex in the code than in the manuscript. The code combines trimmed-mean-style scoring with local-update cosine alignment, norm-ratio checks, sign alignment, Byzantine-mode-aware rejection, and per-peer selection quality. The paper's formula \(\alpha_{ij}\propto \beta_{ij}q_{ij}r_{ij}\) omits these decisive gates.

### Byzantine Threat Model

The paper states an omniscient and colluding Byzantine model with arbitrary, inconsistent updates. The implementation and experiments mainly test sign-flip attacks, and the attack simulator has access to global honest-update statistics to synthesize attacks. The DANTE protocol itself does not prove resilience to omniscient colluding adversaries under arbitrary inconsistent per-recipient messages.

A defensible statement would be:

"We evaluate robustness against strong model-poisoning attacks, including high-magnitude sign-flip attacks with colluding Byzantine vehicles. The convergence theorem is conditional on a bounded post-filter aggregation bias; it does not constitute a universal resilience guarantee against all omniscient Byzantine strategies."

### Learning-Gain Indicator

The paper claims that positive gradient agreement \(G_i^{(t)}\) indicates whether collaboration is helpful. This is too strong. A positive cosine or inner-product alignment can be a first-order sufficient signal under small step sizes and comparable update norms, but it is neither necessary nor sufficient for downstream validation improvement in nonconvex, non-IID personalized learning.

The statement should be narrowed:

"\(G_i^{(t)}>0\) is used as a first-order compatibility proxy: under small aggregation mass and smooth local loss, positively aligned peer updates are less likely to increase the local objective to first order."

## Proof Review

### Proposition 1: Conditional Convergence

The theorem claims an \(O(T^{-1/2})\)-type average gradient bound with an additional Byzantine bias term. The high-level form is plausible, but the proof is not correct as written.

Key problems:

1. The proof treats the update as
   \[
   w_i^{(t+1)} = w_i^{(t)} - \eta_t \tilde g_i^{(t)}.
   \]
   The algorithm actually performs aggregation first and then multiple local SGD steps:
   \[
   \bar w_i^{(t)} = (1-\mu_i^{(t)})w_i^{(t)} + \mu_i^{(t)}\sum_j \alpha_{ij}^{(t)}w_j^{(t)},
   \]
   followed by \(I_i^{(t)}\) stochastic local updates. The proof ignores the aggregation displacement \(\bar w_i^{(t)}-w_i^{(t)}\) and the accumulated local-SGD drift.

2. The step-size schedule \(\eta_t=\eta_0/\sqrt{t+1}\) gives
   \[
   \sum_{t=0}^{T-1}\eta_t^2 = O(\log T).
   \]
   After division by \(\sum_t \eta_t = O(\sqrt T)\), the variance term is \(O(\log T/\sqrt T)\), not \(O(1/\sqrt T)\). The appendix says the logarithm is absorbed into a constant, which is not mathematically valid for an asymptotic rate.

3. Assumption A3 states that the robust aggregate has bounded bias relative to \(\nabla F_i(w_i^{(t)})\). This is effectively the core Byzantine-resilience condition. Since the paper does not derive A3 from the trust and robustness mechanism, the theorem is conditional rather than a proof of Byzantine resilience.

4. The remark that \(\zeta_i=0\) when all selected neighbors are honest is false in non-IID personalized learning. Honest neighbors can still have gradients biased relative to \(\nabla F_i\).

5. The all-Byzantine-neighbor fallback claim is heuristic. Trust and robustness weights may eventually decay, but immediate detection is not guaranteed, especially under adaptive near-center attacks that pass coordinate-wise filters.

Recommended theorem revision:

- State a conditional convergence theorem for the actual aggregate-then-local-update recursion.
- Include a consensus-drift or collaboration-drift term:
  \[
  \Delta_i^{(t)} = \bar w_i^{(t)} - w_i^{(t)}.
  \]
- Use either a constant step size \(\eta=O(1/\sqrt T)\), yielding \(O(1/\sqrt T)\), or keep the \(\log T/\sqrt T\) term for \(\eta_t=\eta_0/\sqrt{t+1}\).
- Replace "Byzantine-resilient convergence" with "convergence under bounded post-filter aggregation bias."

### PPO Stationarity Claim

The paper states that PPO converges to a stationary point of the clipped local objective under slowly varying neighborhoods. This should be softened and fully qualified.

Clipped PPO with nonlinear function approximation, independent multi-agent learning, changing graph topology, and nonstationary rewards does not have a general convergence theorem of the same strength as classical stochastic approximation. The current proof relies on assumptions that are not fully stated: compact parameter iterates, bounded gradients, Lipschitz policy/value networks, controlled policy drift, bounded importance ratios, and sufficiently stationary local data distributions.

Recommended revision:

"Under bounded gradients, compact policy parameter iterates, Lipschitz actor-critic networks, and an asymptotically stationary local environment, the PPO update converges to a stationary point of the empirical clipped surrogate. This result characterizes the policy-optimization subproblem and does not imply global optimality of the coupled decentralized learning process."

## Code-Paper Alignment

### What Matches the Paper

The implementation supports the paper's general system concept:

- Decentralized per-vehicle DANTE agents.
- Hybrid candidate links through sidelink and Internet/Uu-style links.
- Local GAT-style actor-critic policy.
- Neighbor selection with communication-aware constraints.
- Personalized aggregation rather than a single global model.
- Byzantine attack simulation and robust filtering.
- Baselines for FedAvg, D-PSGD, IPPO, DPFL, and pFedGraph.
- Experiment families for dataset, density, place, non-IID severity, dual interface, and Byzantine attack.

### Major Mismatches

| Paper claim                                                | Implementation behavior                                                                                                                                     | Consequence                                                                                                                                 |
| ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Reward uses validation-loss decrease.                      | `DANTE.reward_source` defaults to `"training"`; reward is based mainly on training-loss deltas, surrogate gain, and cost penalties.                         | The learning signal in the paper is not the learning signal used in the experiments.                                                        |
| Trust update uses binary validation-improvement indicator. | Trust uses continuous quality from peer feedback, loss trend, geometry, robust pass, and selection weight.                                                  | The trust equation and interpretation are incorrect.                                                                                        |
| Neighbor selection is top-\(K\) by attention.              | PPO sampling is followed by pruning, budget checks, trust/robustness gates, alignment filters, proposal scoring, and forced probes.                         | The paper omits mechanisms that are central to performance.                                                                                 |
| Aggregation weights are proportional to \(\beta q r\).     | Effective scores multiply base weights, trust, robust pass, geometry quality, and selection quality; self-weight schedules also dominate aggregation.       | The aggregation formula is incomplete.                                                                                                      |
| Robustness is coordinate-wise trimmed-mean scoring.        | Code additionally uses local-update cosine alignment, norm-ratio filters, sign agreement, attack-mode-specific rejection, and fallback behavior.            | The stated robustness mechanism is under-specified.                                                                                         |
| Fully decentralized protocol.                              | The simulator centrally constructs candidate sets, computes attacks, and records global metrics; baselines and attacks use environment-level orchestration. | This is acceptable for simulation, but the protocol description must separate deployable local information from simulator-only information. |
| Hybrid interface improves efficiency.                      | In the dual-interface summary, sidelink-only DANTE achieved slightly better accuracy and lower TX energy than hybrid DANTE.                                 | The hybrid-interface claim should be reframed as adaptive interface selection, not always superiority.                                      |

## Experimental Consistency

### Corrected Saved Results

The saved summaries show the following final test accuracies and communication-energy values.

#### CIFAR-10 Comparison

Experiment: `out/CIFAR10/comparison_dante_dpfl_ippo_pfedgraph_fedavg_d-psgd_20260426_183911/comparison_summary.txt`

| Method    | Final test accuracy | TX energy |
| --------- | -------------------:| ---------:|
| DANTE     | 71.59%              | 49.02 J   |
| DPFL      | 71.56%              | 185.02 J  |
| IPPO      | 70.59%              | 93.57 J   |
| pFedGraph | 72.98%              | 1000.13 J |
| FedAvg    | 48.51%              | 1005.08 J |
| D-PSGD    | 48.55%              | 1005.08 J |

The paper currently states values around 51% for DANTE and 48% for IPPO in the CIFAR-10 paragraph. These numbers do not match the saved artifact above.

#### MNIST Comparison

Experiment: `out/MNIST/comparison_dante_dpfl_ippo_pfedgraph_fedavg_d-psgd_20260427_100151/comparison_summary.txt`

| Method    | Final test accuracy | Rounds to target | TX energy |
| --------- | -------------------:| ----------------:| ---------:|
| DANTE     | 97.43%              | 30               | 8.05 J    |
| DPFL      | 97.68%              | 30               | 45.15 J   |
| IPPO      | 97.54%              | 30               | 12.96 J   |
| pFedGraph | 97.32%              | 40               | 150.74 J  |
| FedAvg    | 93.79%              | 130              | 152.08 J  |
| D-PSGD    | 93.81%              | 130              | 152.08 J  |

DANTE is energy efficient, but it is not the highest-accuracy method in this run. The paper should claim "comparable accuracy at substantially lower communication energy," not accuracy dominance.

#### Byzantine Sign-Flip Comparison

Experiment: `out/byz/comparison_dante-signflip_dpfl-signflip_pfedgraph-signflip_20260427_105239/comparison_summary.txt`

| Method             | Final test accuracy | TX energy |
| ------------------ | -------------------:| ---------:|
| DANTE-SignFlip     | 92.58%              | 0.96 J    |
| DPFL-SignFlip      | 92.07%              | 13.82 J   |
| pFedGraph-SignFlip | 9.31%               | 23.68 J   |

The claim that DPFL collapses under the sign-flip attack is not supported by the saved result. DPFL experiences disruption but recovers to 92.07% final test accuracy. pFedGraph collapses; DANTE remains robust and more energy efficient than DPFL.

#### Dual-Interface Comparison

Experiment: `out/Dual_interface/comparison_dante_dante-sidelink_dante-internet_20260426_173057/comparison_summary.txt`

| Method         | Final test accuracy | TX energy |
| -------------- | -------------------:| ---------:|
| DANTE          | 92.34%              | 0.94 J    |
| DANTE-Sidelink | 92.67%              | 0.33 J    |
| DANTE-Internet | 91.45%              | 0.82 J    |

The hybrid version does not dominate the sidelink-only version in this run. The paper should avoid claiming that hybrid access is uniformly superior.

### Presentation Issues

The evaluation section still contains placeholder language: "In the final version of the figure..." This must be removed before submission.

The caption for the energy figure describes "total energy," while the plotted artifact appears to be communication/TX energy. The caption should distinguish transmit energy, receive energy, training energy, and total energy. The current implementation primarily supports communication-energy accounting; unless local computation energy is included and validated, the paper should not call the plotted quantity total energy.

The evaluation appears to rely on single-seed summaries. A high-impact journal submission should report mean and standard deviation or confidence intervals across at least 3 to 5 seeds.

## Likely Reviewer Questions

1. What exactly is new beyond combining attention, PPO, trust, and robust aggregation?
2. Is DANTE fully decentralized, or does it rely on simulator-level global information?
3. Why is the optimization objective separable if collaboration is central?
4. Does the theory prove Byzantine resilience, or does it assume bounded Byzantine bias?
5. How does the method tolerate omniscient colluding attackers that craft near-median updates?
6. Why does the implementation use training-loss reward when the paper claims validation-loss reward?
7. Why is trust binary in the paper but continuous in the code?
8. Are baselines implemented faithfully, especially DPFL and pFedGraph, or adapted approximations?
9. Are FedAvg and D-PSGD disadvantaged by communication settings or non-personalized evaluation?
10. Why does DANTE not outperform DPFL or pFedGraph in final accuracy on some datasets?
11. Why does hybrid DANTE consume more energy than sidelink-only DANTE in the saved dual-interface run?
12. Are Byzantine vehicles excluded from final evaluation? If so, is the reported accuracy honest-only?
13. How sensitive is DANTE to \(K\), self-weight, trust decay, PPO coefficients, and robust-filter thresholds?
14. Does the wireless model include interference, packet loss, scheduling, queueing, or realistic NR sidelink resource allocation?
15. Does the convergence theorem apply to nonconvex neural networks, multi-agent PPO, and time-varying directed graphs?

## Issue and Suggestion Table

| Severity | Issue or suggestion                                                                | Evidence                                                                                                                                    | Proposed solution                                                                                                                                                       |
| -------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Critical | Reward mechanism in the paper does not match the code.                             | Paper states validation-loss reward; code sets `reward_source = "training"` and computes reward from training/surrogate signals plus costs. | Either implement validation-based reward and trust in code, or revise the paper to describe the actual training/surrogate reward.                                       |
| Critical | Trust update equation is incorrect relative to implementation.                     | Paper uses binary \(\phi_{ij}\); code uses continuous quality and does not use `VALIDATION_LOSS_SLACK` in the trust update.                 | Replace the binary trust equation with the implemented continuous trust-quality update, or modify the code to match the manuscript.                                     |
| Critical | Convergence proof omits aggregate-then-local-update dynamics.                      | The proof uses a one-step SGD recursion, while the algorithm aggregates first and then performs local SGD.                                  | Re-derive the theorem for the actual recursion, including aggregation displacement and local-SGD drift terms.                                                           |
| Critical | Claimed \(O(1/\sqrt{T})\) rate is not justified with \(\eta_t=\eta_0/\sqrt{t+1}\). | The proof ignores \(\sum_t \eta_t^2=O(\log T)\).                                                                                            | Use constant \(\eta=O(1/\sqrt T)\), or state \(O(\log T/\sqrt T)\).                                                                                                     |
| Critical | Byzantine-resilience theorem assumes the hard part.                                | Assumption A3 directly assumes bounded post-filter aggregation bias.                                                                        | Reframe the theorem as conditional convergence under bounded robust-aggregation bias, or prove A3 from explicit adversary and filter assumptions.                       |
| Critical | Threat model is stronger than evaluated attacks.                                   | Paper says omniscient/colluding arbitrary Byzantine; saved experiments mainly show sign-flip behavior.                                      | Narrow the threat model or add experiments for Gaussian, LIE, label-flip, backdoor, and adaptive near-median attacks.                                                   |
| Critical | DPFL Byzantine claim is incorrect.                                                 | Saved summary shows DPFL-SignFlip final test accuracy of 92.07%, not collapse.                                                              | Correct the text: DPFL is disrupted but recovers; pFedGraph collapses.                                                                                                  |
| Critical | CIFAR-10 numbers in the paper do not match saved artifacts.                        | Saved CIFAR summary reports DANTE 71.59%, IPPO 70.59%, pFedGraph 72.98%.                                                                    | Update all CIFAR-10 claims, tables, and figure references from the saved summary or rerun experiments.                                                                  |
| Critical | Hybrid-interface superiority is not supported by the saved dual-interface run.     | Sidelink-only DANTE has 92.67% and 0.33 J, while hybrid DANTE has 92.34% and 0.94 J.                                                        | Claim adaptive interface selection, not uniform dominance; explain when hybrid helps and add scenario-specific analysis.                                                |
| Major    | Objective function is separable and does not encode collaboration.                 | \(\min_{\{w_i\}}\sum_i \rho_i F_i(w_i)\) has no graph or resource coupling.                                                                 | Add graph-regularized personalization or explicitly state that collaboration is an algorithmic accelerator for separable personalized risks.                            |
| Major    | Neighbor selection is under-specified.                                             | Paper says attention top-\(K\); code uses PPO sampling, pruning, budget gates, robust gates, forced probes, and proposal scoring.           | Add an implementation-faithful selection subsection and algorithm lines for filtering, probing, and executed-action masks.                                              |
| Major    | Aggregation formula is incomplete.                                                 | Code uses self-weight schedules, peer-mass caps, robust pass, geometry quality, and selection quality.                                      | Replace \(\alpha\propto\beta q r\) with the full effective-score formula or state simplified vs implemented variants.                                                   |
| Major    | Robustness-score description is incomplete.                                        | Code uses update-space robustness plus local alignment, norm-ratio checks, sign agreement, and Byzantine-mode rejection.                    | Define the full robust gate and distinguish trimmed-mean scoring from geometry-based screening.                                                                         |
| Major    | Honest-neighbor bias is incorrectly treated as zero.                               | Non-IID honest neighbors can have gradients biased relative to \(F_i\).                                                                     | Replace "honest implies \(\zeta=0\)" with a heterogeneity-dependent bias term.                                                                                          |
| Major    | All-Byzantine fallback claim is too strong.                                        | Trust decay and robust scores may not immediately reject adaptive attacks.                                                                  | State fallback as a designed safeguard triggered when effective peer mass vanishes; avoid claiming guaranteed detection without proof.                                  |
| Major    | PPO stationarity claim is overgeneral.                                             | Multi-agent PPO with nonlinear approximation and changing graphs has no simple global convergence guarantee.                                | State a conditional empirical-surrogate stationarity result under compactness, bounded gradients, Lipschitz networks, and asymptotic stationarity.                      |
| Major    | Evaluation likely uses single seed.                                                | Config and summaries show fixed seed behavior.                                                                                              | Report mean +/- std over multiple seeds for each main result.                                                                                                           |
| Major    | Byzantine evaluation appears honest-only.                                          | Environment snapshots exclude Byzantine vehicles from evaluation.                                                                           | Explicitly report honest-only accuracy and, if useful, separate global accuracy including Byzantine nodes.                                                              |
| Major    | Baseline fairness needs stronger disclosure.                                       | DPFL/pFedGraph are adapted to this simulator; FedAvg/D-PSGD communicate heavily over Internet-style links.                                  | Add a baseline-implementation appendix with topology, payload, budget, personalization, and hyperparameter parity.                                                      |
| Major    | Energy metric is ambiguous.                                                        | Figure caption says total energy, but artifact names and code indicate communication/TX energy.                                             | Rename to communication or transmit energy unless computation and receive energy are included.                                                                          |
| Major    | V2X model is more abstract than the paper suggests.                                | Code uses simplified SNR and link helpers rather than full NR sidelink/Uu scheduling.                                                       | Present the channel model as an abstraction and cite 3GPP only for architecture/interface context.                                                                      |
| Major    | Validation split is claimed but not central in code.                               | Paper says validation split drives reward/trust; implementation reward defaults to training.                                                | Revise evaluation text and methodology; if validation is used only for metrics, say so.                                                                                 |
| Major    | The related-work table risks overclaiming "first" or "only."                       | Several works address P2P personalization, graph learning, and robustness partially.                                                        | Frame novelty as the first integrated vehicular hybrid-link, resource-aware, Byzantine-filtered, decentralized personalized learning system under the stated simulator. |
| Major    | No ablation isolates all core modules.                                             | Current artifacts show some interface and non-IID studies but not full trust/robust/PPO/self-weight ablations.                              | Add ablations: no trust, no robust gate, no PPO/GAT, no forced probes, fixed self-weight, no communication costs.                                                       |
| Major    | Attack-aware logic in code may affect robustness claims.                           | Robust rejection changes behavior when `BYZANTINE_FRACTION > 0`.                                                                            | Explain whether Byzantine fraction is known to the protocol. If not deployable, remove attack-mode-dependent gates or justify them as defense-mode configuration.       |
| Major    | Coordinate-wise trimmed-mean tolerance is underqualified.                          | Claim \(f/                                                                                                                                  | \mathcal{C}                                                                                                                                                             |
| Major    | Directed-edge notation is inconsistent.                                            | Paper alternates between \(j\to i\) and \(ij\) quantities.                                                                                  | Standardize all link variables as sender-receiver pairs.                                                                                                                |
| Major    | Latency aggregation assumes parallel transfers.                                    | Constraint uses \(\max_j T_{ij}\).                                                                                                          | State the parallel-transfer assumption or change to scheduled/summed latency.                                                                                           |
| Major    | Communication payload units are unclear.                                           | The paper mixes bandwidth, payload bits, and budget notation.                                                                               | Define \(S\) as model/update payload size in bits and distinguish bandwidth from byte/bit budget.                                                                       |
| Major    | Placeholder text remains in evaluation.                                            | "In the final version of the figure..." appears in the manuscript.                                                                          | Remove placeholder prose and describe the actual plotted figure.                                                                                                        |
| Major    | Claims about practical deployment need local-information audit.                    | Simulator has global orchestration for candidates, attacks, and metrics.                                                                    | Add a paragraph separating decentralized protocol state from simulator-only instrumentation.                                                                            |
| Minor    | Bibliography has unused entries.                                                   | Seven BibTeX keys are unused.                                                                                                               | Remove them or cite them in precise related-work positions.                                                                                                             |
| Minor    | 3GPP citation metadata is incomplete.                                              | Entry lacks version/release details.                                                                                                        | Use formal TS 23.287 metadata with release, version, date, and URL.                                                                                                     |
| Minor    | Some venue metadata and DOIs are incomplete.                                       | Several entries lack DOI or final volume/pages.                                                                                             | Normalize all references to publisher-grade metadata.                                                                                                                   |
| Minor    | LaTeX build has IEEEtran appendix warning.                                         | Build warning: "Ignoring useless \section in Appendix."                                                                                     | Use IEEEtran-compliant appendix formatting, e.g., `\appendices` or correct appendix sectioning.                                                                         |
| Minor    | LaTeX has font and box warnings.                                                   | Small-caps-in-italic and overfull/underfull boxes appear.                                                                                   | Replace problematic text styling and manually break long equations.                                                                                                     |
| Minor    | The term "Internet" is not a precise V2X interface label.                          | Code uses `LINK_INTERNET`; paper discusses Uu.                                                                                              | Use "Uu-assisted infrastructure link" in paper and reserve "Internet" for simulator implementation labels.                                                              |
| Minor    | Figure captions should be more self-contained.                                     | Some captions do not state dataset, metric, Byzantine fraction, or whether accuracy is honest-only.                                         | Rewrite captions to include dataset, attack, metric, and aggregation/evaluation convention.                                                                             |
| Minor    | The introduction can sharpen the known-gap-contribution flow.                      | Current contribution is broad and system-heavy.                                                                                             | End each introduction paragraph with a concrete unresolved limitation, then map each limitation to one DANTE contribution.                                              |
| Minor    | Related work should be more thematic.                                              | Some parts compare methods individually.                                                                                                    | Organize by decentralized optimization, personalized graph collaboration, vehicular FL/V2X, and Byzantine robustness.                                                   |
| Minor    | Mathematical notation for self-weight is absent.                                   | Code uses self-weight schedules and peer-mass caps.                                                                                         | Add \(\omega_i^{(t)}\) or \(\mu_i^{(t)}\) definitions matching implementation.                                                                                          |
| Minor    | Reward weights lack sensitivity discussion.                                        | Code uses fixed cost coefficients.                                                                                                          | Add hyperparameter table and sensitivity plot for reward-cost weights.                                                                                                  |

## Recommended Revision Plan

1. Correct all experiment numbers from saved artifacts or rerun the full suite with fixed seeds and regenerate figures/tables.
2. Decide whether the paper should describe the current implementation or whether the code should be changed to match the simpler paper equations. For journal submission, it is better to describe the implemented system faithfully.
3. Rewrite the trust, reward, selection, and aggregation subsections around the actual code path.
4. Recast the theoretical analysis as conditional convergence under bounded post-filter aggregation bias, with the correct step-size rate.
5. Narrow the Byzantine claims and add attack diversity plus adaptive attacks.
6. Add multi-seed statistics and ablation studies.
7. Strengthen related work with precise thematic gap statements and remove or use unused references.

## Suggested High-Impact Positioning

The strongest defensible positioning is:

"DANTE is a fully decentralized, resource-aware personalized learning framework for vehicular edge networks that jointly learns neighbor selection and robust collaboration under hybrid V2X connectivity. Its contribution is not a new robust aggregation rule in isolation, nor a standalone PPO policy, but an integrated protocol that couples topology adaptation, trust-weighted personalization, and communication-aware defense in a dynamic vehicular graph."

The paper should avoid claiming a universal Byzantine-resilience theorem or uniform accuracy superiority. The strongest supported empirical claim is energy-efficient competitive accuracy with strong robustness against high-magnitude sign-flip attacks, especially compared with pFedGraph, under the implemented SUMO-based simulation.
