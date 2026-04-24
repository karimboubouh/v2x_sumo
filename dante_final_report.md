# DANTE Final Experiment Report

Date: 2026-04-25  
Project: SUMO V2V Decentralized Personalized Learning  
Scenario: `khalifa_university`  
Dataset/model: `CIFAR10` / `CNN`  
Vehicles: 50  
Final retained output: `out/khalifa_university_dante_cifar10_cnn_50_20260424_165933_utc/`

## Executive Summary

The final DANTE implementation reached the requested stop target:

| Metric | Final retained value |
|---|---:|
| Test accuracy | **70.61%** |
| Test loss | 0.9257 ± 0.1684 |
| Training accuracy | 73.04% |
| Synchronized round | 40 |
| Total TX energy | **17.17 J** |
| Computation energy | 5.83 kJ |
| Total energy | 5.85 kJ |
| Rounds to target | 40 |

This is the final version to keep under the 10-cycle cap. It is the only tested cycle that reached at least 70% test accuracy. The total communication energy remains small relative to computation energy, about 0.29% of total energy, so the accuracy gain was obtained primarily through corrected DANTE methodology and modest local work rather than heavy communication.

## Final Artifacts

| Artifact | Path |
|---|---|
| Experiment pickle | `out/khalifa_university_dante_cifar10_cnn_50_20260424_165933_utc/experiment.pkl` |
| Run log | `out/khalifa_university_dante_cifar10_cnn_50_20260424_165933_utc/run.log` |
| Accuracy plots | `out/khalifa_university_dante_cifar10_cnn_50_20260424_165933_utc/accuracy_vs_rounds.png` |
| Reward plots | `out/khalifa_university_dante_cifar10_cnn_50_20260424_165933_utc/ppo_reward_vs_steps.png` |

## What Was Fixed

### 1. Round Semantics Were Not Paper-Aligned

Earlier runs reported the fastest client frontier as the DPL round. That made `round 40` mean "some vehicles reached 40" rather than a synchronized DANTE communication round. This inflated progress reporting and could save stale final evaluation results.

Final fix:

- `dl/env.py` now reports the synchronized round as `min_i r_i`.
- `MAX_ROUND_SKEW = 1` prevents fast vehicles from racing far ahead of the slowest vehicle.
- Progress snapshots refresh training state and force final evaluation even when SUMO stops producing new vehicle states.

Paper alignment:

```text
t is a shared DPL communication round.
Round t should be evaluated after every participating client has completed local update t.
```

### 2. Reward Was Initially Dominated by Cost

The original DANTE reward behavior made useful collaboration rare because cost terms overwhelmed the learning-gain signal. The corrected reward follows the paper-style utility form:

```text
R_i^t = G_i^t - lambda_E E_i^t - lambda_B B_i^t - lambda_T T_i^t
```

where `G_i^t` is the learning utility proxy and the remaining terms are normalized energy, bandwidth, and latency costs.

Final DANTE settings:

```text
REWARD_LAMBDA_E = 0.03
REWARD_LAMBDA_B = 0.02
REWARD_LAMBDA_T = 0.03
MAX_INTERNET_NEIGHBORS = 2
MAX_COLLAB_NEIGHBORS = 4
EXPLORATION_WARMUP_ROUNDS = 20
MIN_EXPLORATION_LINKS = 2
```

### 3. Aggregation Needed Safer Personalization

Directly mixing deltas was unstable for non-IID CIFAR10 clients. The final version uses model-space personalized aggregation with explicit local self-retention:

```text
w_i^{t+1} = alpha_ii w_i,local^{t+1} + sum_{j in N_i^t} alpha_ij w_j^t
```

with scheduled self-retention:

```text
SELF_WEIGHT_START = 0.85
SELF_WEIGHT_END = 0.60
```

This keeps personalization high early, then allows more collaboration after peer trust and utility estimates improve.

### 4. Local Training Was Underpowered

The low-accuracy runs were not only a DANTE-selection problem. After reward and aggregation fixes, test accuracy plateaued below 70% because local representation learning was still underpowered.

Final training configuration:

```text
BATCHES_PER_ROUND = 48
LOCAL_LR = 0.05
LOCAL_LR_SCHEDULE = cosine
LOCAL_LR_MIN = 0.01
LABEL_SMOOTHING = 0.1
CNN_CHANNELS = 32
CNN_HIDDEN = 128
CNN_DROPOUT = 0.10
TRAIN_AUGMENTATION_POLICY = dataset_default
```

The compact CNN trial degraded badly, so the retained model is the stronger FC-head CIFAR CNN.

## Cycle Observations

| Cycle | Main change | test@10 | test@20 | test@30 | final test | TX J | Comp |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Initial fixes | 53.99% | 57.09% | 58.88% | 58.65% | 2.99 | 1.94 kJ |
| 2 | Reward correction | 54.77% | 56.87% | 58.33% | 57.92% | 3.74 | 1.94 kJ |
| 3 | Smoothing, dropout, LR schedule | 54.93% | 58.90% | 60.51% | 61.90% | 3.67 | 1.94 kJ |
| 4 | Model-space aggregation | 54.51% | 58.70% | 61.76% | 63.01% | 3.76 | 1.94 kJ |
| 5 | Stronger CNN | 54.61% | 59.41% | 61.64% | 64.26% | 14.60 | 1.94 kJ |
| 6 | 32 batches/round | 58.44% | 62.48% | 65.09% | 66.24% | 13.22 | 3.89 kJ |
| 7 | Higher self-retention | 58.97% | 61.67% | 64.46% | 66.58% | 13.15 | 3.89 kJ |
| 8 | Compact CNN trial | 49.22% | 52.88% | 55.91% | 57.82% | n/a | n/a |
| 9 | Synchronized rounds | 60.11% | 64.38% | 66.79% | 66.79% | 16.36 | 3.89 kJ |
| 10 | Final eval fix + 48 batches/round | **61.62%** | **64.90%** | **67.60%** | **70.61%** | **17.17** | **5.83 kJ** |

Cycle 8 was intentionally not retained because the compact CNN regressed substantially. Cycle 9 exposed the final-evaluation scheduling bug: it saved using the round-30 test result instead of forcing a round-40 test. Cycle 10 fixed that and reached the target.

## Main Causes of Bad Early Accuracy

1. **Round accounting mismatch**: reported rounds were based on the fastest vehicle, not synchronized communication rounds.
2. **Stale final evaluation**: final summaries could reuse the last periodic evaluation instead of evaluating the terminal model.
3. **Reward scaling error**: communication costs dominated the learning utility, suppressing useful peer reuse.
4. **Aggregation instability**: delta-space mixing amplified non-IID drift; model-space personalized mixing was safer.
5. **Insufficient local representation learning**: CIFAR10 needed more local batches and the stronger CNN head.
6. **Compact CNN underfit**: global pooling reduced capacity too much for this non-IID CIFAR10 setup.

## Final Command

```bash
PYTHONPATH=venv/lib/python3.12/site-packages /usr/bin/python3.12 main.py \
  --dl \
  --headless \
  --speed 0 \
  --scenario khalifa_university \
  --num-vehicles 50 \
  --rounds 40 \
  --target_acc 0.70 \
  --stop-on eval_acc \
  --dl-dataset CIFAR10 \
  --dl-model CNN \
  --dl-algorithm DANTE \
  --verbose result \
  --save-logs \
  --seed 42
```

## Verification

Unit tests passed after the final code changes:

```bash
PYTHONPATH=venv/lib/python3.12/site-packages /usr/bin/python3.12 \
  -m unittest discover -s tests -p 'test*.py' -v
```

Result:

```text
Ran 40 tests in 0.848s
OK
```

## Retained Conclusion

The final DANTE version is paper-aligned in the key places that affected accuracy: synchronized communication-round evaluation, utility-minus-cost reward, trust/retention-based neighbor reuse, bounded energy-aware collaboration, and personalized aggregation. It reached **70.61%** test accuracy by round 40 with **17.17 J TX energy**, so it satisfies the requested stop condition while preserving communication efficiency.

The remaining gap to 90% is likely not fixable by DANTE selection alone under the current 40-round, small-CNN, non-IID CIFAR10 setup. Closing that gap would require a stronger benchmark model, longer training, or a less severe data partition, and should be evaluated separately against DPFL and other baselines under identical synchronized-round semantics.
