# Experiment Scripts

These scripts run paper-ready experiments and copy the final comparison figures
into `paper/assets/`.

They are designed around the current simulator and plotting workflow:

- one saved experiment folder per training run in `out/`
- one comparison folder per multi-run overlay created by `run_plots.py`
- final paper figures copied into `paper/assets/`

## Scripts

- `01_main_comparison.sh`
  Main convergence and energy comparison for `DANTE` against the paper baselines.
- `02_attention_ablation.sh`
  `DANTE` versus `IPPO` to isolate the contribution of graph attention.
- `03_dual_interface_ablation.sh`
  `DANTE` with hybrid links, PC5-only, and Uu-only communication.
- `04_vehicle_density_sweep.sh`
  `DANTE` under increasing numbers of vehicles.
- `05_scenario_transfer.sh`
  `DANTE` across representative road scenarios.
- `06_non_iid_sweep.sh`
  `DANTE` under increasing data heterogeneity.
- `07_byzantine_attack.sh`
  `DANTE` and `DPFL` under a 20% LIE Byzantine model-poisoning attack starting at round 20.

## Notes

- These scripts assume SUMO and the Python dependencies are already installed.
- They run in headless mode with `--rounds 200` and `--target_acc 0.98`.
- The PPO-based methods now report on the `test` split for fair paper figures.
