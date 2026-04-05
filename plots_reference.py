"""
plots.py — Publication-quality result plots for V2X TR experiments
===================================================================
Usage (single run, from main.py):
    from plots import SimResults, plot_all
    results = SimResults(algorithm="GAT_PPO", ...)
    plot_all(results, save_dir="figures")

Usage (multi-run comparison):
    from plots import plot_comparison
    plot_comparison({"GAT-PPO": r1, "IPPO": r2, "FedAvg": r3}, save_dir="figures")

Figures produced
----------------
  convergence.pdf   — 2×2: test acc / test loss  ×  rounds / wall-time
  neighbors.pdf     — avg connections per round (total, sidelink, internet) ± std
  energy.pdf        — bar chart: training energy vs. TX energy breakdown
  reward.pdf        — PPO reward vs. rounds  (GAT_PPO / IPPO only)
  comparison.pdf    — multi-algorithm overlay: acc + loss vs. rounds

Energy model (PC5 sidelink / 5G-NR Uu, ~200 KB DNN model)
-----------------------------------------------------------
  E_COMPUTE_J_PER_SAMPLE  0.5 mJ / sample   (ARM Cortex-A local SGD)
  Shannon TX model — helpers.sl_tx_energy_j / helpers.inet_tx_energy_j:
    Sidelink  E(d) = P_sl · S / (B_sl · log₂(1 + SNR_0·(V2X_RANGE/d)²))
              Distance-dependent: closer vehicles consume less energy.
    Internet  E    = 2·P_inet·S / (B_inet · log₂(1 + SNR_inet))
              Fixed cost — vehicle-to-vehicle distance is irrelevant.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import numpy as np

matplotlib.use("QtAgg")  # change to "QtAgg" or "MacOSX" or "TkAgg" if needed
os.environ["QT_SCALE_FACTOR"] = "0.5"  # reduce to 50%
import matplotlib.pyplot as plt
import matplotlib.ticker

# ── Energy model constants ─────────────────────────────────────────────────────
E_COMPUTE_J_PER_SAMPLE = 5e-4  # J/sample — local SGD compute
# TX energy uses the Shannon capacity model — see helpers.sl_tx_energy_j / inet_tx_energy_j

# ── Color palette (IBM accessible, colorblind-friendly) ───────────────────────
_ALGO_COLORS = {
    "GAT_PPO": "#0072B2",  # blue        — proposed
    "IPPO": "#E69F00",  # amber       — RL ablation
    "FedAvg": "#D55E00",  # vermillion  — static baseline
    "FedProx": "#009E73",  # green       — proximal SOTA
    "D-PSGD": "#CC79A7",  # pink        — gossip SOTA
    "DPFL": "#F0E442",  # yellow      — personalized SOTA
}
_FALLBACK_COLORS = ["#0072B2", "#E69F00", "#D55E00", "#009E73", "#CC79A7", "#F0E442"]
_SL_COLOR = "#0072B2"  # blue   — sidelink
_INET_COLOR = "#E69F00"  # amber  — internet
_TOTAL_COLOR = "#009E73"  # green  — total connections / total TX
_TRAIN_E_COLOR = "#2D6A4F"  # dark green — training energy
_TEST_COLOR = "#0072B2"  # solid  — test metric
_TRAIN_COLOR = "#56B4E9"  # light  — train metric (dashed)

# ── Figure dimensions (IEEE column widths in inches) ──────────────────────────
_W1 = 3.45  # single column
_W2 = 7.16  # double column
_H = 2.3  # row height

# ── Paper rcParams ─────────────────────────────────────────────────────────────
_PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.framealpha": 0.85,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.6,
    "lines.markersize": 3.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
}


# ── SimResults ─────────────────────────────────────────────────────────────────

@dataclass
class SimResults:
    """All metrics collected during one simulation run.

    All list fields have one entry per completed TR round.
    Save/load via pickle for multi-run comparisons:
        results.save("run_gat.pkl")
        r = SimResults.load("run_gat.pkl")
    """
    # Metadata
    algorithm: str = "GAT_PPO"
    dataset: str = "MNIST"
    n_vehicles: int = 10

    # Round index and wall-clock time
    rounds: List[int] = field(default_factory=list)
    wall_times: List[float] = field(default_factory=list)  # seconds from start

    # Test metrics (global avg across all vehicles evaluated on full test set)
    test_acc: List[float] = field(default_factory=list)
    test_loss: List[float] = field(default_factory=list)

    # Training metrics (global avg from mini-batch stats)
    train_acc: List[float] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)  # avg PPO reward

    # Neighbor statistics: mean ± std across vehicles, per round
    n_conn_mean: List[float] = field(default_factory=list)
    n_conn_std: List[float] = field(default_factory=list)
    n_sl_mean: List[float] = field(default_factory=list)
    n_sl_std: List[float] = field(default_factory=list)
    n_inet_mean: List[float] = field(default_factory=list)
    n_inet_std: List[float] = field(default_factory=list)

    # Energy per TR round (Joules), accumulated across all vehicles & steps
    energy_train: List[float] = field(default_factory=list)
    energy_sl: List[float] = field(default_factory=list)
    energy_inet: List[float] = field(default_factory=list)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, filepath: str) -> None:
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"  Results saved → {filepath}")

    @staticmethod
    def load(filepath: str) -> "SimResults":
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def __str__(self) -> str:
        n_rounds = len(self.rounds)
        acc = f"{self.test_acc[-1]:.2%}" if self.test_acc else "n/a"
        return (f"SimResults({self.algorithm}, {self.dataset}, "
                f"n={self.n_vehicles}, rounds={n_rounds}, test_acc={acc})")


# ── Internal helpers ───────────────────────────────────────────────────────────

def _style() -> matplotlib.rc_context:
    """Context manager that applies paper rcParams."""
    return matplotlib.rc_context(_PAPER_RC)


def _save(fig: plt.Figure, name: str, save_dir: Optional[str]) -> None:
    if save_dir is None:
        return
    p = Path(save_dir)
    p.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(p / f"{name}.{ext}")
    print(f"  Saved figures/{name}.pdf + .png")


def _ema(values: list, alpha: float = 0.25) -> list:
    """Exponential moving average for smoothing noisy signals (e.g. rewards)."""
    if not values:
        return []
    s, out = values[0], [values[0]]
    for v in values[1:]:
        s = alpha * v + (1.0 - alpha) * s
        out.append(s)
    return out


def _algo_color(algo: str, idx: int = 0) -> str:
    return _ALGO_COLORS.get(algo, _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)])


def _algo_label(algo: str) -> str:
    return {
        "GAT_PPO": "Ours",
        "IPPO": "IPPO",
        "FedAvg": "FedAvg",
        "FedProx": "FedProx",
        "D-PSGD": "D-PSGD",
        "DPFL": "DPFL",
    }.get(algo, algo)


# ── Plot: convergence (2×2) ────────────────────────────────────────────────────

def plot_convergence(results: SimResults, save_dir: Optional[str] = None) -> plt.Figure:
    """2×2 figure: test accuracy / test loss  ×  TR rounds / wall time.

    Solid lines = test metrics.  Dashed lines = training metrics (for comparison).
    """
    R = results.rounds
    T = results.wall_times if results.wall_times else R

    with _style():
        fig, axes = plt.subplots(2, 2, figsize=(_W2, 2 * _H))
        fig.subplots_adjust(hspace=0.42, wspace=0.32)
        color = _algo_color(results.algorithm)
        label = _algo_label(results.algorithm)

        # ── top row: accuracy ─────────────────────────────────────────────────
        for col, (xs, xlabel) in enumerate([(R, "TR Rounds"), (T, "Wall Time (s)")]):
            ax = axes[0, col]
            ax.plot(xs, results.test_acc, color=color, lw=1.8, label=f"{label} (test)")
            ax.plot(xs, results.train_acc, color=color, lw=1.0, ls="--", alpha=0.6,
                    label="Train (mini-batch)")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Accuracy")
            ax.set_title("Test Accuracy vs. " + xlabel.split()[0])
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
            ax.legend(loc="lower right")

        # ── bottom row: loss ──────────────────────────────────────────────────
        for col, (xs, xlabel) in enumerate([(R, "TR Rounds"), (T, "Wall Time (s)")]):
            ax = axes[1, col]
            ax.plot(xs, results.test_loss, color=color, lw=1.8, label=f"{label} (test)")
            ax.plot(xs, results.train_loss, color=color, lw=1.0, ls="--", alpha=0.6,
                    label="Train (mini-batch)")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Loss")
            ax.set_title("Test Loss vs. " + xlabel.split()[0])
            ax.legend(loc="upper right")

        fig.suptitle(
            f"{results.dataset} · {results.n_vehicles} vehicles · {_algo_label(results.algorithm)}",
            fontsize=9, y=0.99,
        )
        _save(fig, "convergence", save_dir)
    return fig


# ── Plot: neighbor statistics ─────────────────────────────────────────────────

def plot_neighbors(results: SimResults, save_dir: Optional[str] = None) -> plt.Figure:
    """Average number of connections per round with ±1 std shading."""
    R = np.asarray(results.rounds)
    cm = np.asarray(results.n_conn_mean)
    cs = np.asarray(results.n_conn_std)
    sm = np.asarray(results.n_sl_mean)
    ss = np.asarray(results.n_sl_std)
    im = np.asarray(results.n_inet_mean)
    is_ = np.asarray(results.n_inet_std)

    with _style():
        fig, ax = plt.subplots(figsize=(_W2, _H))

        def _band(ax, xs, mean, std, color, label, ls="-"):
            ax.plot(xs, mean, color=color, lw=1.8, ls=ls, label=label)
            ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.15)

        _band(ax, R, cm, cs, _TOTAL_COLOR, "Total connections")
        _band(ax, R, sm, ss, _SL_COLOR, "Sidelink", ls="-")  # Sidelink (PC5)
        _band(ax, R, im, is_, _INET_COLOR, "Internet", ls="--")  # Internet (5G Uu)

        ax.set_xlabel("TR Rounds")
        ax.set_ylabel("Avg. connections / vehicle")
        ax.set_title("Neighbor Connectivity per TR Round")
        ax.legend(ncol=3, loc="upper right")
        ax.set_ylim(bottom=0)

        _save(fig, "neighbors", save_dir)
    return fig


# ── Plot: energy breakdown ─────────────────────────────────────────────────────

def plot_energy(results: SimResults, save_dir: Optional[str] = None) -> plt.Figure:
    """Stacked bar chart: cumulative training energy vs. TX energy breakdown.

    Left panel  — absolute totals (J).
    Right panel — per-round cumulative energy growth.
    """
    e_train = np.asarray(results.energy_train)
    e_sl = np.asarray(results.energy_sl)
    e_inet = np.asarray(results.energy_inet)
    R = np.asarray(results.rounds)

    total_train = float(e_train.sum())
    total_sl = float(e_sl.sum())
    total_inet = float(e_inet.sum())
    total_tx = total_sl + total_inet

    with _style():
        fig, (ax_bar, ax_cum) = plt.subplots(1, 2, figsize=(_W2, _H))
        fig.subplots_adjust(wspace=0.38)

        # ── left: stacked bar totals ──────────────────────────────────────────
        categories = ["Training", "Sidelink TX", "Internet TX", "Total TX"]
        values = [total_train, total_sl, total_inet, total_tx]
        bar_colors = [_TRAIN_E_COLOR, _SL_COLOR, _INET_COLOR, _TOTAL_COLOR]
        bars = ax_bar.bar(categories, values, color=bar_colors, width=0.55,
                          edgecolor="white", linewidth=0.8)

        for bar, val in zip(bars, values):
            ax_bar.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.02,
                        f"{val:.2f} J", ha="center", va="bottom", fontsize=7)

        ax_bar.set_ylabel("Total Energy (J)")
        ax_bar.set_title("Cumulative Energy Breakdown")
        ax_bar.tick_params(axis="x", rotation=15)
        ax_bar.set_ylim(top=max(values) * 1.18)

        # ── right: cumulative per-round ───────────────────────────────────────
        ax_cum.plot(R, np.cumsum(e_train), color=_TRAIN_E_COLOR, lw=1.6,
                    label="Training")
        ax_cum.plot(R, np.cumsum(e_sl), color=_SL_COLOR, lw=1.6,
                    label="Sidelink TX")
        ax_cum.plot(R, np.cumsum(e_inet), color=_INET_COLOR, lw=1.6,
                    ls="--", label="Internet TX")
        ax_cum.plot(R, np.cumsum(e_sl + e_inet), color=_TOTAL_COLOR, lw=1.0,
                    ls=":", label="Total TX")

        ax_cum.set_xlabel("TR Rounds")
        ax_cum.set_ylabel("Cumulative Energy (J)")
        ax_cum.set_title("Energy Consumption over Rounds")
        ax_cum.legend(fontsize=7, ncol=2)
        ax_cum.set_ylim(bottom=0)

        _save(fig, "energy", save_dir)
    return fig


# ── Plot: PPO reward ───────────────────────────────────────────────────────────

def plot_reward(results: SimResults, save_dir: Optional[str] = None) -> Optional[plt.Figure]:
    """PPO reward vs. TR rounds (only meaningful for GAT_PPO / IPPO)."""
    if results.algorithm not in ("GAT_PPO", "IPPO") or not results.rewards:
        return None

    R = results.rounds
    raw = results.rewards
    smooth = _ema(raw, alpha=0.3)

    with _style():
        fig, ax = plt.subplots(figsize=(_W1, _H))
        color = _algo_color(results.algorithm)

        ax.plot(R, raw, color=color, lw=0.6, alpha=0.35, label="Raw reward")
        ax.plot(R, smooth, color=color, lw=1.8, label="EMA (α=0.3)")
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)

        ax.set_xlabel("TR Rounds")
        ax.set_ylabel("Avg. PPO Reward")
        ax.set_title(f"PPO Reward — {_algo_label(results.algorithm)}")
        ax.legend()

        _save(fig, "reward", save_dir)
    return fig


# ── Plot: multi-algorithm comparison ──────────────────────────────────────────

def plot_comparison(
        results_dict: Dict[str, SimResults],
        save_dir: Optional[str] = None,
) -> plt.Figure:
    """Overlay multiple algorithms on acc-vs-rounds and loss-vs-rounds.

    Args:
        results_dict: mapping of display label → SimResults.
                      Example: {"GAT-PPO (ours)": r1, "IPPO": r2, "FedAvg": r3}
    """
    with _style():
        fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(_W2, _H))
        fig.subplots_adjust(wspace=0.32)

        for idx, (label, res) in enumerate(results_dict.items()):
            color = _algo_color(res.algorithm, idx)
            R = res.rounds
            ax_acc.plot(R, res.test_acc, color=color, lw=1.8, label=label)
            ax_loss.plot(R, res.test_loss, color=color, lw=1.8, label=label)

        ax_acc.set_xlabel("TR Rounds")
        ax_acc.set_ylabel("Test Accuracy")
        ax_acc.set_title("Test Accuracy vs. TR Round")
        ax_acc.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax_acc.legend()

        ax_loss.set_xlabel("TR Rounds")
        ax_loss.set_ylabel("Test Loss")
        ax_loss.set_title("Test Loss vs. TR Rounds")
        ax_loss.legend()

        # Dataset / n_vehicles from first result
        first = next(iter(results_dict.values()))
        fig.suptitle(
            f"{first.dataset} · {first.n_vehicles} vehicles — Algorithm Comparison",
            fontsize=9, y=1.01,
        )

        _save(fig, "comparison", save_dir)
    return fig


# ── Plot: all single-run figures ───────────────────────────────────────────────

def plot_all(results: SimResults, save_dir: str = "figures") -> None:
    """Generate and save all single-run publication figures.

    Figures written to `save_dir/`:
        convergence.pdf/.png
        neighbors.pdf/.png
        energy.pdf/.png
        reward.pdf/.png   (GAT_PPO / IPPO only)
    """
    print(f"\n  Generating plots → {save_dir}/")

    plot_convergence(results, save_dir)
    plot_neighbors(results, save_dir)
    plot_energy(results, save_dir)
    plot_reward(results, save_dir)

    plt.show()
