"""Plot results from previously saved experiments in out/."""

import os
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import config
from dl.experiment import load_experiment, plot_saved_experiment

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
    "figure.figsize": (7.16, 4.6),
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
}

_ALGO_COLORS = {
    "DANTE": "#0072B2",
    "IPPO": "#E69F00",
    "FedAvg": "#D55E00",
    "FedProx": "#009E73",
    "D-PSGD": "#CC79A7",
    "DPFL": "#F0E442",
}
_FALLBACK_COLORS = ["#0072B2", "#E69F00", "#D55E00", "#009E73", "#CC79A7", "#F0E442"]


def plot_past(experiment_folder: str, block: bool = True) -> None:
    """Load and plot a saved experiment by its folder name or full path.

    Args:
        experiment_folder: folder name inside out/ (e.g. "dubai_marina_fedavg_..."),
                           or an absolute/relative path to the experiment directory.
        block: if True, block until all plot windows are closed.
    """
    # Resolve path
    if os.path.isabs(experiment_folder) or experiment_folder.startswith("."):
        folder = experiment_folder
    else:
        folder = os.path.join(config.OUT_DIR, experiment_folder)

    pickle_path = os.path.join(folder, "experiment.pkl")
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"No experiment.pkl found in: {folder}")

    print(f"Plotting experiment: {folder}")
    result = plot_saved_experiment(pickle_path, show=True, block=block)
    print(f"Figures saved to: {result['experiment_dir']}")


def _style() -> matplotlib.rc_context:
    return matplotlib.rc_context(_PAPER_RC)


def _resolve_folder(selection: str, experiments: list[str]) -> str:
    """Resolve an index or folder/path token to an experiment folder."""
    token = selection.strip()
    if not token:
        raise ValueError("Empty experiment selection.")

    if re.fullmatch(r"\d+", token):
        idx = int(token)
        try:
            return experiments[idx]
        except IndexError as exc:
            raise IndexError(f"Experiment index out of range: {idx}") from exc

    return token


def _resolve_pickle_path(experiment_folder: str) -> str:
    """Resolve an experiment folder name/path to its experiment pickle path."""
    if os.path.isabs(experiment_folder) or experiment_folder.startswith("."):
        folder = experiment_folder
    else:
        folder = os.path.join(config.OUT_DIR, experiment_folder)

    pickle_path = os.path.join(folder, "experiment.pkl")
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"No experiment.pkl found in: {folder}")
    return os.path.abspath(pickle_path)


def _algo_color(algo: str, idx: int = 0) -> str:
    return _ALGO_COLORS.get(algo, _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)])


def _unique_algorithm_labels(experiments: list[dict]) -> list[str]:
    """Return readable labels, disambiguating duplicate algorithm names."""
    counts: dict[str, int] = {}
    labels = []
    for experiment in experiments:
        algo = experiment.get("metadata", {}).get("algorithm", "Unknown")
        counts[algo] = counts.get(algo, 0) + 1
        label = algo if counts[algo] == 1 else f"{algo} ({counts[algo]})"
        labels.append(label)
    return labels


def _comparison_dir(experiments: list[dict]) -> str:
    """Create an output directory for multi-experiment comparison figures."""
    out_root = Path(config.OUT_DIR)
    algos = [
        experiment.get("metadata", {}).get("algorithm", "unknown").lower()
        for experiment in experiments
    ]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = "_".join(re.sub(r"[^a-z0-9._-]+", "-", algo).strip("-") or "unknown" for algo in algos)
    directory = out_root / f"comparison_{slug}_{stamp}"
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory)


def _save_figure(fig: plt.Figure, path_base: Path) -> None:
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(path_base.with_suffix(f".{ext}"))


def _eval_history(experiment: dict) -> list[dict]:
    return sorted(
        experiment.get("eval_history", experiment.get("test_history", [])),
        key=lambda p: p["round"],
    )


def _reward_history(experiment: dict) -> list[dict]:
    return sorted(experiment.get("reward_history", []), key=lambda p: p["step"])


def _is_ppo_experiment(experiment: dict) -> bool:
    return experiment.get("metadata", {}).get("algorithm") in {"DANTE", "IPPO"}


def _reward_rounds(experiment: dict) -> list[float]:
    """Map reward timestamps onto training rounds for round-based PPO plots."""
    rewards = _reward_history(experiment)
    if not rewards:
        return []

    train_history = sorted(experiment.get("train_history", []), key=lambda p: p["round"])
    if len(train_history) < 2:
        return [point["step"] for point in rewards]

    train_times = np.asarray([point["time"] for point in train_history], dtype=float)
    train_rounds = np.asarray([point["round"] for point in train_history], dtype=float)
    reward_times = np.asarray([point["time"] for point in rewards], dtype=float)

    unique_times, unique_idx = np.unique(train_times, return_index=True)
    unique_rounds = train_rounds[unique_idx]
    if len(unique_times) < 2 or unique_times[-1] <= unique_times[0]:
        return [point["step"] for point in rewards]

    return list(np.interp(reward_times, unique_times, unique_rounds))


def _eval_std_history(experiment: dict, key: str) -> list[float]:
    return [point.get(key, 0.0) for point in _eval_history(experiment)]


def _plot_line_comparison(
    experiments: list[dict],
    labels: list[str],
    output_dir: str,
    filename: str,
    title: str,
    xlabel: str,
    ylabel: str,
    x_getter,
    y_getter,
    *,
    std_getter=None,
    percent: bool = False,
    include_filter=None,
) -> bool:
    """Plot one overlay figure across experiments and save it."""
    fig, ax = plt.subplots()
    plotted = False

    for idx, (label, experiment) in enumerate(zip(labels, experiments)):
        if include_filter is not None and not include_filter(experiment):
            continue

        xs = x_getter(experiment)
        ys = y_getter(experiment)
        if not xs or not ys:
            continue

        algo = experiment.get("metadata", {}).get("algorithm", label)
        ax.plot(xs, ys, label=label, color=_algo_color(algo, idx))
        if std_getter is not None:
            stds = std_getter(experiment)
            if stds and len(stds) == len(ys):
                xs_arr = np.asarray(xs, dtype=float)
                ys_arr = np.asarray(ys, dtype=float)
                std_arr = np.asarray(stds, dtype=float)
                lower = ys_arr - std_arr
                if percent:
                    lower = np.clip(lower, 0.0, 1.0)
                    upper = np.clip(ys_arr + std_arr, 0.0, 1.0)
                else:
                    lower = np.clip(lower, 0.0, None)
                    upper = ys_arr + std_arr
                ax.fill_between(
                    xs_arr,
                    lower,
                    upper,
                    color=_algo_color(algo, idx),
                    alpha=0.07,
                    linewidth=0.0,
                )
        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if percent:
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.PercentFormatter(1.0, decimals=0)
        )
    ax.legend()
    _save_figure(fig, Path(output_dir) / filename)
    return True


def _warn_mixed_evaluation_modes(experiments: list[dict], labels: list[str]) -> None:
    """Print a warning when experiments use different evaluation modes.

    Accuracy numbers from personalized evaluation (each vehicle tested on its
    own non-IID shard) and global evaluation (all vehicles on the shared test
    set) are not directly comparable and must not appear on the same axis
    without this caveat.
    """
    modes = {
        exp.get("config", {}).get("EVALUATION_MODE", "global"): []
        for exp in experiments
    }
    for exp, label in zip(experiments, labels):
        mode = exp.get("config", {}).get("EVALUATION_MODE", "global")
        modes[mode].append(label)

    if len(modes) > 1:
        lines = [
            "WARNING: comparison mixes evaluation modes — accuracy axes are NOT directly comparable.",
            "  Personalized evaluation tests each vehicle on its own non-IID shard (easier).",
            "  Global evaluation tests all vehicles on the shared balanced test set.",
            "  Re-run all experiments with the same EVALUATION_MODE before publishing.",
        ]
        for mode, algo_labels in sorted(modes.items()):
            lines.append(f"  [{mode}] {', '.join(algo_labels)}")
        warning = "\n".join(lines)
        print(warning, file=sys.stderr)


def plot_multi(experiment_folders: list[str], block: bool = True) -> None:
    """Plot comparison figures for multiple saved experiments."""
    pickle_paths = [_resolve_pickle_path(folder) for folder in experiment_folders]
    experiments = [load_experiment(path) for path in pickle_paths]
    labels = _unique_algorithm_labels(experiments)
    output_dir = _comparison_dir(experiments)

    _warn_mixed_evaluation_modes(experiments, labels)

    with _style():
        _plot_line_comparison(
            experiments,
            labels,
            output_dir,
            "accuracy_vs_rounds_comparison",
            "Mean Per-Vehicle Accuracy vs Rounds",
            "Rounds",
            "Accuracy",
            lambda exp: [point["round"] for point in _eval_history(exp)],
            lambda exp: [point["acc"] for point in _eval_history(exp)],
            std_getter=lambda exp: _eval_std_history(exp, "acc_std"),
            percent=True,
        )

        _plot_line_comparison(
            experiments,
            labels,
            output_dir,
            "accuracy_vs_time_comparison",
            "Mean Per-Vehicle Accuracy vs Time",
            "Time (s)",
            "Accuracy",
            lambda exp: [point["time"] for point in _eval_history(exp)],
            lambda exp: [point["acc"] for point in _eval_history(exp)],
            std_getter=lambda exp: _eval_std_history(exp, "acc_std"),
            percent=True,
        )

        _plot_line_comparison(
            experiments,
            labels,
            output_dir,
            "loss_vs_rounds_comparison",
            "Mean Per-Vehicle Loss vs Rounds",
            "Rounds",
            "Loss",
            lambda exp: [point["round"] for point in _eval_history(exp)],
            lambda exp: [point["loss"] for point in _eval_history(exp)],
            std_getter=lambda exp: _eval_std_history(exp, "loss_std"),
        )

        _plot_line_comparison(
            experiments,
            labels,
            output_dir,
            "loss_vs_time_comparison",
            "Mean Per-Vehicle Loss vs Time",
            "Time (s)",
            "Loss",
            lambda exp: [point["time"] for point in _eval_history(exp)],
            lambda exp: [point["loss"] for point in _eval_history(exp)],
            std_getter=lambda exp: _eval_std_history(exp, "loss_std"),
        )

        plotted_ppo_rounds = _plot_line_comparison(
            experiments,
            labels,
            output_dir,
            "ppo_vs_rounds_comparison",
            "PPO Reward vs Rounds",
            "Rounds",
            "Avg. PPO Reward",
            _reward_rounds,
            lambda exp: [point["reward"] for point in _reward_history(exp)],
            include_filter=_is_ppo_experiment,
        )

        plotted_ppo_time = _plot_line_comparison(
            experiments,
            labels,
            output_dir,
            "ppo_vs_time_comparison",
            "PPO Reward vs Time",
            "Time (s)",
            "Avg. PPO Reward",
            lambda exp: [point["time"] for point in _reward_history(exp)],
            lambda exp: [point["reward"] for point in _reward_history(exp)],
            include_filter=_is_ppo_experiment,
        )

        # Total energy comparison
        fig_energy, ax_energy = plt.subplots()
        energy_values = []
        bar_colors = []
        for idx, (label, experiment) in enumerate(zip(labels, experiments)):
            energy_totals = dict(experiment.get("energy_totals", {}))
            total_energy = float(energy_totals.get("computation_energy_j", 0.0))
            total_energy += float(energy_totals.get("total_tx_energy_j", 0.0))
            energy_values.append(total_energy)
            algo = experiment.get("metadata", {}).get("algorithm", label)
            bar_colors.append(_algo_color(algo, idx))

        bars = ax_energy.bar(labels, energy_values, color=bar_colors, width=0.55,
                             edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, energy_values):
            ax_energy.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02 if val > 0 else 0.02,
                f"{val:.2f} J",
                ha="center",
                va="bottom",
                fontsize=7,
            )

        ax_energy.set_title("Total Energy Comparison")
        ax_energy.set_ylabel("Energy (J)")
        ax_energy.tick_params(axis="x", rotation=15)
        ax_energy.set_ylim(top=max(energy_values) * 1.18 if any(energy_values) else 1.0)
        _save_figure(fig_energy, Path(output_dir) / "energy_comparison")

    if not plotted_ppo_rounds or not plotted_ppo_time:
        print("Skipping PPO comparison plots for non-PPO selections.")

    plt.show(block=block)
    print(f"Comparison figures saved to: {output_dir}")


def list_experiments(out_root: str | None = None) -> list[str]:
    """Return all experiment folder names available in out/."""
    root = os.path.abspath(out_root or config.OUT_DIR)
    if not os.path.isdir(root):
        return []
    return sorted(
        name for name in os.listdir(root)
        if os.path.isfile(os.path.join(root, name, "experiment.pkl"))
    )


def _parse_selection(raw: str, experiments: list[str]) -> list[str]:
    """Parse one or more selection tokens into experiment folders."""
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens:
        raise ValueError("No experiment selection provided.")
    return [_resolve_folder(token, experiments) for token in tokens]


if __name__ == "__main__":
    experiments = list_experiments()
    if len(sys.argv) < 2:
        if not experiments:
            print(f"No saved experiments found in {config.OUT_DIR}")
            sys.exit(1)
        print(f"Available experiments in {config.OUT_DIR}:")
        for i, name in enumerate(experiments):
            print(f"  [{i}] {name}")
        selection = input("\nEnter index, comma-separated indexes, or folder name: ").strip()
    else:
        selection = ",".join(sys.argv[1:])

    folders = _parse_selection(selection, experiments)
    if len(folders) == 1:
        plot_past(folders[0], block=True)
    else:
        plot_multi(folders, block=True)
