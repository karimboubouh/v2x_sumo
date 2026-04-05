"""Experiment persistence and plotting for DPL training runs."""

import os
import pickle
import re
import tempfile
from datetime import datetime, timezone

import config

_xdg_cache_home = os.path.join(tempfile.gettempdir(), "sumo-xdg-cache")
_mpl_cache_dir = os.path.join(tempfile.gettempdir(), "sumo-matplotlib-cache")
os.makedirs(_xdg_cache_home, exist_ok=True)
os.makedirs(_mpl_cache_dir, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", _xdg_cache_home)
os.environ.setdefault(
    "MPLCONFIGDIR",
    _mpl_cache_dir,
)

import matplotlib
import matplotlib.pyplot as plt

# ── Style (from plots_reference) ──────────────────────────────────────────────
_TEST_COLOR    = "#0072B2"   # solid  — test metric
_TRAIN_COLOR   = "#56B4E9"   # light  — train metric (dashed)
_PPO_COLOR     = "#7c3aed"   # purple — PPO reward
_TRAIN_E_COLOR = "#2D6A4F"   # dark green — training energy
_SL_COLOR      = "#0072B2"   # blue   — sidelink
_INET_COLOR    = "#E69F00"   # amber  — internet
_TOTAL_COLOR   = "#009E73"   # green  — total TX
_W2 = 7.16   # figure width (inches)
_H  = 2.3    # row height (inches)

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
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
}


def _ema(values: list, alpha: float = 0.3) -> list:
    """Exponential moving average smoothing."""
    if not values:
        return []
    s, out = values[0], [values[0]]
    for v in values[1:]:
        s = alpha * v + (1.0 - alpha) * s
        out.append(s)
    return out


def _backend_supports_show(backend: str) -> bool:
    backend = backend.lower()
    if backend in {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}:
        return False
    if backend.startswith("module://matplotlib_inline"):
        return False
    interactive_markers = (
        "qt",
        "tk",
        "wx",
        "gtk",
        "macosx",
        "nbagg",
        "notebook",
        "webagg",
    )
    return any(marker in backend for marker in interactive_markers)


def _present_open_figures() -> None:
    for fig_num in plt.get_fignums():
        try:
            manager = plt.figure(fig_num).canvas.manager
            window = getattr(manager, "window", None)
            if window is None:
                continue
            toolbar = getattr(manager, "toolbar", None)
            if toolbar is not None and hasattr(toolbar, "setIconSize"):
                from PySide6.QtCore import QSize, Qt as _Qt
                toolbar.setIconSize(QSize(16, 16))
                window.removeToolBar(toolbar)
                window.addToolBar(_Qt.BottomToolBarArea, toolbar)
                toolbar.show()
            if hasattr(window, "show"):
                window.show()
            if hasattr(window, "raise_"):
                window.raise_()
            if hasattr(window, "activateWindow"):
                window.activateWindow()
        except Exception:
            continue


def pump_plot_events() -> None:
    """Keep non-blocking matplotlib windows responsive."""
    if not plt.get_fignums():
        return
    try:
        plt.pause(0.001)
    except Exception:
        pass


def _slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "-", text.strip().lower())
    text = text.strip("-")
    return text or "experiment"


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")


def _experiment_id(experiment: dict) -> str:
    metadata = experiment.get("metadata", {})
    explicit = metadata.get("experiment_id")
    if explicit:
        return _slugify(explicit)

    pieces = [
        metadata.get("scenario", "scenario"),
        metadata.get("algorithm", "algo"),
        metadata.get("dataset", "dataset"),
        metadata.get("model", "model"),
        str(metadata.get("num_vehicles", "n")),
        _timestamp_slug(),
    ]
    return _slugify("_".join(str(piece) for piece in pieces))


def _line_title(experiment: dict) -> str:
    metadata = experiment.get("metadata", {})
    return (
        f"{metadata.get('algorithm', 'DPL')} | "
        f"{metadata.get('dataset', 'dataset')}/{metadata.get('model', 'model')} | "
        f"{metadata.get('scenario_name', metadata.get('scenario', 'scenario'))}"
    )


def _ensure_out_root(out_root: str | None = None) -> str:
    root = os.path.abspath(out_root or config.OUT_DIR)
    os.makedirs(root, exist_ok=True)
    return root


def load_experiment(path: str) -> dict:
    """Load a previously saved experiment pickle."""
    with open(path, "rb") as fh:
        return pickle.load(fh)


def save_experiment(experiment: dict, out_root: str | None = None) -> dict:
    """Persist one experiment pickle into its own folder under out_root."""
    root = _ensure_out_root(out_root)
    experiment_id = _experiment_id(experiment)
    experiment_dir = os.path.join(root, experiment_id)
    os.makedirs(experiment_dir, exist_ok=True)

    experiment = dict(experiment)
    metadata = dict(experiment.get("metadata", {}))
    metadata.setdefault("experiment_id", experiment_id)
    metadata.setdefault("saved_at", datetime.now(timezone.utc).isoformat())
    experiment["metadata"] = metadata

    pickle_path = os.path.join(experiment_dir, "experiment.pkl")
    with open(pickle_path, "wb") as fh:
        pickle.dump(experiment, fh)

    return {
        "experiment": experiment,
        "experiment_id": experiment_id,
        "experiment_dir": experiment_dir,
        "pickle_path": pickle_path,
    }


def _save_figure(fig, path: str) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return path


def _prepare_series(history: list, x_key: str, y_key: str) -> tuple[list, list]:
    return [point[x_key] for point in history], [point[y_key] for point in history]


def plot_experiment(
    experiment: dict,
    output_dir: str,
    show: bool = True,
    block: bool = False,
) -> dict:
    """Generate and save the requested experiment figures."""
    os.makedirs(output_dir, exist_ok=True)
    train_history = sorted(experiment.get("train_history", []), key=lambda p: p["round"])
    test_history  = sorted(experiment.get("test_history",  []), key=lambda p: p["round"])
    energy_totals = dict(experiment.get("energy_totals", {}))
    title = _line_title(experiment)

    matplotlib.rcParams.update(_PAPER_RC)
    figures = {}

    train_rounds, train_acc = _prepare_series(train_history, "round", "acc")
    train_times, _ = _prepare_series(train_history, "time", "acc")
    _, train_loss = _prepare_series(train_history, "round", "loss")
    test_rounds, test_acc = _prepare_series(test_history, "round", "acc")
    test_times, _ = _prepare_series(test_history, "time", "acc")
    _, test_loss = _prepare_series(test_history, "round", "loss")
    reward_history = list(experiment.get("reward_history", []))
    reward_steps, reward_values = _prepare_series(reward_history, "step", "reward")
    reward_times, _ = _prepare_series(reward_history, "time", "reward")

    fig, ax = plt.subplots(figsize=(_W2, _H * 2))
    ax.plot(train_rounds, train_acc, label="Train Accuracy", lw=1.0, ls="--", alpha=0.6, color=_TRAIN_COLOR)
    if test_history:
        ax.plot(test_rounds, test_acc, label="Test Accuracy", lw=1.8, marker="o", color=_TEST_COLOR)
    ax.set_title(f"Accuracy vs Rounds\n{title}")
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Accuracy")
    ax.legend(loc="lower right")
    figures["accuracy_vs_rounds"] = _save_figure(
        fig,
        os.path.join(output_dir, "accuracy_vs_rounds.png"),
    )

    fig, ax = plt.subplots(figsize=(_W2, _H * 2))
    ax.plot(train_times, train_acc, label="Train Accuracy", lw=1.0, ls="--", alpha=0.6, color=_TRAIN_COLOR)
    if test_history:
        ax.plot(test_times, test_acc, label="Test Accuracy", lw=1.8, marker="o", color=_TEST_COLOR)
    ax.set_title(f"Accuracy vs Time\n{title}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Accuracy")
    ax.legend(loc="lower right")
    figures["accuracy_vs_time"] = _save_figure(
        fig,
        os.path.join(output_dir, "accuracy_vs_time.png"),
    )

    fig, ax = plt.subplots(figsize=(_W2, _H * 2))
    ax.plot(train_rounds, train_loss, label="Train Loss", lw=1.0, ls="--", alpha=0.6, color=_TRAIN_COLOR)
    if test_history:
        ax.plot(test_rounds, test_loss, label="Test Loss", lw=1.8, marker="o", color=_TEST_COLOR)
    ax.set_title(f"Loss vs Rounds\n{title}")
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    figures["loss_vs_rounds"] = _save_figure(
        fig,
        os.path.join(output_dir, "loss_vs_rounds.png"),
    )

    fig, ax = plt.subplots(figsize=(_W2, _H * 2))
    ax.plot(train_times, train_loss, label="Train Loss", lw=1.0, ls="--", alpha=0.6, color=_TRAIN_COLOR)
    if test_history:
        ax.plot(test_times, test_loss, label="Test Loss", lw=1.8, marker="o", color=_TEST_COLOR)
    ax.set_title(f"Loss vs Time\n{title}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    figures["loss_vs_time"] = _save_figure(
        fig,
        os.path.join(output_dir, "loss_vs_time.png"),
    )

    fig, ax = plt.subplots(figsize=(_W2, _H * 2))
    energy_labels = ["Computation", "Sidelink TX", "Internet TX", "Total TX"]
    energy_values = [
        energy_totals.get("computation_energy_j", 0.0),
        energy_totals.get("sidelink_tx_energy_j", 0.0),
        energy_totals.get("internet_tx_energy_j", 0.0),
        energy_totals.get("total_tx_energy_j", 0.0),
    ]
    bar_colors = [_TRAIN_E_COLOR, _SL_COLOR, _INET_COLOR, _TOTAL_COLOR]
    bars = ax.bar(energy_labels, energy_values, color=bar_colors, width=0.55,
                  edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, energy_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f"{val:.2f} J", ha="center", va="bottom", fontsize=7)
    ax.set_title(f"Energy Totals Across All Vehicles\n{title}")
    ax.set_ylabel("Energy (J)")
    ax.tick_params(axis="x", rotation=15)
    ax.set_ylim(top=max(energy_values) * 1.18 if any(energy_values) else 1)
    figures["energy_totals"] = _save_figure(
        fig,
        os.path.join(output_dir, "energy_totals.png"),
    )

    if reward_history:
        smooth = _ema(reward_values)

        fig, ax = plt.subplots(figsize=(_W2, _H * 2))
        ax.plot(reward_steps, reward_values, color=_PPO_COLOR, lw=0.6, alpha=0.35, label="Raw reward")
        ax.plot(reward_steps, smooth, color=_PPO_COLOR, lw=1.8, label="EMA (α=0.3)")
        ax.axhline(0.0, color="k", lw=0.5, ls="--", alpha=0.4)
        ax.set_title(f"PPO Reward vs Steps\n{title}")
        ax.set_xlabel("Simulation Step")
        ax.set_ylabel("Avg. PPO Reward")
        ax.legend()
        figures["ppo_reward_vs_steps"] = _save_figure(
            fig,
            os.path.join(output_dir, "ppo_reward_vs_steps.png"),
        )

        fig, ax = plt.subplots(figsize=(_W2, _H * 2))
        ax.plot(reward_times, reward_values, color=_PPO_COLOR, lw=0.6, alpha=0.35, label="Raw reward")
        ax.plot(reward_times, smooth, color=_PPO_COLOR, lw=1.8, label="EMA (α=0.3)")
        ax.axhline(0.0, color="k", lw=0.5, ls="--", alpha=0.4)
        ax.set_title(f"PPO Reward vs Time\n{title}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Avg. PPO Reward")
        ax.legend()
        figures["ppo_reward_vs_time"] = _save_figure(
            fig,
            os.path.join(output_dir, "ppo_reward_vs_time.png"),
        )

    backend = matplotlib.get_backend()
    shown = show and _backend_supports_show(backend)
    if shown:
        plt.show(block=block)
        if not block:
            _present_open_figures()
            pump_plot_events()
    else:
        plt.close("all")

    return {
        "figure_paths": figures,
        "shown": shown,
        "backend": backend,
    }


def save_and_plot_experiment(
    experiment: dict,
    out_root: str | None = None,
    show: bool = True,
    block: bool = False,
) -> dict:
    """Save one experiment bundle then generate/show its plots."""
    saved = save_experiment(experiment, out_root=out_root)
    plotted = plot_experiment(
        saved["experiment"],
        saved["experiment_dir"],
        show=show,
        block=block,
    )
    saved.update(plotted)
    return saved


def plot_saved_experiment(
    pickle_path: str,
    out_root: str | None = None,
    show: bool = True,
    block: bool = True,
) -> dict:
    """Reload an experiment pickle and regenerate its figures."""
    experiment = load_experiment(pickle_path)
    output_dir = os.path.dirname(os.path.abspath(pickle_path))
    if out_root is not None:
        output_dir = save_experiment(experiment, out_root=out_root)["experiment_dir"]
    plotted = plot_experiment(experiment, output_dir, show=show, block=block)
    return {
        "experiment": experiment,
        "experiment_dir": output_dir,
        "pickle_path": os.path.abspath(pickle_path),
        **plotted,
    }
