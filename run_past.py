"""Plot results from previously saved experiments in out/."""

import os
import sys

import config
from dl.experiment import plot_saved_experiment


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


def list_experiments(out_root: str | None = None) -> list[str]:
    """Return all experiment folder names available in out/."""
    root = os.path.abspath(out_root or config.OUT_DIR)
    if not os.path.isdir(root):
        return []
    return sorted(
        name for name in os.listdir(root)
        if os.path.isfile(os.path.join(root, name, "experiment.pkl"))
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        experiments = list_experiments()
        if not experiments:
            print(f"No saved experiments found in {config.OUT_DIR}")
            sys.exit(1)
        print(f"Available experiments in {config.OUT_DIR}:")
        for i, name in enumerate(experiments):
            print(f"  [{i}] {name}")
        choice = input("\nEnter index or folder name: ").strip()
        try:
            folder = experiments[int(choice)]
        except (ValueError, IndexError):
            folder = choice
    else:
        folder = sys.argv[1]

    plot_past(folder, block=True)
