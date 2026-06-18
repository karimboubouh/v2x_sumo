"""Runtime resource tuning for smooth dashboard operation."""

from __future__ import annotations

import os
from dataclasses import dataclass

import config


@dataclass(frozen=True)
class RuntimeTuning:
    train_workers: int
    torch_threads: int
    torch_interop_threads: int


def _positive_int(value, fallback: int) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError):
        return max(int(fallback), 1)
    return max(value, 1)


def resolve_runtime_tuning(args) -> RuntimeTuning:
    """Resolve worker/thread counts without importing torch."""
    cpu_count = os.cpu_count() or 1
    if getattr(args, "train_workers", None) is not None:
        train_workers = _positive_int(args.train_workers, config.N_TRAIN_WORKERS)
    elif getattr(args, "headless", False):
        train_workers = _positive_int(config.N_TRAIN_WORKERS, 1)
    else:
        dashboard_cap = max(1, min(4, cpu_count - 2))
        train_workers = min(_positive_int(config.N_TRAIN_WORKERS, 1), dashboard_cap)

    torch_threads = _positive_int(
        getattr(args, "torch_threads", None),
        getattr(config, "TORCH_NUM_THREADS", 1),
    )
    torch_interop_threads = _positive_int(
        getattr(args, "torch_interop_threads", None),
        getattr(config, "TORCH_NUM_INTEROP_THREADS", 1),
    )

    return RuntimeTuning(
        train_workers=train_workers,
        torch_threads=torch_threads,
        torch_interop_threads=torch_interop_threads,
    )


def configure_runtime(args) -> RuntimeTuning:
    """Apply process-level CPU caps before DL modules import torch."""
    tuning = resolve_runtime_tuning(args)
    config.N_TRAIN_WORKERS = tuning.train_workers
    config.TORCH_NUM_THREADS = tuning.torch_threads
    config.TORCH_NUM_INTEROP_THREADS = tuning.torch_interop_threads

    for env_name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[env_name] = str(tuning.torch_threads)

    if getattr(args, "dl", False):
        try:
            import torch

            torch.set_num_threads(tuning.torch_threads)
            try:
                torch.set_num_interop_threads(tuning.torch_interop_threads)
            except RuntimeError:
                # PyTorch only allows this before parallel work starts.  The env
                # caps above still protect fresh simulator runs.
                pass
        except ModuleNotFoundError:
            pass

    return tuning
