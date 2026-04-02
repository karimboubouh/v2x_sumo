"""
fl/data.py — Dataset Loading & Non-IID Partitioning
=====================================================
Downloads MNIST / FEMNIST / CIFAR10 / CIFAR100 and splits them into
per-vehicle non-IID shards using a Dirichlet distribution.

Adapted from v2x_sim/fl_data.py.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ── Transforms ────────────────────────────────────────────────────────────────

_TRANSFORMS = {
    "MNIST": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]),
    "FEMNIST": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]),
    "CIFAR10": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ]),
    "CIFAR100": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ]),
}


def _build_femnist(root: str, train: bool, download: bool, transform):
    """Use torchvision EMNIST byclass split as the FEMNIST-compatible source."""
    return datasets.EMNIST(
        root=root,
        split="byclass",
        train=train,
        download=download,
        transform=transform,
    )


_BUILDERS = {
    "MNIST": datasets.MNIST,
    "FEMNIST": _build_femnist,
    "CIFAR10": datasets.CIFAR10,
    "CIFAR100": datasets.CIFAR100,
}


def _get_labels(dataset) -> np.ndarray:
    """Extract label array from a torchvision dataset."""
    if hasattr(dataset, "targets"):
        t = dataset.targets
        return t.numpy() if isinstance(t, torch.Tensor) else np.array(t)
    return np.array([y for _, y in dataset])


def partition_dataset(dataset_name: str, n_vehicles: int,
                      alpha: float = 0.5, batch_size: int = 32,
                      data_root: str = "./data") -> tuple:
    """
    Download dataset and partition into n_vehicles non-IID DataLoaders.

    Strategy — Dirichlet partition:
        For each class c, sample proportions p_c ~ Dir(alpha)
        Then assign class-c samples to vehicles according to p_c.

    Returns:
        (list_of_train_loaders, test_loader)
    """
    tf = _TRANSFORMS[dataset_name]
    cls = _BUILDERS[dataset_name]

    full_dataset = cls(root=data_root, train=True, download=True, transform=tf)
    labels = _get_labels(full_dataset)
    n_classes = int(labels.max()) + 1

    class_indices = {c: np.where(labels == c)[0].tolist() for c in range(n_classes)}
    vehicle_indices = [[] for _ in range(n_vehicles)]

    for c in range(n_classes):
        idxs = class_indices[c]
        np.random.shuffle(idxs)
        proportions = np.random.dirichlet(np.repeat(alpha, n_vehicles))
        splits = (np.cumsum(proportions) * len(idxs)).astype(int)
        splits = np.clip(splits, 0, len(idxs))
        prev = 0
        for v, split in enumerate(splits):
            vehicle_indices[v].extend(idxs[prev:split])
            prev = split

    loaders = []
    for v in range(n_vehicles):
        indices = vehicle_indices[v]
        if not indices:
            indices = np.random.choice(len(full_dataset), 50, replace=False).tolist()
        subset = Subset(full_dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size,
                            shuffle=True, drop_last=False)
        loaders.append(loader)

    test_ds = cls(root=data_root, train=False, download=True, transform=tf)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    return loaders, test_loader


def get_n_classes(dataset_name: str) -> int:
    """Return the number of output classes for a dataset."""
    return {"MNIST": 10, "FEMNIST": 62, "CIFAR10": 10, "CIFAR100": 100}[dataset_name]
