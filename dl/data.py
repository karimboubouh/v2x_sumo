"""
fl/data.py — Dataset Loading & Non-IID Partitioning
=====================================================
Downloads MNIST / FEMNIST / CIFAR10 / CIFAR100 and splits them into
per-vehicle non-IID shards using a Dirichlet distribution.

Adapted from v2x_sim/fl_data.py.
"""

import config
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ── Transforms ────────────────────────────────────────────────────────────────

_EVAL_TRANSFORMS = {
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

_DATASET_DEFAULT_AUGMENTATIONS = {
    "MNIST": [
        transforms.RandomCrop(28, padding=2),
    ],
    "FEMNIST": [
        transforms.RandomCrop(28, padding=2),
    ],
    "CIFAR10": [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ],
    "CIFAR100": [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ],
}

_DATASET_DEFAULT_AUGMENTATION_DESCRIPTIONS = {
    "MNIST": "RandomCrop(28, padding=2)",
    "FEMNIST": "RandomCrop(28, padding=2)",
    "CIFAR10": "RandomCrop(32, padding=4) + RandomHorizontalFlip()",
    "CIFAR100": "RandomCrop(32, padding=4) + RandomHorizontalFlip()",
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


def _get_eval_transform(dataset_name: str):
    """Return the deterministic transform used for evaluation and test splits."""
    return _EVAL_TRANSFORMS[dataset_name]


def _get_train_transform(dataset_name: str):
    """Return the training transform based on the configured augmentation policy."""
    policy = str(getattr(config, "TRAIN_AUGMENTATION_POLICY", "none")).strip().lower()
    eval_transform = _get_eval_transform(dataset_name)
    if policy == "none":
        return eval_transform
    if policy == "dataset_default":
        return transforms.Compose([
            *_DATASET_DEFAULT_AUGMENTATIONS[dataset_name],
            *list(eval_transform.transforms),
        ])
    raise ValueError(
        "Unsupported TRAIN_AUGMENTATION_POLICY "
        f"{getattr(config, 'TRAIN_AUGMENTATION_POLICY', None)!r}. "
        "Expected 'none' or 'dataset_default'."
    )


def describe_train_augmentation(dataset_name: str) -> str:
    """Return a human-readable description of the active training augmentation policy."""
    policy = str(getattr(config, "TRAIN_AUGMENTATION_POLICY", "none")).strip().lower()
    if policy == "none":
        return "none"
    if policy == "dataset_default":
        return (
            "dataset_default: "
            f"{_DATASET_DEFAULT_AUGMENTATION_DESCRIPTIONS[dataset_name]}"
        )
    raise ValueError(
        "Unsupported TRAIN_AUGMENTATION_POLICY "
        f"{getattr(config, 'TRAIN_AUGMENTATION_POLICY', None)!r}. "
        "Expected 'none' or 'dataset_default'."
    )


def _build_dataset(
    dataset_name: str,
    *,
    root: str,
    train: bool,
    download: bool,
    transform,
):
    """Instantiate a torchvision dataset builder with a specific transform view."""
    builder = _BUILDERS[dataset_name]
    return builder(
        root=root,
        train=train,
        download=download,
        transform=transform,
    )


def _loader_generator(offset: int = 0):
    """Return a per-loader generator when global seeding is enabled."""
    seed = getattr(config, "SEED", None)
    if seed is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(offset))
    return generator


def _get_labels(dataset) -> np.ndarray:
    """Extract label array from a torchvision dataset."""
    if hasattr(dataset, "targets"):
        t = dataset.targets
        return t.numpy() if isinstance(t, torch.Tensor) else np.array(t)
    return np.array([y for _, y in dataset])


def _split_train_validation_indices(indices: list[int]) -> tuple[list[int], list[int]]:
    """Split one client's assigned indices into train/validation subsets."""
    idxs = list(indices)
    if len(idxs) <= 1:
        return idxs, list(idxs)

    n_val = int(round(len(idxs) * float(config.VALIDATION_FRACTION)))
    n_val = min(max(n_val, 1), len(idxs) - 1)
    val_indices = idxs[:n_val]
    train_indices = idxs[n_val:]
    return train_indices, val_indices


def _allocate_counts(total: int, weights: np.ndarray) -> np.ndarray:
    """Allocate an integer total across clients while preserving proportions."""
    weights = np.asarray(weights, dtype=np.float64)
    if total <= 0 or weights.size == 0:
        return np.zeros(weights.shape, dtype=np.int64)

    weights = np.clip(weights, 0.0, None)
    if float(weights.sum()) <= 0.0:
        weights = np.full(weights.shape, 1.0 / len(weights), dtype=np.float64)
    else:
        weights = weights / float(weights.sum())

    raw = weights * float(total)
    counts = np.floor(raw).astype(np.int64)
    remainder = int(total - counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        counts[order[:remainder]] += 1
    return counts


def _rebalance_empty_test_shards(
    vehicle_test_indices: list[list[int]],
    test_labels: np.ndarray,
    reference_counts: np.ndarray,
) -> None:
    """Move a few samples so every vehicle has at least one held-out test example."""
    empties = [v for v, idxs in enumerate(vehicle_test_indices) if not idxs]
    if not empties:
        return

    donors = sorted(
        range(len(vehicle_test_indices)),
        key=lambda v: len(vehicle_test_indices[v]),
        reverse=True,
    )

    for target in empties:
        preferred_classes = set(np.flatnonzero(reference_counts[target] > 0).tolist())
        moved = False
        for donor in donors:
            if donor == target or len(vehicle_test_indices[donor]) <= 1:
                continue

            donor_indices = vehicle_test_indices[donor]
            move_pos = None
            if preferred_classes:
                for pos, sample_idx in enumerate(donor_indices):
                    if int(test_labels[sample_idx]) in preferred_classes:
                        move_pos = pos
                        break
            if move_pos is None:
                move_pos = len(donor_indices) - 1

            vehicle_test_indices[target].append(donor_indices.pop(move_pos))
            moved = True
            break

        if not moved:
            raise RuntimeError("Unable to construct non-empty per-vehicle test shards.")


def _match_test_shards_to_client_distribution(
    test_labels: np.ndarray,
    client_indices: list[list[int]],
    client_labels: np.ndarray,
    n_classes: int,
) -> list[list[int]]:
    """Partition the shared test split so each client's shard mirrors its train labels."""
    reference_counts = np.zeros((len(client_indices), n_classes), dtype=np.int64)
    for vehicle_id, indices in enumerate(client_indices):
        if not indices:
            continue
        reference_counts[vehicle_id] = np.bincount(
            client_labels[np.asarray(indices, dtype=np.int64)],
            minlength=n_classes,
        )

    vehicle_test_indices = [[] for _ in range(len(client_indices))]
    for class_id in range(n_classes):
        class_test_indices = np.where(test_labels == class_id)[0].tolist()
        np.random.shuffle(class_test_indices)
        counts = _allocate_counts(
            len(class_test_indices),
            reference_counts[:, class_id].astype(np.float64),
        )

        start = 0
        for vehicle_id, count in enumerate(counts.tolist()):
            if count <= 0:
                continue
            stop = start + count
            vehicle_test_indices[vehicle_id].extend(class_test_indices[start:stop])
            start = stop

    _rebalance_empty_test_shards(vehicle_test_indices, test_labels, reference_counts)
    for indices in vehicle_test_indices:
        np.random.shuffle(indices)
    return vehicle_test_indices


def partition_dataset(dataset_name: str, n_vehicles: int,
                      alpha: float = 0.5, batch_size: int = 32,
                      data_root: str = "./data") -> tuple:
    """
    Download dataset and partition into n_vehicles non-IID DataLoaders.

    Strategy — Dirichlet partition:
        For each class c, sample proportions p_c ~ Dir(alpha)
        Then assign class-c samples to vehicles according to p_c.

    Returns:
        (
            train_loaders,
            train_eval_loaders,
            val_loaders,
            local_test_loaders,
            shared_test_loader,
        )
    """
    train_tf = _get_train_transform(dataset_name)
    eval_tf = _get_eval_transform(dataset_name)

    train_dataset = _build_dataset(
        dataset_name,
        root=data_root,
        train=True,
        download=True,
        transform=train_tf,
    )
    train_eval_dataset = _build_dataset(
        dataset_name,
        root=data_root,
        train=True,
        download=True,
        transform=eval_tf,
    )
    labels = _get_labels(train_eval_dataset)
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

    train_loaders = []
    train_eval_loaders = []
    val_loaders = []
    client_reference_indices = []
    for v in range(n_vehicles):
        indices = list(vehicle_indices[v])
        if not indices:
            fallback_count = min(50, len(train_eval_dataset))
            indices = (
                np.random.choice(len(train_eval_dataset), fallback_count, replace=False).tolist()
                if fallback_count > 0
                else []
            )
        np.random.shuffle(indices)
        client_reference_indices.append(list(indices))
        train_indices, val_indices = _split_train_validation_indices(indices)

        train_subset = Subset(train_dataset, train_indices)
        train_eval_subset = Subset(train_eval_dataset, train_indices)
        val_subset = Subset(train_eval_dataset, val_indices)

        train_loaders.append(
            DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                generator=_loader_generator(1000 + v),
            )
        )
        train_eval_loaders.append(
            DataLoader(train_eval_subset, batch_size=256, shuffle=False, drop_last=False)
        )
        val_loaders.append(
            DataLoader(val_subset, batch_size=256, shuffle=False, drop_last=False)
        )

    test_ds = _build_dataset(
        dataset_name,
        root=data_root,
        train=False,
        download=True,
        transform=eval_tf,
    )
    test_labels = _get_labels(test_ds)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    local_test_indices = _match_test_shards_to_client_distribution(
        test_labels,
        client_reference_indices,
        labels,
        n_classes,
    )
    local_test_loaders = [
        DataLoader(Subset(test_ds, indices), batch_size=256, shuffle=False, drop_last=False)
        for indices in local_test_indices
    ]

    return train_loaders, train_eval_loaders, val_loaders, local_test_loaders, test_loader


def get_n_classes(dataset_name: str) -> int:
    """Return the number of output classes for a dataset."""
    return {"MNIST": 10, "FEMNIST": 62, "CIFAR10": 10, "CIFAR100": 100}[dataset_name]
