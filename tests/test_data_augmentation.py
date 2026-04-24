import unittest
from types import SimpleNamespace
from unittest import mock

import config


class _FakeDataset:
    def __init__(self, root, train, download, transform):
        del root, download
        self.train = bool(train)
        self.transform = transform
        if self.train:
            self.targets = [idx % 2 for idx in range(40)]
        else:
            self.targets = [idx % 2 for idx in range(20)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return idx, self.targets[idx]


class DataAugmentationTests(unittest.TestCase):
    def test_dataset_default_train_transforms_match_expected_recipes(self):
        try:
            from dl import data as data_module
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
            raise unittest.SkipTest(f"optional dependency missing: {exc}")

        expected_prefixes = {
            "MNIST": ["RandomCrop", "ToTensor", "Normalize"],
            "FEMNIST": ["RandomCrop", "ToTensor", "Normalize"],
            "CIFAR10": ["RandomCrop", "RandomHorizontalFlip", "ToTensor", "Normalize"],
            "CIFAR100": ["RandomCrop", "RandomHorizontalFlip", "ToTensor", "Normalize"],
        }

        with mock.patch.object(config, "TRAIN_AUGMENTATION_POLICY", "dataset_default"):
            for dataset_name, expected_names in expected_prefixes.items():
                transform = data_module._get_train_transform(dataset_name)
                self.assertEqual(
                    [type(step).__name__ for step in transform.transforms],
                    expected_names,
                )

    def test_partition_dataset_reuses_indices_but_separates_train_and_eval_views(self):
        try:
            from dl import data as data_module
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
            raise unittest.SkipTest(f"optional dependency missing: {exc}")

        with (
            mock.patch.dict(data_module._BUILDERS, {"CIFAR10": _FakeDataset}, clear=False),
            mock.patch.object(config, "TRAIN_AUGMENTATION_POLICY", "dataset_default"),
            mock.patch.object(config, "VALIDATION_FRACTION", 0.25),
        ):
            train_loaders, train_eval_loaders, val_loaders, local_test_loaders, _ = (
                data_module.partition_dataset("CIFAR10", n_vehicles=4, batch_size=8)
            )

        for train_loader, train_eval_loader, val_loader, local_test_loader in zip(
            train_loaders,
            train_eval_loaders,
            val_loaders,
            local_test_loaders,
        ):
            train_subset = train_loader.dataset
            train_eval_subset = train_eval_loader.dataset
            val_subset = val_loader.dataset
            local_test_subset = local_test_loader.dataset

            self.assertEqual(train_subset.indices, train_eval_subset.indices)
            self.assertIsNot(train_subset.dataset, train_eval_subset.dataset)
            self.assertIs(train_eval_subset.dataset, val_subset.dataset)

            train_transform_names = [
                type(step).__name__
                for step in train_subset.dataset.transform.transforms
            ]
            eval_transform_names = [
                type(step).__name__
                for step in train_eval_subset.dataset.transform.transforms
            ]
            test_transform_names = [
                type(step).__name__
                for step in local_test_subset.dataset.transform.transforms
            ]

            self.assertEqual(
                train_transform_names,
                ["RandomCrop", "RandomHorizontalFlip", "ToTensor", "Normalize"],
            )
            self.assertEqual(eval_transform_names, ["ToTensor", "Normalize"])
            self.assertEqual(test_transform_names, ["ToTensor", "Normalize"])

    def test_export_experiment_records_train_augmentation_policy(self):
        try:
            from dl.env import DLEnvironment
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
            raise unittest.SkipTest(f"optional dependency missing: {exc}")

        vehicle = SimpleNamespace(
            id=0,
            sumo_id="mv_0",
            tr_rounds=1,
            current_loss=0.5,
            current_acc=0.75,
            loss_hist=[0.5],
            acc_hist=[0.75],
            reward_hist=[],
            round_time_hist=[0.1],
            computation_energy_hist=[0.01],
            get_energy_snapshot=lambda: {
                "computation_energy_j": 0.01,
                "sidelink_tx_energy_j": 0.0,
                "internet_tx_energy_j": 0.0,
            },
        )
        dummy_env = SimpleNamespace(
            get_progress_snapshot=lambda: {
                "eval_loss": 0.6,
                "eval_loss_std": 0.0,
                "eval_acc": 0.7,
                "eval_acc_std": 0.0,
                "test_loss": 0.6,
                "test_loss_std": 0.0,
                "test_acc": 0.7,
                "test_acc_std": 0.0,
                "rounds_to_target": None,
                "wall_time_to_target_s": None,
                "energy_to_target_j": None,
                "elapsed_time": 1.0,
                "stop_reason": "done",
            },
            algo=SimpleNamespace(),
            algo_config={},
            eval_split="test",
            eval_label="Test",
            evaluation_mode="personalized",
            reward_source=None,
            async_eval=True,
            train_history=[],
            eval_history=[],
            reward_history=[],
            tr_round=1,
            global_loss=0.5,
            global_acc=0.75,
            vehicles=[vehicle],
            _collect_energy_totals=lambda: {
                "computation_energy_j": 0.01,
                "sidelink_tx_energy_j": 0.0,
                "internet_tx_energy_j": 0.0,
                "total_tx_energy_j": 0.0,
            },
        )

        with mock.patch.object(config, "TRAIN_AUGMENTATION_POLICY", "dataset_default"):
            experiment = DLEnvironment.export_experiment(dummy_env, {"algorithm": "DANTE"})

        self.assertEqual(
            experiment["config"]["TRAIN_AUGMENTATION_POLICY"],
            "dataset_default",
        )


if __name__ == "__main__":
    unittest.main()
