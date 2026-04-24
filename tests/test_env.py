import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import config

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - environment-specific
    torch = None


class _DummyVehicle:
    def __init__(self, vid: int, pos):
        self.id = vid
        self.pos = np.asarray(pos, dtype=np.float32)


class _DummyTrainingVehicle:
    def __init__(self, current_acc: float, tr_rounds: int):
        self.current_acc = float(current_acc)
        self.tr_rounds = int(tr_rounds)


class _DummyModelVehicle:
    def __init__(self, vid: int, weight: float, dataset_size: int = 4):
        if torch is None:  # pragma: no cover - environment-specific
            raise RuntimeError("torch is required for model vehicle tests")

        self.id = vid
        self.model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.model.weight.fill_(float(weight))
        init_sd = {key: value.clone() for key, value in self.model.state_dict().items()}
        self._shared_weights = {key: value.clone() for key, value in init_sd.items()}
        self._ref_weights = {key: value.clone() for key, value in init_sd.items()}
        self.shared_weights_bytes = self._state_dict_nbytes(init_sd)
        self._param_vec = None
        self.train_loader = SimpleNamespace(dataset=[0] * dataset_size)

    def _state_dict_nbytes(self, state_dict: dict) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in state_dict.values()
        )

    def get_shared_weights(self) -> dict:
        return {
            key: value.clone()
            for key, value in self._shared_weights.items()
        }


class EnvironmentNeighborDiscoveryTests(unittest.TestCase):
    def test_neighbors_of_uses_unbounded_internet_and_ignores_quality_fields(self):
        try:
            from algorithms.base import LINK_INTERNET, LINK_SIDELINK
            from dl.env import DLEnvironment
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
            raise unittest.SkipTest(f"optional dependency missing: {exc}")

        env = DLEnvironment.__new__(DLEnvironment)
        env.algo = SimpleNamespace(max_sidelink_neighbors=1, max_internet_neighbors=3)
        env.vehicles = [
            _DummyVehicle(0, (0.0, 0.0)),
            _DummyVehicle(1, (10.0, 0.0)),
            _DummyVehicle(2, (20.0, 0.0)),
            _DummyVehicle(3, (5000.0, 0.0)),
            _DummyVehicle(4, (9000.0, 0.0)),
        ]

        neighbors = DLEnvironment.neighbors_of(env, env.vehicles[0])

        self.assertEqual(
            [(nbr.id, dist, link_type) for nbr, dist, link_type in neighbors],
            [
                (1, 10.0, LINK_SIDELINK),
                (2, 20.0, LINK_INTERNET),
                (3, 5000.0, LINK_INTERNET),
                (4, 9000.0, LINK_INTERNET),
            ],
        )

    def test_neighbors_of_prioritizes_nearest_internet_candidates(self):
        try:
            from algorithms.base import LINK_INTERNET, LINK_SIDELINK
            from dl.env import DLEnvironment
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
            raise unittest.SkipTest(f"optional dependency missing: {exc}")

        env = DLEnvironment.__new__(DLEnvironment)
        env.algo = SimpleNamespace(max_sidelink_neighbors=1, max_internet_neighbors=2)
        env.vehicles = [
            _DummyVehicle(0, (0.0, 0.0)),
            _DummyVehicle(1, (10.0, 0.0)),
            _DummyVehicle(2, (20.0, 0.0)),
            _DummyVehicle(3, (30.0, 0.0)),
            _DummyVehicle(4, (4000.0, 0.0)),
        ]

        neighbors = DLEnvironment.neighbors_of(env, env.vehicles[0])

        self.assertEqual(
            [(nbr.id, dist, link_type) for nbr, dist, link_type in neighbors],
            [
                (1, 10.0, LINK_SIDELINK),
                (2, 20.0, LINK_INTERNET),
                (3, 30.0, LINK_INTERNET),
            ],
        )

    def test_export_experiment_omits_removed_internet_config_keys(self):
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
                "train_loss": 0.5,
                "train_acc": 0.75,
                "eval_loss": 0.4,
                "eval_loss_std": 0.0,
                "eval_acc": 0.8,
                "eval_acc_std": 0.0,
                "eval_round": 1,
                "init_eval_loss": 0.9,
                "init_eval_loss_std": 0.0,
                "init_eval_acc": 0.5,
                "init_eval_acc_std": 0.0,
                "test_loss": 0.4,
                "test_loss_std": 0.0,
                "test_acc": 0.8,
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

        experiment = DLEnvironment.export_experiment(dummy_env, {"algorithm": "DANTE"})

        self.assertNotIn("INTERNET_RANGE", experiment["config"])
        self.assertNotIn("INTERNET_QUALITY_THRESHOLD", experiment["config"])


class EnvironmentTerminationTests(unittest.TestCase):
    def test_round_cap_remains_active_when_target_accuracy_is_enabled(self):
        try:
            from dl.env import DLEnvironment
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
            raise unittest.SkipTest(f"optional dependency missing: {exc}")

        env = DLEnvironment.__new__(DLEnvironment)
        vehicle = _DummyTrainingVehicle(current_acc=0.25, tr_rounds=3)

        with mock.patch.object(config, "STOP_ON", "train_acc"), mock.patch.object(
            config, "TARGET_ACCURACY", 0.90
        ), mock.patch.object(config, "MAX_TR_ROUNDS", 3):
            self.assertTrue(DLEnvironment._vehicle_is_done(env, vehicle))
            self.assertTrue(DLEnvironment._vehicle_reached_round_cap(env, vehicle))
            self.assertFalse(DLEnvironment._vehicle_reached_target(env, vehicle))

    def test_stop_reason_reports_mixed_target_and_round_completion(self):
        try:
            from dl.env import DLEnvironment
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
            raise unittest.SkipTest(f"optional dependency missing: {exc}")

        env = DLEnvironment.__new__(DLEnvironment)
        env.vehicles = [
            _DummyTrainingVehicle(current_acc=0.95, tr_rounds=1),
            _DummyTrainingVehicle(current_acc=0.25, tr_rounds=3),
        ]

        with mock.patch.object(config, "STOP_ON", "train_acc"), mock.patch.object(
            config, "TARGET_ACCURACY", 0.90
        ), mock.patch.object(config, "MAX_TR_ROUNDS", 3):
            reason = DLEnvironment.get_stop_reason(env)

        self.assertIn("reached target accuracy", reason)
        self.assertIn("completed 3 training rounds", reason)


class InitialModelPolicyTests(unittest.TestCase):
    def setUp(self):
        if torch is None:  # pragma: no cover - environment-specific
            self.skipTest("optional dependency missing: torch")

    def test_apply_initial_model_policy_syncs_models_when_enabled(self):
        try:
            from dl.env import DLEnvironment
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
            raise unittest.SkipTest(f"optional dependency missing: {exc}")

        env = DLEnvironment.__new__(DLEnvironment)
        env.vehicles = [_DummyModelVehicle(0, 1.0), _DummyModelVehicle(1, 2.0)]

        with mock.patch.object(config, "SHARED_INITIAL_MODEL", True):
            DLEnvironment._apply_initial_model_policy(env)

        first = env.vehicles[0].model.state_dict()["weight"]
        second = env.vehicles[1].model.state_dict()["weight"]
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(
            torch.equal(
                env.vehicles[1]._shared_weights["weight"],
                env.vehicles[0]._shared_weights["weight"],
            )
        )

    def test_apply_initial_model_policy_leaves_models_distinct_when_disabled(self):
        try:
            from dl.env import DLEnvironment
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
            raise unittest.SkipTest(f"optional dependency missing: {exc}")

        env = DLEnvironment.__new__(DLEnvironment)
        env.vehicles = [_DummyModelVehicle(0, 1.0), _DummyModelVehicle(1, 2.0)]

        with mock.patch.object(config, "SHARED_INITIAL_MODEL", False):
            DLEnvironment._apply_initial_model_policy(env)

        first = env.vehicles[0].model.state_dict()["weight"]
        second = env.vehicles[1].model.state_dict()["weight"]
        self.assertFalse(torch.equal(first, second))

    def test_pfedgraph_setup_preserves_distinct_vehicle_initial_models(self):
        try:
            from algorithms.pfedgraph.algorithm import PFedGraphAlgorithm
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
            raise unittest.SkipTest(f"optional dependency missing: {exc}")

        vehicles = [_DummyModelVehicle(0, 1.0), _DummyModelVehicle(1, 2.0)]
        algo = PFedGraphAlgorithm()
        algo.setup(vehicles)

        first = vehicles[0].model.state_dict()["weight"]
        second = vehicles[1].model.state_dict()["weight"]
        self.assertFalse(torch.equal(first, second))
        self.assertTrue(torch.equal(algo._initial_states[0]["weight"], first))
        self.assertTrue(torch.equal(algo._initial_states[1]["weight"], second))

    def test_choco_setup_preserves_distinct_vehicle_initial_models(self):
        try:
            from algorithms.choco_sgd.algorithm import CHOCOSGDAlgorithm
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
            raise unittest.SkipTest(f"optional dependency missing: {exc}")

        vehicles = [_DummyModelVehicle(0, 1.0), _DummyModelVehicle(1, 2.0)]
        algo = CHOCOSGDAlgorithm()
        algo.setup(vehicles)

        first = vehicles[0].model.state_dict()["weight"]
        second = vehicles[1].model.state_dict()["weight"]
        self.assertFalse(torch.equal(first, second))
        self.assertEqual(
            int(torch.count_nonzero(vehicles[0]._choco_public_state["weight"])),
            0,
        )
        self.assertEqual(
            int(torch.count_nonzero(vehicles[1]._choco_public_state["weight"])),
            0,
        )


if __name__ == "__main__":
    unittest.main()
