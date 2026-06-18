import threading
import unittest

import numpy as np

try:
    import torch

    from algorithms.base import LINK_INTERNET
    from algorithms.ippo.algorithm import IPPOAlgorithm
except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
    raise unittest.SkipTest(f"optional dependency missing: {exc}")


class _DummyVehicle:
    def __init__(self, vid: int):
        self.id = vid
        self.pos = np.zeros((2,), dtype=np.float32)
        self._lock = threading.Lock()
        self.training_done = threading.Event()
        self.training_done.set()
        self.tr_rounds = 0
        self.connections = set()
        self.alphas = {}
        self.link_types = {}
        self.reward_hist = []
        self._prev_reward_loss = 1.0
        self.current_reward_loss = 0.9
        self.last_update_vec = np.ones((4,), dtype=np.float32)
        self.model = torch.nn.Linear(1, 1, bias=False)
        self._shared_weights = {
            key: value.clone()
            for key, value in self.model.state_dict().items()
        }

    def own_features(self) -> np.ndarray:
        return np.zeros((5,), dtype=np.float32)

    def get_update_vec(self) -> np.ndarray:
        return self.last_update_vec.copy()

    def get_shared_weights(self) -> dict:
        return {
            key: value.clone()
            for key, value in self._shared_weights.items()
        }


class _SelectEnv:
    def __init__(self, feature_rows):
        self._feature_rows = np.asarray(feature_rows, dtype=np.float32)

    def neighbor_features(self, v, nbrs):
        del v, nbrs
        return self._feature_rows.copy()

    def _vehicle_is_done(self, v):
        del v
        return False


class IPPOAlgorithmTests(unittest.TestCase):
    def test_feature_dimensions_match_vehicle_and_environment_contract(self):
        algo = IPPOAlgorithm()

        self.assertEqual(algo.own_feature_dim, 5)
        self.assertEqual(algo.neighbor_feature_dim, 6)

    def test_select_neighbors_accepts_current_feature_shapes(self):
        algo = IPPOAlgorithm()
        vehicle = _DummyVehicle(0)
        peer = _DummyVehicle(1)
        algo.setup([vehicle, peer])
        env = _SelectEnv([[0.1, 0.2, 0.3, LINK_INTERNET, 1.0, 1.0]])

        connections, alphas, link_types, transition = algo.select_neighbors(
            vehicle,
            [(peer, 10.0, LINK_INTERNET)],
            env,
        )

        self.assertIsInstance(connections, set)
        self.assertIsInstance(alphas, dict)
        self.assertIsInstance(link_types, dict)
        self.assertIsNotNone(transition)
        self.assertEqual(transition["own_state"].shape, (5,))
        self.assertEqual(transition["nbr_features"].shape, (1, 6))


if __name__ == "__main__":
    unittest.main()
