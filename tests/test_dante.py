import threading
import unittest

import numpy as np

import config

try:
    import torch
    from algorithms.base import LINK_INTERNET, LINK_SIDELINK
    from algorithms.dante.algorithm import DANTEAlgorithm
except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
    raise unittest.SkipTest(f"optional dependency missing: {exc}")


class _DummyVehicle:
    def __init__(self, vid: int, pos=(0.0, 0.0)):
        self.id = vid
        self.pos = np.asarray(pos, dtype=np.float32)
        self._lock = threading.Lock()
        self.training_done = threading.Event()
        self.training_done.set()
        self.tr_rounds = 0
        self.connections = set()
        self.alphas = {}
        self.link_types = {}
        self.current_val_loss = 1.0
        self.current_val_acc = 0.5
        self._prev_val_loss = 1.0
        self.reward_hist = []
        self.computation_energy_hist = []
        self.last_grad_vec = np.ones((4,), dtype=np.float32)

    def get_energy_snapshot(self) -> dict:
        return {
            "computation_energy_j": 0.05,
            "sidelink_tx_energy_j": 0.0,
            "internet_tx_energy_j": 0.0,
            "total_tx_energy_j": 0.0,
        }

    def own_features(self) -> np.ndarray:
        return np.zeros((7,), dtype=np.float32)

    def get_grad_vec(self) -> np.ndarray:
        return self.last_grad_vec.copy()


def _pending_transition(algo, selected_neighbors, **overrides):
    transition = {
        "own_state": np.zeros((7,), dtype=np.float32),
        "nbr_features": np.zeros((1, 9), dtype=np.float32),
        "action": np.ones((1,), dtype=np.float32),
        "log_prob": 0.0,
        "value": 0.0,
        "actor_indices": np.zeros((1,), dtype=np.int64),
        "comm_energy_j": 0.01,
        "bandwidth_bits": algo.payload_bits,
        "latency_s": 0.01,
        "energy_budget_j": 0.1,
        "bandwidth_budget_bits": algo.payload_bits,
        "latency_budget_s": 1.0,
        "eligible_retained_ids": [],
        "candidate_counts": {"sl": 0, "in": 0, "total": 0},
        "selected_neighbors": list(selected_neighbors),
    }
    transition.update(overrides)
    return transition


class _CandidateEnv:
    def __init__(self, vehicles, sidelink):
        self.vehicles = vehicles
        self._sidelink = list(sidelink)

    def sidelink_neighbors_of(self, v):
        del v
        return list(self._sidelink)


class DanteAlgorithmTests(unittest.TestCase):
    def test_budgeted_subset_prefers_best_feasible_total_benefit(self):
        algo = DANTEAlgorithm()
        algo._budget_state[0] = {
            "remaining_energy_j": 0.10,
            "remaining_bandwidth_bits": algo.round_bandwidth_budget_bits,
            "last_latency_slack_ratio": 1.0,
        }
        proposals = [
            {
                "nid": 1,
                "energy_j": 0.039,
                "bandwidth_bits": algo.payload_bits,
                "latency_s": 0.01,
                "benefit": 9.0,
            },
            {
                "nid": 2,
                "energy_j": 0.018,
                "bandwidth_bits": algo.payload_bits,
                "latency_s": 0.01,
                "benefit": 5.0,
            },
            {
                "nid": 3,
                "energy_j": 0.018,
                "bandwidth_bits": algo.payload_bits,
                "latency_s": 0.01,
                "benefit": 5.0,
            },
        ]

        selected = algo._select_budgeted_subset(0, proposals)

        self.assertEqual({item["nid"] for item in selected}, {2, 3})

    def test_robustness_flags_large_outlier(self):
        original_byz_frac = config.BYZANTINE_FRACTION
        config.BYZANTINE_FRACTION = 0.34
        try:
            algo = DANTEAlgorithm()
            honest = {"w": torch.zeros(4)}
            outlier = {"w": torch.full((4,), 100.0)}

            distances, scores, passes = algo._robustness_scores([honest, honest, outlier])

            self.assertEqual(distances.shape[0], 3)
            self.assertTrue(bool(passes[0]))
            self.assertTrue(bool(passes[1]))
            self.assertFalse(bool(passes[2]))
            self.assertLess(float(scores[2]), float(scores[0]))
        finally:
            config.BYZANTINE_FRACTION = original_byz_frac

    def test_build_candidates_keeps_only_retained_peers_within_internet_range(self):
        algo = DANTEAlgorithm()
        vehicle = _DummyVehicle(0, pos=(0.0, 0.0))
        peer_sl = _DummyVehicle(1, pos=(50.0, 0.0))
        peer_retained = _DummyVehicle(2, pos=(500.0, 0.0))
        peer_far = _DummyVehicle(3, pos=(1500.0, 0.0))
        vehicles = [vehicle, peer_sl, peer_retained, peer_far]
        env = _CandidateEnv(
            vehicles,
            [(peer_sl, 50.0, LINK_SIDELINK)],
        )
        algo._retained_neighbors[vehicle.id] = {
            peer_retained.id: {"last_good_round": 3},
            peer_far.id: {"last_good_round": 4},
        }
        algo._trust_scores[vehicle.id] = {
            peer_retained.id: 1.0,
            peer_far.id: 1.0,
        }

        candidates = algo.build_candidates(vehicle, env)

        self.assertIn((peer_sl, 50.0, LINK_SIDELINK), candidates)
        self.assertTrue(any(item[0].id == peer_retained.id and item[2] == LINK_INTERNET for item in candidates))
        self.assertFalse(any(item[0].id == peer_far.id for item in candidates))

    def test_reward_matches_normalized_learning_gain_minus_mean_cost(self):
        algo = DANTEAlgorithm()

        reward = algo._compute_reward(
            learning_term=0.25,
            comm_energy_j=0.1,
            bandwidth_bits=100.0,
            latency_s=0.2,
            energy_budget_j=0.2,
            bandwidth_budget_bits=200.0,
            latency_budget_s=0.4,
        )

        self.assertAlmostEqual(reward, -0.25, places=6)

    def test_helpful_sidelink_peer_is_promoted_and_reused_via_internet(self):
        algo = DANTEAlgorithm()
        vehicle = _DummyVehicle(0)
        peer = _DummyVehicle(7, pos=(500.0, 0.0))
        vehicles = [vehicle] + [_DummyVehicle(idx + 1, pos=(2000.0, 0.0)) for idx in range(6)] + [peer]
        algo.setup([vehicle])
        agent = algo._agents[vehicle.id]
        vehicle.tr_rounds = 1
        vehicle._prev_val_loss = 1.0
        vehicle.current_val_loss = 0.8
        agent.store_pending(
            _pending_transition(
                algo,
                [{"nid": 7, "link_type": LINK_SIDELINK}],
                candidate_counts={"sl": 1, "in": 0, "total": 1},
            ),
            target_round=1,
        )
        algo._round_feedback[vehicle.id] = {
            "neighbors": {7: {"robust_pass": True, "alpha": 0.5, "robust_score": 1.0}},
            "fallback": False,
        }

        rewards = algo.post_step([vehicle], transitions={}, step_n=1)

        self.assertIn(vehicle.id, rewards)
        self.assertIn(7, algo._retained_neighbors[vehicle.id])
        candidates = algo.build_candidates(vehicle, _CandidateEnv(vehicles, []))
        self.assertTrue(any(item[0].id == 7 and item[2] == LINK_INTERNET for item in candidates))

    def test_skipped_retained_peer_stays_retained(self):
        algo = DANTEAlgorithm()
        vehicle = _DummyVehicle(0)
        algo.setup([vehicle])
        agent = algo._agents[vehicle.id]
        algo._retained_neighbors[vehicle.id] = {7: {"last_good_round": 1}}
        vehicle.tr_rounds = 2
        vehicle._prev_val_loss = 1.0
        vehicle.current_val_loss = 0.9
        agent.store_pending(
            _pending_transition(
                algo,
                [],
                eligible_retained_ids=[7],
                action=np.zeros((1,), dtype=np.float32),
                actor_indices=np.zeros((0,), dtype=np.int64),
                nbr_features=np.zeros((0, 9), dtype=np.float32),
                candidate_counts={"sl": 0, "in": 1, "total": 1},
            ),
            target_round=2,
        )
        algo._round_feedback[vehicle.id] = {"neighbors": {}, "fallback": True}

        algo.post_step([vehicle], transitions={}, step_n=1)

        self.assertIn(7, algo._retained_neighbors[vehicle.id])

    def test_harmful_selected_retained_peer_decays_but_stays_retained(self):
        algo = DANTEAlgorithm()
        vehicle = _DummyVehicle(0)
        algo.setup([vehicle])
        agent = algo._agents[vehicle.id]
        algo._retained_neighbors[vehicle.id] = {7: {"last_good_round": 1}}
        algo._retention_values[vehicle.id] = {7: 0.8}
        algo._trust_scores[vehicle.id] = {7: 1.0}
        vehicle.tr_rounds = 2
        vehicle._prev_val_loss = 1.0
        vehicle.current_val_loss = 1.1
        agent.store_pending(
            _pending_transition(
                algo,
                [{"nid": 7, "link_type": LINK_INTERNET}],
                eligible_retained_ids=[7],
                candidate_counts={"sl": 0, "in": 1, "total": 1},
            ),
            target_round=2,
        )
        algo._round_feedback[vehicle.id] = {
            "neighbors": {7: {"robust_pass": False, "alpha": 0.0, "robust_score": 0.2}},
            "fallback": True,
        }

        algo.post_step([vehicle], transitions={}, step_n=1)

        self.assertIn(7, algo._retained_neighbors[vehicle.id])
        self.assertLess(algo.get_trust_score(vehicle.id, 7), 1.0)
        self.assertLess(algo.get_retention_value(vehicle.id, 7), 0.8)

    def test_nonpositive_benefit_subset_returns_empty(self):
        algo = DANTEAlgorithm()
        algo._budget_state[0] = {
            "remaining_energy_j": 0.10,
            "remaining_bandwidth_bits": algo.round_bandwidth_budget_bits,
            "last_latency_slack_ratio": 1.0,
        }
        proposals = [
            {
                "nid": 1,
                "energy_j": 0.01,
                "bandwidth_bits": algo.payload_bits,
                "latency_s": 0.01,
                "benefit": 0.0,
            },
            {
                "nid": 2,
                "energy_j": 0.01,
                "bandwidth_bits": algo.payload_bits,
                "latency_s": 0.01,
                "benefit": -0.5,
            },
        ]

        selected = algo._select_budgeted_subset(0, proposals)

        self.assertEqual(selected, [])

    def test_failed_robust_peer_is_down_ranked_for_next_round(self):
        algo = DANTEAlgorithm()
        vehicle = _DummyVehicle(0)
        algo.setup([vehicle])
        agent = algo._agents[vehicle.id]
        vehicle.tr_rounds = 2
        vehicle._prev_val_loss = 1.0
        vehicle.current_val_loss = 0.95
        before = algo._proposal_benefit(0.8, 1.0, 0.9, algo.get_last_robust_score(vehicle.id, 7))
        agent.store_pending(
            _pending_transition(
                algo,
                [{"nid": 7, "link_type": LINK_SIDELINK}],
            ),
            target_round=2,
        )
        algo._round_feedback[vehicle.id] = {
            "neighbors": {7: {"robust_pass": False, "alpha": 0.0, "robust_score": 0.2}},
            "fallback": True,
        }

        algo.post_step([vehicle], transitions={}, step_n=1)

        after_score = algo.get_last_robust_score(vehicle.id, 7)
        after = algo._proposal_benefit(0.8, 1.0, 0.9, after_score)
        self.assertFalse(algo.get_last_robust_pass(vehicle.id, 7))
        self.assertLess(after_score, 1.0)
        self.assertLess(after, before)

    def test_positive_validation_gain_with_modest_cost_yields_positive_reward(self):
        algo = DANTEAlgorithm()
        reward = algo._compute_reward(
            learning_term=0.3,
            comm_energy_j=0.01,
            bandwidth_bits=0.1 * algo.round_bandwidth_budget_bits,
            latency_s=0.1 * algo.round_latency_budget_s,
            energy_budget_j=algo.round_energy_budget_j,
            bandwidth_budget_bits=algo.round_bandwidth_budget_bits,
            latency_budget_s=algo.round_latency_budget_s,
        )

        self.assertGreater(reward, 0.0)

    def test_reward_sign_is_stable_when_pending_budget_shrinks(self):
        vehicle_a = _DummyVehicle(0)
        vehicle_b = _DummyVehicle(1)
        algo_a = DANTEAlgorithm()
        algo_b = DANTEAlgorithm()
        algo_a.setup([vehicle_a])
        algo_b.setup([vehicle_b])
        vehicle_a.tr_rounds = 2
        vehicle_b.tr_rounds = 2
        vehicle_a._prev_val_loss = 1.0
        vehicle_b._prev_val_loss = 1.0
        vehicle_a.current_val_loss = 0.8
        vehicle_b.current_val_loss = 0.8
        algo_a._agents[vehicle_a.id].store_pending(
            _pending_transition(
                algo_a,
                [],
                comm_energy_j=0.0,
                bandwidth_bits=0.0,
                latency_s=0.0,
                pre_comp_energy_j=0.05,
                energy_budget_j=algo_a.round_energy_budget_j,
            ),
            target_round=2,
        )
        algo_b._agents[vehicle_b.id].store_pending(
            _pending_transition(
                algo_b,
                [],
                comm_energy_j=0.0,
                bandwidth_bits=0.0,
                latency_s=0.0,
                pre_comp_energy_j=0.05,
                energy_budget_j=algo_b.round_energy_budget_j * 0.1,
            ),
            target_round=2,
        )
        algo_a._round_feedback[vehicle_a.id] = {"neighbors": {}, "fallback": True}
        algo_b._round_feedback[vehicle_b.id] = {"neighbors": {}, "fallback": True}

        rewards_a = algo_a.post_step([vehicle_a], transitions={}, step_n=1)
        rewards_b = algo_b.post_step([vehicle_b], transitions={}, step_n=1)

        self.assertGreater(rewards_a[vehicle_a.id], 0.0)
        self.assertGreater(rewards_b[vehicle_b.id], 0.0)
        self.assertAlmostEqual(rewards_a[vehicle_a.id], rewards_b[vehicle_b.id], places=6)

    def test_debug_logs_include_retention_and_neighbor_summary(self):
        algo = DANTEAlgorithm()
        vehicle = _DummyVehicle(0)
        algo.setup([vehicle])
        agent = algo._agents[vehicle.id]
        vehicle.tr_rounds = 2
        vehicle._prev_val_loss = 1.0
        vehicle.current_val_loss = 0.7
        agent.store_pending(
            _pending_transition(
                algo,
                [
                    {"nid": 7, "link_type": LINK_SIDELINK},
                    {"nid": 9, "link_type": LINK_INTERNET},
                ],
                comm_energy_j=0.02,
                eligible_retained_ids=[9],
                candidate_counts={"sl": 2, "in": 1, "total": 3},
            ),
            target_round=2,
        )
        algo._retained_neighbors[vehicle.id] = {9: {"last_good_round": 1}}
        algo._last_selected_neighbors[vehicle.id] = {5}
        algo._round_feedback[vehicle.id] = {
            "neighbors": {
                7: {"robust_pass": True, "alpha": 0.4, "robust_score": 1.0},
                9: {"robust_pass": True, "alpha": 0.3, "robust_score": 0.9},
            },
            "fallback": False,
        }

        algo.post_step([vehicle], transitions={}, step_n=1)
        lines = algo.consume_debug_logs()

        self.assertEqual(len(lines), 2)
        self.assertIn("retained pool/offered/sel/skip", lines[0])
        self.assertIn("selected [7:SL/new", lines[1])
        self.assertIn("retained [7(q=", lines[1])
        self.assertIn("9(q=", lines[1])
        self.assertIn("dropped_prev [5]", lines[1])

    def test_retained_candidate_ordering_uses_retention_trust_and_robustness(self):
        algo = DANTEAlgorithm()
        vehicle = _DummyVehicle(0)
        algo.setup([vehicle])
        peer_a = _DummyVehicle(1, pos=(500.0, 0.0))
        peer_b = _DummyVehicle(2, pos=(520.0, 0.0))
        env = _CandidateEnv([vehicle, peer_a, peer_b], [])
        algo._retained_neighbors[vehicle.id] = {
            1: {"last_good_round": 3},
            2: {"last_good_round": 4},
        }
        algo._retention_values[vehicle.id] = {1: 0.9, 2: 0.4}
        algo._trust_scores[vehicle.id] = {1: 0.8, 2: 1.0}
        algo._peer_robustness[vehicle.id] = {
            1: {"last_robust_score": 0.9, "last_robust_pass": True},
            2: {"last_robust_score": 1.0, "last_robust_pass": True},
        }

        candidates = algo.build_candidates(vehicle, env)

        self.assertEqual([item[0].id for item in candidates], [1, 2])

    def test_select_neighbors_stores_executed_actor_mask(self):
        class _SelectEnv:
            def __init__(self):
                self.vehicles = []

            def neighbor_features(self, v, nbrs):
                del v, nbrs
                return np.array([
                    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, LINK_SIDELINK, 1.0, 0.0],
                    [0.8, 0.1, 0.0, 0.0, 0.0, 0.0, LINK_SIDELINK, 1.0, 0.0],
                ], dtype=np.float32)

            def _vehicle_is_done(self, v):
                del v
                return False

        algo = DANTEAlgorithm()
        vehicle = _DummyVehicle(0)
        algo.setup([vehicle])
        algo._budget_state[vehicle.id] = {
            "remaining_energy_j": 0.04,
            "remaining_bandwidth_bits": algo.round_bandwidth_budget_bits,
            "last_latency_slack_ratio": 1.0,
        }
        peer_a = _DummyVehicle(1, pos=(10.0, 0.0))
        peer_b = _DummyVehicle(2, pos=(12.0, 0.0))
        env = _SelectEnv()
        agent = algo._agents[vehicle.id]
        agent.act = lambda own_state, nbr_features, actor_indices=None: {
            "own_state": own_state.astype(np.float32, copy=True),
            "nbr_features": nbr_features.astype(np.float32, copy=True),
            "action": np.ones((2,), dtype=np.float32),
            "value": 0.0,
            "actor_indices": np.array([0, 1], dtype=np.int64),
            "selector_logits": np.zeros((2,), dtype=np.float32),
            "selector_prob": np.array([0.9, 0.8], dtype=np.float32),
            "attention": np.array([0.7, 0.3], dtype=np.float32),
        }
        algo._neighbor_energy_j = lambda link_type, dist: 0.03
        algo._neighbor_bandwidth_bits = lambda link_type, dist: 1.0
        algo._neighbor_latency_s = lambda link_type, dist: 0.01

        connections, _, _, transition = algo.select_neighbors(
            vehicle,
            [(peer_a, 10.0, LINK_SIDELINK), (peer_b, 12.0, LINK_SIDELINK)],
            env,
        )

        self.assertEqual(connections, {1})
        self.assertListEqual(transition["action"].tolist(), [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
