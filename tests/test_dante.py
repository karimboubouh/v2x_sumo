import threading
import unittest

import numpy as np

import config

try:
    import torch
    from torch.distributions import Bernoulli

    from algorithms.base import LINK_INTERNET, LINK_SIDELINK
    from algorithms.dante.algorithm import DANTEAlgorithm
    from algorithms.dante.config import SELF_WEIGHT_END, SELF_WEIGHT_START
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
        self.current_reward_loss = self.current_val_loss
        self.current_reward_acc = self.current_val_acc
        self._prev_reward_loss = self._prev_val_loss
        self._prev_reward_acc = self.current_val_acc
        self.reward_hist = []
        self.computation_energy_hist = []
        self.last_update_vec = np.ones((4,), dtype=np.float32)
        self._shared_weights = {
            "w": torch.ones((4,), dtype=torch.float32) * float(vid + 1)
        }
        self._ref_weights = {
            "w": torch.zeros((4,), dtype=torch.float32)
        }
        self._shared_update = {
            "w": self._shared_weights["w"] - self._ref_weights["w"]
        }

    def get_energy_snapshot(self) -> dict:
        return {
            "computation_energy_j": 0.05,
            "sidelink_tx_energy_j": 0.0,
            "internet_tx_energy_j": 0.0,
            "total_tx_energy_j": 0.0,
        }

    def own_features(self) -> np.ndarray:
        return np.zeros((5,), dtype=np.float32)

    def get_update_vec(self) -> np.ndarray:
        return self.last_update_vec.copy()

    def get_grad_vec(self) -> np.ndarray:
        return self.get_update_vec()

    def get_shared_weights(self) -> dict:
        return {key: value.clone() for key, value in self._shared_weights.items()}

    def get_shared_update(self) -> dict:
        return {key: value.clone() for key, value in self._shared_update.items()}


def _pending_transition(algo, selected_neighbors, **overrides):
    transition = {
        "own_state": np.zeros((algo.own_feature_dim,), dtype=np.float32),
        "nbr_features": np.zeros((0, algo.neighbor_feature_dim), dtype=np.float32),
        "action": np.zeros((0,), dtype=np.float32),
        "log_prob": 0.0,
        "value": 0.0,
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


class _SelectEnv:
    def __init__(self, vehicles, feature_rows):
        self.vehicles = vehicles
        self._feature_rows = np.asarray(feature_rows, dtype=np.float32)

    def neighbor_features(self, v, nbrs):
        del v, nbrs
        return self._feature_rows.copy()

    def _vehicle_is_done(self, v):
        del v
        return False


class DanteAlgorithmTests(unittest.TestCase):
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

    def test_build_candidates_reuses_retained_peer_without_pairwise_internet_range_gate(self):
        algo = DANTEAlgorithm()
        vehicle = _DummyVehicle(0, pos=(0.0, 0.0))
        peer_sl = _DummyVehicle(1, pos=(50.0, 0.0))
        peer_retained = _DummyVehicle(2, pos=(5000.0, 0.0))
        peer_untrusted = _DummyVehicle(3, pos=(6000.0, 0.0))
        vehicles = [vehicle, peer_sl, peer_retained, peer_untrusted]
        env = _CandidateEnv(vehicles, [(peer_sl, 50.0, LINK_SIDELINK)])
        algo._retained_neighbors[vehicle.id] = {
            peer_retained.id: {"last_good_round": 3},
            peer_untrusted.id: {"last_good_round": 4},
        }
        algo._trust_scores[vehicle.id] = {
            peer_retained.id: 1.0,
            peer_untrusted.id: 0.0,
        }
        algo._peer_robustness[vehicle.id] = {
            peer_retained.id: {"last_robust_score": 1.0, "last_robust_pass": True},
            peer_untrusted.id: {"last_robust_score": 1.0, "last_robust_pass": True},
        }

        candidates = algo.build_candidates(vehicle, env)

        self.assertIn((peer_sl, 50.0, LINK_SIDELINK), candidates)
        self.assertTrue(
            any(item[0].id == peer_retained.id and item[2] == LINK_INTERNET for item in candidates)
        )
        self.assertFalse(any(item[0].id == peer_untrusted.id for item in candidates))

    def test_reward_matches_weighted_normalized_learning_gain_minus_cost(self):
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

        self.assertAlmostEqual(reward, 0.21, places=6)

    def test_self_weight_schedule_decays_from_start_to_end(self):
        algo = DANTEAlgorithm()

        first = algo._resolve_self_weight(0, 2)
        last = algo._resolve_self_weight(algo._max_rounds - 1, 2)
        middle = algo._resolve_self_weight((algo._max_rounds - 1) // 2, 2)

        self.assertAlmostEqual(first, SELF_WEIGHT_START, places=6)
        self.assertAlmostEqual(last, SELF_WEIGHT_END, places=6)
        self.assertGreater(first, middle)
        self.assertGreater(middle, last)

    def test_bernoulli_joint_log_prob_is_sum_over_feasible_peers(self):
        algo = DANTEAlgorithm()
        algo.setup([_DummyVehicle(0)])
        agent = algo._agents[0]
        own_state = np.zeros((algo.own_feature_dim,), dtype=np.float32)
        nbr_features = np.array(
            [
                [0.6, 0.2, 0.2, LINK_SIDELINK, 1.0, 1.0],
                [0.5, 0.4, 0.2, LINK_INTERNET, 0.7, 1.0],
            ],
            dtype=np.float32,
        )
        action = torch.tensor([1.0, 0.0], dtype=torch.float32)
        own_t = torch.as_tensor(own_state, dtype=torch.float32)
        nbr_t = torch.as_tensor(nbr_features, dtype=torch.float32)
        out = agent.policy(own_t, nbr_t)

        log_prob, _, _ = agent.policy.evaluate_actions(own_t, nbr_t, action)
        expected = Bernoulli(logits=out["selector_logits"]).log_prob(action).sum()

        self.assertAlmostEqual(float(log_prob.item()), float(expected.item()), places=6)

    def test_positive_sampled_feasible_peers_survive_final_admissibility(self):
        algo = DANTEAlgorithm()
        vehicle = _DummyVehicle(0)
        algo.setup([vehicle])
        algo.round_energy_budget_j = algo._round_compute_energy_j + 0.04
        algo._budget_state[vehicle.id] = {
            "remaining_energy_j": algo.round_energy_budget_j,
            "remaining_bandwidth_bits": algo.round_bandwidth_budget_bits,
            "last_latency_slack_ratio": 1.0,
        }
        peer_a = _DummyVehicle(1, pos=(10.0, 0.0))
        peer_b = _DummyVehicle(2, pos=(12.0, 0.0))
        env = _SelectEnv(
            [vehicle, peer_a, peer_b],
            [
                [0.9, 0.0, 0.0, LINK_SIDELINK, 1.0, 1.0],
                [0.8, 0.0, 0.0, LINK_SIDELINK, 1.0, 1.0],
            ],
        )
        agent = algo._agents[vehicle.id]
        agent.act = lambda own_state, nbr_features: {
            "own_state": own_state.astype(np.float32, copy=True),
            "nbr_features": nbr_features.astype(np.float32, copy=True),
            "action": np.ones((2,), dtype=np.float32),
            "value": 0.0,
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

    def test_selection_keeps_positive_alignment_new_peer_without_history_bonus(self):
        algo = DANTEAlgorithm()
        vehicle = _DummyVehicle(0)
        algo.setup([vehicle])
        peer_a = _DummyVehicle(1, pos=(10.0, 0.0))
        peer_b = _DummyVehicle(2, pos=(12.0, 0.0))
        env = _SelectEnv(
            [vehicle, peer_a, peer_b],
            [
                [-0.6, 0.1, 0.1, LINK_SIDELINK, 1.0, 1.0],
                [0.7, 0.1, 0.1, LINK_SIDELINK, 1.0, 0.0],
            ],
        )
        agent = algo._agents[vehicle.id]
        agent.act = lambda own_state, nbr_features: {
            "own_state": own_state.astype(np.float32, copy=True),
            "nbr_features": nbr_features.astype(np.float32, copy=True),
            "action": np.ones((2,), dtype=np.float32),
            "value": 0.0,
            "selector_logits": np.zeros((2,), dtype=np.float32),
            "selector_prob": np.array([0.7, 0.6], dtype=np.float32),
            "attention": np.array([0.6, 0.4], dtype=np.float32),
        }

        connections, _, _, transition = algo.select_neighbors(
            vehicle,
            [(peer_a, 10.0, LINK_SIDELINK), (peer_b, 12.0, LINK_SIDELINK)],
            env,
        )

        self.assertEqual(connections, {2})
        self.assertListEqual(transition["action"].tolist(), [0.0, 1.0])

    def test_forced_probe_selects_one_positive_sidelink_when_actor_picks_none(self):
        algo = DANTEAlgorithm()
        vehicle = _DummyVehicle(0)
        algo.setup([vehicle])
        peer_a = _DummyVehicle(1, pos=(10.0, 0.0))
        peer_b = _DummyVehicle(2, pos=(12.0, 0.0))
        env = _SelectEnv(
            [vehicle, peer_a, peer_b],
            [
                [0.9, 0.1, 0.1, LINK_SIDELINK, 1.0, 1.0],
                [-0.4, 0.1, 0.1, LINK_SIDELINK, 1.0, 1.0],
            ],
        )
        agent = algo._agents[vehicle.id]
        agent.act = lambda own_state, nbr_features: {
            "own_state": own_state.astype(np.float32, copy=True),
            "nbr_features": nbr_features.astype(np.float32, copy=True),
            "action": np.zeros((2,), dtype=np.float32),
            "value": 0.0,
            "selector_logits": np.zeros((2,), dtype=np.float32),
            "selector_prob": np.array([0.4, 0.3], dtype=np.float32),
            "attention": np.array([0.6, 0.4], dtype=np.float32),
        }

        connections, _, _, transition = algo.select_neighbors(
            vehicle,
            [(peer_a, 10.0, LINK_SIDELINK), (peer_b, 12.0, LINK_SIDELINK)],
            env,
        )

        self.assertEqual(connections, {1})
        self.assertListEqual(transition["action"].tolist(), [1.0, 0.0])

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

    def test_unhelpful_new_peer_is_not_retained(self):
        algo = DANTEAlgorithm()
        vehicle = _DummyVehicle(0)
        algo.setup([vehicle])
        agent = algo._agents[vehicle.id]
        vehicle.tr_rounds = 1
        vehicle._prev_val_loss = 1.0
        vehicle.current_val_loss = 1.1
        agent.store_pending(
            _pending_transition(
                algo,
                [{"nid": 7, "link_type": LINK_SIDELINK}],
                candidate_counts={"sl": 1, "in": 0, "total": 1},
            ),
            target_round=1,
        )
        algo._round_feedback[vehicle.id] = {
            "neighbors": {7: {"robust_pass": False, "alpha": 0.0, "robust_score": 0.2}},
            "fallback": True,
        }

        algo.post_step([vehicle], transitions={}, step_n=1)

        self.assertNotIn(7, algo._retained_neighbors[vehicle.id])

    def test_skipped_retained_peer_stays_retained(self):
        algo = DANTEAlgorithm()
        vehicle = _DummyVehicle(0)
        algo.setup([vehicle])
        agent = algo._agents[vehicle.id]
        algo._retained_neighbors[vehicle.id] = {7: {"last_good_round": 1}}
        algo._peer_robustness[vehicle.id] = {7: {"last_robust_score": 1.0, "last_robust_pass": True}}
        vehicle.tr_rounds = 2
        vehicle._prev_val_loss = 1.0
        vehicle.current_val_loss = 0.9
        agent.store_pending(
            _pending_transition(
                algo,
                [],
                eligible_retained_ids=[7],
                action=np.zeros((1,), dtype=np.float32),
                nbr_features=np.zeros((1, algo.neighbor_feature_dim), dtype=np.float32),
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
        algo._trust_scores[vehicle.id] = {7: 0.8}
        algo._peer_robustness[vehicle.id] = {7: {"last_robust_score": 1.0, "last_robust_pass": True}}
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
        self.assertLess(algo.get_trust_score(vehicle.id, 7), 0.8)

    def test_trust_update_uses_binary_phi(self):
        algo = DANTEAlgorithm()
        vehicle = _DummyVehicle(0)
        algo.setup([vehicle])
        agent = algo._agents[vehicle.id]
        algo._trust_scores[vehicle.id] = {7: 0.8}
        vehicle.tr_rounds = 3
        vehicle._prev_val_loss = 1.0
        vehicle.current_val_loss = 0.7
        agent.store_pending(
            _pending_transition(
                algo,
                [{"nid": 7, "link_type": LINK_SIDELINK}],
            ),
            target_round=3,
        )
        algo._round_feedback[vehicle.id] = {
            "neighbors": {7: {"robust_pass": True, "alpha": 0.5, "robust_score": 1.0}},
            "fallback": False,
        }

        algo.post_step([vehicle], transitions={}, step_n=1)

        expected_phi = 0.3
        expected = (1.0 - algo.trust_smoothing) * 0.8 + algo.trust_smoothing * expected_phi
        self.assertAlmostEqual(algo.get_trust_score(vehicle.id, 7), expected, places=6)

    def test_trust_update_accepts_paper_validation_slack(self):
        algo = DANTEAlgorithm()
        vehicle = _DummyVehicle(0)
        algo.setup([vehicle])
        agent = algo._agents[vehicle.id]
        algo._trust_scores[vehicle.id] = {7: 0.2}
        vehicle.tr_rounds = 3
        vehicle._prev_val_loss = 1.0
        vehicle.current_val_loss = 1.0 + (algo.validation_loss_slack * 0.5)
        agent.store_pending(
            _pending_transition(
                algo,
                [{"nid": 7, "link_type": LINK_SIDELINK}],
            ),
            target_round=3,
        )
        algo._round_feedback[vehicle.id] = {
            "neighbors": {7: {"robust_pass": True, "alpha": 0.5, "robust_score": 1.0}},
            "fallback": False,
        }

        algo.post_step([vehicle], transitions={}, step_n=1)

        expected = (1.0 - algo.trust_smoothing) * 0.2
        self.assertAlmostEqual(algo.get_trust_score(vehicle.id, 7), expected, places=6)

    def test_budget_accounting_subtracts_compute_and_accepted_comm_energy(self):
        algo = DANTEAlgorithm()
        vehicle = _DummyVehicle(0)
        algo.setup([vehicle])
        agent = algo._agents[vehicle.id]
        vehicle.tr_rounds = 1
        vehicle._prev_val_loss = 1.0
        vehicle.current_val_loss = 0.8
        vehicle.computation_energy_hist.append(0.2)
        initial_energy = algo._budget_state[vehicle.id]["remaining_energy_j"]
        agent.store_pending(
            _pending_transition(
                algo,
                [{"nid": 7, "link_type": LINK_SIDELINK}],
                comm_energy_j=0.03,
                bandwidth_bits=0.0,
                latency_s=0.0,
                energy_budget_j=1.0,
            ),
            target_round=1,
        )
        algo._round_feedback[vehicle.id] = {
            "neighbors": {7: {"robust_pass": True, "alpha": 0.5, "robust_score": 1.0}},
            "fallback": False,
        }

        algo.post_step([vehicle], transitions={}, step_n=1)

        self.assertAlmostEqual(
            algo._budget_state[vehicle.id]["remaining_energy_j"],
            initial_energy - 0.23,
            places=6,
        )

    def test_fallback_next_transition_records_no_tx_cost(self):
        algo = DANTEAlgorithm()
        vehicle = _DummyVehicle(0)
        algo.setup([vehicle])
        transition = _pending_transition(
            algo,
            [{"nid": 7, "link_type": LINK_INTERNET}],
            comm_energy_j=0.2,
            bandwidth_bits=1000.0,
            latency_s=0.5,
            target_round=1,
        )
        algo._round_feedback[vehicle.id] = {"neighbors": {}, "fallback": True}

        algo.post_step([vehicle], transitions={vehicle.id: transition}, step_n=1)

        pending = algo._agents[vehicle.id].pending_transition
        self.assertEqual(pending["selected_neighbors"], [])
        self.assertEqual(pending["comm_energy_j"], 0.0)
        self.assertEqual(pending["bandwidth_bits"], 0.0)
        self.assertEqual(pending["latency_s"], 0.0)

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
                energy_budget_j=algo_b.round_energy_budget_j * 0.1,
            ),
            target_round=2,
        )
        algo_a._round_feedback[vehicle_a.id] = {"neighbors": {}, "fallback": True}
        algo_b._round_feedback[vehicle_b.id] = {"neighbors": {}, "fallback": True}

        rewards_a = algo_a.post_step([vehicle_a], transitions={}, step_n=1)
        rewards_b = algo_b.post_step([vehicle_b], transitions={}, step_n=1)

        self.assertEqual(rewards_a[vehicle_a.id], 0.0)
        self.assertEqual(rewards_b[vehicle_b.id], 0.0)
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
        self.assertIn("comm_cost", lines[0])
        self.assertIn("gain/base_reward/adv", lines[0])
        self.assertIn("selected [7:SL/new", lines[1])
        self.assertIn("retained [7(q=", lines[1])
        self.assertIn("9(q=", lines[1])
        self.assertIn("dropped_prev [5]", lines[1])

    def test_retained_candidate_ordering_uses_retention_trust_and_robustness(self):
        algo = DANTEAlgorithm()
        algo.max_internet_neighbors = 2
        vehicle = _DummyVehicle(0)
        algo.setup([vehicle])
        peer_a = _DummyVehicle(1, pos=(500.0, 0.0))
        peer_b = _DummyVehicle(2, pos=(520.0, 0.0))
        env = _CandidateEnv([vehicle, peer_a, peer_b], [])
        algo._retained_neighbors[vehicle.id] = {
            1: {"last_good_round": 3},
            2: {"last_good_round": 4},
        }
        algo._trust_scores[vehicle.id] = {1: 0.8, 2: 0.6}
        algo._peer_robustness[vehicle.id] = {
            1: {"last_robust_score": 0.9, "last_robust_pass": True},
            2: {"last_robust_score": 1.0, "last_robust_pass": True},
        }

        candidates = algo.build_candidates(vehicle, env)

        self.assertEqual([item[0].id for item in candidates[:2]], [1, 2])

    def test_neighbor_features_have_new_dimensions_and_link_type(self):
        from dl.env import DLEnvironment  # local import to avoid module side effects at import time

        algo = DANTEAlgorithm()
        self.assertEqual(algo.own_feature_dim, 5)
        self.assertEqual(algo.neighbor_feature_dim, 6)

        env = DLEnvironment.__new__(DLEnvironment)
        env.algo = algo
        vehicle = _DummyVehicle(0)
        peer = _DummyVehicle(1)
        algo.setup([vehicle])
        features = DLEnvironment.neighbor_features(
            env,
            vehicle,
            [(peer, 10.0, LINK_INTERNET)],
        )

        self.assertEqual(features.shape, (1, 6))
        self.assertAlmostEqual(float(features[0, 3]), float(LINK_INTERNET), places=6)

    def test_export_diagnostics_reports_reward_windows_and_collaboration_advantage(self):
        algo = DANTEAlgorithm()
        algo._diagnostic_totals["completed_rounds"] = 3
        algo._diagnostic_totals["retained_offered"] = 4
        algo._diagnostic_totals["retained_selected"] = 2
        algo._diagnostic_totals["retained_skipped"] = 2
        algo._diagnostic_totals["selected_total"] = 3
        algo._diagnostic_totals["selected_useful"] = 2
        algo._diagnostic_totals["retained_positive_reused"] = 1
        algo._diagnostic_totals["selection_overlap_sum"] = 1.5
        algo._diagnostic_totals["baseline_reward_sum"] = 0.12
        algo._diagnostic_totals["normalized_gain_sum"] = 0.18
        algo._diagnostic_totals["normalized_comm_cost_sum"] = 0.09
        algo._diagnostic_totals["collaboration_advantage_sum"] = 0.15
        algo._diagnostic_totals["reward_positive_count"] = 2
        algo._diagnostic_totals["usefulness_sum"] = 1.2
        algo._diagnostic_totals["usefulness_count"] = 2
        algo._reward_records = [
            {"reward": -0.1, "normalized_gain": 0.0, "normalized_comm_cost": 0.1, "collaboration_advantage": -0.05},
            {"reward": 0.1, "normalized_gain": 0.2, "normalized_comm_cost": 0.1, "collaboration_advantage": 0.08},
            {"reward": 0.2, "normalized_gain": 0.3, "normalized_comm_cost": 0.1, "collaboration_advantage": 0.18},
        ]

        diagnostics = algo.export_diagnostics()

        self.assertIn("reward_window_stats", diagnostics)
        self.assertAlmostEqual(diagnostics["late_round_mean_reward"], 0.2, places=6)
        self.assertAlmostEqual(diagnostics["late_round_collaboration_advantage"], 0.18, places=6)
        self.assertAlmostEqual(diagnostics["useful_selection_rate"], 2.0 / 3.0, places=6)
        self.assertAlmostEqual(diagnostics["retained_positive_reuse_rate"], 0.5, places=6)
        self.assertAlmostEqual(diagnostics["mean_normalized_comm_cost"], 0.03, places=6)


if __name__ == "__main__":
    unittest.main()
