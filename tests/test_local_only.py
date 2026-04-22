import unittest

from algorithms.local_only.algorithm import LocalOnlyAlgorithm


class LocalOnlyAlgorithmTests(unittest.TestCase):
    def test_select_neighbors_always_returns_empty_collaboration(self):
        algo = LocalOnlyAlgorithm()

        connections, alphas, link_types, transition = algo.select_neighbors(None, [], None)

        self.assertEqual(connections, set())
        self.assertEqual(alphas, {})
        self.assertEqual(link_types, {})
        self.assertIsNone(transition)


if __name__ == "__main__":
    unittest.main()
