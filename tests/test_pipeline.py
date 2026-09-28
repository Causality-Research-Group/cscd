"""Tests for the revision pipeline helpers and record analysis."""

import unittest

import numpy as np
import pandas as pd

from cscd.analysis import claim_metrics, mb_error_bound_threshold, pr_summary, stability_graph
from cscd.mixed_graph import from_dag_adjacency, skeleton
from cscd.pipeline import (
    add_pair_coverage,
    markov_blanket_subsets,
    separation_cover,
    weighted_skeleton_distance,
)
from cscd.scenarios import make_scenario
from cscd.selfcompat import uncovered_pairs


def _adj(names, edges):
    A = pd.DataFrame(0, index=names, columns=names, dtype=int)
    for a, b in edges:
        A.loc[a, b] = 1
    return A


class TestWeightedDistance(unittest.TestCase):
    def test_extra_and_missing_weights(self):
        names = ["A", "B", "C"]
        P = skeleton(from_dag_adjacency(_adj(names, [("A", "B"), ("B", "C")])))
        Q = skeleton(from_dag_adjacency(_adj(names, [("A", "B")])))
        # One extra edge (B-C) in P out of three pairs.
        self.assertAlmostEqual(weighted_skeleton_distance(P, Q, 1.0, 1.0), 1 / 3)
        self.assertAlmostEqual(weighted_skeleton_distance(P, Q, 2.0, 1.0), 2 / 6)
        self.assertAlmostEqual(weighted_skeleton_distance(P, Q, 1.0, 2.0), 1 / 6)
        self.assertEqual(weighted_skeleton_distance(P, P, 2.0, 1.0), 0.0)


class TestSubsets(unittest.TestCase):
    def test_separation_cover_chain(self):
        names = ["A", "B", "C"]
        marks = from_dag_adjacency(_adj(names, [("A", "B"), ("B", "C")]))
        # A, C non-adjacent; only a subset containing B separates them.
        self.assertEqual(separation_cover(marks, names, [["A", "C"]]), 0.0)
        self.assertEqual(separation_cover(marks, names, [["A", "B", "C"]]), 1.0)

    def test_markov_blanket_subsets_contain_blanket(self):
        names = ["A", "B", "C", "D", "E"]
        A = _adj(names, [("A", "C"), ("B", "C"), ("C", "D")])
        subsets = markov_blanket_subsets(A, 4, np.random.default_rng(0))
        self.assertEqual(len(subsets), len(names))
        self.assertTrue({"A", "B", "C"} <= set(subsets[0]))  # spouse B of A
        self.assertTrue(all(len(s) == 4 for s in subsets))
        covered = add_pair_coverage(names, subsets, 4, np.random.default_rng(0))
        self.assertEqual(uncovered_pairs(names, covered), set())


class TestAnalysis(unittest.TestCase):
    def test_claim_metrics_and_pr(self):
        truth = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        freq = np.array([[0, 1.0, 0.5], [0, 0, 0.95], [0, 0.2, 0]])
        m = claim_metrics(freq, truth, 0.9)
        self.assertEqual(m["precision"], 1.0)
        self.assertEqual(m["recall"], 1.0)
        self.assertAlmostEqual(m["unresolved"], 2 / 6)  # 0.5 and 0.2
        pr = pr_summary(freq, truth)
        self.assertAlmostEqual(pr["avg_precision"], 1.0)
        self.assertEqual(pr["prec_at_rec0.8"], 1.0)

    def test_stability_graph_and_bound(self):
        boot = np.zeros((10, 3, 3), dtype=int)
        boot[:, 0, 1] = 1
        boot[:7, 1, 2] = 1
        A = stability_graph(boot, 0.6)
        self.assertEqual(A[0, 1], 1)
        self.assertEqual(A[1, 2], 1)
        self.assertEqual(stability_graph(boot, 0.8)[1, 2], 0)
        thr = mb_error_bound_threshold(boot, ev=1.0)
        self.assertTrue(0.5 < thr <= 1.0)


class TestScenarios(unittest.TestCase):
    def test_hidden_hubs_and_quadratic(self):
        sc = make_scenario("linear", 12, 200, 100, np.random.default_rng(1), n_latent=3, latent_mode="hubs")
        self.assertEqual(sc.data.shape, (200, 9))
        self.assertEqual(sc.test_data.shape, (100, 9))
        self.assertIsNone(sc.weights)
        sc = make_scenario("quadratic", 8, 300, 0, np.random.default_rng(2), quadratic_frac=1.0)
        self.assertTrue(np.allclose(sc.data.std(ddof=0).values, 1.0, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
