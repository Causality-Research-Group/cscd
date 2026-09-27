"""Unit tests for the baselines and the Bayesian order MCMC sampler."""

import unittest

import numpy as np
import pandas as pd

from cscd.baselines import (
    bootstrap_compatibility,
    bootstrap_threshold,
    claims_from_edge_probabilities,
    stability_selection_graph,
)
from cscd.bayes import OrderMCMC
from cscd.config import ExperimentConfig
from cscd.mixed_graph import ARROW, TAIL, empty_marks, set_edge
from cscd.selfcompat import conformal_threshold


def adjacency(names, edges):
    A = pd.DataFrame(0, index=names, columns=names, dtype=int)
    for a, b in edges:
        A.loc[a, b] = 1
    return A


class TestThresholds(unittest.TestCase):

    def test_conformal_threshold_order_statistic(self):
        scores = np.arange(1, 20) / 100.0  # N = 19
        self.assertEqual(conformal_threshold(scores, 0.10), 0.18)  # ceil(20 * 0.9) = 18
        self.assertEqual(conformal_threshold(scores, 0.05), 0.19)  # ceil(20 * 0.95) = 19
        self.assertEqual(conformal_threshold(scores[:5], 0.05), float("inf"))

    def test_bootstrap_threshold_is_quantile(self):
        self.assertAlmostEqual(bootstrap_threshold(np.array([0.0, 0.1, 0.2, 0.3, 0.4]), 0.5), 0.2)
        self.assertEqual(bootstrap_threshold(np.array([]), 0.1), float("inf"))


class TestStabilitySelection(unittest.TestCase):

    def test_keeps_frequent_pairs_and_orients_by_majority(self):
        names = ["A", "B", "C"]
        graphs = [adjacency(names, [("A", "B")])] * 3 + [adjacency(names, [("B", "A"), ("B", "C")])]
        result = stability_selection_graph(graphs, names, threshold=0.6)
        self.assertEqual(result.loc["A", "B"], 1)
        self.assertEqual(result.loc["B", "A"], 0)
        self.assertEqual(result.loc["B", "C"] + result.loc["C", "B"], 0)


class TestBootstrapCompatibility(unittest.TestCase):

    def test_identical_bootstraps_admit_only_matching_candidates(self):
        names = ["A", "B", "C"]
        chain = empty_marks(names)
        set_edge(chain, "A", "B", TAIL, ARROW)
        set_edge(chain, "B", "C", TAIL, ARROW)
        other = empty_marks(names)
        set_edge(other, "A", "C", TAIL, ARROW)
        result = bootstrap_compatibility([chain] * 4, [chain, other], names, "skeleton")
        self.assertTrue(np.all(result["reference"] == 0.0))
        self.assertEqual(result["candidate"][0], 0.0)
        self.assertGreater(result["candidate"][1], 0.0)


class TestClaims(unittest.TestCase):

    def test_claims_from_probabilities(self):
        prob = pd.DataFrame([[0.0, 0.95], [0.05, 0.0]], index=["A", "B"], columns=["A", "B"])
        claims = claims_from_edge_probabilities(prob, ExperimentConfig()).set_index(["parent", "child"])
        self.assertEqual(claims.loc[("A", "B"), "status"], "accepted")
        self.assertEqual(claims.loc[("B", "A"), "status"], "rejected")


class TestOrderMCMC(unittest.TestCase):

    def test_recovers_strong_dependencies(self):
        rng = np.random.default_rng(0)
        n = 2000
        a = rng.normal(size=n)
        b = a + 0.5 * rng.normal(size=n)
        c = b + 0.5 * rng.normal(size=n)
        d = rng.normal(size=n)
        data = pd.DataFrame({"A": a, "B": b, "C": c, "D": d})
        post = OrderMCMC(data, max_parents=2, max_candidates=3).edge_posterior(
            np.random.default_rng(1), n_iterations=1500, burn_in=300
        )
        adjacency_posterior = post + post.T
        self.assertGreater(adjacency_posterior.loc["A", "B"], 0.95)
        self.assertGreater(adjacency_posterior.loc["B", "C"], 0.95)
        self.assertLess(adjacency_posterior.loc["A", "C"], 0.1)
        self.assertLess(adjacency_posterior.loc["A", "D"], 0.1)

    def test_order_score_is_finite_for_every_order(self):
        rng = np.random.default_rng(2)
        data = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("WXYZ"))
        sampler = OrderMCMC(data, max_parents=2, max_candidates=3)
        for _ in range(5):
            self.assertTrue(np.isfinite(sampler.order_log_score(list(rng.permutation(4)))))


if __name__ == "__main__":
    unittest.main()
