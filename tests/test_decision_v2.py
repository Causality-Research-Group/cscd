"""Unit tests for local IDA effect estimation (cscd/decision_v2.py)."""

import unittest

import networkx as nx
import numpy as np
import pandas as pd

from cscd.data import generate_random_dag, simulate_linear_sem
from cscd.decision import true_total_effect
from cscd.decision_v2 import (
    aggregate_effect,
    bma_dag_samples,
    decision_task_v2,
    local_ida_estimates,
)
from cscd.mixed_graph import ARROW, TAIL, empty_marks, from_dag_adjacency, set_edge


def _weighted_dag(names, edges, rng):
    """Build a labelled 0/1 adjacency DataFrame and weight matrix for `edges`."""
    p = len(names)
    B = pd.DataFrame(0, index=names, columns=names, dtype=int)
    for a, b in edges:
        B.loc[a, b] = 1
    W = pd.DataFrame(0.0, index=names, columns=names)
    for a, b in edges:
        W.loc[a, b] = rng.choice([-1.0, 1.0]) * rng.uniform(0.8, 1.5)
    return B, W


def _simulate(names, edges, n, rng):
    """Simulate a linear Gaussian SEM over `names` with the given directed `edges`."""
    B, W = _weighted_dag(names, edges, rng)
    dag = nx.DiGraph()
    dag.add_nodes_from(names)
    dag.add_edges_from(edges)
    order = list(nx.topological_sort(dag))

    data = pd.DataFrame(0.0, index=range(n), columns=names)
    for node in order:
        parents = [a for a, b in edges if b == node]
        signal = np.zeros(n)
        for pa in parents:
            signal += W.loc[pa, node] * data[pa].values
        data[node] = signal + rng.normal(size=n)
    return data, B, W


class TestLocalIdaTrueDag(unittest.TestCase):
    """Local IDA on the fully-oriented true DAG recovers the true effect."""

    def setUp(self):
        self.rng = np.random.default_rng(0)
        # U is a confounder of A -> C: U -> A, U -> C, A -> C.
        self.names = ["U", "A", "B", "C"]
        self.edges = [("U", "A"), ("U", "C"), ("A", "B"), ("B", "C")]
        self.data, self.B, self.W = _simulate(self.names, self.edges, n=20000, rng=self.rng)

    def test_recovers_true_effect_with_adjustment(self):
        marks = from_dag_adjacency(self.B)
        estimates = local_ida_estimates(self.data, marks, "A", "C")
        # The true DAG is fully oriented, so A has one definite parent (U) and
        # no ambiguous siblings: there should be exactly one estimate.
        self.assertEqual(len(estimates), 1)

        Wnp = self.W.values
        theta = true_total_effect(Wnp, self.names.index("A"), self.names.index("C"))
        self.assertAlmostEqual(estimates[0], theta, delta=0.05)

    def test_outcome_as_parent_gives_zero_effect(self):
        # Effect of C on U: U is a definite parent of C, so the "effect" of
        # C on U under that orientation must be reported as 0.
        marks = from_dag_adjacency(self.B)
        estimates = local_ida_estimates(self.data, marks, "C", "U")
        self.assertTrue(all(e == 0.0 for e in estimates))

    def test_aggregate_mean_and_min_abs(self):
        estimates = [1.0, -0.2, 0.5]
        self.assertAlmostEqual(aggregate_effect(estimates, "mean"), np.mean(estimates))
        self.assertAlmostEqual(aggregate_effect(estimates, "min_abs"), -0.2)
        self.assertTrue(np.isnan(aggregate_effect([], "mean")))


class TestLocalIdaUndirectedSibling(unittest.TestCase):
    """An undirected edge t--c (c a confounder) yields both estimates."""

    def setUp(self):
        self.rng = np.random.default_rng(1)
        # C is a confounder of T -> Y: C -> T, C -> Y, T -> Y.
        self.names = ["T", "C", "Y"]
        self.edges = [("C", "T"), ("C", "Y"), ("T", "Y")]
        self.data, self.B, self.W = _simulate(self.names, self.edges, n=20000, rng=self.rng)

    def test_ambiguous_sibling_gives_adjusted_and_unadjusted_estimates(self):
        marks = empty_marks(self.names)
        set_edge(marks, "T", "C", TAIL, TAIL)  # T -- C, undirected (ambiguous)
        set_edge(marks, "T", "Y", TAIL, ARROW)  # T -> Y, definite

        estimates = local_ida_estimates(self.data, marks, "T", "Y")
        self.assertEqual(len(estimates), 2)

        Wnp = self.W.values
        theta = true_total_effect(Wnp, self.names.index("T"), self.names.index("Y"))

        adjusted = min(estimates, key=lambda e: abs(e - theta))
        unadjusted = max(estimates, key=lambda e: abs(e - theta))
        self.assertAlmostEqual(adjusted, theta, delta=0.05)
        # The unadjusted (confounded) estimate should be meaningfully biased.
        self.assertGreater(abs(unadjusted - theta), 0.1)


class TestDecisionTaskV2(unittest.TestCase):
    def setUp(self):
        # A denser random DAG makes confounded (treatment, outcome) pairs
        # (needed by every test below) very likely; try a few seeds to avoid
        # a flaky, structure-dependent skip.
        p = 8
        for seed in range(10):
            rng = np.random.default_rng(seed)
            B = generate_random_dag(p, expected_degree=3.0, rng=rng)
            data, W = simulate_linear_sem(B, n=5000, noise="gaussian", rng=rng, return_weights=True)
            var_names = list(data.columns)
            true_graph_adj = pd.DataFrame(B, index=var_names, columns=var_names)
            from cscd.decision import _descendants, _is_confounded

            has_confounded = any(
                _is_confounded(true_graph_adj, t, y)
                for t in var_names
                for y in _descendants(true_graph_adj, t)
            )
            if has_confounded:
                break
        self.rng = np.random.default_rng(seed + 1000)
        self.B, self.data, self.W = B, data, W
        self.var_names = var_names
        self.true_graph_adj = true_graph_adj

    def test_runs_and_oracle_bias_is_small(self):
        true_marks = from_dag_adjacency(self.true_graph_adj)
        graph_sets = {
            "single": [true_marks],
            "pool": [true_marks, true_marks],
        }
        result = decision_task_v2(
            self.data, self.W, self.var_names, self.true_graph_adj,
            graph_sets, self.rng, n_pairs=5,
        )
        if result["dec2_n_pairs"] == 0:
            self.skipTest("No confounded pairs sampled for this random DAG/seed.")

        self.assertFalse(np.isnan(result["dec2_bias_oracle"]))
        self.assertLess(result["dec2_bias_oracle"], 0.2)
        self.assertIn("dec2_bias_single", result)
        self.assertIn("dec2_cover_single", result)
        self.assertIn("dec2_width_single", result)
        self.assertFalse(np.isnan(result["dec2_bias_naive"]))

    def test_empty_graph_sets_give_nan(self):
        result = decision_task_v2(
            self.data, self.W, self.var_names, self.true_graph_adj,
            {"empty_set": []}, self.rng, n_pairs=3,
        )
        self.assertTrue(np.isnan(result["dec2_bias_empty_set"]))
        self.assertTrue(np.isnan(result["dec2_cover_empty_set"]))

    def test_accepts_01_adjacency_as_graphs(self):
        graph_sets = {"adj_only": [self.true_graph_adj]}
        result = decision_task_v2(
            self.data, self.W, self.var_names, self.true_graph_adj,
            graph_sets, self.rng, n_pairs=5,
        )
        if result["dec2_n_pairs"] == 0:
            self.skipTest("No confounded pairs sampled for this random DAG/seed.")
        self.assertFalse(np.isnan(result["dec2_bias_adj_only"]))


class TestBmaDagSamples(unittest.TestCase):
    def test_returns_acyclic_dags_of_correct_shape(self):
        rng = np.random.default_rng(7)
        p = 5
        B = generate_random_dag(p, expected_degree=2.0, rng=rng)
        data = simulate_linear_sem(B, n=500, noise="gaussian", rng=rng)

        samples = bma_dag_samples(
            data, rng, n_samples=10, n_iterations=200, max_parents=2, max_candidates=4
        )
        self.assertGreater(len(samples), 0)
        for adj in samples:
            self.assertEqual(list(adj.index), list(data.columns))
            self.assertEqual(list(adj.columns), list(data.columns))
            self.assertTrue(set(np.unique(adj.values)).issubset({0, 1}))
            dag = nx.DiGraph()
            dag.add_nodes_from(adj.index)
            for a in adj.index:
                for b in adj.columns:
                    if a != b and adj.loc[a, b] == 1:
                        dag.add_edge(a, b)
            self.assertTrue(nx.is_directed_acyclic_graph(dag))


if __name__ == "__main__":
    unittest.main()
