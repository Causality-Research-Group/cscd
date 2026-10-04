"""Unit tests for the external falsification detectors.

Run from the repository root with:

    python -m unittest tests.test_detectors -v
"""

import unittest
import warnings

import networkx as nx
import numpy as np
import pandas as pd

from cscd.config import ExperimentConfig
from cscd.detectors import (
    dag_to_cpdag_marks,
    eulig_permutation_test,
    faller_kappa_g,
    latent_projection,
    lovo_edge_might_exist,
    lovo_path_adjustment_set,
    lovo_scores,
    marks_to_observed_dag,
    structural_hamming_distance,
    three_step_lovo_predictor,
)
from cscd.mixed_graph import ARROW, TAIL, dag_marginal_pag, empty_marks, set_edge
from cscd.scenarios import make_scenario

NAMES = ["A", "B", "C", "D", "E"]
# A -> B -> C <- D, C -> E: a chain into a collider with a descendant.
EDGES = [("A", "B"), ("B", "C"), ("D", "C"), ("C", "E")]
SUBSETS = [["A", "C", "D", "E"], ["B", "C", "E"], ["A", "D", "E"], ["A", "B", "D", "E"], ["A", "E", "B"]]


def true_dag() -> nx.DiGraph:
    dag = nx.DiGraph()
    dag.add_nodes_from(NAMES)
    dag.add_edges_from(EDGES)
    return dag


def directed_marks(names, edges) -> pd.DataFrame:
    marks = empty_marks(names)
    for a, b in edges:
        set_edge(marks, a, b, TAIL, ARROW)
    return marks


class TestGraphHelpers(unittest.TestCase):

    def test_latent_projection(self):
        directed, bidirected = latent_projection(true_dag(), ["A", "C", "E"])
        self.assertEqual(directed, {("A", "C"), ("C", "E")})
        self.assertEqual(bidirected, set())
        # Hiding the common cause C of a two-child fork gives a bidirected edge.
        fork = nx.DiGraph([("C", "X"), ("C", "Y")])
        directed, bidirected = latent_projection(fork, ["X", "Y"])
        self.assertEqual(directed, set())
        self.assertEqual(bidirected, {("X", "Y")})

    def test_cpdag_of_dag(self):
        cpdag = dag_to_cpdag_marks(NAMES, set(EDGES))
        # v-structure B -> C <- D and, by Meek R1, C -> E; A - B stays undirected.
        self.assertEqual((cpdag.loc["B", "C"], cpdag.loc["C", "B"]), (ARROW, TAIL))
        self.assertEqual((cpdag.loc["C", "E"], cpdag.loc["E", "C"]), (ARROW, TAIL))
        self.assertEqual((cpdag.loc["A", "B"], cpdag.loc["B", "A"]), (TAIL, TAIL))

    def test_shd_counts_pairs(self):
        P = directed_marks(["A", "B", "C"], [("A", "B")])
        Q = directed_marks(["A", "B", "C"], [("B", "A"), ("B", "C")])
        self.assertEqual(structural_hamming_distance(P, Q), 2)
        self.assertEqual(structural_hamming_distance(P, P), 0)


class TestFallerKappaG(unittest.TestCase):

    def test_zero_for_exact_cpdag_projections(self):
        dag = true_dag()
        full = dag_to_cpdag_marks(NAMES, set(EDGES))
        local = [dag_to_cpdag_marks(s, latent_projection(dag, s)[0]) for s in SUBSETS]
        self.assertEqual(faller_kappa_g(full, local, SUBSETS, "pc"), 0.0)
        self.assertEqual(faller_kappa_g(full, local, SUBSETS, "ges"), 0.0)

    def test_zero_for_exact_pag_projections(self):
        dag = true_dag()
        full = dag_marginal_pag(dag, NAMES)
        local = [dag_marginal_pag(dag, s) for s in SUBSETS]
        self.assertEqual(faller_kappa_g(full, local, SUBSETS, "fci"), 0.0)

    def test_positive_for_wrong_full_graph(self):
        dag = true_dag()
        wrong_edges = {("A", "B"), ("C", "B"), ("D", "E")}
        wrong = dag_to_cpdag_marks(NAMES, wrong_edges)
        local = [dag_to_cpdag_marks(s, latent_projection(dag, s)[0]) for s in SUBSETS]
        self.assertGreater(faller_kappa_g(wrong, local, SUBSETS, "pc"), 0.0)
        local_pag = [dag_marginal_pag(dag, s) for s in SUBSETS]
        wrong_pag = dag_marginal_pag(nx.DiGraph(list(wrong_edges)), NAMES)
        self.assertGreater(faller_kappa_g(wrong_pag, local_pag, SUBSETS, "fci"), 0.0)


def strong_linear_data(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 6))
    X[:, 0] = rng.normal(size=n)
    X[:, 1] = 0.8 * X[:, 0] + rng.normal(size=n)
    X[:, 2] = 0.8 * X[:, 0] + rng.normal(size=n)
    X[:, 3] = 0.8 * X[:, 1] + 0.8 * X[:, 2] + rng.normal(size=n)
    X[:, 4] = 0.8 * X[:, 3] + rng.normal(size=n)
    X[:, 5] = 0.8 * X[:, 4] + rng.normal(size=n)
    return pd.DataFrame(X, columns=[f"X{i}" for i in range(6)])


TRUE_EDGES_6 = [("X0", "X1"), ("X0", "X2"), ("X1", "X3"), ("X2", "X3"), ("X3", "X4"), ("X4", "X5")]


class TestEuligPermutationTest(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")
        self.data = strong_linear_data()
        self.names = list(self.data.columns)

    def test_true_dag_not_falsified(self):
        result = eulig_permutation_test(self.data, directed_marks(self.names, TRUE_EDGES_6), 100,
                                        np.random.default_rng(1))
        self.assertFalse(result["falsified"])
        self.assertTrue(result["falsifiable"])
        self.assertLessEqual(result["p_value_lmc"], 0.05)
        self.assertLess(result["n_violations"], result["perm_violations_mean"])

    def test_wrong_dag_falsified(self):
        wrong = [("X0", "X5"), ("X3", "X5"), ("X2", "X1"), ("X4", "X1"), ("X1", "X5")]
        result = eulig_permutation_test(self.data, directed_marks(self.names, wrong), 100,
                                        np.random.default_rng(1))
        self.assertTrue(result["falsified"])
        self.assertGreater(result["p_value_lmc"], 0.05)

    def test_empty_graph_not_falsifiable(self):
        # Every permutation of the empty graph is the empty graph itself.
        result = eulig_permutation_test(self.data, empty_marks(self.names), 20, np.random.default_rng(1))
        self.assertEqual(result["n_violations"], result["n_tests"])
        self.assertFalse(result["falsifiable"])
        self.assertTrue(result["not_better_than_random"])

    def test_reproducible_and_restores_global_rng(self):
        np.random.seed(7)
        before = np.random.get_state()[1].copy()
        marks = directed_marks(self.names, TRUE_EDGES_6)
        a = eulig_permutation_test(self.data, marks, 30, np.random.default_rng(3))
        b = eulig_permutation_test(self.data, marks, 30, np.random.default_rng(3))
        self.assertEqual(a, b)
        np.testing.assert_array_equal(np.random.get_state()[1], before)

    def test_bidirected_edge_becomes_acyclic_orientation(self):
        marks = directed_marks(["A", "B", "C"], [("A", "B"), ("B", "C")])
        set_edge(marks, "A", "C", ARROW, ARROW)
        dag = marks_to_observed_dag(marks)
        self.assertTrue(nx.is_directed_acyclic_graph(dag))
        self.assertEqual(set(dag.nodes), {"A", "B", "C"})
        self.assertTrue(dag.has_edge("A", "C"))


class TestLOVO(unittest.TestCase):

    def test_three_step_predictor_on_chain(self):
        rng = np.random.default_rng(0)
        n = 20000
        x = rng.normal(size=n)
        z = 0.8 * x + rng.normal(size=n)
        y = 0.8 * z + rng.normal(size=n)
        df = pd.DataFrame({"X": x, "Z": z, "Y": y})
        df = (df - df.mean()) / df.std()
        rho = float(np.corrcoef(x, y)[0, 1])
        pred = three_step_lovo_predictor("X", "Y", df[["X", "Z"]], df[["Y", "Z"]], ["Z"])
        self.assertAlmostEqual(pred, rho, delta=0.02)
        self.assertEqual(three_step_lovo_predictor("X", "Y", df[["X", "Z"]], df[["Y", "Z"]], []), 0.0)

    def test_lemma7_and_path_adjustment(self):
        # G: X -> Z1 <- W -> Z2 <- Y.  Leaving out Y keeps X -> Z1 <- W -> Z2,
        # and Z1 is not a neighbor of Y in the graph without X, so X - Y is excluded.
        g_x = dag_to_cpdag_marks(["X", "Z1", "W", "Z2"], {("X", "Z1"), ("W", "Z1"), ("W", "Z2")})
        g_y = dag_to_cpdag_marks(["Y", "Z1", "W", "Z2"], {("W", "Z1"), ("W", "Z2"), ("Y", "Z2")})
        self.assertFalse(lovo_edge_might_exist("X", "Y", g_x, g_y, "cpdag"))
        self.assertEqual(lovo_path_adjustment_set("X", "Y", g_x, g_y, "cpdag"), ["W", "Z1", "Z2"])
        # A common possible child of X and Y forces abstention.
        g_x2 = directed_marks(["X", "M"], [("X", "M")])
        g_y2 = directed_marks(["Y", "M"], [("Y", "M")])
        self.assertIsNone(lovo_path_adjustment_set("X", "Y", g_x2, g_y2, "cpdag"))

    def test_lovo_small_and_reproducible_for_pc(self):
        warnings.simplefilter("ignore")
        data = make_scenario("linear", 8, 3000, 0, np.random.default_rng(1)).data
        config = ExperimentConfig()
        a = lovo_scores(data, "pc", config, np.random.default_rng(0), n_pairs=28)
        b = lovo_scores(data, "pc", config, np.random.default_rng(0), n_pairs=28)
        self.assertGreater(a["n_evaluated"], 0)
        self.assertTrue(np.isfinite(a["lovo_loss"]))
        self.assertLess(a["lovo_loss"], 0.1)
        self.assertLessEqual(a["lovo_loss"], a["baseline_loss"] + 0.02)
        self.assertEqual(a["lovo_loss"], b["lovo_loss"])
        np.testing.assert_array_equal(a["pair_losses"], b["pair_losses"])
        self.assertEqual(len(a["pairs"]), 28)
        self.assertEqual(len(set(a["pairs"])), 28)


if __name__ == "__main__":
    unittest.main()
