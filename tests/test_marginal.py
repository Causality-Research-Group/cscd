"""Unit tests for marginalization and the self-compatibility distance.

Run from the repository root with:

    python -m unittest discover -s tests -v
"""

import unittest

import networkx as nx
import numpy as np
import pandas as pd

from cscd.config import DomainKnowledge
from cscd.mixed_graph import (
    ARROW,
    CIRCLE,
    NO_EDGE,
    TAIL,
    canonical_dag,
    dag_from_adjacency,
    dag_marginal_mag,
    directed_adjacency,
    empty_marks,
    from_dag_adjacency,
    is_directed,
    marginal_mag,
    marginal_pag,
    set_edge,
    skeleton,
    to_adjacency,
)
from cscd.selfcompat import (
    candidate_graph_score,
    graph_distance,
    project,
    project_local_graphs,
    sample_variable_subsets,
    self_compatibility_score_among_local_graphs,
    single_edge_edit,
    uncovered_pairs,
)
from cscd.utils import apply_hard_domain_constraints


def marks_from_edges(names, directed=(), undirected=(), bidirected=(), circle_arrow=(), circle_circle=()):
    """Build an endpoint-mark graph from lists of edges."""
    marks = empty_marks(names)
    for a, b in directed:
        set_edge(marks, a, b, TAIL, ARROW)
    for a, b in undirected:
        set_edge(marks, a, b, TAIL, TAIL)
    for a, b in bidirected:
        set_edge(marks, a, b, ARROW, ARROW)
    for a, b in circle_arrow:
        set_edge(marks, a, b, CIRCLE, ARROW)
    for a, b in circle_circle:
        set_edge(marks, a, b, CIRCLE, CIRCLE)
    return marks


def induced_subgraph_skeleton(marks, keep):
    """The old, incorrect restriction: drop hidden variables and their edges."""
    return skeleton(marks.loc[keep, keep])


class TestDagMarginalization(unittest.TestCase):
    """Latent projection of DAGs onto a subset, as a MAG."""

    def test_chain_becomes_direct_edge(self):
        # The reviewer's example: A -> B -> C with B hidden must give A -> C.
        dag = marks_from_edges(["A", "B", "C"], directed=[("A", "B"), ("B", "C")])
        mag = marginal_mag(dag, ["A", "C"])
        self.assertTrue(is_directed(mag, "A", "C"))
        # The induced subgraph has no edge, which is why the old score was wrong.
        self.assertEqual(induced_subgraph_skeleton(dag, ["A", "C"]).loc["A", "C"], 0)

    def test_fork_becomes_bidirected_edge(self):
        dag = marks_from_edges(["A", "B", "C"], directed=[("B", "A"), ("B", "C")])
        mag = marginal_mag(dag, ["A", "C"])
        self.assertEqual(mag.loc["A", "C"], ARROW)
        self.assertEqual(mag.loc["C", "A"], ARROW)

    def test_collider_gives_no_edge(self):
        dag = marks_from_edges(["A", "B", "C"], directed=[("A", "B"), ("C", "B")])
        mag = marginal_mag(dag, ["A", "C"])
        self.assertEqual(mag.loc["A", "C"], NO_EDGE)

    def test_inducing_path_adds_an_edge(self):
        # X -> Z <- L -> Y and Z -> Y.  Conditioning on Z opens X -> Z <- L -> Y,
        # so X and Y cannot be separated once L is hidden.
        dag = marks_from_edges(
            ["X", "Z", "Y", "L"],
            directed=[("X", "Z"), ("L", "Z"), ("L", "Y"), ("Z", "Y")],
        )
        mag = marginal_mag(dag, ["X", "Z", "Y"])
        self.assertTrue(is_directed(mag, "X", "Y"))
        self.assertTrue(is_directed(mag, "X", "Z"))
        self.assertTrue(is_directed(mag, "Z", "Y"))

    def test_keeping_every_variable_returns_the_dag(self):
        rng = np.random.default_rng(0)
        names = [f"V{i}" for i in range(8)]
        adj = pd.DataFrame(np.triu(rng.random((8, 8)) < 0.4, k=1).astype(int), index=names, columns=names)
        mag = marginal_mag(from_dag_adjacency(adj), names)
        pd.testing.assert_frame_equal(directed_adjacency(mag), adj)

    def test_true_observed_graph_uses_marginal(self):
        from cscd.data import true_observed_graph

        full = pd.DataFrame(0, index=["X0", "L0", "X1"], columns=["X0", "L0", "X1"])
        full.loc["X0", "L0"] = 1
        full.loc["L0", "X1"] = 1
        truth = true_observed_graph(full, ["X0", "X1"])
        self.assertEqual(truth.loc["X0", "X1"], 1)
        self.assertEqual(truth.loc["X1", "X0"], 0)


class TestEquivalenceClassMarginalization(unittest.TestCase):
    """Marginals must not depend on the member chosen from a CPDAG or PAG."""

    def test_cpdag_members_share_marginal_skeleton_and_pag(self):
        names = ["A", "B", "C"]
        members = [
            marks_from_edges(names, directed=[("A", "B"), ("B", "C")]),
            marks_from_edges(names, directed=[("C", "B"), ("B", "A")]),
            marks_from_edges(names, directed=[("B", "A"), ("B", "C")]),
        ]
        cpdag = marks_from_edges(names, undirected=[("A", "B"), ("B", "C")])
        reference_skeleton = project(cpdag, ["A", "C"], "skeleton")
        reference_pag = project(cpdag, ["A", "C"], "pag")
        for member in members:
            pd.testing.assert_frame_equal(project(member, ["A", "C"], "skeleton"), reference_skeleton)
            pd.testing.assert_frame_equal(project(member, ["A", "C"], "pag"), reference_pag)
        self.assertEqual(reference_skeleton.loc["A", "C"], 1)

    def test_canonical_dag_of_cpdag_adds_no_collider(self):
        # A -- B -- C -- D -- A with chord A -- C: a chordal chain component.
        names = ["A", "B", "C", "D"]
        cpdag = marks_from_edges(
            names, undirected=[("A", "B"), ("B", "C"), ("C", "D"), ("D", "A"), ("A", "C")]
        )
        dag, latents = canonical_dag(cpdag)
        self.assertEqual(latents, [])
        self.assertTrue(nx.is_directed_acyclic_graph(dag))
        for v in dag.nodes:
            parents = list(dag.predecessors(v))
            for i, p in enumerate(parents):
                for q in parents[i + 1:]:
                    self.assertTrue(dag.has_edge(p, q) or dag.has_edge(q, p), f"collider at {v}")

    def test_bidirected_edge_becomes_latent_confounder(self):
        pag = marks_from_edges(["A", "B"], bidirected=[("A", "B")])
        dag, latents = canonical_dag(pag)
        self.assertEqual(len(latents), 1)
        self.assertEqual(set(dag.successors(latents[0])), {"A", "B"})

    def test_pag_collider_marginal_has_no_edge(self):
        pag = marks_from_edges(["A", "B", "C"], circle_arrow=[("A", "B"), ("C", "B")])
        self.assertEqual(project(pag, ["A", "C"], "skeleton").loc["A", "C"], 0)

    def test_marginal_pag_orients_collider_through_hidden_mediator(self):
        # X -> M -> Z <- Y with M hidden: the PAG over {X, Z, Y} is X o-> Z <-o Y.
        dag = marks_from_edges(
            ["X", "M", "Z", "Y"], directed=[("X", "M"), ("M", "Z"), ("Y", "Z")]
        )
        pag = marginal_pag(dag, ["X", "Z", "Y"])
        self.assertEqual(pag.loc["X", "Z"], ARROW)
        self.assertEqual(pag.loc["Y", "Z"], ARROW)
        self.assertEqual(pag.loc["Z", "X"], CIRCLE)
        self.assertEqual(pag.loc["Z", "Y"], CIRCLE)
        self.assertEqual(pag.loc["X", "Y"], NO_EDGE)

    def test_invalid_graph_with_cycle_is_repaired(self):
        cyclic = marks_from_edges(["A", "B", "C"], directed=[("A", "B"), ("B", "C"), ("C", "A")])
        dag, _ = canonical_dag(cyclic)
        self.assertTrue(nx.is_directed_acyclic_graph(dag))


class TestDistance(unittest.TestCase):

    def test_identical_graphs_have_zero_distance(self):
        g = skeleton(marks_from_edges(["A", "B", "C"], directed=[("A", "B")]))
        self.assertEqual(graph_distance(g, g, "skeleton"), 0.0)

    def test_skeleton_distance_counts_pairs(self):
        names = ["A", "B", "C"]
        full = skeleton(marks_from_edges(names, directed=[("A", "B"), ("B", "C"), ("A", "C")]))
        empty = skeleton(empty_marks(names))
        self.assertEqual(graph_distance(full, empty, "skeleton"), 1.0)

    def test_pag_distance_charges_half_per_endpoint(self):
        names = ["A", "B"]
        forward = marks_from_edges(names, directed=[("A", "B")])
        backward = marks_from_edges(names, directed=[("B", "A")])
        circles = marks_from_edges(names, circle_circle=[("A", "B")])
        self.assertEqual(graph_distance(forward, backward, "pag"), 1.0)
        self.assertEqual(graph_distance(forward, circles, "pag"), 1.0)
        self.assertEqual(graph_distance(forward, forward, "pag"), 0.0)
        self.assertEqual(graph_distance(forward, backward, "skeleton"), 0.0)

    def test_distance_is_order_invariant(self):
        g = skeleton(marks_from_edges(["A", "B", "C"], directed=[("A", "B")]))
        h = skeleton(marks_from_edges(["A", "B", "C"], directed=[("B", "C")]))
        h_shuffled = h.loc[["C", "A", "B"], ["C", "A", "B"]]
        self.assertEqual(graph_distance(g, h, "skeleton"), graph_distance(g, h_shuffled, "skeleton"))


class TestSelfCompatibilityOfTheTruth(unittest.TestCase):
    """If every local graph is the true marginal, the true graph scores zero."""

    def setUp(self):
        rng = np.random.default_rng(1)
        self.names = [f"X{i}" for i in range(10)]
        order = rng.permutation(10)
        adj = pd.DataFrame(0, index=self.names, columns=self.names)
        for i in range(10):
            for j in range(i + 1, 10):
                if rng.random() < 0.3:
                    adj.iloc[order[i], order[j]] = 1
        self.truth = from_dag_adjacency(adj)
        self.subsets = sample_variable_subsets(self.names, m=8, subset_frac=0.5, rng=rng)

    def oracle_local_graphs(self):
        # What a consistent base algorithm returns on each subset in the limit.
        return {j: marginal_mag(self.truth, s) for j, s in enumerate(self.subsets)}

    def test_true_graph_scores_zero(self):
        local = self.oracle_local_graphs()
        for mode in ("skeleton", "pag"):
            projections = project_local_graphs(local, self.subsets, mode)
            score = candidate_graph_score(self.truth, projections, self.subsets, mode)
            self.assertEqual(score, 0.0, mode)

    def test_oracle_local_graphs_are_mutually_compatible(self):
        local = self.oracle_local_graphs()
        for mode in ("skeleton", "pag"):
            self.assertEqual(self_compatibility_score_among_local_graphs(local, self.subsets, mode), 0.0)

    def test_induced_subgraph_would_penalize_the_truth(self):
        # Sanity check that the fix matters on this graph: some oracle local
        # graph has an edge that the induced subgraph of the truth lacks.
        local = self.oracle_local_graphs()
        disagreements = [
            graph_distance(induced_subgraph_skeleton(self.truth, s), skeleton(local[j]), "skeleton")
            for j, s in enumerate(self.subsets)
        ]
        self.assertGreater(max(disagreements), 0.0)


class TestSubsetsAndCandidates(unittest.TestCase):

    def test_subsets_cover_every_pair(self):
        rng = np.random.default_rng(3)
        names = [f"X{i}" for i in range(20)]
        subsets = sample_variable_subsets(names, m=3, subset_frac=0.3, rng=rng)
        self.assertEqual(uncovered_pairs(names, subsets), set())
        self.assertGreater(len(subsets), 3)
        self.assertTrue(all(len(s) == len(set(s)) == 6 for s in subsets))

    def test_single_edge_edit_changes_one_pair(self):
        rng = np.random.default_rng(4)
        g = marks_from_edges(["A", "B", "C", "D"], directed=[("A", "B"), ("B", "C")], undirected=[("C", "D")])
        for _ in range(20):
            edited = single_edge_edit(g, rng)
            changed = {tuple(sorted((a, b))) for a in g.index for b in g.index if g.loc[a, b] != edited.loc[a, b]}
            self.assertEqual(len(changed), 1)


class TestHardConstraints(unittest.TestCase):

    def knowledge(self, forbidden=(), required=(), order=None):
        return DomainKnowledge(order=order or {}, forbidden_edges=set(forbidden), required_edges=set(required))

    def test_forbidden_direction_keeps_adjacency_of_definite_edge(self):
        g = marks_from_edges(["A", "B"], directed=[("A", "B")])
        out = apply_hard_domain_constraints(g, self.knowledge(forbidden=[("A", "B")]))
        self.assertEqual((out.loc["B", "A"], out.loc["A", "B"]), (ARROW, ARROW))

    def test_legacy_deletion_of_definite_edge(self):
        g = marks_from_edges(["A", "B"], directed=[("A", "B")])
        out = apply_hard_domain_constraints(g, self.knowledge(forbidden=[("A", "B")]), delete_forbidden_definite=True)
        self.assertEqual(out.loc["A", "B"], NO_EDGE)

    def test_forbidden_direction_orients_undirected_edge(self):
        g = marks_from_edges(["A", "B"], undirected=[("A", "B")])
        out = apply_hard_domain_constraints(g, self.knowledge(forbidden=[("A", "B")]))
        self.assertTrue(is_directed(out, "B", "A"))

    def test_forbidden_direction_turns_circle_arrow_into_bidirected(self):
        g = marks_from_edges(["A", "B"], circle_arrow=[("A", "B")])
        out = apply_hard_domain_constraints(g, self.knowledge(forbidden=[("A", "B")]))
        self.assertEqual((out.loc["B", "A"], out.loc["A", "B"]), (ARROW, ARROW))

    def test_order_constraint(self):
        g = marks_from_edges(["A", "B"], undirected=[("A", "B")])
        out = apply_hard_domain_constraints(g, self.knowledge(order={"A": 0, "B": 1}))
        self.assertTrue(is_directed(out, "A", "B"))

    def test_required_edge(self):
        g = empty_marks(["A", "B"])
        out = apply_hard_domain_constraints(g, self.knowledge(required=[("A", "B")]))
        self.assertTrue(is_directed(out, "A", "B"))


class TestConversions(unittest.TestCase):

    def test_to_adjacency_keeps_old_convention(self):
        g = marks_from_edges(["A", "B", "C"], directed=[("A", "B")], undirected=[("B", "C")])
        adj = to_adjacency(g)
        self.assertEqual(adj.loc["A", "B"], 1)
        self.assertEqual(adj.loc["B", "A"], 0)
        self.assertEqual(adj.loc["B", "C"], 1)
        self.assertEqual(adj.loc["C", "B"], 1)

    def test_from_causallearn(self):
        from causallearn.graph.Edge import Edge
        from causallearn.graph.Endpoint import Endpoint
        from causallearn.graph.GeneralGraph import GeneralGraph
        from causallearn.graph.GraphNode import GraphNode

        from cscd.mixed_graph import from_causallearn

        nodes = [GraphNode(f"X{i + 1}") for i in range(3)]
        g = GeneralGraph(nodes)
        g.add_edge(Edge(nodes[0], nodes[1], Endpoint.TAIL, Endpoint.ARROW))
        g.add_edge(Edge(nodes[1], nodes[2], Endpoint.CIRCLE, Endpoint.ARROW))
        marks = from_causallearn(g, ["A", "B", "C"])
        self.assertTrue(is_directed(marks, "A", "B"))
        self.assertEqual((marks.loc["C", "B"], marks.loc["B", "C"]), (CIRCLE, ARROW))

    def test_dag_from_adjacency_roundtrip(self):
        adj = pd.DataFrame([[0, 1], [0, 0]], index=["A", "B"], columns=["A", "B"])
        self.assertEqual(list(dag_from_adjacency(adj).edges), [("A", "B")])
        mag = dag_marginal_mag(dag_from_adjacency(adj), ["A", "B"])
        self.assertTrue(is_directed(mag, "A", "B"))


if __name__ == "__main__":
    unittest.main()
