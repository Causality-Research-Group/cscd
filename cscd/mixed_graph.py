"""Endpoint-mark graphs and their marginalization onto variable subsets.

Discovery algorithms return different graph classes: PC and GES return
CPDAGs, FCI returns PAGs.  To compare a graph over all variables V with a
graph learned on a subset S, the global graph must be *marginalized* onto S,
not restricted to the induced subgraph.  For example, A -> B -> C becomes
A -> C once B is hidden, whereas the induced subgraph on {A, C} is empty.

Representation
--------------
A graph is a square ``pandas.DataFrame`` of endpoint marks indexed by the
variable names.  ``marks.loc[a, b]`` is the mark *at b* on the edge between
a and b (the pcalg convention):

    a -> b     marks[a, b] = ARROW, marks[b, a] = TAIL
    a <-> b    marks[a, b] = ARROW, marks[b, a] = ARROW
    a -- b     marks[a, b] = TAIL,  marks[b, a] = TAIL   (CPDAG undirected)
    a o-> b    marks[a, b] = ARROW, marks[b, a] = CIRCLE
    a o-o b    marks[a, b] = CIRCLE, marks[b, a] = CIRCLE

Marginalization
---------------
Every CPDAG or PAG is first turned into one member of its equivalence class,
written as a DAG over the observed variables plus one latent node per
bidirected edge (``canonical_dag``).  The member DAG is then projected onto
S as a maximal ancestral graph (``marginal_mag``).  Two quantities of that
MAG do not depend on which member was chosen, because Markov-equivalent
graphs have the same independence model over S:

* its skeleton, used by the ``skeleton`` distance;
* its PAG, used by the ``pag`` distance (``marginal_pag``).
"""

import logging
from typing import Iterable, List, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NO_EDGE = 0
TAIL = 1
ARROW = 2
CIRCLE = 3

LATENT_PREFIX = "__latent_"


# ---------------------------------------------------------------------------
# Construction and conversion
# ---------------------------------------------------------------------------

def empty_marks(var_names: List[str]) -> pd.DataFrame:
    return pd.DataFrame(NO_EDGE, index=list(var_names), columns=list(var_names), dtype=int)


def set_edge(marks: pd.DataFrame, a: str, b: str, mark_at_a: int, mark_at_b: int) -> None:
    """Write the edge a *-* b with the given marks (in place)."""
    marks.loc[b, a] = mark_at_a
    marks.loc[a, b] = mark_at_b


def remove_edge(marks: pd.DataFrame, a: str, b: str) -> None:
    set_edge(marks, a, b, NO_EDGE, NO_EDGE)


def has_edge(marks: pd.DataFrame, a: str, b: str) -> bool:
    return marks.loc[a, b] != NO_EDGE


def is_directed(marks: pd.DataFrame, a: str, b: str) -> bool:
    """True if the edge is a -> b."""
    return marks.loc[b, a] == TAIL and marks.loc[a, b] == ARROW


def from_dag_adjacency(adj: pd.DataFrame) -> pd.DataFrame:
    """Marks of a DAG given as a 0/1 adjacency matrix (adj[a, b] = 1 means a -> b)."""
    marks = empty_marks(list(adj.index))
    for a in adj.index:
        for b in adj.columns:
            if a != b and adj.loc[a, b] == 1:
                set_edge(marks, a, b, TAIL, ARROW)
    return marks


def dag_from_adjacency(adj: pd.DataFrame) -> nx.DiGraph:
    """networkx DAG from a labelled 0/1 adjacency matrix."""
    dag = nx.DiGraph()
    dag.add_nodes_from(adj.index)
    for a in adj.index:
        for b in adj.columns:
            if a != b and adj.loc[a, b] == 1:
                dag.add_edge(a, b)
    return dag


def directed_adjacency(marks: pd.DataFrame) -> pd.DataFrame:
    """0/1 matrix with 1 only for definite directed edges a -> b."""
    names = list(marks.index)
    adj = pd.DataFrame(0, index=names, columns=names, dtype=int)
    for a in names:
        for b in names:
            if a != b and is_directed(marks, a, b):
                adj.loc[a, b] = 1
    return adj


def to_adjacency(marks: pd.DataFrame) -> pd.DataFrame:
    """Flatten marks into the 0/1 adjacency used by claims and baselines.

    A definite a -> b gives adj[a, b] = 1.  Every other edge type (undirected,
    bidirected, or with circle marks) gives adj[a, b] = adj[b, a] = 1.  This is
    the convention the rest of the code base has always used for claims.
    """
    names = list(marks.index)
    adj = pd.DataFrame(0, index=names, columns=names, dtype=int)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if not has_edge(marks, a, b):
                continue
            if is_directed(marks, a, b):
                adj.loc[a, b] = 1
            elif is_directed(marks, b, a):
                adj.loc[b, a] = 1
            else:
                adj.loc[a, b] = 1
                adj.loc[b, a] = 1
    return adj


def skeleton(marks: pd.DataFrame) -> pd.DataFrame:
    """Symmetric 0/1 matrix with 1 where the two variables are adjacent."""
    return (marks != NO_EDGE).astype(int)


def from_causallearn(graph_obj, var_names: List[str]) -> pd.DataFrame:
    """Convert a causal-learn GeneralGraph into endpoint marks.

    causal-learn names nodes X1, X2, ... unless names are passed, so nodes are
    matched by position in ``get_nodes()``.
    """
    from causallearn.graph.Endpoint import Endpoint

    def to_mark(endpoint):
        # Endpoint defines __eq__ without __hash__, so it cannot key a dict.
        if endpoint == Endpoint.TAIL:
            return TAIL
        if endpoint == Endpoint.ARROW:
            return ARROW
        if endpoint == Endpoint.CIRCLE:
            return CIRCLE
        return None

    marks = empty_marks(var_names)
    node_index = {node: i for i, node in enumerate(graph_obj.get_nodes())}
    for edge in graph_obj.get_graph_edges():
        a = var_names[node_index[edge.get_node1()]]
        b = var_names[node_index[edge.get_node2()]]
        mark_at_a = to_mark(edge.get_endpoint1())
        mark_at_b = to_mark(edge.get_endpoint2())
        if mark_at_a is None or mark_at_b is None:
            logger.debug("Skipping edge %s *-* %s with unsupported endpoints.", a, b)
            continue
        set_edge(marks, a, b, mark_at_a, mark_at_b)
    return marks


# ---------------------------------------------------------------------------
# From an equivalence class to one member DAG with latent nodes
# ---------------------------------------------------------------------------

def _maximum_cardinality_order(graph: nx.Graph) -> List[str]:
    """Maximum cardinality search order of an undirected graph.

    If the graph is chordal, orienting every edge from the earlier to the later
    node in this order creates no unshielded collider.  The undirected part of
    a CPDAG and the circle part of a PAG are chordal, so this orientation gives
    a member of the equivalence class (Meek 1995; Zhang 2008, Theorem 2).
    """
    order: List[str] = []
    weight = {node: 0 for node in graph.nodes}
    while weight:
        # Ties are broken by name so the result is deterministic.
        node = max(sorted(weight), key=lambda v: weight[v])
        order.append(node)
        del weight[node]
        for neighbour in graph.neighbors(node):
            if neighbour in weight:
                weight[neighbour] += 1
    return order


def canonical_dag(marks: pd.DataFrame) -> Tuple[nx.DiGraph, List[str]]:
    """One member DAG of the class encoded by ``marks``, with latent nodes.

    * An edge with an arrowhead at exactly one end (a -> b, a o-> b) becomes
      the directed edge a -> b.
    * An edge with arrowheads at both ends (a <-> b) becomes a new latent
      node L with L -> a and L -> b.
    * Edges without arrowheads (a -- b, a o-o b, a --o b) are oriented along a
      maximum cardinality search order.

    Finite-sample outputs are not always valid CPDAGs or PAGs, so the result
    may contain a directed cycle.  Cycles are broken by deleting one edge per
    cycle, as in Faller et al. (2024).

    Returns the DAG and the list of latent node names it added.
    """
    names = list(marks.index)
    dag = nx.DiGraph()
    dag.add_nodes_from(names)
    undetermined = nx.Graph()
    latents: List[str] = []

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if not has_edge(marks, a, b):
                continue
            arrow_at_a = marks.loc[b, a] == ARROW
            arrow_at_b = marks.loc[a, b] == ARROW
            if arrow_at_a and arrow_at_b:
                latent = f"{LATENT_PREFIX}{len(latents)}"
                latents.append(latent)
                dag.add_edge(latent, a)
                dag.add_edge(latent, b)
            elif arrow_at_b:
                dag.add_edge(a, b)
            elif arrow_at_a:
                dag.add_edge(b, a)
            else:
                undetermined.add_edge(a, b)

    position = {node: k for k, node in enumerate(_maximum_cardinality_order(undetermined))}
    for a, b in undetermined.edges:
        if position[a] < position[b]:
            dag.add_edge(a, b)
        else:
            dag.add_edge(b, a)

    while not nx.is_directed_acyclic_graph(dag):
        cycle = nx.find_cycle(dag, orientation="original")
        tail, head = cycle[0][0], cycle[0][1]
        logger.debug("Input is not a valid equivalence class; removing %s -> %s to break a cycle.", tail, head)
        dag.remove_edge(tail, head)

    return dag, latents


# ---------------------------------------------------------------------------
# Marginalization
# ---------------------------------------------------------------------------

def _d_separated(dag: nx.DiGraph, x: str, y: str, given: Set[str]) -> bool:
    """d-separation test that works across networkx versions."""
    if hasattr(nx, "is_d_separator"):
        return nx.is_d_separator(dag, {x}, {y}, set(given))
    return nx.d_separated(dag, {x}, {y}, set(given))


def dag_marginal_mag(dag: nx.DiGraph, keep: Iterable[str]) -> pd.DataFrame:
    """Maximal ancestral graph over ``keep`` of a DAG whose other nodes are latent.

    Adjacency: x and y are adjacent in the MAG iff no subset of the kept
    variables d-separates them.  It suffices to test the single set
    (An(x) u An(y)) n keep minus {x, y} (Richardson and Spirtes 2002).

    Orientation: the mark at y is a tail if y is an ancestor of x in the DAG,
    and an arrowhead otherwise.
    """
    keep = list(keep)
    keep_set = set(keep)
    ancestors = {v: nx.ancestors(dag, v) for v in keep}
    mag = empty_marks(keep)

    for i, x in enumerate(keep):
        for y in keep[i + 1:]:
            conditioning = ((ancestors[x] | ancestors[y]) & keep_set) - {x, y}
            if _d_separated(dag, x, y, conditioning):
                continue
            mark_at_x = TAIL if x in ancestors[y] else ARROW
            mark_at_y = TAIL if y in ancestors[x] else ARROW
            set_edge(mag, x, y, mark_at_x, mark_at_y)
    return mag


def marginal_mag(marks: pd.DataFrame, keep: Iterable[str]) -> pd.DataFrame:
    """MAG over ``keep`` of one member of the class encoded by ``marks``.

    Only representation-invariant parts of the result (its skeleton, its PAG)
    should be compared across graphs.
    """
    dag, _ = canonical_dag(marks)
    return dag_marginal_mag(dag, keep)


def _import_dag2pag():
    """Import causal-learn's dag2pag.

    causal-learn 0.1.3.x imports ``networkx.algorithms.d_separated``, which
    networkx 3.5 removed in favour of ``is_d_separator``.  Restore the old
    name so the import works with either networkx version.
    """
    import networkx.algorithms

    if not hasattr(networkx.algorithms, "d_separated"):
        networkx.algorithms.d_separated = nx.is_d_separator
    from causallearn.utils.DAG2PAG import dag2pag

    return dag2pag


def dag_marginal_pag(dag: nx.DiGraph, keep: Iterable[str]) -> pd.DataFrame:
    """PAG over ``keep`` of a DAG whose other nodes are latent."""
    from causallearn.graph.Dag import Dag
    from causallearn.graph.GraphNode import GraphNode

    dag2pag = _import_dag2pag()
    keep = list(keep)
    nodes = {name: GraphNode(name) for name in dag.nodes}
    cl_dag = Dag(list(nodes.values()))
    for a, b in dag.edges:
        cl_dag.add_directed_edge(nodes[a], nodes[b])
    latent_nodes = [nodes[name] for name in dag.nodes if name not in set(keep)]
    pag = dag2pag(cl_dag, islatent=latent_nodes)

    observed_names = [node.get_name() for node in pag.get_nodes()]
    return from_causallearn(pag, observed_names).loc[keep, keep]


def marginal_pag(marks: pd.DataFrame, keep: Iterable[str]) -> pd.DataFrame:
    """PAG over ``keep`` of the class encoded by ``marks``.

    This does not depend on which member of the class is used, since all
    members imply the same independences over ``keep``.
    """
    dag, _ = canonical_dag(marks)
    return dag_marginal_pag(dag, keep)
