import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import DomainKnowledge


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    arr = StandardScaler().fit_transform(df.values)
    return pd.DataFrame(arr, columns=df.columns, index=df.index)


def make_domain_knowledge(
    var_names: List[str],
    order: Optional[List[str]] = None,
) -> "DomainKnowledge":
    # order=None → no temporal ordering constraint (empty dict).
    # Passing order=var_names would impose Xi→Xj only when i<j by variable
    # index, which is wrong for synthetic experiments where the true causal
    # order is a random permutation of the variable indices.
    order_dict = {v: i for i, v in enumerate(order)} if order is not None else {}
    return DomainKnowledge(
        order=order_dict,
        forbidden_edges=set(),
        required_edges=set(),
    )


def restrict_knowledge(
    knowledge: "DomainKnowledge",
    subset: List[str],
) -> "DomainKnowledge":
    subset_set = set(subset)
    return DomainKnowledge(
        order={v: knowledge.order[v] for v in subset if v in knowledge.order},
        forbidden_edges={(a, b) for (a, b) in knowledge.forbidden_edges
                         if a in subset_set and b in subset_set},
        required_edges={(a, b) for (a, b) in knowledge.required_edges
                        if a in subset_set and b in subset_set},
    )


def forbids_edge(knowledge: "DomainKnowledge", parent: str, child: str) -> bool:
    """True if the knowledge rules out parent -> child."""
    violates_order = (
        parent in knowledge.order
        and child in knowledge.order
        and knowledge.order[parent] >= knowledge.order[child]
    )
    return violates_order or (parent, child) in knowledge.forbidden_edges


def apply_hard_domain_constraints(
    marks: pd.DataFrame,
    knowledge: "DomainKnowledge",
    delete_forbidden_definite: bool = False,
) -> pd.DataFrame:
    """Enforce hard knowledge on an endpoint-mark graph.

    A forbidden a -> b means that a is not an ancestor of b, so in any
    ancestral graph the mark at a must be an arrowhead.  We therefore put an
    arrowhead at a and keep the adjacency: a -- b becomes a <- b, a o-> b and
    a -> b become a <-> b.  The knowledge constrains orientations, not
    adjacencies, so a correct constraint never removes a true adjacency.

    ``delete_forbidden_definite=True`` reproduces the original behaviour,
    which deleted a definite a -> b edge.  That turns an orientation error of
    the base algorithm into an adjacency error, and is what made correct hard
    knowledge lower the inclusion of the true graph in the first experiments.

    A required a -> b is written as a definite a -> b edge.
    """
    from .mixed_graph import ARROW, TAIL, has_edge, is_directed, remove_edge, set_edge

    marks = marks.copy()
    vars_ = list(marks.index)

    for a in vars_:
        for b in vars_:
            if a == b or not has_edge(marks, a, b) or not forbids_edge(knowledge, a, b):
                continue
            if delete_forbidden_definite and is_directed(marks, a, b):
                remove_edge(marks, a, b)
            elif marks.loc[b, a] != ARROW:
                marks.loc[b, a] = ARROW

    for a, b in knowledge.required_edges:
        if a in marks.index and b in marks.index:
            set_edge(marks, a, b, TAIL, ARROW)

    return marks


def knowledge_from_true_graph(
    true_graph: pd.DataFrame,
    frac: float,
    rng: np.random.Generator,
) -> "DomainKnowledge":
    """Derive correct partial knowledge from the true DAG.

    Computes a topological order of the true DAG, then samples a fraction
    ``frac`` of all ordered variable pairs (i precedes j) and adds forbidden
    edges (j, i) — i.e. "j cannot cause i" (temporal-ordering style).
    """
    import networkx as nx

    vars_ = list(true_graph.index)
    G = nx.DiGraph()
    G.add_nodes_from(vars_)
    for u in vars_:
        for v in vars_:
            if true_graph.loc[u, v] == 1:
                G.add_edge(u, v)

    topo = list(nx.topological_sort(G))
    # All ordered pairs (i, j) where i precedes j in topological order.
    ordered_pairs = [
        (topo[i], topo[j])
        for i in range(len(topo))
        for j in range(i + 1, len(topo))
    ]

    n_sample = max(1, int(np.round(frac * len(ordered_pairs))))
    n_sample = min(n_sample, len(ordered_pairs))
    chosen = rng.choice(len(ordered_pairs), size=n_sample, replace=False)

    forbidden: Set[Tuple[str, str]] = set()
    for idx in chosen:
        i_var, j_var = ordered_pairs[idx]
        # i_var precedes j_var, so forbid j_var -> i_var
        forbidden.add((j_var, i_var))

    return DomainKnowledge(order={}, forbidden_edges=forbidden, required_edges=set())


def corrupt_knowledge(
    knowledge: "DomainKnowledge",
    noise_frac: float,
    rng: np.random.Generator,
) -> "DomainKnowledge":
    """Return a copy where a fraction ``noise_frac`` of forbidden edges are reversed.

    Reversed constraints forbid the correct direction, potentially blocking a
    true edge and misleading the discovery algorithm.
    """
    forbidden = list(knowledge.forbidden_edges)
    n_corrupt = max(0, int(np.round(noise_frac * len(forbidden))))
    n_corrupt = min(n_corrupt, len(forbidden))

    if n_corrupt == 0 or not forbidden:
        return DomainKnowledge(
            order=dict(knowledge.order),
            forbidden_edges=set(knowledge.forbidden_edges),
            required_edges=set(knowledge.required_edges),
        )

    chosen = rng.choice(len(forbidden), size=n_corrupt, replace=False)
    chosen_set = set(chosen)
    new_forbidden: Set[Tuple[str, str]] = set()
    for i, (a, b) in enumerate(forbidden):
        if i in chosen_set:
            new_forbidden.add((b, a))
        else:
            new_forbidden.add((a, b))

    return DomainKnowledge(
        order=dict(knowledge.order),
        forbidden_edges=new_forbidden,
        required_edges=set(knowledge.required_edges),
    )


def knowledge_penalty(
    G: pd.DataFrame,
    knowledge: "DomainKnowledge",
) -> float:
    """R_K(G): fraction of knowledge constraints violated by G.

    Counts violations of forbidden_edges, required_edges, and order
    constraints, restricted to variables present in G.  Returns 0.0 when
    there are no applicable constraints.
    """
    vars_set = set(G.index)
    total = 0
    violated = 0

    for (a, b) in knowledge.forbidden_edges:
        if a in vars_set and b in vars_set:
            total += 1
            if G.loc[a, b] == 1:
                violated += 1

    for (a, b) in knowledge.required_edges:
        if a in vars_set and b in vars_set:
            total += 1
            if G.loc[a, b] != 1:
                violated += 1

    vars_ = list(G.index)
    for a in vars_:
        for b in vars_:
            if a == b:
                continue
            if a in knowledge.order and b in knowledge.order:
                # order[a] >= order[b] means a should not precede b (b is
                # earlier), so edge a->b violates ordering.
                if knowledge.order[a] >= knowledge.order[b]:
                    total += 1
                    if G.loc[a, b] == 1:
                        violated += 1

    if total == 0:
        return 0.0
    return float(violated) / float(total)


def empty_adj(var_names: List[str]) -> pd.DataFrame:
    return pd.DataFrame(0, index=var_names, columns=var_names, dtype=int)


def numpy_to_adj(B: "np.ndarray", var_names: List[str]) -> pd.DataFrame:
    import numpy as np

    return pd.DataFrame(np.asarray(B, dtype=int), index=var_names, columns=var_names)
