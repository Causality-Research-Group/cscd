from typing import List

import numpy as np
import pandas as pd

from .utils import apply_hard_domain_constraints, empty_adj

try:
    from causallearn.search.ConstraintBased.PC import pc
except Exception:
    pc = None

try:
    from causallearn.search.ConstraintBased.FCI import fci
except Exception:
    fci = None

try:
    from causallearn.search.ScoreBased.GES import ges
except Exception:
    ges = None

try:
    from causallearn.graph.Endpoint import Endpoint
except Exception:
    Endpoint = None


def causallearn_graph_to_directed_adjacency(
    graph_obj,
    var_names: List[str],
    keep_undirected_as_bidirected: bool = True,
) -> pd.DataFrame:
    A = empty_adj(var_names)

    try:
        nodes = graph_obj.get_nodes()
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        edges = graph_obj.get_graph_edges()
    except Exception:
        if hasattr(graph_obj, "graph"):
            mat = np.asarray(graph_obj.graph)
            p = len(var_names)
            for i in range(p):
                for j in range(p):
                    if i != j and mat[i, j] != 0:
                        A.iloc[i, j] = 1
            return A
        raise RuntimeError("Could not parse causal-learn graph object.")

    for e in edges:
        try:
            n1 = e.get_node1()
            n2 = e.get_node2()
            ep1 = e.get_endpoint1()
            ep2 = e.get_endpoint2()
        except Exception:
            continue

        i = node_to_idx.get(n1)
        j = node_to_idx.get(n2)
        if i is None or j is None:
            continue
        if i >= len(var_names) or j >= len(var_names):
            continue

        v1 = var_names[i]
        v2 = var_names[j]

        if Endpoint is not None and ep1 == Endpoint.TAIL and ep2 == Endpoint.ARROW:
            A.loc[v1, v2] = 1
        elif Endpoint is not None and ep1 == Endpoint.ARROW and ep2 == Endpoint.TAIL:
            A.loc[v2, v1] = 1
        else:
            if keep_undirected_as_bidirected:
                A.loc[v1, v2] = 1
                A.loc[v2, v1] = 1

    return A


def ges_record_to_adjacency(record, var_names: List[str]) -> pd.DataFrame:
    if isinstance(record, dict) and "G" in record:
        return causallearn_graph_to_directed_adjacency(record["G"], var_names)
    return causallearn_graph_to_directed_adjacency(record, var_names)


def discover_graph(
    data: pd.DataFrame,
    method: str,
    knowledge,
    config,
    random_state: int = 0,
) -> pd.DataFrame:
    method = method.lower()
    data = data.copy()
    X = data.values
    var_names = list(data.columns)

    if method == "pc":
        if pc is None:
            raise ImportError("Could not import causal-learn PC. Install causal-learn.")
        try:
            cg = pc(
                X,
                alpha=config.pc_alpha,
                indep_test=config.independence_test,
                stable=True,
                node_names=var_names,
                verbose=False,
            )
        except TypeError:
            cg = pc(
                X,
                alpha=config.pc_alpha,
                indep_test=config.independence_test,
                stable=True,
                verbose=False,
            )
        A = causallearn_graph_to_directed_adjacency(cg.G, var_names)

    elif method == "fci":
        if fci is None:
            raise ImportError("Could not import causal-learn FCI. Install causal-learn.")
        try:
            g, _ = fci(
                X,
                independence_test_method=config.independence_test,
                alpha=config.fci_alpha,
                verbose=False,
                node_names=var_names,
            )
        except TypeError:
            g, _ = fci(
                X,
                independence_test_method=config.independence_test,
                alpha=config.fci_alpha,
                verbose=False,
            )
        A = causallearn_graph_to_directed_adjacency(g, var_names)

    elif method == "ges":
        if ges is None:
            raise ImportError("Could not import causal-learn GES. Install causal-learn.")
        record = ges(
            X,
            score_func=config.ges_score,
        )
        A = ges_record_to_adjacency(record, var_names)

    elif method == "rcd":
        raise NotImplementedError(
            "The RCD (i.e., LinGAM) causal algorithm is not yet implemented. "
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    A = apply_hard_domain_constraints(A, knowledge)
    return A
