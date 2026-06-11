from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def edge_frequency(
    graphs: List[pd.DataFrame],
    var_names: List[str],
) -> pd.DataFrame:
    if len(graphs) == 0:
        return pd.DataFrame(0.0, index=var_names, columns=var_names)

    freq = np.zeros((len(var_names), len(var_names)))
    for G in graphs:
        freq += G.loc[var_names, var_names].values
    freq /= len(graphs)
    np.fill_diagonal(freq, 0.0)
    return pd.DataFrame(freq, index=var_names, columns=var_names)


def summarize_claims(
    graphs: List[pd.DataFrame],
    var_names: List[str],
    accepted_threshold: float,
    rejected_threshold: float,
) -> pd.DataFrame:
    rows = []

    if len(graphs) == 0:
        # C_\alpha = no emptyset: every claim is unresolved by definition.
        for parent in var_names:
            for child in var_names:
                if parent != child:
                    rows.append({
                        "parent": parent,
                        "child": child,
                        "frequency": float("nan"),
                        "status": "unresolved",
                    })
        return pd.DataFrame(rows)

    freq = edge_frequency(graphs, var_names)

    for parent in var_names:
        for child in var_names:
            if parent == child:
                continue
            f = float(freq.loc[parent, child])
            if f >= accepted_threshold:
                status = "accepted"
            elif f <= rejected_threshold:
                status = "rejected"
            else:
                status = "unresolved"
            rows.append({
                "parent": parent,
                "child": child,
                "frequency": f,
                "status": status,
            })

    return pd.DataFrame(rows)


def shd(A: pd.DataFrame, B: pd.DataFrame) -> int:
    vars_ = list(A.index)
    B = B.loc[vars_, vars_]
    mask = ~np.eye(len(vars_), dtype=bool)
    return int(np.abs(A.values[mask] - B.values[mask]).sum())


def skeleton_f1(A: pd.DataFrame, B: pd.DataFrame) -> Tuple[float, float, float]:
    vars_ = list(A.index)
    A_skel = ((A.loc[vars_, vars_].values + A.loc[vars_, vars_].values.T) > 0).astype(int)
    B_skel = ((B.loc[vars_, vars_].values + B.loc[vars_, vars_].values.T) > 0).astype(int)
    mask = np.triu(np.ones_like(A_skel, dtype=bool), k=1)
    pred = A_skel[mask] == 1
    true = B_skel[mask] == 1
    tp = np.sum(pred & true)
    fp = np.sum(pred & ~true)
    fn = np.sum(~pred & true)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return float(prec), float(rec), float(f1)


def accepted_precision_recall(
    claim_df: pd.DataFrame,
    true_graph: Optional[pd.DataFrame],
) -> Dict[str, float]:
    if true_graph is None:
        return {
            "accepted_precision": np.nan,
            "accepted_recall": np.nan,
            "n_true_edges": np.nan,
        }

    true_edges = set()
    for i in true_graph.index:
        for j in true_graph.columns:
            if i != j and true_graph.loc[i, j] == 1:
                true_edges.add((i, j))

    accepted = claim_df[claim_df["status"] == "accepted"]
    accepted_edges = set(zip(accepted["parent"], accepted["child"]))
    tp = len(accepted_edges & true_edges)
    precision = tp / max(len(accepted_edges), 1)
    recall = tp / max(len(true_edges), 1)
    return {
        "accepted_precision": precision,
        "accepted_recall": recall,
        "n_true_edges": len(true_edges),
    }


def status_counts(claim_df: pd.DataFrame) -> Dict[str, int]:
    counts = claim_df["status"].value_counts().to_dict()
    return {
        "n_accepted": counts.get("accepted", 0),
        "n_rejected": counts.get("rejected", 0),
        "n_unresolved": counts.get("unresolved", 0),
    }
