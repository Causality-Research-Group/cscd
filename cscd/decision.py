"""Downstream decision experiment for CSCD.

Evaluates treatment-effect estimation bias when adjustment sets are chosen
(a) from the true graph (oracle), (b) from a single discovered graph, or
(c) only from conformally accepted parent claims.

In a linear Gaussian DAG, adjusting for Pa(T) identifies the total causal
effect of T on Y via the backdoor criterion.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def true_total_effect(W: np.ndarray, t: int, y: int) -> float:
    """Total causal effect of node t on node y in a linear SCM.

    For a DAG weight matrix W (W[i,j] = coefficient of i->j), the matrix of
    total effects is (I - W)^{-1} - I, so entry [t, y] gives the total effect
    of t on y via all directed paths.  This formula is exact for any DAG since
    (I - W) is invertible (W is strictly lower/upper triangular in topo order).
    """
    p = W.shape[0]
    total_effects = np.linalg.inv(np.eye(p) - W) - np.eye(p)
    return float(total_effects[t, y])


def estimate_effect_ols(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    adjustment: List[str],
) -> float:
    """Estimate the total effect of treatment on outcome via OLS with adjustment.

    Fits outcome ~ intercept + treatment + adjustment variables using numpy
    lstsq; returns the coefficient on treatment.  The adjustment set is
    sanitised to exclude treatment and outcome if present (defensive guard).
    """
    # Guard: remove treatment and outcome from adjustment
    adj_clean = [v for v in adjustment if v != treatment and v != outcome]

    cols = [treatment] + adj_clean
    X = data[cols].values.astype(float)
    # Add intercept column
    X = np.column_stack([np.ones(len(X)), X])
    y = data[outcome].values.astype(float)

    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    # coef[0] = intercept, coef[1] = treatment coefficient
    return float(coef[1])


def parents_from_graph(G: pd.DataFrame, node: str) -> List[str]:
    """Return variables v with G.loc[v, node] == 1, excluding node itself.

    Note: undirected/bidirected edges are stored as 1 in both directions;
    any v with G.loc[v, node] == 1 is treated as a parent (conservative).
    """
    if node not in G.columns:
        return []
    return [v for v in G.index if v != node and G.loc[v, node] == 1]


def parents_from_claims(claims: pd.DataFrame, node: str) -> List[str]:
    """Return parents v where the claim (parent=v, child=node) is 'accepted'."""
    mask = (claims["child"] == node) & (claims["status"] == "accepted")
    return list(claims.loc[mask, "parent"])


def _descendants(adj: pd.DataFrame, node: str) -> List[str]:
    """Return all descendants of node in the adjacency DataFrame (BFS)."""
    visited = set()
    queue = [node]
    while queue:
        cur = queue.pop()
        for v in adj.columns:
            if v != cur and v not in visited and adj.loc[cur, v] == 1:
                visited.add(v)
                queue.append(v)
    return list(visited)


def _is_confounded(true_graph: pd.DataFrame, t: str, y: str) -> bool:
    """True if some parent of t is still an ancestor of y once t is removed.

    Such a parent opens a backdoor path t <- pa ~> y, so the naive regression
    of y on t is biased and adjustment is genuinely required.  Restricting the
    decision task to confounded pairs keeps it informative: on unconfounded
    pairs every strategy (including no adjustment) is consistent.
    """
    parents = parents_from_graph(true_graph, t)
    if not parents:
        return False
    sub = true_graph.drop(index=t, columns=t)
    for pa in parents:
        if pa == y or y in _descendants(sub, pa):
            return True
    return False


def decision_task_summary(
    data_eval: pd.DataFrame,
    W: np.ndarray,
    var_names: List[str],
    true_graph: pd.DataFrame,
    full_graph: pd.DataFrame,
    claims: pd.DataFrame,
    rng: np.random.Generator,
    n_pairs: int = 5,
) -> Dict[str, float]:
    """Compute mean absolute bias for four adjustment strategies.

    Strategies
    ----------
    oracle   : adjust for true parents of treatment (from true_graph).
    single   : adjust for parents of treatment in the single discovered graph.
    accepted : selective — abstain when any parent claim for the treatment is
               unresolved; otherwise adjust for accepted parents.  Reported
               only over acted pairs, alongside the single-run bias on the
               same pairs (decision_bias_single_acted) and the abstention
               rate.
    naive    : no adjustment (empty set).

    Parameters
    ----------
    data_eval : pd.DataFrame
        Dataset used for OLS estimation.
    W : np.ndarray
        Linear SEM weight matrix; W[i,j] = coefficient of edge i->j.
        Rows/columns correspond to var_names order.
    var_names : list of str
        Variable names in the same order as W's rows/columns.
    true_graph : pd.DataFrame
        True adjacency matrix (var_names x var_names).
    full_graph : pd.DataFrame
        Single-run discovered graph (adjacency DataFrame).
    claims : pd.DataFrame
        Claims DataFrame with columns parent, child, status.
    rng : np.random.Generator
        For reproducible (treatment, outcome) pair selection.
    n_pairs : int
        Maximum number of (treatment, outcome) pairs to evaluate.

    Returns
    -------
    dict with keys decision_bias_oracle, decision_bias_single,
    decision_bias_accepted, decision_bias_naive, decision_n_pairs.
    """
    nan_result = {
        "decision_bias_oracle": float("nan"),
        "decision_bias_single": float("nan"),
        "decision_bias_accepted": float("nan"),
        "decision_bias_single_acted": float("nan"),
        "decision_bias_naive": float("nan"),
        "decision_abstain_rate": float("nan"),
        "decision_n_pairs": 0,
        "decision_n_acted": 0,
    }

    p = len(var_names)
    if p < 2:
        return nan_result

    # Find nodes that have at least one descendant in the true graph
    nodes_with_desc = [
        v for v in var_names if len(_descendants(true_graph, v)) > 0
    ]
    if not nodes_with_desc:
        return nan_result

    # Sample up to n_pairs confounded (treatment, outcome) pairs
    pairs = []
    seen = set()
    attempts = 0
    max_attempts = n_pairs * 40
    while len(pairs) < n_pairs and attempts < max_attempts:
        attempts += 1
        t_name = nodes_with_desc[int(rng.integers(len(nodes_with_desc)))]
        desc = _descendants(true_graph, t_name)
        if not desc:
            continue
        y_name = desc[int(rng.integers(len(desc)))]
        if (t_name, y_name) in seen:
            continue
        seen.add((t_name, y_name))
        if _is_confounded(true_graph, t_name, y_name):
            pairs.append((t_name, y_name))

    if not pairs:
        return nan_result

    bias_oracle = []
    bias_single = []
    bias_naive = []
    bias_accepted_acted = []
    bias_single_acted = []
    n_abstained = 0

    for t_name, y_name in pairs:
        t_idx = var_names.index(t_name)
        y_idx = var_names.index(y_name)
        theta = true_total_effect(W, t_idx, y_idx)

        if abs(theta) < 1e-8:
            continue  # skip negligible effects

        # Adjustment sets
        adj_oracle = parents_from_graph(true_graph, t_name)
        adj_single = parents_from_graph(full_graph, t_name)
        adj_accepted = parents_from_claims(claims, t_name)
        adj_empty: List[str] = []

        # Selective rule: CSCD acts only when every parent claim for the
        # treatment is resolved (accepted or rejected).  An unresolved parent
        # claim means the adjustment set itself is uncertain, so the
        # calibrated answer is to abstain rather than estimate.
        parent_claims = claims[(claims["child"] == t_name) & (claims["parent"] != t_name)]
        cscd_acts = not (parent_claims["status"] == "unresolved").any()

        try:
            est_oracle = estimate_effect_ols(data_eval, t_name, y_name, adj_oracle)
            est_single = estimate_effect_ols(data_eval, t_name, y_name, adj_single)
            est_naive = estimate_effect_ols(data_eval, t_name, y_name, adj_empty)
            if cscd_acts:
                est_accepted = estimate_effect_ols(data_eval, t_name, y_name, adj_accepted)
        except Exception as exc:
            logger.debug("decision_task_summary OLS failed for (%s, %s): %s", t_name, y_name, exc)
            continue

        bias_oracle.append(abs(est_oracle - theta))
        bias_single.append(abs(est_single - theta))
        bias_naive.append(abs(est_naive - theta))
        if cscd_acts:
            bias_accepted_acted.append(abs(est_accepted - theta))
            bias_single_acted.append(abs(est_single - theta))
        else:
            n_abstained += 1

    n_eval = len(bias_oracle)
    if n_eval == 0:
        return nan_result

    n_acted = len(bias_accepted_acted)
    return {
        "decision_bias_oracle": float(np.mean(bias_oracle)),
        "decision_bias_single": float(np.mean(bias_single)),
        "decision_bias_accepted": float(np.mean(bias_accepted_acted)) if n_acted else float("nan"),
        "decision_bias_single_acted": float(np.mean(bias_single_acted)) if n_acted else float("nan"),
        "decision_bias_naive": float(np.mean(bias_naive)),
        "decision_abstain_rate": float(n_abstained / n_eval),
        "decision_n_pairs": n_eval,
        "decision_n_acted": n_acted,
    }
