"""Local IDA effect estimation on top of the CPDAG/PAG-aware graph decision task.

``cscd/decision.py`` picks adjustment sets by ``parents_from_graph``, which
treats any node adjacent through an undirected CPDAG edge as a definite
parent.  That is anti-conservative: an undirected edge t--v in a CPDAG means
v could be a parent *or* a child of t in the underlying DAG, and adjusting
for a child of the treatment can itself induce bias.  It also evaluates
effect estimates on the same data used for discovery, which underestimates
uncertainty from structure search.

This module implements local IDA (Maathuis, Kalisch and Buhlmann, 2009,
"Estimating high-dimensional intervention effects from observational data",
Ann. Statist.), which is the conservative alternative: only nodes definitely
oriented into the treatment are used as parents; nodes attached by an
edge that is not definitely oriented *out* of the treatment are treated as
"ambiguous siblings" and enumerated over every locally consistent orientation
(one that introduces no new v-structure at the treatment).  Each orientation
gives one OLS effect estimate, and the resulting multiset of estimates is
summarised either by its mean or by IDA's usual conservative summary,
the minimum absolute value.
"""

import logging
from itertools import combinations
from typing import Dict, List

import numpy as np
import pandas as pd

from .decision import (
    _descendants,
    _is_confounded,
    estimate_effect_ols,
    parents_from_graph,
    true_total_effect,
)
from .mixed_graph import ARROW, CIRCLE, TAIL, empty_marks, set_edge

logger = logging.getLogger(__name__)

MAX_SIBLINGS = 8


def _to_marks(G: pd.DataFrame) -> pd.DataFrame:
    """Return an endpoint-mark DataFrame for G, which may already be marks or 0/1 adjacency."""
    values = G.values
    is_marks = np.isin(values, [0, TAIL, ARROW, CIRCLE]).all() and (values > 1).any()
    if is_marks:
        return G
    # Treat as 0/1 adjacency: a->b iff A[a,b]==1 and A[b,a]==0; both 1 => undirected tail-tail.
    names = list(G.index)
    marks = empty_marks(names)
    for a in names:
        for b in names:
            if a == b:
                continue
            ab = G.loc[a, b]
            ba = G.loc[b, a]
            if ab == 1 and ba == 0:
                set_edge(marks, a, b, TAIL, ARROW)
            elif ab == 1 and ba == 1:
                set_edge(marks, a, b, TAIL, TAIL)
    return marks


def _definite_parents_and_siblings(marks: pd.DataFrame, node: str):
    """Split node's neighbours into definite parents and ambiguous siblings.

    A definite parent v has v -> node (mark at node is ARROW, mark at v is
    TAIL).  A definite child v (node -> v: mark at node is TAIL, mark at v is
    ARROW) is excluded outright -- it is never a valid parent under any
    consistent orientation.  Any other adjacent node (undirected,
    circle-marked, or bidirected) is an ambiguous sibling: its status
    relative to node is not determined by the graph.
    """
    names = list(marks.index)
    parents: List[str] = []
    siblings: List[str] = []
    for v in names:
        if v == node:
            continue
        mark_at_node = marks.loc[v, node]
        mark_at_v = marks.loc[node, v]
        if mark_at_node == 0 and mark_at_v == 0:
            continue
        if mark_at_node == ARROW and mark_at_v == TAIL:
            parents.append(v)
        elif mark_at_node == TAIL and mark_at_v == ARROW:
            continue  # definite child of node: never a parent
        else:
            siblings.append(v)
    return parents, siblings


def _adjacent(marks: pd.DataFrame, a: str, b: str) -> bool:
    return marks.loc[a, b] != 0 or marks.loc[b, a] != 0


def _creates_new_vstructure(
    marks: pd.DataFrame, node: str, parents: List[str], into: List[str]
) -> bool:
    """True if orienting `into` -> node (with the existing `parents` -> node)
    creates a new unshielded collider at node.

    A new v-structure a -> node <- b appears whenever a, b are both in
    parents u into, a != b, at least one of them is in `into` (the newly
    oriented set), and a, b are not adjacent in the original graph.
    """
    combined = list(parents) + list(into)
    into_set = set(into)
    for a, b in combinations(combined, 2):
        if a not in into_set and b not in into_set:
            continue  # both were already parents; not a new collider
        if not _adjacent(marks, a, b):
            return True
    return False


def _select_top_siblings(data: pd.DataFrame, node: str, siblings: List[str], cap: int) -> List[str]:
    if len(siblings) <= cap:
        return siblings
    corrs = data[siblings].corrwith(data[node]).abs()
    ranked = corrs.sort_values(ascending=False).index.tolist()
    return ranked[:cap]


def local_ida_estimates(
    data: pd.DataFrame,
    marks: pd.DataFrame,
    treatment: str,
    outcome: str,
) -> List[float]:
    """Local IDA effect estimates of treatment on outcome under ``marks``.

    Enumerates every subset S' of the ambiguous siblings of ``treatment``
    such that orienting S' -> treatment (together with the definite parents)
    introduces no new v-structure at treatment.  Each valid S' contributes
    one OLS estimate of the treatment coefficient in
    ``outcome ~ 1 + treatment + definite_parents + S'``.  If ``outcome`` is
    itself a definite parent or an admitted sibling of ``treatment`` for a
    given configuration, that configuration's causal effect is defined as
    0.0 (outcome sits upstream of/adjacent-as-parent to treatment, so
    treatment cannot cause it under that orientation).

    Always returns at least one estimate (falling back to the definite
    parents alone if no sibling subset is admissible, which can only happen
    if the treatment has no siblings).
    """
    if treatment not in marks.index or treatment == outcome:
        return [0.0]

    parents, siblings = _definite_parents_and_siblings(marks, treatment)
    siblings = _select_top_siblings(data, treatment, siblings, MAX_SIBLINGS)

    estimates: List[float] = []
    n = len(siblings)
    for k in range(n + 1):
        for subset in combinations(siblings, k):
            subset = list(subset)
            if _creates_new_vstructure(marks, treatment, parents, subset):
                continue
            adjustment = parents + subset
            if outcome in adjustment:
                estimates.append(0.0)
                continue
            try:
                est = estimate_effect_ols(data, treatment, outcome, adjustment)
            except Exception as exc:
                logger.debug(
                    "local_ida_estimates OLS failed for (%s, %s) with adj=%s: %s",
                    treatment, outcome, adjustment, exc,
                )
                continue
            estimates.append(est)

    if not estimates:
        if outcome in parents:
            return [0.0]
        try:
            estimates.append(estimate_effect_ols(data, treatment, outcome, parents))
        except Exception as exc:
            logger.debug(
                "local_ida_estimates fallback OLS failed for (%s, %s): %s",
                treatment, outcome, exc,
            )
            estimates.append(0.0)

    return estimates


def aggregate_effect(estimates: List[float], how: str = "mean") -> float:
    """Summarise a multiset of local IDA estimates.

    'mean'    : plain average of the estimates.
    'min_abs' : IDA's usual conservative summary -- the estimate with the
                smallest absolute value (a lower bound on |effect|).
    """
    if not estimates:
        return float("nan")
    if how == "mean":
        return float(np.mean(estimates))
    if how == "min_abs":
        arr = np.asarray(estimates, dtype=float)
        return float(arr[np.argmin(np.abs(arr))])
    raise ValueError(f"Unknown aggregation: {how}")


def _mean_local_ida(data: pd.DataFrame, G, treatment: str, outcome: str) -> List[float]:
    """All local IDA estimates for one candidate graph G (marks or 0/1 adjacency)."""
    marks = _to_marks(G)
    return local_ida_estimates(data, marks, treatment, outcome)


def decision_task_v2(
    data_eval: pd.DataFrame,
    W: np.ndarray,
    var_names: List[str],
    true_graph_adj: pd.DataFrame,
    graph_sets: Dict[str, List[pd.DataFrame]],
    rng: np.random.Generator,
    n_pairs: int = 5,
) -> Dict[str, float]:
    """Bias/coverage of local-IDA-based effect estimation for named graph sets.

    For each named list of candidate graphs in ``graph_sets`` (each graph
    either an endpoint-mark DataFrame or a 0/1 adjacency DataFrame), the
    per-graph estimate is the mean of that graph's local IDA estimates; the
    set's overall estimate is the average of the per-graph estimates.  Also
    reports 'oracle' (true parents of treatment, no discovery uncertainty)
    and 'naive' (no adjustment).  For every set, records whether the true
    effect falls in [min, max] of *all* per-graph-per-configuration IDA
    estimates ('range coverage') and that range's width.

    (Treatment, outcome) pairs are sampled exactly as in
    ``cscd.decision.decision_task_summary``: confounded pairs drawn from the
    true DAG adjacency.
    """
    result: Dict[str, float] = {"dec2_n_pairs": 0}
    set_names = list(graph_sets.keys()) + ["oracle", "naive"]
    for name in set_names:
        result[f"dec2_bias_{name}"] = float("nan")
        result[f"dec2_sq_{name}"] = float("nan")
        result[f"dec2_cover_{name}"] = float("nan")
        result[f"dec2_width_{name}"] = float("nan")

    p = len(var_names)
    if p < 2:
        return result

    nodes_with_desc = [v for v in var_names if len(_descendants(true_graph_adj, v)) > 0]
    if not nodes_with_desc:
        return result

    pairs = []
    seen = set()
    attempts = 0
    max_attempts = n_pairs * 40
    while len(pairs) < n_pairs and attempts < max_attempts:
        attempts += 1
        t_name = nodes_with_desc[int(rng.integers(len(nodes_with_desc)))]
        desc = _descendants(true_graph_adj, t_name)
        if not desc:
            continue
        y_name = desc[int(rng.integers(len(desc)))]
        if (t_name, y_name) in seen:
            continue
        seen.add((t_name, y_name))
        if _is_confounded(true_graph_adj, t_name, y_name):
            pairs.append((t_name, y_name))

    if not pairs:
        return result

    per_set_biases: Dict[str, List[float]] = {name: [] for name in set_names}
    per_set_sqs: Dict[str, List[float]] = {name: [] for name in set_names}
    per_set_covers: Dict[str, List[float]] = {name: [] for name in set_names}
    per_set_widths: Dict[str, List[float]] = {name: [] for name in set_names}

    n_eval = 0
    for t_name, y_name in pairs:
        t_idx = var_names.index(t_name)
        y_idx = var_names.index(y_name)
        theta = true_total_effect(W, t_idx, y_idx)
        if abs(theta) < 1e-8:
            continue

        # oracle and naive
        try:
            est_oracle = estimate_effect_ols(
                data_eval, t_name, y_name, parents_from_graph(true_graph_adj, t_name)
            )
            est_naive = estimate_effect_ols(data_eval, t_name, y_name, [])
        except Exception as exc:
            logger.debug("dec2 oracle/naive OLS failed for (%s, %s): %s", t_name, y_name, exc)
            continue

        per_set_biases["oracle"].append(abs(est_oracle - theta))
        per_set_sqs["oracle"].append((est_oracle - theta) ** 2)
        per_set_biases["naive"].append(abs(est_naive - theta))
        per_set_sqs["naive"].append((est_naive - theta) ** 2)

        any_valid = False
        for name, graphs in graph_sets.items():
            per_graph_means = []
            all_estimates: List[float] = []
            for G in graphs:
                try:
                    ests = _mean_local_ida(data_eval, G, t_name, y_name)
                except Exception as exc:
                    logger.debug(
                        "dec2 local IDA failed for set=%s (%s, %s): %s",
                        name, t_name, y_name, exc,
                    )
                    continue
                if not ests:
                    continue
                per_graph_means.append(float(np.mean(ests)))
                all_estimates.extend(ests)

            if not per_graph_means:
                continue
            any_valid = True
            set_estimate = float(np.mean(per_graph_means))
            per_set_biases[name].append(abs(set_estimate - theta))
            per_set_sqs[name].append((set_estimate - theta) ** 2)
            lo, hi = min(all_estimates), max(all_estimates)
            per_set_covers[name].append(1.0 if lo <= theta <= hi else 0.0)
            per_set_widths[name].append(hi - lo)

        if any_valid:
            n_eval += 1

    result["dec2_n_pairs"] = n_eval
    if n_eval == 0:
        return result

    for name in set_names:
        if per_set_biases[name]:
            result[f"dec2_bias_{name}"] = float(np.mean(per_set_biases[name]))
            result[f"dec2_sq_{name}"] = float(np.mean(per_set_sqs[name]))
        if per_set_covers[name]:
            result[f"dec2_cover_{name}"] = float(np.mean(per_set_covers[name]))
            result[f"dec2_width_{name}"] = float(np.mean(per_set_widths[name]))

    return result


def bma_dag_samples(
    data: pd.DataFrame,
    rng: np.random.Generator,
    n_samples: int = 100,
    n_iterations: int = 4000,
    max_parents: int = 3,
    max_candidates: int = 10,
) -> List[pd.DataFrame]:
    """Draw posterior DAG samples from order MCMC (Friedman and Koller, 2003).

    Runs the same Metropolis-Hastings chain over topological orders as
    ``cscd.bayes.OrderMCMC.edge_posterior``, but instead of accumulating an
    edge-probability matrix, collects the individual sampled DAG adjacencies
    (one per post-burn-in, thinned iteration) as 0/1 DataFrames.  Burn-in is
    ``n_iterations // 4`` and samples are thinned evenly to return at most
    ``n_samples`` DAGs.
    """
    from .bayes import OrderMCMC

    sampler = OrderMCMC(data, max_parents=max_parents, max_candidates=max_candidates)
    burn_in = n_iterations // 4
    n_post = max(n_iterations - burn_in, 1)
    thin = max(n_post // n_samples, 1)

    order = list(rng.permutation(sampler.p))
    current = sampler.order_log_score(order)
    samples: List[pd.DataFrame] = []
    for it in range(n_iterations):
        i, j = rng.choice(sampler.p, size=2, replace=False)
        proposal = order.copy()
        proposal[i], proposal[j] = proposal[j], proposal[i]
        proposed = sampler.order_log_score(proposal)
        if np.log(rng.random()) < proposed - current:
            order, current = proposal, proposed
        if it >= burn_in and (it - burn_in) % thin == 0 and len(samples) < n_samples:
            adj = sampler.sample_dag(order, rng)
            samples.append(pd.DataFrame(adj, index=sampler.names, columns=sampler.names))

    return samples
