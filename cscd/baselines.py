"""Baselines compared with conformal self-compatibility (CSCD).

Claim-level baselines (independent of alpha):
  * single run of the base algorithm on D_train;
  * stability selection over bootstrap runs (Meinshausen and Buehlmann, 2010);
  * uncalibrated self-compatibility (Faller et al., 2024): the candidate set is
    thresholded at the raw self-compatibility score instead of tau_alpha;
  * Bayesian order MCMC edge posteriors (Friedman and Koller, 2003).

Localization ablations (one result per alpha).  Both score the *same*
candidate pool as CSCD, so they differ from it only in how the reference and
the threshold are built:
  * bootstrap compatibility: candidates are compared with full-variable
    bootstrap graphs, and admitted if their mean distance is at most the
    (1 - alpha) quantile of the bootstrap graphs' own mean distances;
  * conformal without subsets: CSCD with a single subset equal to V.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .bayes import bayesian_edge_posterior
from .graph import discover_marks
from .metrics import accepted_precision_recall, shd, skeleton_f1, status_counts, summarize_claims
from .selfcompat import calibration_scores, conformal_threshold, graph_distance, project

logger = logging.getLogger(__name__)


def directed_edge_precision_recall(
    pred_graph: pd.DataFrame,
    true_graph: pd.DataFrame,
) -> Dict[str, float]:
    vars_ = list(pred_graph.index)
    true_graph = true_graph.loc[vars_, vars_]
    mask = ~np.eye(len(vars_), dtype=bool)

    pred_edges = pred_graph.values[mask] == 1
    true_edges = true_graph.values[mask] == 1

    tp = int(np.sum(pred_edges & true_edges))
    fp = int(np.sum(pred_edges & ~true_edges))
    fn = int(np.sum(~pred_edges & true_edges))
    return {"precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)}


def bootstrap_marks(
    data_train: pd.DataFrame,
    knowledge,
    method: str,
    config,
    rng: np.random.Generator,
    B: int,
) -> List[pd.DataFrame]:
    """Base algorithm on B bootstrap resamples of D_train (all variables)."""
    n = len(data_train)
    graphs = []
    for b in range(B):
        idx = rng.choice(n, size=n, replace=True)
        try:
            graphs.append(
                discover_marks(
                    data_train.iloc[idx].reset_index(drop=True),
                    method=method,
                    knowledge=knowledge,
                    config=config,
                    random_state=int(rng.integers(1_000_000)),
                )
            )
        except Exception as exc:
            logger.debug("Bootstrap %d failed (%s); skipping.", b, exc)
    return graphs


def stability_selection_graph(
    bootstrap_adjs: List[pd.DataFrame],
    var_names: List[str],
    threshold: float,
) -> pd.DataFrame:
    """Keep pairs adjacent in at least ``threshold`` of the bootstrap graphs.

    Each kept pair is oriented in its more frequent direction; ties are kept
    in both directions (the undirected/bidirected convention).
    """
    p = len(var_names)
    skeleton_freq = np.zeros((p, p))
    directed_freq = np.zeros((p, p))
    for G in bootstrap_adjs:
        values = G.loc[var_names, var_names].values
        skeleton_freq += ((values + values.T) > 0)
        directed_freq += (values == 1)
    denom = max(len(bootstrap_adjs), 1)
    skeleton_freq /= denom

    A = pd.DataFrame(0, index=var_names, columns=var_names, dtype=int)
    for i in range(p):
        for j in range(i + 1, p):
            if skeleton_freq[i, j] < threshold:
                continue
            vi, vj = var_names[i], var_names[j]
            if directed_freq[i, j] > directed_freq[j, i]:
                A.loc[vi, vj] = 1
            elif directed_freq[j, i] > directed_freq[i, j]:
                A.loc[vj, vi] = 1
            else:
                A.loc[vi, vj] = 1
                A.loc[vj, vi] = 1
    return A


def uncalibrated_selfcompat_claims(
    candidates: List[pd.DataFrame],
    candidate_scores: List[float],
    raw_self_compat: float,
    var_names: List[str],
    config,
):
    admitted = [G for G, s in zip(candidates, candidate_scores) if s <= raw_self_compat]
    claims = summarize_claims(
        admitted,
        var_names,
        accepted_threshold=config.accepted_threshold,
        rejected_threshold=config.rejected_threshold,
    )
    return claims, len(admitted)


def claims_from_edge_probabilities(prob: pd.DataFrame, config) -> pd.DataFrame:
    """Accepted / rejected / unresolved claims from edge probabilities."""
    rows = []
    for parent in prob.index:
        for child in prob.columns:
            if parent == child:
                continue
            f = float(prob.loc[parent, child])
            if f >= config.accepted_threshold:
                status = "accepted"
            elif f <= config.rejected_threshold:
                status = "rejected"
            else:
                status = "unresolved"
            rows.append({"parent": parent, "child": child, "frequency": f, "status": status})
    return pd.DataFrame(rows)


def _claim_columns(prefix: str, claims: pd.DataFrame, true_graph, n_vars: int) -> Dict[str, float]:
    pr = accepted_precision_recall(claims, true_graph)
    counts = status_counts(claims)
    total = n_vars * (n_vars - 1)
    return {
        f"{prefix}_precision": pr["accepted_precision"],
        f"{prefix}_recall": pr["accepted_recall"],
        f"{prefix}_unresolved_rate": counts["n_unresolved"] / total if total else float("nan"),
        f"{prefix}_n_accepted": float(counts["n_accepted"]),
    }


# ---------------------------------------------------------------------------
# Alpha-independent baselines
# ---------------------------------------------------------------------------

def baseline_summary(
    full_graph: pd.DataFrame,
    data_train: pd.DataFrame,
    config,
    rng: np.random.Generator,
    candidates: List[pd.DataFrame],
    candidate_scores: List[float],
    raw_self_compat: float,
    var_names: List[str],
    true_graph: Optional[pd.DataFrame],
    bootstrap_adjs: List[pd.DataFrame],
) -> Dict[str, float]:
    """Metrics of the baselines that do not depend on alpha."""
    columns: Dict[str, float] = {}

    # Single run.
    if true_graph is not None:
        sr = directed_edge_precision_recall(full_graph, true_graph)
        columns.update({
            "single_run_precision": sr["precision"],
            "single_run_recall": sr["recall"],
            "single_run_shd": float(shd(full_graph, true_graph)),
            "single_run_skeleton_f1": skeleton_f1(full_graph, true_graph)[2],
        })

    # Stability selection.
    stab_graph = stability_selection_graph(bootstrap_adjs, var_names, config.stability_threshold)
    if true_graph is not None:
        st = directed_edge_precision_recall(stab_graph, true_graph)
        columns.update({
            "stability_precision": st["precision"],
            "stability_recall": st["recall"],
            "stability_shd": float(shd(stab_graph, true_graph)),
            "stability_skeleton_f1": skeleton_f1(stab_graph, true_graph)[2],
        })

    # Uncalibrated self-compatibility.
    uncal_claims, uncal_n = uncalibrated_selfcompat_claims(
        candidates, candidate_scores, raw_self_compat, var_names, config
    )
    columns.update(_claim_columns("uncal", uncal_claims, true_graph, len(var_names)))
    columns["uncal_n_selected_graphs"] = float(uncal_n)

    # Bayesian order MCMC.
    if config.run_bayesian:
        try:
            posterior = bayesian_edge_posterior(
                data_train, rng, n_iterations=config.bayes_n_iterations
            )
            bayes_claims = claims_from_edge_probabilities(posterior, config)
            columns.update(_claim_columns("bayes", bayes_claims, true_graph, len(var_names)))
        except Exception as exc:
            logger.warning("Bayesian baseline failed (%s).", exc)

    return columns


# ---------------------------------------------------------------------------
# Localization ablations (per alpha)
# ---------------------------------------------------------------------------

def bootstrap_compatibility(
    bootstrap: List[pd.DataFrame],
    candidates: List[pd.DataFrame],
    var_names: List[str],
    distance_mode: str,
    true_full_dag_marks: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    """Scores for the full-variable bootstrap compatibility baseline.

    Returns the candidates' mean distances to the bootstrap graphs, the
    bootstrap graphs' own leave-one-out mean distances (the reference
    distribution for the threshold), and the true graph's mean distance.
    """
    boot_proj = [project(G, var_names, distance_mode) for G in bootstrap]

    def mean_distance(proj, skip=None):
        ds = [graph_distance(proj, q, distance_mode) for k, q in enumerate(boot_proj) if k != skip]
        return float(np.mean(ds)) if ds else 0.0

    reference = [mean_distance(q, skip=k) for k, q in enumerate(boot_proj)]
    candidate = [mean_distance(project(G, var_names, distance_mode)) for G in candidates]
    truth = (
        mean_distance(project(true_full_dag_marks, var_names, distance_mode))
        if true_full_dag_marks is not None
        else float("nan")
    )
    return {"reference": np.array(reference), "candidate": candidate, "truth": truth}


def conformal_without_subsets(
    data_cal: pd.DataFrame,
    full_graph_marks: pd.DataFrame,
    candidates: List[pd.DataFrame],
    knowledge,
    method: str,
    config,
    rng: np.random.Generator,
    distance_mode: str,
    n_chunks: int,
    chunk_size: Optional[int],
    true_full_dag_marks: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    """Scores for CSCD run with the single subset S = V."""
    var_names = list(full_graph_marks.index)
    subsets = [var_names]
    reference = {0: project(full_graph_marks, var_names, distance_mode)}
    cal = calibration_scores(
        data_cal, subsets, reference, knowledge, method, config,
        n_resamples=n_chunks, rng=rng, distance_mode=distance_mode, chunk_size=chunk_size,
    )
    candidate = [
        graph_distance(project(G, var_names, distance_mode), reference[0], distance_mode)
        for G in candidates
    ]
    truth = (
        graph_distance(project(true_full_dag_marks, var_names, distance_mode), reference[0], distance_mode)
        if true_full_dag_marks is not None
        else float("nan")
    )
    return {"calibration": cal, "candidate": candidate, "truth": truth}


def ablation_columns(
    prefix: str,
    admitted: List[pd.DataFrame],
    truth_included: float,
    var_names: List[str],
    true_graph: Optional[pd.DataFrame],
    config,
) -> Dict[str, float]:
    """Claim metrics of a set of admitted candidate graphs (adjacency form)."""
    claims = summarize_claims(
        admitted, var_names,
        accepted_threshold=config.accepted_threshold,
        rejected_threshold=config.rejected_threshold,
    )
    columns = _claim_columns(prefix, claims, true_graph, len(var_names))
    columns[f"{prefix}_n_selected_graphs"] = float(len(admitted))
    columns[f"{prefix}_true_graph_included"] = truth_included
    return columns


def bootstrap_threshold(reference: np.ndarray, alpha: float) -> float:
    """(1 - alpha) empirical quantile of the bootstrap reference distances."""
    if len(reference) == 0:
        return float("inf")
    return float(np.quantile(reference, 1 - alpha))

