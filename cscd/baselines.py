import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .graph import discover_graph
from .metrics import accepted_precision_recall, shd, skeleton_f1, status_counts, summarize_claims

logger = logging.getLogger(__name__)


def directed_edge_precision_recall(
    pred_graph: pd.DataFrame,
    true_graph: pd.DataFrame,
) -> Dict[str, float]:
    vars_ = list(pred_graph.index)
    true_graph = true_graph.loc[vars_, vars_]
    mask = ~np.eye(len(vars_), dtype=bool)

    pred_vals = pred_graph.values[mask]
    true_vals = true_graph.values[mask]

    pred_edges = pred_vals == 1
    true_edges = true_vals == 1

    tp = int(np.sum(pred_edges & true_edges))
    fp = int(np.sum(pred_edges & ~true_edges))
    fn = int(np.sum(~pred_edges & true_edges))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"precision": float(precision), "recall": float(recall)}


def stability_selection_graph(
    data_train: pd.DataFrame,
    knowledge,
    method: str,
    config,
    rng: np.random.Generator,
    B: Optional[int] = None,
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    if B is None:
        B = config.stability_n_bootstraps
    if threshold is None:
        threshold = config.stability_threshold

    var_names = list(data_train.columns)
    n = len(data_train)

    # skeleton_counts[i, j] counts bootstraps where the unordered pair (i,j) was present
    p = len(var_names)
    skeleton_counts = np.zeros((p, p), dtype=int)
    # directed_counts[i, j] counts bootstraps where directed edge i->j was present
    directed_counts = np.zeros((p, p), dtype=int)

    n_success = 0
    for b in range(B):
        try:
            idx = rng.choice(np.arange(n), size=n, replace=True)
            data_b = data_train.iloc[idx].reset_index(drop=True)
            G_b = discover_graph(
                data_b,
                method=method,
                knowledge=knowledge,
                config=config,
                random_state=int(rng.integers(1_000_000)),
            )
            G_vals = G_b.loc[var_names, var_names].values
            for i in range(p):
                for j in range(i + 1, p):
                    if G_vals[i, j] == 1 or G_vals[j, i] == 1:
                        skeleton_counts[i, j] += 1
                        skeleton_counts[j, i] += 1
            directed_counts += (G_vals == 1).astype(int)
            n_success += 1
        except Exception as exc:
            logger.debug("stability bootstrap %d discovery failed (%s); skipping.", b, exc)

    denom = max(n_success, 1)
    skeleton_freq = skeleton_counts / denom

    A = pd.DataFrame(0, index=var_names, columns=var_names, dtype=int)
    for i in range(p):
        for j in range(i + 1, p):
            if skeleton_freq[i, j] >= threshold:
                vi, vj = var_names[i], var_names[j]
                fwd = directed_counts[i, j]
                bwd = directed_counts[j, i]
                if fwd > bwd:
                    A.loc[vi, vj] = 1
                elif bwd > fwd:
                    A.loc[vj, vi] = 1
                else:
                    # equal counts or both zero — undirected/bidirected convention
                    A.loc[vi, vj] = 1
                    A.loc[vj, vi] = 1

    return A


def uncalibrated_selfcompat_claims(
    candidates: List[pd.DataFrame],
    candidate_scores: List[float],
    raw_self_compat: float,
    var_names: List[str],
    config,
) -> Tuple[pd.DataFrame, int]:
    admitted = [G for G, s in zip(candidates, candidate_scores) if s <= raw_self_compat]
    claims = summarize_claims(
        admitted,
        var_names,
        accepted_threshold=config.accepted_threshold,
        rejected_threshold=config.rejected_threshold,
    )
    return claims, len(admitted)


def baseline_summary(
    full_graph: pd.DataFrame,
    data_train: pd.DataFrame,
    knowledge,
    method: str,
    config,
    rng: np.random.Generator,
    candidates: List[pd.DataFrame],
    candidate_scores: List[float],
    raw_self_compat: float,
    var_names: List[str],
    true_graph: Optional[pd.DataFrame],
) -> Dict[str, float]:
    # --- single run ---
    if true_graph is not None:
        sr_pr = directed_edge_precision_recall(full_graph, true_graph)
        sr_shd = float(shd(full_graph, true_graph))
        _, _, sr_sk_f1 = skeleton_f1(full_graph, true_graph)
    else:
        sr_pr = {"precision": float("nan"), "recall": float("nan")}
        sr_shd = float("nan")
        sr_sk_f1 = float("nan")

    # --- stability selection ---
    stab_graph = stability_selection_graph(
        data_train, knowledge, method, config, rng
    )
    if true_graph is not None:
        stab_pr = directed_edge_precision_recall(stab_graph, true_graph)
        stab_shd = float(shd(stab_graph, true_graph))
        _, _, stab_sk_f1 = skeleton_f1(stab_graph, true_graph)
    else:
        stab_pr = {"precision": float("nan"), "recall": float("nan")}
        stab_shd = float("nan")
        stab_sk_f1 = float("nan")

    # --- uncalibrated self-compatibility ---
    uncal_claims, uncal_n_selected = uncalibrated_selfcompat_claims(
        candidates, candidate_scores, raw_self_compat, var_names, config
    )
    uncal_pr = accepted_precision_recall(uncal_claims, true_graph)
    uncal_counts = status_counts(uncal_claims)
    total_claims = len(var_names) * (len(var_names) - 1)
    uncal_unresolved_rate = (
        uncal_counts["n_unresolved"] / total_claims if total_claims > 0 else float("nan")
    )

    return {
        "single_run_precision": sr_pr["precision"],
        "single_run_recall": sr_pr["recall"],
        "single_run_shd": sr_shd,
        "single_run_skeleton_f1": sr_sk_f1,
        "stability_precision": stab_pr["precision"],
        "stability_recall": stab_pr["recall"],
        "stability_shd": stab_shd,
        "stability_skeleton_f1": stab_sk_f1,
        "uncal_precision": uncal_pr["accepted_precision"],
        "uncal_recall": uncal_pr["accepted_recall"],
        "uncal_unresolved_rate": uncal_unresolved_rate,
        "uncal_n_selected_graphs": float(uncal_n_selected),
    }
