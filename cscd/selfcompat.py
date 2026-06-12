import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .graph import discover_graph
from .utils import knowledge_penalty, restrict_knowledge

logger = logging.getLogger(__name__)


def sample_variable_subsets(
    var_names: List[str],
    m: int,
    subset_frac: float,
    rng: np.random.Generator,
) -> List[List[str]]:
    p = len(var_names)
    subset_size = max(3, int(np.ceil(subset_frac * p)))
    subsets = []

    for _ in range(m):
        subsets.append(list(rng.choice(var_names, size=subset_size, replace=False)))

    return subsets


def graph_distance(
    A: pd.DataFrame,
    B: pd.DataFrame,
    vars_: List[str],
    mode: str = "directed",
) -> float:
    vars_ = [v for v in vars_ if v in A.index and v in B.index]
    if len(vars_) < 2:
        return 0.0

    A_sub = A.loc[vars_, vars_].values
    B_sub = B.loc[vars_, vars_].values

    if mode == "skeleton":
        A_sub = ((A_sub + A_sub.T) > 0).astype(int)
        B_sub = ((B_sub + B_sub.T) > 0).astype(int)
    elif mode != "directed":
        raise ValueError(f"Unknown distance mode: {mode}")

    mask = ~np.eye(len(vars_), dtype=bool)
    return float(np.abs(A_sub[mask] - B_sub[mask]).sum() / mask.sum())


def train_local_graphs(
    data_train: pd.DataFrame,
    subsets: List[List[str]],
    knowledge,
    method: str,
    config,
    rng: np.random.Generator,
) -> Dict[int, pd.DataFrame]:
    local_graphs = {}
    for j, subset in enumerate(subsets):
        k_sub = restrict_knowledge(knowledge, subset)
        A = discover_graph(
            data_train[subset],
            method=method,
            knowledge=k_sub,
            config=config,
            random_state=int(rng.integers(1_000_000)),
        )
        local_graphs[j] = A
    return local_graphs


def self_compatibility_score_among_local_graphs(
    local_graphs: Dict[int, pd.DataFrame],
    subsets: List[List[str]],
    distance_mode: str = "directed",
) -> float:
    losses = []
    keys = list(local_graphs.keys())

    for a_idx in range(len(keys)):
        for b_idx in range(a_idx + 1, len(keys)):
            a = keys[a_idx]
            b = keys[b_idx]
            overlap = sorted(set(subsets[a]) & set(subsets[b]))
            if len(overlap) < 2:
                continue
            losses.append(
                graph_distance(
                    local_graphs[a],
                    local_graphs[b],
                    overlap,
                    mode=distance_mode,
                )
            )

    return float(np.mean(losses)) if losses else 0.0


def candidate_graph_score(
    candidate: pd.DataFrame,
    local_graphs: Dict[int, pd.DataFrame],
    subsets: List[List[str]],
    distance_mode: str = "directed",
    knowledge=None,
    lambda_penalty: float = 0.0,
) -> float:
    losses = []
    for j, subset in enumerate(subsets):
        common = [v for v in subset if v in candidate.index and v in local_graphs[j].index]
        losses.append(
            graph_distance(
                candidate,
                local_graphs[j],
                common,
                mode=distance_mode,
            )
        )
    base = float(np.mean(losses)) if losses else 0.0
    if knowledge is not None and lambda_penalty > 0.0:
        base += lambda_penalty * knowledge_penalty(candidate, knowledge)
    return base


def calibration_scores(
    data_cal: pd.DataFrame,
    subsets: List[List[str]],
    train_local_graphs_dict: Dict[int, pd.DataFrame],
    knowledge,
    method: str,
    config,
    n_resamples: int,
    rng: np.random.Generator,
    distance_mode: str = "directed",
    soft_knowledge=None,
    lambda_penalty: float = 0.0,
) -> np.ndarray:
    n = len(data_cal)
    scores = []

    # Partition D_cal into n_resamples disjoint chunks.  Bootstrap resampling
    # (with replacement) would break the exchangeability of (A_1,...,A_N,A_test)
    # that underlies Proposition 1.
    perm = rng.permutation(n)
    chunk_size = max(1, n // n_resamples)

    for b in range(n_resamples):
        start = b * chunk_size
        end = min(start + chunk_size, n)
        if start >= n:
            break
        data_b = data_cal.iloc[perm[start:end]].reset_index(drop=True)
        losses = []
        for j, subset in enumerate(subsets):
            try:
                k_sub = restrict_knowledge(knowledge, subset)
                A_cal = discover_graph(
                    data_b[subset],
                    method=method,
                    knowledge=k_sub,
                    config=config,
                    random_state=int(rng.integers(1_000_000)),
                )
                A_train = train_local_graphs_dict[j]
                d = graph_distance(
                    A_cal,
                    A_train,
                    subset,
                    mode=distance_mode,
                )
                if soft_knowledge is not None and lambda_penalty > 0.0:
                    k_sub_soft = restrict_knowledge(soft_knowledge, subset)
                    d += lambda_penalty * knowledge_penalty(A_cal, k_sub_soft)
                losses.append(d)
            except Exception as exc:
                logger.debug(
                    "calibration chunk %d subset %d discovery failed (%s); skipping.",
                    b, j, exc,
                )
        if losses:
            scores.append(float(np.mean(losses)))

    return np.array(scores)


def conformal_threshold(scores: np.ndarray, alpha: float) -> float:
    N = len(scores)
    k = int(np.ceil((N + 1) * (1 - alpha)))
    if k > N:
        # Not enough calibration scores to set a finite threshold at this alpha;
        # accept all candidates (τ_α = +∞ is the standard CP convention).
        return float("inf")
    return float(np.sort(scores)[k - 1])


def generate_candidate_graphs(
    data_train: pd.DataFrame,
    knowledge,
    method: str,
    config,
    n_candidates: int,
    rng: np.random.Generator,
) -> List[pd.DataFrame]:
    n = len(data_train)
    candidates = []
    for _ in range(n_candidates):
        # Individual bootstrap candidates can fail (e.g. near-collinear
        # resample triggering a NumPy 2.0 scalar-conversion bug in causal-learn
        # GES).  Skip silently — the full-data run below is always included.
        try:
            idx = rng.choice(np.arange(n), size=n, replace=True)
            data_b = data_train.iloc[idx].reset_index(drop=True)
            candidates.append(
                discover_graph(
                    data_b,
                    method=method,
                    knowledge=knowledge,
                    config=config,
                    random_state=int(rng.integers(1_000_000)),
                )
            )
        except Exception as exc:
            logger.debug("candidate bootstrap discovery failed (%s); skipping.", exc)
    # Always include the full-data run as a baseline candidate.
    candidates.append(
        discover_graph(
            data_train,
            method=method,
            knowledge=knowledge,
            config=config,
            random_state=int(rng.integers(1_000_000)),
        )
    )
    return candidates


def conformal_graph_set(
    candidates: List[pd.DataFrame],
    local_graphs: Dict[int, pd.DataFrame],
    subsets: List[List[str]],
    tau: float,
    distance_mode: str = "directed",
    knowledge=None,
    lambda_penalty: float = 0.0,
) -> Tuple[List[pd.DataFrame], List[float]]:
    selected = []
    scores = []

    for G in candidates:
        s = candidate_graph_score(
            G, local_graphs, subsets, distance_mode,
            knowledge=knowledge, lambda_penalty=lambda_penalty,
        )
        scores.append(s)
        if s <= tau:
            selected.append(G)

    return selected, scores
