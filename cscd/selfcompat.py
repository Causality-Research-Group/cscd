"""Self-compatibility scores, conformal calibration, and candidate graphs.

All graphs here are endpoint-mark DataFrames (see ``mixed_graph``).  A graph
is compared with a graph learned on a variable subset S only after it has
been *marginalized* onto S.  ``project`` does this and returns an object whose
value does not depend on which member of an equivalence class was used:

* ``skeleton`` mode: the skeleton of the marginal MAG over S.
* ``pag`` mode: the marginal PAG over S (adjacencies and endpoint marks).
"""

import itertools
import logging
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .graph import default_hyperparameter, discover_marks
from .mixed_graph import (
    ARROW,
    NO_EDGE,
    TAIL,
    has_edge,
    is_directed,
    marginal_mag,
    marginal_pag,
    remove_edge,
    set_edge,
    skeleton,
    to_adjacency,
)
from .utils import apply_hard_domain_constraints, knowledge_penalty, restrict_knowledge

logger = logging.getLogger(__name__)

DISTANCE_MODES = ("skeleton", "pag")


# ---------------------------------------------------------------------------
# Variable subsets
# ---------------------------------------------------------------------------

def uncovered_pairs(var_names: Sequence[str], subsets: List[List[str]]) -> Set[Tuple[str, str]]:
    """Unordered variable pairs that do not appear together in any subset."""
    covered = set()
    for subset in subsets:
        covered.update(itertools.combinations(sorted(subset), 2))
    return set(itertools.combinations(sorted(var_names), 2)) - covered


def sample_variable_subsets(
    var_names: List[str],
    m: int,
    subset_frac: float,
    rng: np.random.Generator,
    require_pair_coverage: bool = True,
) -> List[List[str]]:
    """Draw m random subsets of size ceil(subset_frac * p).

    If ``require_pair_coverage`` is set, extra subsets are added until every
    pair of variables appears together in at least one subset.  Each extra
    subset starts from an uncovered pair and is filled greedily with the
    variables that cover the most remaining uncovered pairs.  Without this,
    a pairwise claim may never be checked by any local discovery.
    """
    p = len(var_names)
    subset_size = min(p, max(3, int(np.ceil(subset_frac * p))))
    subsets = [list(rng.choice(var_names, size=subset_size, replace=False)) for _ in range(m)]

    if not require_pair_coverage:
        return subsets

    missing = uncovered_pairs(var_names, subsets)
    n_added = 0
    while missing:
        first_pair = sorted(missing)[int(rng.integers(len(missing)))]
        subset = list(first_pair)
        while len(subset) < subset_size:
            gains = {
                v: sum(tuple(sorted((v, u))) in missing for u in subset)
                for v in var_names
                if v not in subset
            }
            best_gain = max(gains.values())
            best = [v for v, g in gains.items() if g == best_gain]
            subset.append(best[int(rng.integers(len(best)))])
        subsets.append(subset)
        missing -= set(itertools.combinations(sorted(subset), 2))
        n_added += 1

    if n_added:
        logger.debug("Added %d subsets to cover every variable pair.", n_added)
    return subsets


# ---------------------------------------------------------------------------
# Marginal projections and distances
# ---------------------------------------------------------------------------

def project(marks: pd.DataFrame, keep: Sequence[str], mode: str) -> pd.DataFrame:
    """Marginalize a graph onto ``keep`` and return what the distance compares."""
    keep = list(keep)
    if mode == "skeleton":
        return skeleton(marginal_mag(marks, keep))
    if mode == "pag":
        return marginal_pag(marks, keep)
    raise ValueError(f"Unknown distance mode {mode!r}; expected one of {DISTANCE_MODES}.")


def graph_distance(P: pd.DataFrame, Q: pd.DataFrame, mode: str) -> float:
    """Normalized disagreement between two projections over the same variables.

    Averaged over unordered pairs of variables.  A pair costs 1 if exactly one
    graph has the edge.  In ``pag`` mode, a pair adjacent in both graphs also
    costs 1/2 for each endpoint mark that differs.  The result is in [0, 1].
    """
    names = list(P.index)
    Q = Q.loc[names, names]
    if len(names) < 2:
        return 0.0

    total = 0.0
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            in_P = P.loc[a, b] != NO_EDGE
            in_Q = Q.loc[a, b] != NO_EDGE
            if in_P != in_Q:
                total += 1.0
            elif in_P and mode == "pag":
                total += 0.5 * (P.loc[a, b] != Q.loc[a, b])
                total += 0.5 * (P.loc[b, a] != Q.loc[b, a])
    n_pairs = len(names) * (len(names) - 1) / 2
    return float(total / n_pairs)


# ---------------------------------------------------------------------------
# Local graphs and self-compatibility scores
# ---------------------------------------------------------------------------

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
        local_graphs[j] = discover_marks(
            data_train[subset],
            method=method,
            knowledge=restrict_knowledge(knowledge, subset),
            config=config,
            random_state=int(rng.integers(1_000_000)),
        )
    return local_graphs


def project_local_graphs(
    local_graphs: Dict[int, pd.DataFrame],
    subsets: List[List[str]],
    mode: str,
) -> Dict[int, pd.DataFrame]:
    """Projection of each local graph onto its own subset (the fixed references)."""
    return {j: project(local_graphs[j], subsets[j], mode) for j in local_graphs}


def self_compatibility_score_among_local_graphs(
    local_graphs: Dict[int, pd.DataFrame],
    subsets: List[List[str]],
    distance_mode: str = "skeleton",
) -> float:
    """Mean disagreement of pairs of local graphs on their shared variables.

    Both local graphs are marginalized onto the overlap before comparison.
    """
    losses = []
    keys = list(local_graphs.keys())
    for a, b in itertools.combinations(keys, 2):
        overlap = sorted(set(subsets[a]) & set(subsets[b]))
        if len(overlap) < 2:
            continue
        losses.append(
            graph_distance(
                project(local_graphs[a], overlap, distance_mode),
                project(local_graphs[b], overlap, distance_mode),
                distance_mode,
            )
        )
    return float(np.mean(losses)) if losses else 0.0


def candidate_graph_score(
    candidate: pd.DataFrame,
    local_projections: Dict[int, pd.DataFrame],
    subsets: List[List[str]],
    distance_mode: str = "skeleton",
    knowledge=None,
    lambda_penalty: float = 0.0,
) -> float:
    """A(G): mean distance between the marginals of G and the local graphs.

    Adds the optional soft knowledge penalty lambda * R_K(G).
    """
    losses = [
        graph_distance(project(candidate, subsets[j], distance_mode), local_projections[j], distance_mode)
        for j in local_projections
    ]
    score = float(np.mean(losses)) if losses else 0.0
    if knowledge is not None and lambda_penalty > 0.0:
        score += lambda_penalty * knowledge_penalty(to_adjacency(candidate), knowledge)
    return score


def calibration_scores(
    data_cal: pd.DataFrame,
    subsets: List[List[str]],
    local_projections: Dict[int, pd.DataFrame],
    knowledge,
    method: str,
    config,
    n_resamples: int,
    rng: np.random.Generator,
    distance_mode: str = "skeleton",
    chunk_size: Optional[int] = None,
    soft_knowledge=None,
    lambda_penalty: float = 0.0,
) -> np.ndarray:
    """Scores A_b on disjoint chunks of D_cal.

    For each chunk, local discovery is rerun on every subset and compared with
    the training local graph of the same subset.  Both graphs live on the same
    variables, so the projection only normalizes their representation; no
    variable is marginalized out.

    Chunks are disjoint (not bootstrap resamples) so that the scores stay
    exchangeable, which Proposition 1 requires.  With ``chunk_size=None`` the
    chunk size is len(D_cal) // n_resamples; otherwise only full chunks of the
    requested size are used, at most ``n_resamples`` of them.
    """
    n = len(data_cal)
    perm = rng.permutation(n)
    if chunk_size is None:
        chunk_size = max(1, n // n_resamples)
        n_chunks = n_resamples
    else:
        n_chunks = min(n_resamples, n // chunk_size)

    scores = []
    for b in range(n_chunks):
        start = b * chunk_size
        if start >= n:
            break
        data_b = data_cal.iloc[perm[start:start + chunk_size]].reset_index(drop=True)
        losses = []
        for j, subset in enumerate(subsets):
            try:
                local_b = discover_marks(
                    data_b[subset],
                    method=method,
                    knowledge=restrict_knowledge(knowledge, subset),
                    config=config,
                    random_state=int(rng.integers(1_000_000)),
                )
            except Exception as exc:
                logger.debug("Calibration chunk %d, subset %d: discovery failed (%s).", b, j, exc)
                continue
            d = graph_distance(project(local_b, subset, distance_mode), local_projections[j], distance_mode)
            if soft_knowledge is not None and lambda_penalty > 0.0:
                d += lambda_penalty * knowledge_penalty(
                    to_adjacency(local_b), restrict_knowledge(soft_knowledge, subset)
                )
            losses.append(d)
        if losses:
            scores.append(float(np.mean(losses)))

    return np.array(scores)


def conformal_threshold(scores: np.ndarray, alpha: float) -> float:
    """tau_alpha = the ceil((N+1)(1-alpha))-th smallest score, or +inf if N is too small."""
    N = len(scores)
    k = int(np.ceil((N + 1) * (1 - alpha)))
    if k > N:
        return float("inf")
    return float(np.sort(scores)[k - 1])


def conformal_graph_set(
    candidates: List[pd.DataFrame],
    candidate_scores: Sequence[float],
    tau: float,
) -> List[pd.DataFrame]:
    """C_alpha: the candidates whose score does not exceed tau."""
    return [G for G, s in zip(candidates, candidate_scores) if s <= tau]


# ---------------------------------------------------------------------------
# Candidate graphs
# ---------------------------------------------------------------------------

def hyperparameter_grid(method: str, config) -> List[Optional[float]]:
    """Grid used to build candidates: significance levels or maximum parents."""
    method = method.lower()
    if method in ("pc", "fci"):
        return list(config.candidate_alpha_grid)
    if method == "ges":
        return list(config.candidate_max_parents_grid)
    return []


def single_edge_edit(
    marks: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """One random edge addition, removal, or reversal of a graph."""
    edited = marks.copy()
    names = list(marks.index)
    pairs = list(itertools.combinations(names, 2))
    adjacent = [(a, b) for a, b in pairs if has_edge(marks, a, b)]
    non_adjacent = [(a, b) for a, b in pairs if not has_edge(marks, a, b)]
    directed = [(a, b) for a in names for b in names if a != b and is_directed(marks, a, b)]

    operations = []
    if non_adjacent:
        operations.append("add")
    if adjacent:
        operations.append("remove")
    if directed:
        operations.append("reverse")
    operation = operations[int(rng.integers(len(operations)))]

    if operation == "add":
        a, b = non_adjacent[int(rng.integers(len(non_adjacent)))]
        if rng.random() < 0.5:
            a, b = b, a
        set_edge(edited, a, b, TAIL, ARROW)
    elif operation == "remove":
        a, b = adjacent[int(rng.integers(len(adjacent)))]
        remove_edge(edited, a, b)
    else:
        a, b = directed[int(rng.integers(len(directed)))]
        set_edge(edited, b, a, TAIL, ARROW)
    return edited


def generate_candidate_graphs(
    data_train: pd.DataFrame,
    knowledge,
    method: str,
    config,
    n_candidates: int,
    rng: np.random.Generator,
) -> List[pd.DataFrame]:
    """The finite candidate family used to approximate C_alpha.

    It contains, in order:
      1. the base algorithm on the full D_train;
      2. the base algorithm on D_train for each value of the hyperparameter grid;
      3. the base algorithm on bootstrap resamples of D_train;
      4. single-edge additions, removals, or reversals of the full-data graph.
    Bootstraps take a fraction ``config.candidate_bootstrap_frac`` of the
    ``n_candidates`` slots left after the grid; edits fill the rest.  Every
    candidate satisfies the hard knowledge.
    """
    def run(data, hyperparameter=None):
        return discover_marks(
            data,
            method=method,
            knowledge=knowledge,
            config=config,
            random_state=int(rng.integers(1_000_000)),
            hyperparameter=hyperparameter,
        )

    full_graph = run(data_train)
    candidates = [full_graph]

    default = default_hyperparameter(method, config)
    for value in hyperparameter_grid(method, config):
        if value == default:
            continue
        try:
            candidates.append(run(data_train, value))
        except Exception as exc:
            logger.debug("Grid candidate %s failed (%s).", value, exc)

    remaining = max(0, n_candidates - (len(candidates) - 1))
    n_bootstrap = int(round(config.candidate_bootstrap_frac * remaining))
    n = len(data_train)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        try:
            candidates.append(run(data_train.iloc[idx].reset_index(drop=True)))
        except Exception as exc:
            # e.g. a near-collinear resample can make GES fail.
            logger.debug("Bootstrap candidate failed (%s).", exc)

    n_edits = max(0, n_candidates + 1 - len(candidates))
    for _ in range(n_edits):
        edited = single_edge_edit(full_graph, rng)
        candidates.append(apply_hard_domain_constraints(edited, knowledge))

    return candidates
