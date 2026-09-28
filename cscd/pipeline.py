"""One repeat of CSCD with everything the revision analyses need.

``fit_repeat`` runs the method once and returns a *record*: calibration,
held-out and candidate scores for several distances, the candidate graphs,
bootstrap graphs on D_train and on the full data, Bayesian edge posteriors,
the ablation scores, local error indicators, the separation-cover status of
the subsets, and timings.  Claims, thresholds and all baselines at any
alpha, gamma or selection threshold are then computed from the record by
``cscd.analysis``, so sweeps do not rerun discovery.

Chunk design.  ``n_cal_chunks`` disjoint chunks of D_cal set the threshold.
With ``cal_chunk_size=None`` they split D_cal evenly, so a small number of
chunks gives large chunks.  Held-out chunks of the same size come from a
fresh sample of the same model (synthetic data) or from what is left of
D_cal, and are used only to estimate the exceedance rate.
"""

import itertools
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .baselines import bootstrap_marks
from .bayes import bayesian_edge_posterior
from .graph import default_hyperparameter, discover_marks
from .mixed_graph import NO_EDGE, from_dag_adjacency, to_adjacency
from .selfcompat import (
    generate_candidate_graphs,
    hyperparameter_grid,
    graph_distance,
    project,
    sample_variable_subsets,
    self_compatibility_score_among_local_graphs,
    uncovered_pairs,
)
from .utils import knowledge_penalty, restrict_knowledge

logger = logging.getLogger(__name__)


@dataclass
class PipelineOptions:
    distances: Sequence[str] = ("skeleton",)
    n_cal_chunks: int = 9
    cal_chunk_size: Optional[int] = None
    n_test_chunks: int = 10
    train_frac: float = 0.5
    train_size: Optional[int] = None
    subset_mode: str = "random"          # random | mb
    run_bayes: bool = True
    run_full_data_baselines: bool = True
    run_ablations: bool = True
    n_bma_samples: int = 0               # posterior DAG samples for the decision task


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------

def base_mode(distance: str) -> str:
    """Projection used by a distance: 'pag' or 'skeleton' (weighted variants)."""
    return "pag" if distance == "pag" else "skeleton"


def weighted_skeleton_distance(P: pd.DataFrame, Q: pd.DataFrame, w_extra: float, w_miss: float) -> float:
    """Skeleton disagreement with separate costs for extra and missing edges.

    ``P`` is the graph being scored (candidate or chunk graph) and ``Q`` the
    reference (training local graph).  An edge in P but not in Q costs
    ``w_extra``, an edge in Q but not in P costs ``w_miss``.  Normalized to
    [0, 1] by the number of pairs times the larger weight.
    """
    names = list(P.index)
    Q = Q.loc[names, names]
    k = len(names)
    if k < 2:
        return 0.0
    iu = np.triu_indices(k, 1)
    inP = P.values[iu] != NO_EDGE
    inQ = Q.values[iu] != NO_EDGE
    total = w_extra * np.sum(inP & ~inQ) + w_miss * np.sum(~inP & inQ)
    return float(total / (len(iu[0]) * max(w_extra, w_miss)))


def distance_value(P: pd.DataFrame, Q: pd.DataFrame, distance: str) -> float:
    if distance in ("skeleton", "pag"):
        return graph_distance(P, Q, distance)
    if distance == "skel_extra2":
        return weighted_skeleton_distance(P, Q, 2.0, 1.0)
    if distance == "skel_miss2":
        return weighted_skeleton_distance(P, Q, 1.0, 2.0)
    raise ValueError(f"Unknown distance {distance!r}.")


# ---------------------------------------------------------------------------
# Subsets
# ---------------------------------------------------------------------------

def markov_blanket_subsets(
    full_adj: pd.DataFrame,
    subset_size: int,
    rng: np.random.Generator,
) -> List[List[str]]:
    """One subset per variable: the variable, its estimated Markov blanket, fill-up.

    The blanket is read from the full-data graph (adjacencies plus co-parents
    of children, undirected edges counted as adjacencies).  If it is larger
    than the subset size, adjacencies are kept first.  Subsets are filled with
    random variables up to ``subset_size``.
    """
    names = list(full_adj.index)
    A = full_adj.loc[names, names].values
    adjacent = (A + A.T) > 0
    directed = (A == 1) & (A.T == 0)
    subsets = []
    for i, v in enumerate(names):
        neighbours = [names[j] for j in np.flatnonzero(adjacent[i])]
        children = np.flatnonzero(directed[i])
        spouses = {names[s] for c in children for s in np.flatnonzero(directed[:, c]) if s != i}
        spouses -= set(neighbours)
        order = list(rng.permutation(neighbours)) + list(rng.permutation(sorted(spouses)))
        subset = [v] + order[: subset_size - 1]
        rest = [u for u in names if u not in subset]
        fill = list(rng.permutation(rest)[: max(0, subset_size - len(subset))])
        subsets.append(subset + fill)
    return subsets


def add_pair_coverage(var_names: List[str], subsets: List[List[str]], size: int, rng) -> List[List[str]]:
    """Greedily add subsets until every pair of variables co-occurs (as in selfcompat)."""
    missing = uncovered_pairs(var_names, subsets)
    subsets = [list(s) for s in subsets]
    while missing:
        first = sorted(missing)[int(rng.integers(len(missing)))]
        subset = list(first)
        while len(subset) < size:
            gains = {v: sum(tuple(sorted((v, u))) in missing for u in subset) for v in var_names if v not in subset}
            best = max(gains.values())
            cands = [v for v, g in gains.items() if g == best]
            subset.append(cands[int(rng.integers(len(cands)))])
        subsets.append(subset)
        missing -= set(itertools.combinations(sorted(subset), 2))
    return subsets


def separation_cover(true_marks: pd.DataFrame, var_names: List[str], subsets: List[List[str]]) -> float:
    """Fraction of truly non-adjacent pairs that some subset separates (Def. 3).

    A pair non-adjacent in the marginal over V is certified by subset S if it
    is also non-adjacent in the marginal over S.  Returns 1.0 if there are no
    non-adjacent pairs.
    """
    full = project(true_marks, var_names, "skeleton")
    targets = [(a, b) for a, b in itertools.combinations(var_names, 2) if full.loc[a, b] == NO_EDGE]
    if not targets:
        return 1.0
    covered = set()
    for S in subsets:
        proj = project(true_marks, S, "skeleton")
        for a, b in itertools.combinations(S, 2):
            if proj.loc[a, b] == NO_EDGE:
                covered.add(tuple(sorted((a, b))))
    return float(np.mean([tuple(sorted(t)) in covered for t in targets]))


# ---------------------------------------------------------------------------
# One repeat
# ---------------------------------------------------------------------------

def _adj_array(G: pd.DataFrame, names: List[str]) -> np.ndarray:
    return to_adjacency(G).loc[names, names].values.astype(np.int8)


def _local(data: pd.DataFrame, subsets, knowledge, method, config, rng) -> List[Optional[pd.DataFrame]]:
    out = []
    for S in subsets:
        try:
            out.append(discover_marks(data[S], method, restrict_knowledge(knowledge, S), config,
                                      random_state=int(rng.integers(1_000_000))))
        except Exception as exc:  # recorded as a failure count
            logger.debug("Local discovery failed (%s).", exc)
            out.append(None)
    return out


def _chunks(data: pd.DataFrame, n_chunks: int, size: int, rng) -> List[pd.DataFrame]:
    perm = rng.permutation(len(data))
    n_chunks = min(n_chunks, len(data) // max(size, 1))
    return [data.iloc[perm[b * size:(b + 1) * size]].reset_index(drop=True) for b in range(n_chunks)]


def fit_repeat(
    data: pd.DataFrame,
    method: str,
    config,
    knowledge,
    opts: PipelineOptions,
    rng_seed: int,
    true_full_dag: Optional[pd.DataFrame] = None,
    true_graph: Optional[pd.DataFrame] = None,
    test_data: Optional[pd.DataFrame] = None,
    soft_knowledge=None,
    lambda_penalty: float = 0.0,
    weights: Optional[np.ndarray] = None,
    decision_alpha: float = 0.1,
    decision_n_pairs: int = 5,
) -> Dict:
    rng = np.random.default_rng(rng_seed)
    names = list(data.columns)
    times: Dict[str, float] = {}
    t0 = time.perf_counter()

    # Split.
    n = len(data)
    perm = rng.permutation(n)
    n_tr = opts.train_size if opts.train_size is not None else int(round(opts.train_frac * n))
    D_tr = data.iloc[perm[:n_tr]].reset_index(drop=True)
    D_cal = data.iloc[perm[n_tr:]].reset_index(drop=True)
    # Without a fresh sample (real data), held-out chunks are carved from D_cal
    # next to the calibration chunks, all of the same size.
    n_split = opts.n_cal_chunks + (opts.n_test_chunks if test_data is None else 0)
    chunk = opts.cal_chunk_size or max(1, len(D_cal) // n_split)

    # Full-data run first (also used by Markov-blanket subsets).
    k = min(len(names), max(3, int(np.ceil(config.subset_frac * len(names)))))
    candidates = generate_candidate_graphs(D_tr, knowledge, method, config, config.n_candidate_graphs, rng)
    times["candidates"] = time.perf_counter() - t0
    # Candidates: [full run, grid runs, bootstrap runs, single-edge edits].
    default = default_hyperparameter(method, config)
    n_grid = sum(v != default for v in hyperparameter_grid(method, config))

    # Subsets.
    if opts.subset_mode == "mb":
        subsets = markov_blanket_subsets(to_adjacency(candidates[0]), k, rng)
        if config.require_pair_coverage:
            subsets = add_pair_coverage(names, subsets, k, rng)
    else:
        subsets = sample_variable_subsets(names, config.n_subsets, config.subset_frac, rng,
                                          require_pair_coverage=config.require_pair_coverage)

    t = time.perf_counter()
    local = _local(D_tr, subsets, knowledge, method, config, rng)
    keep = [j for j, g in enumerate(local) if g is not None]
    n_dropped = len(local) - len(keep)
    subsets = [subsets[j] for j in keep]
    local = [local[j] for j in keep]
    modes = sorted({base_mode(d) for d in opts.distances})
    ref = {md: [project(g, S, md) for g, S in zip(local, subsets)] for md in modes}
    times["local_train"] = time.perf_counter() - t

    true_marks = from_dag_adjacency(true_full_dag) if true_full_dag is not None else None
    truth_proj = (
        {md: [project(true_marks, S, md) for S in subsets] for md in modes} if true_marks is not None else None
    )

    def score_graph_on_subsets(proj_by_mode, penalty_graph=None) -> Dict[str, float]:
        out = {}
        for d in opts.distances:
            md = base_mode(d)
            out[d] = float(np.mean([distance_value(P, R, d) for P, R in zip(proj_by_mode[md], ref[md])]))
        return out

    # Calibration and held-out chunk scores (and local errors on chunks).
    def chunk_scores(chunks: List[pd.DataFrame]):
        scores = {d: [] for d in opts.distances}
        errors, failures = [], 0
        for C in chunks:
            loc = _local(C, subsets, knowledge, method, config, rng)
            failures += sum(g is None for g in loc)
            per = {d: [] for d in opts.distances}
            err = []
            for j, g in enumerate(loc):
                if g is None:
                    continue
                pen = 0.0
                if soft_knowledge is not None and lambda_penalty > 0:
                    pen = lambda_penalty * knowledge_penalty(to_adjacency(g), restrict_knowledge(soft_knowledge, subsets[j]))
                projs = {md: project(g, subsets[j], md) for md in modes}
                for d in opts.distances:
                    md = base_mode(d)
                    per[d].append(distance_value(projs[md], ref[md][j], d) + pen)
                if truth_proj is not None:
                    err.append(float(graph_distance(projs["skeleton"] if "skeleton" in projs else project(g, subsets[j], "skeleton"),
                                                    project(true_marks, subsets[j], "skeleton"), "skeleton") > 0))
            for d in opts.distances:
                if per[d]:
                    scores[d].append(float(np.mean(per[d])))
            if err:
                errors.append(float(np.mean(err)))
        return {d: np.array(v) for d, v in scores.items()}, errors, failures

    t = time.perf_counter()
    all_chunks = _chunks(D_cal, n_split, chunk, rng)
    cal_chunks = all_chunks[:opts.n_cal_chunks]
    cal, cal_err, cal_fail = chunk_scores(cal_chunks)
    times["calibration"] = time.perf_counter() - t

    if test_data is not None:
        test_chunks = _chunks(test_data, opts.n_test_chunks, chunk, rng)
    else:
        test_chunks = all_chunks[opts.n_cal_chunks:]
    t = time.perf_counter()
    test, _, test_fail = chunk_scores(test_chunks)
    times["test_chunks"] = time.perf_counter() - t

    # Candidate and truth scores.
    t = time.perf_counter()
    cand_scores = {d: [] for d in opts.distances}
    for G in candidates:
        projs = {md: [project(G, S, md) for S in subsets] for md in modes}
        s = score_graph_on_subsets(projs)
        pen = lambda_penalty * knowledge_penalty(to_adjacency(G), soft_knowledge) if (soft_knowledge is not None and lambda_penalty > 0) else 0.0
        for d in opts.distances:
            cand_scores[d].append(s[d] + pen)
    true_score = {}
    if truth_proj is not None:
        s = score_graph_on_subsets(truth_proj)
        pen = lambda_penalty * knowledge_penalty(true_graph, soft_knowledge) if (soft_knowledge is not None and lambda_penalty > 0 and true_graph is not None) else 0.0
        true_score = {d: s[d] + pen for d in opts.distances}
    times["candidate_scores"] = time.perf_counter() - t

    local_err_train = None
    if truth_proj is not None:
        local_err_train = [
            float(graph_distance(project(g, S, "skeleton"), project(true_marks, S, "skeleton"), "skeleton") > 0)
            for g, S in zip(local, subsets)
        ]

    rec: Dict = {
        "var_names": names,
        "method": method,
        "n": n, "n_train": len(D_tr), "chunk_size": chunk,
        "n_cal": len(cal_chunks), "n_test": len(test_chunks),
        "subsets": subsets,
        "cal_scores": cal, "test_scores": test,
        "cand_scores": {d: np.array(v) for d, v in cand_scores.items()},
        "true_score": true_score,
        "cand_adj": np.stack([_adj_array(G, names) for G in candidates]),
        "cand_marks": np.stack([G.loc[names, names].values.astype(np.int8) for G in candidates]),
        "n_grid": n_grid,
        "raw_selfcompat": self_compatibility_score_among_local_graphs(
            {j: g for j, g in enumerate(local)}, subsets, distance_mode="skeleton") if len(local) < 80 else float("nan"),
        "local_err_train": local_err_train,
        "local_err_chunks": cal_err,
        "failures": {"cal": cal_fail, "test": test_fail, "train_subsets_dropped": n_dropped},
        "true_graph": true_graph.loc[names, names].values.astype(np.int8) if true_graph is not None else None,
        "sep_cover": separation_cover(true_marks, names, subsets) if true_marks is not None else float("nan"),
    }

    # Baselines: bootstrap graphs, Bayesian posterior, full-data variants.
    t = time.perf_counter()
    boot = bootstrap_marks(D_tr, knowledge, method, config, rng, config.stability_n_bootstraps)
    rec["boot_adj"] = np.stack([_adj_array(G, names) for G in boot]) if boot else None
    times["bootstrap_train"] = time.perf_counter() - t
    if opts.run_full_data_baselines:
        t = time.perf_counter()
        full_run = discover_marks(data, method, knowledge, config, random_state=int(rng.integers(1_000_000)))
        rec["single_full_adj"] = _adj_array(full_run, names)
        bootF = bootstrap_marks(data, knowledge, method, config, rng, config.stability_n_bootstraps)
        rec["boot_full_adj"] = np.stack([_adj_array(G, names) for G in bootF]) if bootF else None
        times["bootstrap_full"] = time.perf_counter() - t
    if opts.run_bayes:
        t = time.perf_counter()
        try:
            rec["bayes_train"] = bayesian_edge_posterior(D_tr, rng, n_iterations=config.bayes_n_iterations).loc[names, names].values
            if opts.run_full_data_baselines:
                rec["bayes_full"] = bayesian_edge_posterior(data, rng, n_iterations=config.bayes_n_iterations).loc[names, names].values
        except Exception as exc:
            logger.warning("Bayesian baseline failed (%s).", exc)
        times["bayes"] = time.perf_counter() - t

    # Localization ablations (skeleton projection over V; first distance).
    if opts.run_ablations:
        t = time.perf_counter()
        d0 = opts.distances[0]
        md = base_mode(d0)
        boot_proj = [project(G, names, md) for G in boot]

        def mean_to_boot(P, skip=None):
            ds = [distance_value(P, Q, d0) for i, Q in enumerate(boot_proj) if i != skip]
            return float(np.mean(ds)) if ds else 0.0

        rec["bootcompat_ref"] = np.array([mean_to_boot(Q, skip=i) for i, Q in enumerate(boot_proj)])
        rec["bootcompat_cand"] = np.array([mean_to_boot(project(G, names, md)) for G in candidates])
        rec["bootcompat_truth"] = mean_to_boot(project(true_marks, names, md)) if true_marks is not None else float("nan")
        # Conformal without subsets: reference = full-data run on D_tr, chunks of D_cal on all variables.
        refV = project(candidates[0], names, md)
        ns_cal, ns_fail = [], 0
        for C in cal_chunks:
            try:
                g = discover_marks(C, method, knowledge, config, random_state=int(rng.integers(1_000_000)))
                ns_cal.append(distance_value(project(g, names, md), refV, d0))
            except Exception as exc:
                ns_fail += 1
                logger.debug("No-subset calibration chunk failed (%s).", exc)
        rec["failures"]["nosub_cal"] = ns_fail
        rec["nosub_cal"] = np.array(ns_cal)
        rec["nosub_cand"] = np.array([distance_value(project(G, names, md), refV, d0) for G in candidates])
        rec["nosub_truth"] = distance_value(project(true_marks, names, md), refV, d0) if true_marks is not None else float("nan")
        times["ablations"] = time.perf_counter() - t

    # Downstream decision task (linear, fully observed models only), on the
    # fresh held-out sample so that no estimate reuses D_train.
    if weights is not None and true_graph is not None and test_data is not None and len(test_data) > 0:
        from .decision_v2 import bma_dag_samples, decision_task_v2
        from .selfcompat import conformal_threshold

        t = time.perf_counter()
        d0 = opts.distances[0]
        tau = conformal_threshold(cal[d0], decision_alpha)
        admitted = [G for G, s in zip(candidates, cand_scores[d0]) if s <= tau]
        sets = {
            "single": [candidates[0]],
            "cscd_set": admitted,
            "pool": candidates,
            "bootstrap_avg": boot,
        }
        if opts.n_bma_samples > 0:
            try:
                sets["bma"] = bma_dag_samples(D_tr, rng, n_samples=opts.n_bma_samples,
                                              n_iterations=config.bayes_n_iterations)
            except Exception as exc:
                logger.warning("BMA sampling failed (%s).", exc)
        try:
            rec["decision"] = decision_task_v2(
                test_data, weights, names, true_graph.loc[names, names], sets,
                np.random.default_rng(rng_seed + 7919), n_pairs=decision_n_pairs,
            )
        except Exception as exc:
            logger.warning("Decision task failed (%s).", exc)
        times["decision"] = time.perf_counter() - t

    rec["times"] = times
    rec["time_total"] = time.perf_counter() - t0
    rec["_D_tr_index"] = perm[:n_tr]
    return rec
