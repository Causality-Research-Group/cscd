"""One repeat of CSCD for the second revision (review of the AISTATS draft).

``fit_repeat_v3`` extends ``pipeline.fit_repeat`` with everything the second
revision needs, and stores raw objects rather than summaries so that every
analysis can be recomputed from the record:

* every local graph (training, calibration and held-out chunks) as endpoint
  marks, together with the true projection of each subset, so that the
  pairwise decomposition of E[A_b | D_train] - A(G*) can be computed;
* a tagged candidate pool (``pool_arms``), from which the candidate-family
  arms are formed afterwards (bootstrap only, edits only, grid only,
  multi-algorithm, order-MCMC samples, pool with G*, pool padded with random
  sparse DAGs);
* calibration scores with and without the soft penalty (``calnopen``), so
  that the penalty can be applied to candidates only;
* external detectors (Faller et al.'s kappa_G, the permutation test of
  Eulig et al., LOVO of Schkoda et al.) next to CSCD, bootstrap
  compatibility and conformal without subsets;
* multi-chain order MCMC with convergence diagnostics and DAG samples.

``score_only`` skips candidates and baselines except the full-data run, for
the large-repeat validity check.
"""

import itertools
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .baselines import bootstrap_marks
from .graph import default_hyperparameter, discover_marks
from .mixed_graph import NO_EDGE, from_dag_adjacency, marginal_mag, to_adjacency
from .pipeline import (
    _chunks,
    _local,
    add_pair_coverage,
    markov_blanket_subsets,
    separation_cover,
)
from .selfcompat import (
    graph_distance,
    hyperparameter_grid,
    project,
    sample_variable_subsets,
    self_compatibility_score_among_local_graphs,
    single_edge_edit,
)
from .utils import apply_hard_domain_constraints, knowledge_penalty, restrict_knowledge

logger = logging.getLogger(__name__)

FAILED = -1  # marker for a failed local run in the stored arrays


@dataclass
class OptionsV3:
    n_cal_chunks: int = 9
    cal_chunk_size: Optional[int] = None
    n_test_chunks: int = 10
    train_frac: float = 0.5
    train_size: Optional[int] = None
    subset_mode: str = "random"          # random | mb
    score_only: bool = False             # validity check: no candidates or baselines
    pool: str = "standard"               # standard | arms
    n_boot_baseline: int = 50            # bootstrap graphs for stability / bootstrap compatibility
    run_full_data_baselines: bool = False
    run_bayes: bool = False              # multi-chain order MCMC on D_train (and D if full baselines)
    mcmc_chains: int = 4
    mcmc_iterations: int = 10000
    run_ablations: bool = True           # bootstrap compatibility, conformal without subsets
    run_external: bool = False           # kappa_G, Eulig permutation test, LOVO
    eulig_permutations: int = 100
    lovo_pairs: int = 20
    run_decision: bool = False
    decision_n_pairs: int = 5
    store_local: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _skel(marks: pd.DataFrame, names: List[str]) -> np.ndarray:
    return (project(marks, names, "skeleton").loc[names, names].values != NO_EDGE)


def _dist_arrays(P: np.ndarray, Q: np.ndarray) -> float:
    k = P.shape[0]
    iu = np.triu_indices(k, 1)
    return float(np.mean(P[iu] != Q[iu])) if len(iu[0]) else 0.0


def _stack_local(graphs: List[Optional[pd.DataFrame]], subsets: List[List[str]]) -> np.ndarray:
    k = len(subsets[0])
    out = np.full((len(graphs), k, k), FAILED, dtype=np.int8)
    for j, (g, S) in enumerate(zip(graphs, subsets)):
        if g is not None and len(S) == k:
            out[j] = g.loc[S, S].values.astype(np.int8)
    return out


def random_sparse_dag_marks(names: List[str], n_edges: int, rng: np.random.Generator) -> pd.DataFrame:
    """Marks of a uniformly random DAG with ``n_edges`` edges (random order)."""
    p = len(names)
    order = rng.permutation(p)
    pairs = [(order[i], order[j]) for i in range(p) for j in range(i + 1, p)]
    chosen = rng.choice(len(pairs), size=min(n_edges, len(pairs)), replace=False)
    A = np.zeros((p, p), dtype=int)
    for c in chosen:
        a, b = pairs[c]
        A[a, b] = 1
    return from_dag_adjacency(pd.DataFrame(A, index=names, columns=names))


def mec_key(marks: pd.DataFrame, names: List[str]) -> bytes:
    """Key of the Markov equivalence class of a candidate.

    The marginal MAG over V of one member is computed, and the class is keyed
    by its skeleton and its unshielded colliders.  This identifies the class
    exactly for DAG-like outputs (Verma and Pearl, 1990); for MAGs with
    discriminating paths it can merge classes, so deduplication is slightly
    conservative there.
    """
    mag = marginal_mag(marks, names).loc[names, names].values
    adj = mag != NO_EDGE
    p = len(names)
    coll = np.zeros((p, p, p), dtype=bool)
    for c in range(p):
        nb = np.flatnonzero(adj[c])
        for a, b in itertools.combinations(nb, 2):
            if not adj[a, b] and mag[a, c] == 2 and mag[b, c] == 2:
                coll[a, b, c] = coll[b, a, c] = True
    return adj.tobytes() + np.packbits(coll).tobytes()


# ---------------------------------------------------------------------------
# Candidate pool
# ---------------------------------------------------------------------------

def candidate_pool(
    D_tr: pd.DataFrame,
    knowledge,
    method: str,
    config,
    rng: np.random.Generator,
    arms: bool,
    true_marks: Optional[pd.DataFrame] = None,
    mcmc_samples: Optional[np.ndarray] = None,
    n_arm: int = 100,
):
    """Candidates and their tags.

    The standard family (``arms=False``) is that of the paper: full-data run,
    hyperparameter grid, bootstrap runs on half of the remaining slots, and
    single-edge edits of the full-data run for the rest.  With ``arms=True``
    each source contributes up to ``n_arm`` candidates of its own, and the
    candidate-family arms are subsets of the pool selected by tag.
    """
    names = list(D_tr.columns)

    def run(data, meth=method, hp=None):
        return discover_marks(data, meth, knowledge, config, random_state=int(rng.integers(1_000_000)),
                              hyperparameter=hp)

    full = run(D_tr)
    cands, tags = [full], ["full"]
    default = default_hyperparameter(method, config)
    for v in hyperparameter_grid(method, config):
        if v == default:
            continue
        try:
            cands.append(run(D_tr, hp=v))
            tags.append("grid")
        except Exception as exc:
            logger.debug("grid candidate failed (%s)", exc)

    n = len(D_tr)
    if not arms:
        remaining = max(0, config.n_candidate_graphs - (len(cands) - 1))
        n_boot = int(round(config.candidate_bootstrap_frac * remaining))
        n_edit = None
    else:
        n_boot = n_arm
        n_edit = n_arm
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        try:
            cands.append(run(D_tr.iloc[idx].reset_index(drop=True)))
            tags.append("boot")
        except Exception as exc:
            logger.debug("bootstrap candidate failed (%s)", exc)
    if n_edit is None:
        n_edit = max(0, config.n_candidate_graphs + 1 - len(cands))
    for _ in range(n_edit):
        cands.append(apply_hard_domain_constraints(single_edge_edit(full, rng), knowledge))
        tags.append("edit")

    if arms:
        # Multi-algorithm: every base algorithm, its grid, and 10 bootstraps each.
        for meth in ("pc", "ges", "fci", "lingam"):
            hps = [None] + [v for v in hyperparameter_grid(meth, config) if v != default_hyperparameter(meth, config)]
            for hp in hps:
                try:
                    cands.append(run(D_tr, meth=meth, hp=hp))
                    tags.append("multi")
                except Exception as exc:
                    logger.debug("multi candidate failed (%s)", exc)
            for _ in range(10):
                idx = rng.choice(n, size=n, replace=True)
                try:
                    cands.append(run(D_tr.iloc[idx].reset_index(drop=True), meth=meth))
                    tags.append("multi")
                except Exception as exc:
                    logger.debug("multi bootstrap failed (%s)", exc)
        # Order-MCMC DAG samples.
        if mcmc_samples is not None and len(mcmc_samples):
            pick = rng.choice(len(mcmc_samples), size=min(n_arm, len(mcmc_samples)), replace=False)
            for i in pick:
                A = pd.DataFrame(mcmc_samples[i].astype(int), index=names, columns=names)
                cands.append(apply_hard_domain_constraints(from_dag_adjacency(A), knowledge))
                tags.append("mcmc")
        # Random sparse DAGs with as many edges as the full-data run.
        n_edges = int((to_adjacency(full).values + to_adjacency(full).values.T > 0).sum() // 2)
        for _ in range(n_arm):
            cands.append(apply_hard_domain_constraints(random_sparse_dag_marks(names, max(n_edges, 1), rng), knowledge))
            tags.append("random")
        if true_marks is not None:
            cands.append(true_marks.loc[names, names].copy())
            tags.append("truth")
    return cands, tags


# ---------------------------------------------------------------------------
# One repeat
# ---------------------------------------------------------------------------

def fit_repeat_v3(
    data: pd.DataFrame,
    method: str,
    config,
    knowledge,
    opts: OptionsV3,
    rng_seed: int,
    true_full_dag: Optional[pd.DataFrame] = None,
    true_graph: Optional[pd.DataFrame] = None,
    test_data: Optional[pd.DataFrame] = None,
    soft_knowledge=None,
    lambda_penalty: float = 0.0,
    weights: Optional[np.ndarray] = None,
) -> Dict:
    rng = np.random.default_rng(rng_seed)
    names = list(data.columns)
    times: Dict[str, float] = {}
    t0 = time.perf_counter()

    n = len(data)
    perm = rng.permutation(n)
    n_tr = opts.train_size if opts.train_size is not None else int(round(opts.train_frac * n))
    D_tr = data.iloc[perm[:n_tr]].reset_index(drop=True)
    D_cal = data.iloc[perm[n_tr:]].reset_index(drop=True)
    n_split = opts.n_cal_chunks + (opts.n_test_chunks if test_data is None else 0)
    chunk = opts.cal_chunk_size or max(1, len(D_cal) // n_split)
    k = min(len(names), max(3, int(np.ceil(config.subset_frac * len(names)))))
    true_marks = from_dag_adjacency(true_full_dag) if true_full_dag is not None else None

    # Bayesian posterior first: its samples feed the MCMC arm of the pool.
    rec: Dict = {}
    mcmc = None
    if opts.run_bayes and not opts.score_only:
        from .bayes import multi_chain_order_mcmc

        t = time.perf_counter()
        try:
            mcmc = multi_chain_order_mcmc(D_tr, rng, n_chains=opts.mcmc_chains, n_iterations=opts.mcmc_iterations)
            rec["bayes_train"] = mcmc["posterior"]
            rec["bayes_rhat"] = mcmc["rhat_logscore"]
            rec["bayes_max_edge_diff"] = mcmc["max_edge_diff"]
            rec["bayes_samples"] = mcmc["samples"][:: max(1, len(mcmc["samples"]) // 200)]
            if opts.run_full_data_baselines:
                mF = multi_chain_order_mcmc(data, rng, n_chains=opts.mcmc_chains, n_iterations=opts.mcmc_iterations)
                rec["bayes_full"] = mF["posterior"]
                rec["bayesF_rhat"] = mF["rhat_logscore"]
                rec["bayesF_samples"] = mF["samples"][:: max(1, len(mF["samples"]) // 200)]
        except Exception as exc:
            logger.warning("Order MCMC failed (%s).", exc)
        times["bayes"] = time.perf_counter() - t

    # Candidates (or only the full-data run).
    t = time.perf_counter()
    if opts.score_only:
        cands = [discover_marks(D_tr, method, knowledge, config, random_state=int(rng.integers(1_000_000)))]
        tags = ["full"]
    else:
        cands, tags = candidate_pool(D_tr, knowledge, method, config, rng, arms=(opts.pool == "arms"),
                                     true_marks=true_marks, mcmc_samples=None if mcmc is None else mcmc["samples"])
    times["candidates"] = time.perf_counter() - t

    # Subsets.
    if opts.subset_mode == "mb":
        subsets = markov_blanket_subsets(to_adjacency(cands[0]), k, rng)
        if config.require_pair_coverage:
            subsets = add_pair_coverage(names, subsets, k, rng)
    else:
        subsets = sample_variable_subsets(names, config.n_subsets, config.subset_frac, rng,
                                          require_pair_coverage=config.require_pair_coverage)

    t = time.perf_counter()
    local = _local(D_tr, subsets, knowledge, method, config, rng)
    keep = [j for j, g in enumerate(local) if g is not None]
    subsets = [subsets[j] for j in keep]
    local = [local[j] for j in keep]
    ref = [(project(g, S, "skeleton").loc[S, S].values != NO_EDGE) for g, S in zip(local, subsets)]
    times["local_train"] = time.perf_counter() - t
    truth_proj = None
    if true_marks is not None:
        truth_proj = [(project(true_marks, S, "skeleton").loc[S, S].values != NO_EDGE) for S in subsets]

    def chunk_pass(chunks: List[pd.DataFrame]):
        scores, scores_nopen, graphs = [], [], []
        for C in chunks:
            loc = _local(C, subsets, knowledge, method, config, rng)
            per, per_np = [], []
            for j, g in enumerate(loc):
                if g is None:
                    continue
                S = subsets[j]
                d = _dist_arrays(project(g, S, "skeleton").loc[S, S].values != NO_EDGE, ref[j])
                pen = 0.0
                if soft_knowledge is not None and lambda_penalty > 0:
                    pen = lambda_penalty * knowledge_penalty(to_adjacency(g), restrict_knowledge(soft_knowledge, S))
                per.append(d + pen)
                per_np.append(d)
            if per:
                scores.append(float(np.mean(per)))
                scores_nopen.append(float(np.mean(per_np)))
            graphs.append(_stack_local(loc, subsets) if opts.store_local else None)
        return np.array(scores), np.array(scores_nopen), graphs

    t = time.perf_counter()
    all_chunks = _chunks(D_cal, n_split, chunk, rng)
    cal_chunks = all_chunks[:opts.n_cal_chunks]
    cal, cal_np, cal_graphs = chunk_pass(cal_chunks)
    test_chunks = _chunks(test_data, opts.n_test_chunks, chunk, rng) if test_data is not None else all_chunks[opts.n_cal_chunks:]
    test, test_np, test_graphs = chunk_pass(test_chunks)
    times["chunks"] = time.perf_counter() - t

    def score(G: pd.DataFrame, with_pen: bool = True) -> float:
        s = float(np.mean([_dist_arrays(project(G, S, "skeleton").loc[S, S].values != NO_EDGE, R)
                           for S, R in zip(subsets, ref)]))
        if with_pen and soft_knowledge is not None and lambda_penalty > 0:
            s += lambda_penalty * knowledge_penalty(to_adjacency(G), soft_knowledge)
        return s

    t = time.perf_counter()
    cand_scores = np.array([score(G) for G in cands])
    times["candidate_scores"] = time.perf_counter() - t
    true_score = None
    if truth_proj is not None:
        true_score = float(np.mean([_dist_arrays(P, R) for P, R in zip(truth_proj, ref)]))
        if soft_knowledge is not None and lambda_penalty > 0 and true_graph is not None:
            true_score += lambda_penalty * knowledge_penalty(true_graph, soft_knowledge)

    rec.update({
        "var_names": names, "method": method, "n": n, "n_train": len(D_tr), "chunk_size": chunk,
        "n_cal": len(cal), "n_test": len(test), "subsets": subsets,
        "cal_scores": {"skeleton": cal, "calnopen": cal_np},
        "test_scores": {"skeleton": test, "calnopen": test_np},
        "cand_scores": {"skeleton": cand_scores, "calnopen": cand_scores},
        "true_score": {"skeleton": true_score, "calnopen": true_score} if true_score is not None else {},
        "cand_tags": tags,
        "cand_marks": np.stack([G.loc[names, names].values.astype(np.int8) for G in cands]),
        "cand_adj": np.stack([to_adjacency(G).loc[names, names].values.astype(np.int8) for G in cands]),
        "true_graph": true_graph.loc[names, names].values.astype(np.int8) if true_graph is not None else None,
        "true_skel": _skel(true_marks, names).astype(np.int8) if true_marks is not None else None,
        "sep_cover": separation_cover(true_marks, names, subsets) if true_marks is not None else float("nan"),
        "local_train": _stack_local(local, subsets) if opts.store_local else None,
        "local_cal": np.stack(cal_graphs) if opts.store_local and cal_graphs else None,
        "local_test": np.stack(test_graphs) if opts.store_local and test_graphs else None,
        "truth_proj": np.stack(truth_proj).astype(np.int8) if truth_proj is not None else None,
        "raw_selfcompat": self_compatibility_score_among_local_graphs(
            {j: g for j, g in enumerate(local)}, subsets, distance_mode="skeleton") if len(local) < 80 else float("nan"),
        "_D_tr_index": perm[:n_tr],
    })
    if true_marks is not None:
        rec["mec_true"] = mec_key(true_marks, names)
        # Causal sufficiency of each subset: no bidirected edge in the true marginal MAG.
        def _sufficient(S):
            m = marginal_mag(true_marks, S).values
            return not bool(((m == 2) & (m.T == 2)).any())

        rec["subset_sufficient"] = np.array([_sufficient(S) for S in subsets])
    if not opts.score_only:
        t = time.perf_counter()
        rec["cand_mec"] = [mec_key(G, names) for G in cands]
        times["mec"] = time.perf_counter() - t

    # Baselines.
    if not opts.score_only:
        t = time.perf_counter()
        boot = bootstrap_marks(D_tr, knowledge, method, config, rng, opts.n_boot_baseline)
        rec["boot_marks"] = np.stack([G.loc[names, names].values.astype(np.int8) for G in boot]) if boot else None
        rec["boot_adj"] = np.stack([to_adjacency(G).loc[names, names].values.astype(np.int8) for G in boot]) if boot else None
        if opts.run_full_data_baselines:
            single = discover_marks(data, method, knowledge, config, random_state=int(rng.integers(1_000_000)))
            rec["single_full_marks"] = single.loc[names, names].values.astype(np.int8)
            rec["single_full_adj"] = to_adjacency(single).loc[names, names].values.astype(np.int8)
            bootF = bootstrap_marks(data, knowledge, method, config, rng, opts.n_boot_baseline)
            rec["boot_full_adj"] = np.stack([to_adjacency(G).loc[names, names].values.astype(np.int8) for G in bootF]) if bootF else None
        times["bootstrap"] = time.perf_counter() - t

        if opts.run_ablations and boot:
            t = time.perf_counter()
            bsk = np.stack([_skel(G, names) for G in boot])
            iu = np.triu_indices(len(names), 1)
            bflat = bsk[:, iu[0], iu[1]]

            def mean_to_boot(sk_flat, skip=None):
                d = np.mean(bflat != sk_flat[None, :], axis=1)
                if skip is not None:
                    d = np.delete(d, skip)
                return float(d.mean())

            rec["bootcompat_ref"] = np.array([mean_to_boot(bflat[i], skip=i) for i in range(len(bflat))])
            cflat = [(_skel(G, names))[iu] for G in cands]
            rec["bootcompat_cand"] = np.array([mean_to_boot(c) for c in cflat])
            rec["bootcompat_truth"] = mean_to_boot(_skel(true_marks, names)[iu]) if true_marks is not None else float("nan")
            refV = _skel(cands[0], names)[iu]
            ns_cal = []
            for C in cal_chunks:
                try:
                    g = discover_marks(C, method, knowledge, config, random_state=int(rng.integers(1_000_000)))
                    ns_cal.append(float(np.mean(_skel(g, names)[iu] != refV)))
                except Exception as exc:
                    logger.debug("no-subset chunk failed (%s)", exc)
            rec["nosub_cal"] = np.array(ns_cal)
            rec["nosub_cand"] = np.array([float(np.mean(c != refV)) for c in cflat])
            rec["nosub_truth"] = float(np.mean(_skel(true_marks, names)[iu] != refV)) if true_marks is not None else float("nan")
            times["ablations"] = time.perf_counter() - t

    if opts.run_external:
        from .detectors import eulig_permutation_test, faller_kappa_g, lovo_scores

        ext: Dict[str, object] = {}
        t = time.perf_counter()
        try:
            ext["kappa_g"] = faller_kappa_g(cands[0], local, subsets, method)
            if true_marks is not None:
                ext["kappa_g_truth"] = faller_kappa_g(true_marks.loc[names, names], local, subsets, method)
        except Exception as exc:
            logger.warning("kappa_G failed (%s).", exc)
        times["kappa_g"] = time.perf_counter() - t
        t = time.perf_counter()
        try:
            ext["eulig"] = eulig_permutation_test(D_tr, cands[0], opts.eulig_permutations, rng)
        except Exception as exc:
            logger.warning("Eulig test failed (%s).", exc)
        times["eulig"] = time.perf_counter() - t
        t = time.perf_counter()
        try:
            ext["lovo"] = lovo_scores(D_tr, method, config, rng, n_pairs=opts.lovo_pairs)
        except Exception as exc:
            logger.warning("LOVO failed (%s).", exc)
        times["lovo"] = time.perf_counter() - t
        rec["external"] = ext

    if opts.run_decision and weights is not None and true_graph is not None and test_data is not None and len(test_data):
        from .decision_v2 import decision_task_v2
        from .selfcompat import conformal_threshold

        t = time.perf_counter()
        tau = conformal_threshold(cal, 0.1)
        admitted = [G for G, s, tg in zip(cands, cand_scores, tags) if s <= tau and tg != "truth"]
        std = [G for G, tg in zip(cands, tags) if tg in ("full", "grid", "boot", "edit")]
        sets = {"single": [cands[0]], "cscd_set": admitted, "pool": std,
                "bootstrap_avg": boot if not opts.score_only else []}
        if mcmc is not None:
            names_df = lambda A: pd.DataFrame(A.astype(int), index=names, columns=names)
            sets["bma"] = [names_df(A) for A in mcmc["samples"][:: max(1, len(mcmc["samples"]) // 100)]]
        try:
            rec["decision"] = decision_task_v2(test_data, weights, names, true_graph.loc[names, names], sets,
                                               np.random.default_rng(rng_seed + 7919), n_pairs=opts.decision_n_pairs)
        except Exception as exc:
            logger.warning("Decision task failed (%s).", exc)
        times["decision"] = time.perf_counter() - t

    rec["times"] = times
    rec["time_total"] = time.perf_counter() - t0
    return rec
