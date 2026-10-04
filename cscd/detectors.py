"""External falsification detectors used as comparison points.

Three published ways to check a causal discovery output without ground truth:

* ``faller_kappa_g``: the graphical incompatibility score kappa_G of Faller,
  Vankadara, Mastakouri, Locatello and Janzing, "Self-Compatibility:
  Evaluating Causal Discovery without Ground Truth" (AISTATS 2024,
  arXiv:2307.09552), Definition 10 with the practical choices of App. A14.2.
* ``eulig_permutation_test``: the node-permutation test of Eulig, Mastakouri,
  Bloebaum, Hardt and Janzing, "Toward Falsifying Causal Graphs Using a
  Permutation-Based Test" (AAAI 2025, arXiv:2305.09565), run through DoWhy's
  reference implementation ``dowhy.gcm.falsify.falsify_graph``.
* ``lovo_scores``: Leave-One-Variable-Out cross-validation of Schkoda, Faller,
  Bloebaum and Janzing, "Cross-validating causal discovery via
  Leave-One-Variable-Out" (CLeaR 2025, PMLR 275; arXiv:2411.05625), with the
  parent / path adjustment LOVO predictor and the MaxEnt baseline.

All graphs are endpoint-mark DataFrames (see ``mixed_graph``).
"""

import contextlib
import io
import itertools
import logging
import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

from .graph import discover_marks
from .mixed_graph import (
    ARROW,
    CIRCLE,
    LATENT_PREFIX,
    NO_EDGE,
    TAIL,
    canonical_dag,
    dag_marginal_pag,
    empty_marks,
    set_edge,
)
from .utils import make_domain_knowledge

logger = logging.getLogger(__name__)

CPDAG_METHODS = ("pc", "ges")
PAG_METHODS = ("fci",)
DAG_METHODS = ("lingam",)


def _graph_class(method: str) -> str:
    method = method.lower()
    if method in CPDAG_METHODS:
        return "cpdag"
    if method in PAG_METHODS:
        return "pag"
    if method in DAG_METHODS:
        return "dag"
    raise ValueError(f"Unknown method {method!r}; expected one of {CPDAG_METHODS + PAG_METHODS + DAG_METHODS}.")


# ---------------------------------------------------------------------------
# Shared graph helpers
# ---------------------------------------------------------------------------

def structural_hamming_distance(P: pd.DataFrame, Q: pd.DataFrame) -> int:
    """Number of unordered pairs whose edge differs between two mark graphs.

    A pair costs 1 if exactly one graph has an edge, or if both have one but
    either endpoint mark differs.  Unnormalized, as in Faller et al. (2024).
    """
    names = list(P.index)
    Pv = P.loc[names, names].to_numpy()
    Qv = Q.loc[names, names].to_numpy()
    differs = (Pv != Qv) | (Pv.T != Qv.T)
    return int(np.triu(differs, k=1).sum())


def latent_projection(dag: nx.DiGraph, keep: Sequence[str]) -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, str]]]:
    """Latent ADMG of a DAG over ``keep`` (Faller et al. 2024, Definition 5).

    * x -> y iff there is a directed path x -> ... -> y whose intermediate
      nodes are all outside ``keep``.
    * x <-> y iff there is a confounding path x <- ... <- u -> ... -> y whose
      non-endpoint nodes (u included) are all outside ``keep``.  If the two
      directed halves share a node w, w is itself a valid fork, so it
      suffices to collect, for every hidden u, the kept nodes reachable from
      u through hidden nodes only.

    Returns the directed edges (ordered pairs) and the bidirected edges
    (pairs sorted by name).  A pair can carry both kinds of edge.
    """
    keep = list(keep)
    keep_set = set(keep)

    def kept_reachable(start: str) -> Set[str]:
        found: Set[str] = set()
        stack = list(dag.successors(start))
        seen = set(stack)
        while stack:
            node = stack.pop()
            if node in keep_set:
                found.add(node)
                continue
            for child in dag.successors(node):
                if child not in seen:
                    seen.add(child)
                    stack.append(child)
        return found

    directed = {(x, y) for x in keep for y in kept_reachable(x) if y != x}
    bidirected: Set[Tuple[str, str]] = set()
    for u in dag.nodes:
        if u in keep_set:
            continue
        reached = sorted(kept_reachable(u))
        bidirected.update(itertools.combinations(reached, 2))
    return directed, bidirected


def dag_to_cpdag_marks(names: Sequence[str], directed_edges: Set[Tuple[str, str]]) -> pd.DataFrame:
    """CPDAG (as marks) of the DAG given by ``directed_edges`` over ``names``.

    Keeps the v-structures and closes the orientations under Meek's rules
    R1-R3, which give the CPDAG for a pattern without background knowledge
    (Meek 1995).
    """
    names = list(names)
    adjacent = {v: set() for v in names}
    parents = {v: set() for v in names}
    for a, b in directed_edges:
        adjacent[a].add(b)
        adjacent[b].add(a)
        parents[b].add(a)

    oriented: Set[Tuple[str, str]] = set()
    for z in names:
        for x, y in itertools.combinations(sorted(parents[z]), 2):
            if y not in adjacent[x]:
                oriented.add((x, z))
                oriented.add((y, z))

    def undirected(a: str, b: str) -> bool:
        return b in adjacent[a] and (a, b) not in oriented and (b, a) not in oriented

    changed = True
    while changed:
        changed = False
        for a in names:
            for b in names:
                if a == b or not undirected(a, b):
                    continue
                # R1: c -> a - b, c and b not adjacent  =>  a -> b
                r1 = any((c, a) in oriented and b not in adjacent[c] and c != b for c in adjacent[a])
                # R2: a -> c -> b and a - b  =>  a -> b
                r2 = any((a, c) in oriented and (c, b) in oriented for c in adjacent[a])
                # R3: a - c -> b, a - d -> b, c and d not adjacent  =>  a -> b
                r3 = False
                if not (r1 or r2):
                    cands = [c for c in adjacent[a] if undirected(a, c) and (c, b) in oriented]
                    r3 = any(d not in adjacent[c] for c, d in itertools.combinations(cands, 2))
                if r1 or r2 or r3:
                    oriented.add((a, b))
                    changed = True

    marks = empty_marks(names)
    for a, b in directed_edges:
        if (a, b) in oriented:
            set_edge(marks, a, b, TAIL, ARROW)
        else:
            set_edge(marks, a, b, TAIL, TAIL)
    return marks


# ---------------------------------------------------------------------------
# 1. Faller et al. (2024): graphical incompatibility score kappa_G
# ---------------------------------------------------------------------------

def faller_kappa_g(
    full_marks: pd.DataFrame,
    local_marks: List[pd.DataFrame],
    subsets: List[List[str]],
    method: str,
    fci_projection: str = "pag",
) -> float:
    """Graphical incompatibility score kappa_G of Faller et al. (2024).

    kappa_G = (1/k) sum_i SHD(L(A(X), S_i), A(X_{S_i})), Definition 10, where
    ``full_marks`` is A(X), ``local_marks[i]`` is A(X_{S_i}) and L is the
    latent projection onto S_i.  SHD counts unordered pairs that differ in
    adjacency or in either endpoint mark (1 per pair, unnormalized).

    The projection depends on the output class of ``method``:

    * PC / GES (CPDAGs), App. A14.2 "CPDAGs and hidden confounders": the
      latent ADMG is computed, its bidirected edges are deleted, and the CPDAG
      of the remaining DAG is compared with the local CPDAG.
    * FCI (PAGs): by default (``fci_projection="pag"``) the latent PAG of
      Definition 24, i.e. the PAG of the latent MAG of a member of the full
      PAG (``mixed_graph.dag_marginal_pag``).  With ``fci_projection="admg"``
      the latent ADMG itself is written as marks (x -> y, x <-> y; a pair with
      both a directed and a bidirected edge is written as the directed edge)
      and compared with the local PAG; this never matches a PAG that has
      circle marks, so it is not the default.
    * DirectLiNGAM (DAGs): the latent ADMG with bidirected edges deleted is
      compared with the local DAG directly (the analogue of the CPDAG rule;
      the paper does not run LiNGAM).

    Simplifications:

    * The full output is a CPDAG/PAG, not a single graph.  One member DAG is
      taken with ``canonical_dag`` (bidirected edges become explicit latent
      nodes ``__latent_*``, which are treated as outside every S_i).  For PC
      and GES, the CPDAG of the projected DAG can depend on the member chosen
      when S_i is not causally sufficient for that member; the paper is
      silent on this, and we use the canonical member.  The latent PAG is
      member-independent.
    * Cycles in invalid finite-sample outputs are broken by ``canonical_dag``
      (deleting one edge per cycle), as Faller et al. do.
    """
    if len(local_marks) != len(subsets):
        raise ValueError("local_marks and subsets must have the same length.")
    if not subsets:
        return float("nan")
    graph_class = _graph_class(method)
    if fci_projection not in ("pag", "admg"):
        raise ValueError("fci_projection must be 'pag' or 'admg'.")

    dag, _ = canonical_dag(full_marks)
    total = 0
    for subset, local in zip(subsets, local_marks):
        subset = list(subset)
        local = local.loc[subset, subset]
        if graph_class == "pag" and fci_projection == "pag":
            # causal-learn's dag2pag prints progress; keep this function silent.
            with contextlib.redirect_stdout(io.StringIO()):
                projected = dag_marginal_pag(dag, subset)
        else:
            directed, bidirected = latent_projection(dag, subset)
            if graph_class == "cpdag":
                projected = dag_to_cpdag_marks(subset, directed)
            else:
                projected = empty_marks(subset)
                if graph_class == "pag":
                    for a, b in bidirected:
                        set_edge(projected, a, b, ARROW, ARROW)
                for a, b in directed:
                    set_edge(projected, a, b, TAIL, ARROW)
        total += structural_hamming_distance(projected, local)
    return float(total / len(subsets))


# ---------------------------------------------------------------------------
# 2. Eulig et al. (2025): permutation-based falsification test (DoWhy)
# ---------------------------------------------------------------------------

def _as_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a.reshape(len(a), -1)


def _residualize(v: np.ndarray, Z: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(v)), Z])
    coef, *_ = np.linalg.lstsq(design, v, rcond=None)
    return v - design @ coef


def _fisher_z_p_value(x: np.ndarray, y: np.ndarray, n_cond: int) -> float:
    n = len(x)
    sx, sy = np.std(x), np.std(y)
    if sx < 1e-12 or sy < 1e-12:
        return 1.0
    r = float(np.clip(np.corrcoef(x, y)[0, 1], -0.999999, 0.999999))
    dof = n - n_cond - 3
    if dof <= 0:
        return 1.0
    statistic = np.sqrt(dof) * np.arctanh(r)
    return float(2.0 * stats.norm.sf(abs(statistic)))


def fisher_z_independence_test(X: np.ndarray, Y: np.ndarray) -> float:
    """p-value of the Fisher-z test of zero correlation (DoWhy test signature).

    Multivariate X or Y are handled by Bonferroni over column pairs.
    """
    X, Y = _as_2d(X), _as_2d(Y)
    p_values = [_fisher_z_p_value(X[:, i], Y[:, j], 0) for i in range(X.shape[1]) for j in range(Y.shape[1])]
    return float(min(1.0, min(p_values) * len(p_values)))


def fisher_z_conditional_independence_test(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> float:
    """p-value of the Fisher-z test of zero partial correlation given Z."""
    X, Y, Z = _as_2d(X), _as_2d(Y), _as_2d(Z)
    p_values = []
    for i in range(X.shape[1]):
        rx = _residualize(X[:, i], Z)
        for j in range(Y.shape[1]):
            ry = _residualize(Y[:, j], Z)
            p_values.append(_fisher_z_p_value(rx, ry, Z.shape[1]))
    return float(min(1.0, min(p_values) * len(p_values)))


def marks_to_observed_dag(marks: pd.DataFrame) -> nx.DiGraph:
    """A DAG over the observed variables of a mark graph.

    ``canonical_dag`` gives a member DAG in which each bidirected edge is a
    latent node with two children.  Each latent is replaced by a directed
    edge between its two children, oriented a -> b (a before b by name)
    unless that would close a cycle, in which case b -> a.  If the children
    are already adjacent, nothing is added.
    """
    dag, latents = canonical_dag(marks)
    out = nx.DiGraph()
    out.add_nodes_from(marks.index)
    out.add_edges_from((a, b) for a, b in dag.edges if not str(a).startswith(LATENT_PREFIX))
    for latent in latents:
        a, b = sorted(dag.successors(latent))
        if out.has_edge(a, b) or out.has_edge(b, a):
            continue
        if nx.has_path(out, b, a):
            out.add_edge(b, a)
        else:
            out.add_edge(a, b)
    return out


def eulig_permutation_test(
    data: pd.DataFrame,
    marks: pd.DataFrame,
    n_permutations: int,
    rng: np.random.Generator,
    significance: float = 0.05,
    significance_ci: float = 0.05,
) -> Dict[str, object]:
    """Node-permutation falsification test of Eulig et al. (2025), via DoWhy.

    The graph is converted to a DAG over the observed variables with
    ``marks_to_observed_dag`` (member DAG of the equivalence class; each
    bidirected edge replaced by an acyclic orientation between its two
    endpoints).  ``dowhy.gcm.falsify.falsify_graph`` is then run with linear
    Fisher-z (partial-correlation) tests at level ``significance_ci``,
    ``suggestions=False``, ``n_jobs=1`` and no plotting.

    The test statistic is the fraction of violated local Markov conditions
    (LMC; node independent of each non-descendant non-parent given its
    parents, unconditional tests for roots included).  The LMC p-value is the
    fraction of node-permuted graphs whose violation fraction is <= that of
    the given graph, so a *small* p-value means the graph is significantly
    better than random.  Following DoWhy/the paper, the graph is

    * falsifiable if the fraction of permutations in the given graph's Markov
      equivalence class (p_value_tpa) is below ``significance``; and
    * falsified if it is falsifiable and p_value_lmc > ``significance``
      (not better than random relabelings).

    Graphs whose permutations are all Markov equivalent (e.g. the empty or
    complete graph) are never falsifiable, hence never falsified.

    DoWhy draws permutations from the global ``np.random`` state.  It is
    seeded from ``rng`` and both ``np.random`` and ``random`` states are
    restored afterwards, so the result is a deterministic function of ``rng``.

    Returns a dict with keys ``p_value_lmc``, ``p_value_tpa``,
    ``n_violations``, ``n_tests``, ``frac_violations``,
    ``perm_violations_mean``, ``perm_frac_violations_mean``,
    ``perm_violations_quantiles`` (0.05, 0.5, 0.95), ``n_permutations``,
    ``falsifiable``, ``falsified``, ``not_better_than_random``
    (p_value_lmc > significance, ignoring falsifiability), ``can_evaluate``.
    """
    from dowhy.gcm.falsify import FalsifyConst, falsify_graph

    marks = marks.loc[list(data.columns), list(data.columns)]
    dag = marks_to_observed_dag(marks)

    np_state = np.random.get_state()
    py_state = random.getstate()
    seed = int(rng.integers(2 ** 32 - 1))
    np.random.seed(seed)
    random.seed(seed)
    try:
        result = falsify_graph(
            dag,
            data,
            suggestions=False,
            independence_test=fisher_z_independence_test,
            conditional_independence_test=fisher_z_conditional_independence_test,
            significance_level=significance,
            significance_ci=significance_ci,
            n_permutations=n_permutations,
            show_progress_bar=False,
            n_jobs=1,
            plot_histogram=False,
        )
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)

    lmc = result.summary[FalsifyConst.VALIDATE_LMC]
    tpa = result.summary[FalsifyConst.VALIDATE_TPA]
    perm = np.asarray(lmc[FalsifyConst.PERM_VIOLATIONS], dtype=float)
    perm_frac = np.asarray(lmc[FalsifyConst.F_PERM_VIOLATIONS], dtype=float)
    p_lmc = float(lmc[FalsifyConst.P_VALUE])
    return {
        "p_value_lmc": p_lmc,
        "p_value_tpa": float(tpa[FalsifyConst.P_VALUE]),
        "n_violations": int(lmc[FalsifyConst.GIVEN_VIOLATIONS]),
        "n_tests": int(lmc[FalsifyConst.N_TESTS]),
        "frac_violations": float(lmc[FalsifyConst.F_GIVEN_VIOLATIONS]),
        "perm_violations_mean": float(perm.mean()),
        "perm_frac_violations_mean": float(perm_frac.mean()),
        "perm_violations_quantiles": {q: float(np.quantile(perm, q)) for q in (0.05, 0.5, 0.95)},
        "n_permutations": int(len(perm)),
        "falsifiable": bool(result.falsifiable) if result.can_evaluate else False,
        "falsified": bool(result.falsified) if result.can_evaluate else False,
        "not_better_than_random": p_lmc > significance,
        "can_evaluate": bool(result.can_evaluate),
    }


# ---------------------------------------------------------------------------
# 3. Schkoda et al. (2025): Leave-One-Variable-Out cross-validation
# ---------------------------------------------------------------------------

def _definite_children(marks: pd.DataFrame, x: str) -> List[str]:
    return [c for c in marks.columns if c != x and marks.loc[x, c] == ARROW and marks.loc[c, x] == TAIL]


def _definite_parents(marks: pd.DataFrame, x: str) -> List[str]:
    return [c for c in marks.columns if c != x and marks.loc[c, x] == ARROW and marks.loc[x, c] == TAIL]


def _neighbors(marks: pd.DataFrame, x: str) -> Set[str]:
    return {c for c in marks.columns if c != x and marks.loc[x, c] != NO_EDGE}


def _pag_possible_child_or_sibling(marks: pd.DataFrame, y: str, c: str) -> bool:
    """c is a possible child or possible sibling of y in a PAG (pywhy-graphs semantics)."""
    at_c, at_y = marks.loc[y, c], marks.loc[c, y]
    if at_c == NO_EDGE:
        return False
    possible_child = at_y != ARROW and not (at_y == TAIL and at_c == TAIL)
    possible_sibling = (at_y == ARROW and at_c == ARROW) or at_y == CIRCLE or at_c == CIRCLE
    return possible_child or possible_sibling


def lovo_edge_might_exist(x: str, y: str, graph_with_x: pd.DataFrame, graph_with_y: pd.DataFrame, graph_class: str) -> bool:
    """Whether the marginal graphs fail to exclude an edge between x and y.

    ``graph_with_x`` is learned on V \\ {y}, ``graph_with_y`` on V \\ {x}.

    * DAG / CPDAG outputs, Lemma 7 (App. B.2-B.3): the pair is unlinked if x
      has a definite child in its graph that is not a neighbor of y in the
      other graph, or vice versa.
    * PAG outputs, Lemma 2 with PAG uncertainty (App. B.3): the pair is
      unlinked if x has a definite child that is neither a possible child nor
      a possible sibling of y, or vice versa.
    """
    for a, b, Ga, Gb in ((x, y, graph_with_x, graph_with_y), (y, x, graph_with_y, graph_with_x)):
        children = _definite_children(Ga, a)
        if graph_class in ("dag", "cpdag"):
            nb = _neighbors(Gb, b)
            if any(c not in nb for c in children):
                return False
        else:
            if any(not _pag_possible_child_or_sibling(Gb, b, c) for c in children):
                return False
    return True


def _possible_child_combined(marks: pd.DataFrame, a: str, c: str, graph_class: str) -> bool:
    """c is a possible child of a in the combined PAG (pywhy ``possible_children``).

    CPDAG undirected edges are read as o-o, as in the reference code.
    """
    at_c, at_a = marks.loc[a, c], marks.loc[c, a]
    if at_c == NO_EDGE:
        return False
    if graph_class == "cpdag" and at_a == TAIL and at_c == TAIL:
        return True
    return at_a != ARROW and not (at_a == TAIL and at_c == TAIL)


def lovo_path_adjustment_set(
    x: str, y: str, graph_with_x: pd.DataFrame, graph_with_y: pd.DataFrame, graph_class: str
) -> Optional[List[str]]:
    """Adjustment set for CPDAG/PAG outputs (App. B.3, reference code).

    The two marginal graphs are merged into one graph over V (union of the
    skeletons).  If some common neighbor m of x and y is a possible child of
    both (a possible collider x *-> m <-* y, including a definite one), the
    predictor abstains (returns None).  Otherwise the adjustment set is every
    node lying on some simple x-y path of the merged skeleton.  These are the
    nodes of the biconnected component containing the virtual edge x-y.
    """
    skeleton = nx.Graph()
    skeleton.add_nodes_from(set(graph_with_x.index) | set(graph_with_y.index))
    for G in (graph_with_x, graph_with_y):
        names = list(G.index)
        values = G.to_numpy()
        for i, j in zip(*np.nonzero(np.triu(values != NO_EDGE, k=1))):
            skeleton.add_edge(names[i], names[j])

    common = (set(skeleton.neighbors(x)) & set(skeleton.neighbors(y))) - {x, y}
    for m in sorted(common):
        if _possible_child_combined(graph_with_x, x, m, graph_class) and _possible_child_combined(
            graph_with_y, y, m, graph_class
        ):
            return None

    skeleton.add_edge(x, y)
    for component in nx.biconnected_component_edges(skeleton):
        nodes = {u for edge in component for u in edge}
        if x in nodes and y in nodes and any(set(edge) == {x, y} for edge in component):
            return sorted(nodes - {x, y})
    return []


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    std = df.std(axis=0).replace(0.0, 1.0)
    return (df - df.mean(axis=0)) / std


def three_step_lovo_predictor(
    x: str, y: str, data_xz: pd.DataFrame, data_yz: pd.DataFrame, adjustment_set: Sequence[str]
) -> float:
    """Predicted corr(x, y) from P(x, Z) and P(y, Z) (Section 3.2, three steps).

    (1) regress y on Z_S on the (y, Z) sample; (2) predict y-hat on the
    (x, Z) sample; (3) regress y-hat on x.  On standardized data the slope is
    the predicted correlation.  An empty adjustment set predicts 0.
    """
    adjustment_set = list(adjustment_set)
    if not adjustment_set:
        return 0.0
    Zy = data_yz[adjustment_set].to_numpy()
    design = np.column_stack([np.ones(len(Zy)), Zy])
    coef, *_ = np.linalg.lstsq(design, data_yz[y].to_numpy(), rcond=None)
    y_hat = data_xz[adjustment_set].to_numpy() @ coef[1:]
    xv = data_xz[x].to_numpy()
    xc = xv - xv.mean()
    return float(xc @ (y_hat - y_hat.mean()) / max(xc @ xc, 1e-12))


def lovo_scores(
    data: pd.DataFrame,
    method: str,
    config,
    rng: np.random.Generator,
    n_pairs: int = 20,
    learn_frac: float = 0.5,
    n_baseline_draws: int = 100,
    random_state: int = 0,
) -> Dict[str, object]:
    """LOVO cross-validation loss of Schkoda et al. (2025), linear version.

    Procedure (following the paper and its reference code,
    github.com/DanielaSchkoda/LOVO):

    1. Rows are shuffled with ``rng``.  The first ``learn_frac`` of them
       (standardized) are used to learn graphs, the rest is the validation
       sample, split into three equal parts used for P(X, Z), P(Y, Z) and the
       empirical corr(X, Y) respectively (as in Section 5.1).
    2. ``n_pairs`` unordered pairs (X, Y) are drawn without replacement.  The
       base method is run on V \\ {Y} and on V \\ {X} (one run per left-out
       variable, cached; no domain knowledge).
    3. Unlinked pairs are certified with Lemma 7 (DAG/CPDAG outputs) or
       Lemma 2 with PAG uncertainty (PAG outputs); otherwise the predictor
       abstains.
    4. Adjustment set: parents pa(X) in G_X union pa(Y) in G_Y for DAG
       outputs (LiNGAM, Theorem 4); for CPDAG/PAG outputs the path
       adjustment set of App. B.3 (``lovo_path_adjustment_set``), abstaining
       on a possible collider.
    5. The three-step predictor gives rho_LOVO; the loss is
       |rho_LOVO - rho_hat|.  The baseline is the MaxEnt predictor
       (Definition 6) with random adjustment sets of the same size, averaged
       over ``n_baseline_draws`` draws (all of Z when the LOVO set is empty),
       as recommended in Section 4.

    Deviations: the parent-adjustment / path-adjustment predictor is used
    for every method, including LiNGAM (the "LOVO via LiNGAM" predictor of
    Section 3.3 and the deep-learning predictor are not implemented);
    marks are read with the pywhy-graphs semantics of the reference code
    (definite children; CPDAG undirected edges as o-o); bidirected edges in
    PC/GES outputs (conflicting orientations) count as adjacencies but not
    as children.

    Returns ``lovo_loss`` (mean over evaluated pairs, NaN if none),
    ``baseline_loss`` (mean over the same pairs), ``pair_losses`` and
    ``baseline_pair_losses`` (arrays with NaN for abstentions), ``pairs``,
    ``per_pair`` (list of dicts with status, adjustment set, predicted and
    empirical correlation), ``n_evaluated``, ``n_abstained`` and
    ``lovo_better_than_baseline``.
    """
    graph_class = _graph_class(method)
    names = list(data.columns)
    if len(names) < 3:
        raise ValueError("LOVO needs at least three variables.")

    shuffled = data.iloc[rng.permutation(len(data))].reset_index(drop=True)
    n_learn = int(round(learn_frac * len(shuffled)))
    learn = _standardize(shuffled.iloc[:n_learn])
    validation = shuffled.iloc[n_learn:].reset_index(drop=True)
    third = len(validation) // 3
    if third < 5:
        raise ValueError("Too few validation rows for the three-way split.")

    all_pairs = list(itertools.combinations(names, 2))
    chosen = rng.choice(len(all_pairs), size=min(n_pairs, len(all_pairs)), replace=False)
    pairs = [all_pairs[int(k)] for k in chosen]

    graphs: Dict[str, pd.DataFrame] = {}

    def graph_without(v: str) -> pd.DataFrame:
        if v not in graphs:
            sub = learn.drop(columns=[v])
            graphs[v] = discover_marks(
                sub, method, make_domain_knowledge(list(sub.columns)), config, random_state=random_state
            )
        return graphs[v]

    per_pair = []
    losses = np.full(len(pairs), np.nan)
    base_losses = np.full(len(pairs), np.nan)
    for k, (x, y) in enumerate(pairs):
        g_x = graph_without(y)  # contains x
        g_y = graph_without(x)  # contains y
        record = {"x": x, "y": y, "status": "evaluated", "adjustment_set": None,
                  "predicted_corr": np.nan, "empirical_corr": np.nan,
                  "lovo_loss": np.nan, "baseline_loss": np.nan}

        if lovo_edge_might_exist(x, y, g_x, g_y, graph_class):
            record["status"] = "abstain_edge_might_exist"
            per_pair.append(record)
            continue
        if graph_class == "dag":
            adjustment = sorted((set(_definite_parents(g_x, x)) | set(_definite_parents(g_y, y))) - {x, y})
        else:
            adjustment = lovo_path_adjustment_set(x, y, g_x, g_y, graph_class)
            if adjustment is None:
                record["status"] = "abstain_possible_collider"
                per_pair.append(record)
                continue

        data_xz = _standardize(validation.drop(columns=[y]).iloc[:third])
        data_yz = _standardize(validation.drop(columns=[x]).iloc[third:2 * third])
        test = validation[[x, y]].iloc[2 * third:]
        empirical = float(np.corrcoef(test[x], test[y])[0, 1])
        predicted = three_step_lovo_predictor(x, y, data_xz, data_yz, adjustment)
        loss = abs(predicted - empirical)

        z_vars = [v for v in names if v not in (x, y)]
        if adjustment:
            base = np.mean([
                abs(three_step_lovo_predictor(
                    x, y, data_xz, data_yz, list(rng.choice(z_vars, size=len(adjustment), replace=False))
                ) - empirical)
                for _ in range(n_baseline_draws)
            ])
        else:
            base = abs(three_step_lovo_predictor(x, y, data_xz, data_yz, z_vars) - empirical)

        losses[k], base_losses[k] = loss, float(base)
        record.update(adjustment_set=adjustment, predicted_corr=predicted, empirical_corr=empirical,
                      lovo_loss=loss, baseline_loss=float(base))
        per_pair.append(record)

    evaluated = ~np.isnan(losses)
    lovo_loss = float(losses[evaluated].mean()) if evaluated.any() else float("nan")
    baseline_loss = float(base_losses[evaluated].mean()) if evaluated.any() else float("nan")
    return {
        "lovo_loss": lovo_loss,
        "baseline_loss": baseline_loss,
        "pair_losses": losses,
        "baseline_pair_losses": base_losses,
        "pairs": pairs,
        "per_pair": per_pair,
        "n_evaluated": int(evaluated.sum()),
        "n_abstained": int((~evaluated).sum()),
        "lovo_better_than_baseline": bool(lovo_loss < baseline_loss) if evaluated.any() else False,
    }
