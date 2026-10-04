"""Metrics for second-revision records (``pipeline_v3.fit_repeat_v3``).

Main entry points
-----------------
* ``standard_mask`` / ``arm_masks``: which candidates form each candidate
  family (the paper's standard family and the review's arms).
* ``claim_frequencies``: claim frequencies of three claim types over a set
  of endpoint-mark graphs -- adjacency, definite orientation (only edges
  oriented in the graph's class), ancestral (x is a definite ancestor of y).
  The legacy convention of the paper (undirected edges count in both
  directions) is kept as ``dir_any``.
* ``claims_row``: precision / recall at gamma, precision at matched recall
  and average precision for one frequency matrix.
* ``inclusion_diagnostics``: rank of A(G*) among the calibration scores,
  margin Delta = E[A_b | D_train] - A(G*), Cantelli bound, and the pairwise
  decomposition of Delta (review item 1).
"""

import itertools
from typing import Dict, Iterable, List, Optional

import numpy as np
from scipy.stats import binom

from .selfcompat import conformal_threshold

TAIL, ARROW, NO_EDGE = 1, 2, 0
RECALLS = (0.5, 0.6, 0.7, 0.8)


# ---------------------------------------------------------------------------
# Candidate families
# ---------------------------------------------------------------------------

def standard_mask(tags: List[str], n_candidate_graphs: int = 100, boot_frac: float = 0.5) -> np.ndarray:
    """The paper's family: full run, grid, then bootstraps and edits as in generate_candidate_graphs."""
    tags = np.array(tags)
    n_grid = int(np.sum(tags == "grid"))
    remaining = max(0, n_candidate_graphs - n_grid)
    n_boot = int(round(boot_frac * remaining))
    n_edit = max(0, n_candidate_graphs + 1 - (1 + n_grid + n_boot))
    mask = np.isin(tags, ["full", "grid"])
    for tag, n_take in (("boot", n_boot), ("edit", n_edit)):
        idx = np.flatnonzero(tags == tag)[:n_take]
        mask[idx] = True
    return mask


def arm_masks(tags: List[str], n_candidate_graphs: int = 100) -> Dict[str, np.ndarray]:
    """Candidate-family arms of review item 2 (all include the full-data run)."""
    t = np.array(tags)
    full = t == "full"
    std = standard_mask(tags, n_candidate_graphs)
    arms = {
        "standard": std,
        "boot_only": full | (t == "boot"),
        "edit_only": full | (t == "edit"),
        "grid_only": full | (t == "grid"),
        "multi_alg": full | (t == "multi"),
        "mcmc": full | (t == "mcmc"),
        "standard+truth": std | (t == "truth"),
        "standard+random": std | (t == "random"),
    }
    return {k: v for k, v in arms.items() if v.sum() > 1 or k == "standard"}


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------

def definite_directed(marks: np.ndarray) -> np.ndarray:
    """D[a, b] = 1 iff a -> b is a definite directed edge (tail at a, arrow at b)."""
    return ((marks == ARROW) & (marks.T == TAIL)).astype(np.int8)


def any_directed(marks: np.ndarray) -> np.ndarray:
    """Legacy claim convention: a definite edge one way, any other edge both ways."""
    adj = marks != NO_EDGE
    d = definite_directed(marks).astype(bool)
    return (d | (adj & ~d & ~d.T)).astype(np.int8)


def transitive_closure(D: np.ndarray) -> np.ndarray:
    R = D.astype(bool).copy()
    p = R.shape[0]
    for k in range(p):
        R |= R[:, [k]] & R[[k], :]
    np.fill_diagonal(R, False)
    return R.astype(np.int8)


def claim_matrix(marks: np.ndarray, kind: str) -> np.ndarray:
    if kind == "adj":
        return (marks != NO_EDGE).astype(np.int8)
    if kind == "dir_any":
        return any_directed(marks)
    if kind == "dir_def":
        return definite_directed(marks)
    if kind == "anc":
        return transitive_closure(definite_directed(marks))
    raise ValueError(kind)


def dag_claim_matrix(A: np.ndarray, kind: str) -> np.ndarray:
    """Claims of a DAG adjacency matrix (MCMC samples, truth)."""
    if kind == "adj":
        return ((A + A.T) > 0).astype(np.int8)
    if kind in ("dir_any", "dir_def"):
        return A.astype(np.int8)
    if kind == "anc":
        return transitive_closure(A)
    raise ValueError(kind)


def claim_frequencies(marks_stack: np.ndarray, kind: str, weights: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    if marks_stack is None or len(marks_stack) == 0:
        return None
    C = np.stack([claim_matrix(M, kind) for M in marks_stack]).astype(float)
    if weights is None:
        return C.mean(axis=0)
    w = np.asarray(weights, float)
    return np.tensordot(w / w.sum(), C, axes=1)


def dedup_weights(keys: List[bytes]) -> np.ndarray:
    """Weight 1/|class| per graph so that every Markov equivalence class counts once."""
    counts: Dict[bytes, int] = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    return np.array([1.0 / counts[k] for k in keys])


def _pairs(freq: np.ndarray, truth: np.ndarray, symmetric: bool):
    p = truth.shape[0]
    if symmetric:
        iu = np.triu_indices(p, 1)
        return np.maximum(freq, freq.T)[iu], ((truth + truth.T) > 0)[iu]
    mask = ~np.eye(p, dtype=bool)
    return freq[mask], truth[mask] > 0


def claims_row(freq: Optional[np.ndarray], truth: np.ndarray, kind: str, gamma: float = 0.9) -> Dict[str, float]:
    """Precision/recall of accepted claims, precision at matched recall, average precision."""
    if freq is None:
        return {"prec": np.nan, "rec": 0.0, "n_acc": 0.0, **{f"p@r{r}": np.nan for r in RECALLS}, "ap": np.nan}
    s, t = _pairs(freq, truth, symmetric=(kind == "adj"))
    acc = s >= gamma - 1e-12
    tp = np.sum(acc & t)
    out = {"prec": tp / acc.sum() if acc.sum() else np.nan, "rec": tp / max(t.sum(), 1), "n_acc": float(acc.sum())}
    order = np.argsort(-s, kind="stable")
    ss, tt = s[order], t[order]
    precs, recs = [], []
    ap, hits, prev, i = 0.0, 0, 0.0, 0
    while i < len(ss):
        j = i
        while j < len(ss) and ss[j] == ss[i]:
            j += 1
        if ss[i] <= 0:
            break
        hits += int(np.sum(tt[i:j]))
        rec = hits / max(t.sum(), 1)
        prec = hits / j
        ap += (rec - prev) * prec
        precs.append(prec)
        recs.append(rec)
        prev, i = rec, j
    precs, recs = np.array(precs), np.array(recs)
    for r in RECALLS:
        ok = recs >= r - 1e-12
        out[f"p@r{r}"] = float(precs[ok].max()) if ok.any() else np.nan
    out["ap"] = float(ap)
    return out


def point_row(claims: np.ndarray, truth: np.ndarray, kind: str) -> Dict[str, float]:
    """Precision/recall of a single graph's claims (no score, no curve)."""
    s, t = _pairs(claims.astype(float), truth, symmetric=(kind == "adj"))
    acc = s > 0.5
    tp = np.sum(acc & t)
    return {"prec": tp / acc.sum() if acc.sum() else np.nan, "rec": tp / max(t.sum(), 1), "n_acc": float(acc.sum())}


def truth_claims(true_dag: np.ndarray, true_skel: Optional[np.ndarray], kind: str) -> np.ndarray:
    if kind == "adj":
        return true_skel if true_skel is not None else dag_claim_matrix(true_dag, "adj")
    return dag_claim_matrix(true_dag, kind)


# ---------------------------------------------------------------------------
# Inclusion diagnostics (review item 1)
# ---------------------------------------------------------------------------

def k_alpha(N: int, alpha: float) -> int:
    return int(np.ceil((N + 1) * (1 - alpha)))


def exclusion_probability(u: float, N: int, alpha: float) -> float:
    """beta_{N,alpha}(u) = P(Bin(N, u) >= k_alpha): exact conditional exclusion probability."""
    k = k_alpha(N, alpha)
    if k > N:
        return 0.0
    return float(binom.sf(k - 1, N, u))


def inclusion_diagnostics(rec: Dict, alpha: float = 0.1) -> Dict[str, float]:
    """Rank of A(G*), margin, Cantelli bound and the pairwise decomposition of the margin.

    The conditional law of a chunk score given D_train is estimated from all
    chunk scores of the record (calibration and held-out chunks are i.i.d.
    given D_train).
    """
    cal = np.asarray(rec["cal_scores"]["skeleton"])
    test = np.asarray(rec["test_scores"]["skeleton"])
    a_star = rec["true_score"]["skeleton"]
    allc = np.concatenate([cal, test])
    N = len(cal)
    tau = conformal_threshold(cal, alpha)
    mu, sd = allc.mean(), allc.std(ddof=1)
    delta = mu - a_star
    u_hat = float(np.mean(allc < a_star))
    cantelli = sd ** 2 / (sd ** 2 + delta ** 2) if delta > 0 else 1.0
    out = {
        "rank": int(np.sum(cal < a_star)), "N": N, "a_star": a_star, "tau": tau,
        "included": float(a_star <= tau), "mu": mu, "sd": sd, "delta": delta, "u_hat": u_hat,
        "pred_excl": exclusion_probability(u_hat, N, alpha),
        "cantelli_u": cantelli, "cantelli_excl": exclusion_probability(min(1.0, cantelli), N, alpha),
        "full_score": float(rec["cand_scores"]["skeleton"][0]),
        "full_rank": int(np.sum(cal < rec["cand_scores"]["skeleton"][0])),
    }
    # Pairwise decomposition: Delta = (1/mK) sum e (1 - 2 w).
    if rec.get("local_train") is not None and rec.get("truth_proj") is not None and rec.get("local_cal") is not None:
        T = rec["truth_proj"].astype(bool)                    # (m, k, k)
        L = rec["local_train"]
        chunks = [c for c in (rec["local_cal"], rec.get("local_test")) if c is not None]
        Cg = np.concatenate(chunks, axis=0)                    # (B, m, k, k)
        k = T.shape[1]
        iu = np.triu_indices(k, 1)
        ok_train = L[:, 0, 0] != -1
        w = (L[:, iu[0], iu[1]] != NO_EDGE) != T[:, iu[0], iu[1]]               # (m, K)
        valid = Cg[:, :, 0, 0] != -1                                             # (B, m)
        chunk_wrong = (Cg[:, :, iu[0], iu[1]] != NO_EDGE) != T[None, :, iu[0], iu[1]]
        e = np.where(valid[..., None], chunk_wrong, np.nan)
        e = np.nanmean(e, axis=0)                                                # (m, K)
        e, w = e[ok_train], w[ok_train]
        out["delta_pairwise"] = float(np.nanmean(e * (1 - 2 * w)))
        out["frac_wrong_pairs"] = float(w.mean())
        out["e_right"] = float(np.nanmean(np.where(~w, e, np.nan)))
        out["e_wrong"] = float(np.nanmean(np.where(w, e, np.nan))) if w.any() else np.nan
        out["eps_train"] = float(w.any(axis=1).mean())
        out["eps_chunk"] = float(np.nanmean(np.where(valid[:, ok_train], chunk_wrong[:, ok_train].any(axis=2), np.nan)))
        out["sum_eps_train"] = float(w.any(axis=1).sum())
    if rec.get("subset_sufficient") is not None:
        out["frac_sufficient"] = float(np.mean(rec["subset_sufficient"]))
    return out


def eta_slack(u: Iterable[float]) -> float:
    """eta = sup_t [P(u > t) - (1 - t)]_+, estimated from per-repeat u values."""
    u = np.sort(np.asarray(list(u), float))
    if len(u) == 0:
        return np.nan
    ts = np.concatenate([[0.0], u, [1.0]])
    surv = np.array([np.mean(u > t) for t in ts])
    return float(max(0.0, np.max(surv - (1 - ts))))


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def auroc_ci(pos, neg, rng, n_boot: int = 2000, level: float = 0.95):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan, np.nan
    vals = [auroc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(n_boot)]
    a = (1 - level) / 2
    return float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))


def power_at_matched_fpr(pos, neg, fpr: float = 0.1) -> Dict[str, float]:
    """Threshold at the (1 - fpr) quantile of the control scores; flag if score > threshold."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return {"thr": np.nan, "power": np.nan, "fpr": np.nan}
    thr = float(np.quantile(neg, 1 - fpr, method="higher"))
    return {"thr": thr, "power": float(np.mean(pos > thr)), "fpr": float(np.mean(neg > thr))}


def spearman(x, y) -> float:
    from scipy.stats import spearmanr

    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return np.nan
    return float(spearmanr(x[ok], y[ok]).correlation)
