"""Metrics computed from pipeline records (see ``cscd.pipeline.fit_repeat``).

Claim convention (as in the paper): a directed claim x -> y holds in a graph
if its 0/1 adjacency has A[x, y] = 1, so undirected edges count in both
directions.  Claims are accepted at frequency >= gamma and rejected at
frequency <= 1 - gamma over a set of graphs.
"""

from typing import Dict, Optional, Sequence

import numpy as np

from .selfcompat import conformal_threshold

RECALL_LEVELS = (0.5, 0.6, 0.7, 0.8)
SCORE_GRID = np.linspace(0.0, 1.0, 41)


def _offdiag(p: int) -> np.ndarray:
    return ~np.eye(p, dtype=bool)


def claim_metrics(freq: np.ndarray, truth: np.ndarray, gamma: float) -> Dict[str, float]:
    """Precision / recall / unresolved rate of claims from a frequency matrix."""
    mask = _offdiag(truth.shape[0])
    f, t = freq[mask], truth[mask] == 1
    acc = f >= gamma - 1e-12
    rej = f <= 1 - gamma + 1e-12
    tp = np.sum(acc & t)
    return {
        "precision": tp / max(acc.sum(), 1),
        "recall": tp / max(t.sum(), 1),
        "n_accepted": float(acc.sum()),
        "unresolved": float(np.mean(~acc & ~rej)),
        "false_rejections": float(np.sum(rej & t) / max(t.sum(), 1)),
    }


def skeleton_claim_precision(freq: np.ndarray, truth: np.ndarray, gamma: float) -> float:
    """Precision of accepted adjacencies (pair accepted if either direction is)."""
    p = truth.shape[0]
    iu = np.triu_indices(p, 1)
    fs = np.maximum(freq, freq.T)[iu]
    ts = ((truth + truth.T) > 0)[iu]
    acc = fs >= gamma - 1e-12
    return float(np.sum(acc & ts) / max(acc.sum(), 1))


def freq_of(adjs: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if adjs is None or len(adjs) == 0:
        return None
    return adjs.mean(axis=0)


def pr_summary(score: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    """Average precision and interpolated precision at fixed recall levels."""
    mask = _offdiag(truth.shape[0])
    s, t = score[mask], truth[mask] == 1
    out: Dict[str, float] = {}
    precs, recs = [], []
    for thr in SCORE_GRID[1:]:
        sel = s >= thr - 1e-12
        tp = np.sum(sel & t)
        precs.append(tp / sel.sum() if sel.sum() else np.nan)
        recs.append(tp / max(t.sum(), 1))
    precs, recs = np.array(precs), np.array(recs)
    for r in RECALL_LEVELS:
        ok = (recs >= r) & ~np.isnan(precs)
        out[f"prec_at_rec{r}"] = float(precs[ok].max()) if ok.any() else 0.0
    # Average precision over distinct score values (ties grouped).
    order = np.argsort(-s, kind="stable")
    s_sorted, t_sorted = s[order], t[order]
    ap, tp, prev_rec = 0.0, 0, 0.0
    i = 0
    while i < len(s_sorted):
        j = i
        while j < len(s_sorted) and s_sorted[j] == s_sorted[i]:
            j += 1
        tp += np.sum(t_sorted[i:j])
        rec = tp / max(t.sum(), 1)
        if s_sorted[i] > 0:
            ap += (rec - prev_rec) * tp / j
        prev_rec = rec
        i = j
    out["avg_precision"] = float(ap)
    return out


def stability_graph(boot: np.ndarray, threshold: float) -> np.ndarray:
    """Stability selection as in the paper: adjacency frequency >= threshold, majority orientation."""
    p = boot.shape[1]
    skel = ((boot + boot.transpose(0, 2, 1)) > 0).mean(axis=0)
    directed = ((boot == 1) & (boot.transpose(0, 2, 1) == 0)).mean(axis=0)
    A = np.zeros((p, p), dtype=int)
    for i in range(p):
        for j in range(i + 1, p):
            if skel[i, j] < threshold:
                continue
            if directed[i, j] > directed[j, i]:
                A[i, j] = 1
            elif directed[j, i] > directed[i, j]:
                A[j, i] = 1
            else:
                A[i, j] = A[j, i] = 1
    return A


def graph_pr(A: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    mask = _offdiag(truth.shape[0])
    a, t = A[mask] == 1, truth[mask] == 1
    tp = np.sum(a & t)
    return {"precision": tp / max(a.sum(), 1), "recall": tp / max(t.sum(), 1)}


def mb_error_bound_threshold(boot: np.ndarray, ev: float = 1.0) -> float:
    """Selection threshold with the Meinshausen-Buehlmann bound E[V] <= q^2 / ((2 pi - 1) P) <= ev.

    q is the mean number of selected directed claims per bootstrap graph and
    P = p (p - 1).  Returns a threshold in (0.5, 1], or 1.0 if unattainable.
    """
    p = boot.shape[1]
    q = boot.reshape(len(boot), -1).sum(axis=1).mean()
    P = p * (p - 1)
    pi = 0.5 * (q ** 2 / (ev * P) + 1.0)
    return float(min(max(pi, 0.5 + 1e-9), 1.0))


def record_metrics(rec: Dict, alpha: float = 0.1, gamma: float = 0.9, distance: Optional[str] = None) -> Dict[str, float]:
    """Every scalar metric of one record at (alpha, gamma)."""
    d = distance or list(rec["cal_scores"].keys())[0]
    cal, test, cand = rec["cal_scores"][d], rec["test_scores"][d], rec["cand_scores"][d]
    truth = rec["true_graph"]
    tau = conformal_threshold(cal, alpha) if len(cal) else float("inf")
    adm = cand <= tau
    out: Dict[str, float] = {
        "tau": tau if np.isfinite(tau) else np.nan,
        "tau_finite": float(np.isfinite(tau)),
        "exceedance": float(np.mean(test > tau)) if len(test) else np.nan,
        "admission": float(np.mean(adm)),
        "n_admitted": float(adm.sum()),
        "empty_set": float(adm.sum() == 0),
        "full_admitted": float(cand[0] <= tau),
        "full_score": float(cand[0]),
        "chunk_size": rec["chunk_size"], "n_cal": rec["n_cal"], "n_train": rec["n_train"],
        "sep_cover": rec.get("sep_cover", np.nan),
        "n_subsets": len(rec["subsets"]),
        "time_total": rec.get("time_total", np.nan),
        "cal_failures": rec["failures"]["cal"], "test_failures": rec["failures"]["test"],
    }
    if rec.get("true_score"):
        ts = rec["true_score"][d]
        out["true_score"] = ts
        out["true_included"] = float(ts <= tau)
    if rec.get("local_err_train") is not None:
        out["eps_train"] = float(np.mean(rec["local_err_train"]))
        out["eps_chunk"] = float(np.mean(rec["local_err_chunks"])) if rec["local_err_chunks"] else np.nan
    if truth is None:
        return out

    cand_adj = rec["cand_adj"]
    sets = {
        "cscd": cand_adj[adm] if adm.any() else None,
        "nofilter": cand_adj,
        "uncal": cand_adj[cand <= rec["raw_selfcompat"]] if np.isfinite(rec.get("raw_selfcompat", np.nan)) else None,
    }
    for name, adjs in sets.items():
        f = freq_of(adjs)
        if f is None:
            out.update({f"{name}_precision": np.nan, f"{name}_recall": 0.0, f"{name}_unresolved": 1.0})
            continue
        m = claim_metrics(f, truth, gamma)
        out.update({f"{name}_{k}": v for k, v in m.items()})
        out[f"{name}_skel_precision"] = skeleton_claim_precision(f, truth, gamma)
        for k, v in pr_summary(f, truth).items():
            out[f"{name}_{k}"] = v

    # Single run (D_train and full data) and its hyperparameter path.
    out.update({f"single_{k}": v for k, v in graph_pr(cand_adj[0], truth).items()})
    if rec.get("single_full_adj") is not None:
        out.update({f"singleF_{k}": v for k, v in graph_pr(rec["single_full_adj"], truth).items()})
    path = cand_adj[: 1 + rec.get("n_grid", 0)]
    prs = [graph_pr(A, truth) for A in path]
    for r in RECALL_LEVELS:
        ok = [x["precision"] for x in prs if x["recall"] >= r]
        out[f"single_path_prec_at_rec{r}"] = max(ok) if ok else 0.0

    # Stability selection / bootstrap edge frequencies.
    for tag, key in (("stab", "boot_adj"), ("stabF", "boot_full_adj")):
        boot = rec.get(key)
        if boot is None:
            continue
        out.update({f"{tag}_{k}": v for k, v in graph_pr(stability_graph(boot, 0.6), truth).items()})
        f = boot.mean(axis=0)
        m = claim_metrics(f, truth, gamma)
        out.update({f"{tag}freq_{k}": v for k, v in m.items()})
        for k, v in pr_summary(f, truth).items():
            out[f"{tag}freq_{k}"] = v
        thr = mb_error_bound_threshold(boot, ev=1.0)
        m = claim_metrics(f, truth, thr)
        out[f"{tag}MB_threshold"] = thr
        out[f"{tag}MB_precision"] = m["precision"]
        out[f"{tag}MB_recall"] = m["recall"]

    for tag, key in (("bayes", "bayes_train"), ("bayesF", "bayes_full")):
        post = rec.get(key)
        if post is None:
            continue
        m = claim_metrics(post, truth, gamma)
        out.update({f"{tag}_{k}": v for k, v in m.items()})
        for k, v in pr_summary(post, truth).items():
            out[f"{tag}_{k}"] = v

    # Localization ablations.
    if "bootcompat_ref" in rec:
        ref = rec["bootcompat_ref"]
        for tag, thr in (("bcmed", np.median(ref)), ("bcq", np.quantile(ref, 1 - alpha))):
            a = rec["bootcompat_cand"] <= thr
            f = freq_of(cand_adj[a]) if a.any() else None
            out[f"{tag}_admission"] = float(a.mean())
            out[f"{tag}_full_admitted"] = float(rec["bootcompat_cand"][0] <= thr)
            out[f"{tag}_true_included"] = float(rec["bootcompat_truth"] <= thr)
            out[f"{tag}_full_score"] = float(rec["bootcompat_cand"][0] - thr)
            if f is not None:
                m = claim_metrics(f, truth, gamma)
                out[f"{tag}_precision"], out[f"{tag}_recall"] = m["precision"], m["recall"]
        if len(rec.get("nosub_cal", [])):
            tv = conformal_threshold(rec["nosub_cal"], alpha)
            a = rec["nosub_cand"] <= tv
            out["nosub_admission"] = float(a.mean())
            out["nosub_full_admitted"] = float(rec["nosub_cand"][0] <= tv)
            out["nosub_true_included"] = float(rec["nosub_truth"] <= tv)
            if a.any():
                m = claim_metrics(freq_of(cand_adj[a]), truth, gamma)
                out["nosub_precision"], out["nosub_recall"] = m["precision"], m["recall"]

    # Structural error of the full-data run (for the systematic-error study).
    t = truth
    A0 = cand_adj[0]
    # True adjacencies over the observed variables: the marginal MAG skeleton
    # (includes bidirected edges from hidden confounders), else the claims.
    t_skel = rec["true_skel"] if rec.get("true_skel") is not None else ((t + t.T) > 0)
    skel_err = ((A0 + A0.T) > 0) != (t_skel > 0)
    out["full_skel_errors"] = float(skel_err[np.triu_indices(t.shape[0], 1)].sum())
    out["full_shd"] = float(np.abs(A0 - t)[_offdiag(t.shape[0])].sum())

    if rec.get("decision"):
        out.update(rec["decision"])
    return out
