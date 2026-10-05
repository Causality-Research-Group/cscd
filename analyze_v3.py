"""Analyse the second-revision records (results_v3) and selected v2 records.

Usage:
    python analyze_v3.py [section ...]     # default: every section

Writes CSV summaries to results_v3/summary/<section>.csv.  Sections:
validity, inclusion, claims, arms, detection, knowledge, alpha, mcmc,
selective, scale, semi, sachs, decision.
"""

import glob
import os
import pickle
import sys
import warnings
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from scipy.stats import binomtest, kstest

from cscd.analysis_v3 import (
    arm_masks,
    auroc,
    auroc_ci,
    claim_frequencies,
    claims_row,
    dag_claim_matrix,
    dedup_weights,
    eta_slack,
    exclusion_probability,
    inclusion_diagnostics,
    k_alpha,
    point_row,
    power_at_matched_fpr,
    spearman,
    standard_mask,
    truth_claims,
)
from cscd.selfcompat import conformal_threshold

warnings.filterwarnings("ignore")
R3, R2 = "results_v3", "results_v2"
OUT = os.path.join(R3, "summary")
ALPHA, GAMMA = 0.1, 0.9
KINDS = ("adj", "dir_any", "dir_def", "anc")


def records(exp: str, root: str = R3):
    for path in sorted(glob.glob(os.path.join(root, exp, "rec_*.pkl"))):
        with open(path, "rb") as f:
            yield pickle.load(f)


def save(df: pd.DataFrame, name: str) -> pd.DataFrame:
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
    print(f"[{name}] {len(df)} rows")
    return df


def mse(x) -> str:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return "nan"
    return f"{x.mean():.3f}±{x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0:.3f}"


def agg(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    num = [c for c in df.columns if c not in keys and pd.api.types.is_numeric_dtype(df[c])]
    g = df.groupby(keys)[num]
    m = g.mean().add_suffix("_mean")
    se = (g.std(ddof=1) / np.sqrt(g.count())).add_suffix("_se")
    n = df.groupby(keys).size().rename("n")
    return pd.concat([m, se, n], axis=1).reset_index()


def tau_of(rec, alpha=ALPHA, key="skeleton"):
    return conformal_threshold(np.asarray(rec["cal_scores"][key]), alpha)


# ---------------------------------------------------------------------------
# validity: pooled exceedance and training-conditional exceedance (R5)
# ---------------------------------------------------------------------------

def sec_validity():
    rows, cond = [], []
    for exp in ("valid_N4", "valid_N9", "valid_N19"):
        for rec in records(exp):
            N = rec["n_cal"]
            for a in (0.05, 0.1, 0.2):
                k = k_alpha(N, a)
                if k > N:
                    continue
                tau = tau_of(rec, a)
                test = np.asarray(rec["test_scores"]["skeleton"])
                cond.append(dict(experiment=exp, method=rec["method"], repeat=rec["repeat"], N=N, alpha=a,
                                 exceed=float(np.mean(test > tau)), n_test=len(test),
                                 ties=float(np.mean(test == tau))))
    df = pd.DataFrame(cond)
    save(df, "validity_per_repeat")
    for (exp, meth, N, a), g in df.groupby(["experiment", "method", "N", "alpha"]):
        k = k_alpha(N, a)
        hits = int(round((g.exceed * g.n_test).sum()))
        tot = int(g.n_test.sum())
        ci = binomtest(hits, tot).proportion_ci(0.95)
        # Repeat-level CI (exceedances within a repeat share tau).
        se = g.exceed.std(ddof=1) / np.sqrt(len(g))
        ref = beta_dist(N + 1 - k, k)
        ks = kstest(g.exceed.values, ref.cdf)
        rows.append(dict(experiment=exp, method=meth, N=N, alpha=a, nominal=(N + 1 - k) / (N + 1),
                         n_repeats=len(g), exceed=g.exceed.mean(), se_repeat=se,
                         ci_lo=ci.low, ci_hi=ci.high, ref_mean=ref.mean(),
                         p_gt_2a=float(np.mean(g.exceed > 2 * a)), ref_p_gt_2a=float(ref.sf(2 * a)),
                         ks_stat=ks.statistic, ks_p=ks.pvalue, ties=g.ties.mean()))
    return save(pd.DataFrame(rows), "validity")


# ---------------------------------------------------------------------------
# inclusion: rank of A(G*), margin, eta, decomposition (R1, R4)
# ---------------------------------------------------------------------------

INCLUSION_SETS = [
    ("valid_N9", R3, "main, $N{=}9$ (500)"),
    ("main_v3", R3, "main, $N{=}9$"),
    ("alpha_N4", R3, "$N{=}4$ ($\\alpha{=}0.2$)"),
    ("alpha_N19", R3, "$N{=}19$ ($\\alpha{=}0.05$)"),
    ("design_N20_c40", R2, "$N{=}20$, $n_c{=}40$"),
    ("design_sym_N9", R2, "symmetric"),
    ("hubs_v3", R3, "hidden hubs"),
    ("lowdim_n1000_v3", R3, "$p{=}20$, $n{=}1000$"),
    ("highdim_v3", R3, "$p{=}50$, $n{=}1000$"),
    ("quadratic_v3", R3, "quadratic"),
    ("n200_N9", R2, "$n{=}200$"),
    ("n400_N9", R2, "$n{=}400$"),
    ("n800_N9", R2, "$n{=}800$"),
    ("n1600_N9", R2, "$n{=}1600$"),
    ("n6400_N9", R2, "$n{=}6400$"),
    ("n12800_N9", R2, "$n{=}12800$"),
]


def alpha_of(exp: str) -> float:
    return {"alpha_N4": 0.2, "alpha_N19": 0.05, "valid_N4": 0.2, "valid_N19": 0.05}.get(exp, 0.1)


def sec_inclusion():
    per = []
    for exp, root, label in INCLUSION_SETS:
        a = alpha_of(exp)
        for rec in records(exp, root):
            if not rec.get("true_score"):
                continue
            d = inclusion_diagnostics(rec, alpha=a)
            d.update(experiment=exp, label=label, method=rec["method"], repeat=rec["repeat"], alpha=a)
            per.append(d)
    df = save(pd.DataFrame(per), "inclusion_per_repeat")
    rows = []
    for (exp, label, meth), g in df.groupby(["experiment", "label", "method"], sort=False):
        N = int(g.N.iloc[0])
        a = g.alpha.iloc[0]
        k = k_alpha(N, a)
        rows.append(dict(
            experiment=exp, label=label, method=meth, n=len(g), N=N, alpha=a,
            included=g.included.mean(), rank_ge_k=float(np.mean(g["rank"] >= k)),
            mean_rank_frac=float(np.mean(g["rank"] / N)),
            rank0=float(np.mean(g["rank"] == 0)), rankN=float(np.mean(g["rank"] == N)),
            a_star=g.a_star.mean(), tau=g.tau.mean(), delta=g.delta.mean(), delta_se=g.delta.std(ddof=1) / np.sqrt(len(g)),
            frac_delta_pos=float(np.mean(g.delta > 0)),
            pred_excl=g.pred_excl.mean(), cantelli_excl=g.cantelli_excl.mean(),
            eta=eta_slack(g.u_hat), u_mean=g.u_hat.mean(),
            delta_pairwise=g.get("delta_pairwise", pd.Series(dtype=float)).mean(),
            frac_wrong_pairs=g.get("frac_wrong_pairs", pd.Series(dtype=float)).mean(),
            e_right=g.get("e_right", pd.Series(dtype=float)).mean(),
            e_wrong=g.get("e_wrong", pd.Series(dtype=float)).mean(),
            eps_train=g.get("eps_train", pd.Series(dtype=float)).mean(),
            eps_train_max=g.get("eps_train", pd.Series(dtype=float)).max(),
            eps_chunk=g.get("eps_chunk", pd.Series(dtype=float)).mean(),
            eps_chunk_max=g.get("eps_chunk", pd.Series(dtype=float)).max(),
            sum_eps_train=g.get("sum_eps_train", pd.Series(dtype=float)).mean(),
            union_bound=1 - g.get("sum_eps_train", pd.Series(dtype=float)).mean(),
            frac_sufficient=g.get("frac_sufficient", pd.Series(dtype=float)).mean(),
            full_rank_ge_k=float(np.mean(g.full_rank >= k)),
        ))
    return save(pd.DataFrame(rows), "inclusion")


# ---------------------------------------------------------------------------
# claims: three claim types, matched recall, full-data baselines (R6, R7, R2)
# ---------------------------------------------------------------------------

def _estimators(rec: Dict, kind: str, std_mask: np.ndarray):
    """Frequency matrices (or point claims) of every estimator for one record."""
    tau = tau_of(rec)
    cs = np.asarray(rec["cand_scores"]["skeleton"])
    marks = rec["cand_marks"]
    adm = std_mask & (cs <= tau)
    est = {}
    est["CSCD"] = claim_frequencies(marks[adm], kind) if adm.any() else None
    if adm.any() and rec.get("cand_mec") is not None:
        keys = [rec["cand_mec"][i] for i in np.flatnonzero(adm)]
        est["CSCD, MEC-dedup."] = claim_frequencies(marks[adm], kind, dedup_weights(keys))
    est["Pool, no filter"] = claim_frequencies(marks[std_mask], kind)
    raw = rec.get("raw_selfcompat", np.nan)
    if np.isfinite(raw):
        un = std_mask & (cs <= raw)
        est["Uncal. threshold"] = claim_frequencies(marks[un], kind) if un.any() else None
    if rec.get("boot_marks") is not None:
        est["Bootstrap freq., $D_{tr}$"] = claim_frequencies(rec["boot_marks"], kind)
    if rec.get("boot_full_adj") is not None:
        # Full-data bootstraps are stored as adjacencies: definite edges one way only.
        A = rec["boot_full_adj"]
        if kind == "adj":
            est["Bootstrap freq., $D$"] = ((A + A.transpose(0, 2, 1)) > 0).mean(0)
        elif kind == "dir_any":
            est["Bootstrap freq., $D$"] = A.mean(0)
        else:
            D = (A == 1) & (A.transpose(0, 2, 1) == 0)
            est["Bootstrap freq., $D$"] = np.stack([dag_claim_matrix(x.astype(np.int8), kind) for x in D]).mean(0)
    for tag, key in (("Order MCMC, $D_{tr}$", "bayes_samples"), ("Order MCMC, $D$", "bayesF_samples")):
        if rec.get(key) is not None:
            est[tag] = np.stack([dag_claim_matrix(S, kind) for S in rec[key]]).mean(0)
    return est


def _points(rec: Dict, kind: str):
    from cscd.analysis import stability_graph
    from cscd.analysis_v3 import claim_matrix

    pts = {"Single run, $D_{tr}$": claim_matrix(rec["cand_marks"][0], kind)}
    if rec.get("single_full_marks") is not None:
        pts["Single run, $D$"] = claim_matrix(rec["single_full_marks"], kind)
    for tag, key in (("Stability sel., $D_{tr}$", "boot_adj"), ("Stability sel., $D$", "boot_full_adj")):
        if rec.get(key) is not None:
            S = stability_graph(rec[key], 0.6)
            D = ((S == 1) & (S.T == 0)).astype(np.int8)
            pts[tag] = dag_claim_matrix(D, kind) if kind in ("dir_def", "anc") else (
                ((S + S.T) > 0).astype(np.int8) if kind == "adj" else S.astype(np.int8))
    return pts


def sec_claims(exps=("main_v3",)):
    rows = []
    for exp in exps:
        for rec in records(exp):
            if rec.get("true_graph") is None:
                continue
            std = standard_mask(rec["cand_tags"]) if "cand_tags" in rec else np.ones(len(rec["cand_marks"]), bool)
            for kind in KINDS:
                T = truth_claims(rec["true_graph"], rec.get("true_skel"), kind)
                for name, f in _estimators(rec, kind, std).items():
                    r = claims_row(f, T, kind, GAMMA)
                    rows.append(dict(experiment=exp, method=rec["method"], repeat=rec["repeat"], kind=kind,
                                     estimator=name, **r))
                for name, C in _points(rec, kind).items():
                    rows.append(dict(experiment=exp, method=rec["method"], repeat=rec["repeat"], kind=kind,
                                     estimator=name, **point_row(C, T, kind)))
    df = save(pd.DataFrame(rows), "claims_per_repeat")
    return save(agg(df.drop(columns=["repeat"]), ["experiment", "method", "kind", "estimator"]), "claims")


# ---------------------------------------------------------------------------
# arms: candidate-family composition (R2)
# ---------------------------------------------------------------------------

def sec_arms(exp="main_v3"):
    rows = []
    for rec in records(exp):
        tags = rec["cand_tags"]
        tau = tau_of(rec)
        cs = np.asarray(rec["cand_scores"]["skeleton"])
        marks = rec["cand_marks"]
        g0 = marks[0] != 0
        for arm, mask in arm_masks(tags).items():
            adm = mask & (cs <= tau)
            row = dict(method=rec["method"], repeat=rec["repeat"], arm=arm, size=int(mask.sum()),
                       admitted=float(adm.sum() / mask.sum()), empty=float(not adm.any()),
                       n_mec=len(set(rec["cand_mec"][i] for i in np.flatnonzero(mask))),
                       n_mec_adm=len(set(rec["cand_mec"][i] for i in np.flatnonzero(adm))))
            for kind in ("adj", "dir_def"):
                T = truth_claims(rec["true_graph"], rec.get("true_skel"), kind)
                f = claim_frequencies(marks[adm], kind) if adm.any() else None
                r = claims_row(f, T, kind, GAMMA)
                row[f"{kind}_prec"], row[f"{kind}_rec"] = r["prec"], r["rec"]
                if adm.any():
                    w = dedup_weights([rec["cand_mec"][i] for i in np.flatnonzero(adm)])
                    rd = claims_row(claim_frequencies(marks[adm], kind, w), T, kind, GAMMA)
                    row[f"{kind}_prec_dedup"], row[f"{kind}_rec_dedup"] = rd["prec"], rd["rec"]
            if adm.any():
                f = claim_frequencies(marks[adm], "adj")
                iu = np.triu_indices(len(g0), 1)
                acc = np.maximum(f, f.T)[iu] >= GAMMA
                ghat = g0[iu]
                row["jaccard_ghat"] = float(np.sum(acc & ghat) / max(np.sum(acc | ghat), 1))
            if "truth" in tags and arm == "standard+truth":
                row["truth_admitted"] = float(cs[tags.index("truth")] <= tau)
            rows.append(row)
    df = save(pd.DataFrame(rows), "arms_per_repeat")
    return save(agg(df.drop(columns=["repeat"]), ["method", "arm"]), "arms")


# ---------------------------------------------------------------------------
# detection (R3)
# ---------------------------------------------------------------------------

def detector_scores(rec: Dict) -> Dict[str, float]:
    tau = tau_of(rec)
    cs = np.asarray(rec["cand_scores"]["skeleton"])
    out = {"CSCD margin $A(\\widehat G)-\\tau_\\alpha$": cs[0] - tau,
           "CSCD rank of $A(\\widehat G)$": float(np.sum(np.asarray(rec["cal_scores"]["skeleton"]) < cs[0])),
           "Raw $A(\\widehat G)$": cs[0]}
    if "bootcompat_cand" in rec:
        ref = rec["bootcompat_ref"]
        out["Bootstrap compat."] = rec["bootcompat_cand"][0]
        out["Bootstrap compat. (own quantile)"] = rec["bootcompat_cand"][0] - np.quantile(ref, 1 - ALPHA)
        if len(rec.get("nosub_cal", [])):
            out["Conformal, no subsets"] = rec["nosub_cand"][0] - conformal_threshold(rec["nosub_cal"], ALPHA)
    if np.isfinite(rec.get("raw_selfcompat", np.nan)):
        out["Uncal. threshold"] = cs[0] - rec["raw_selfcompat"]
    ext = rec.get("external", {})
    if "kappa_g" in ext:
        out["$\\kappa_G$ (Faller et al.)"] = ext["kappa_g"]
    if "eulig" in ext:
        out["Permutation test (Eulig et al.)"] = ext["eulig"]["p_value_lmc"]
        out["Eulig violations (ratio)"] = ext["eulig"]["frac_violations"] / max(ext["eulig"]["perm_frac_violations_mean"], 1e-12)
    if "lovo" in ext:
        out["LOVO loss (Schkoda et al.)"] = ext["lovo"]["lovo_loss"]
    return out


DET_PAIRS = [
    ("Hidden hubs", "hubs_v3", "main_v3", ["pc", "ges", "fci"]),
    ("$p{=}50$, $n{=}1000$", "highdim_v3", "lowdim_n1000_v3", ["pc", "ges", "fci"]),
    ("Wrong knowledge", "know_hard_rev50", "know_none", ["pc", "fci"]),
    ("Quadratic (shared)", "quadratic_v3", "main_v3", ["pc", "fci"]),
]


def _shd(A: np.ndarray, T: np.ndarray) -> float:
    off = ~np.eye(len(T), dtype=bool)
    return float(np.abs(A.astype(int) - T.astype(int))[off].sum())


def _skel_err(marks: np.ndarray, t_skel: np.ndarray) -> float:
    iu = np.triu_indices(len(t_skel), 1)
    return float(np.sum((marks != 0)[iu] != (t_skel > 0)[iu]))


def sec_detection():
    rng = np.random.default_rng(0)
    cache = {}

    def load(exp):
        if exp not in cache:
            rows = []
            for rec in records(exp):
                if rec.get("true_graph") is None:
                    continue
                tau = tau_of(rec)
                cs = np.asarray(rec["cand_scores"]["skeleton"])
                t_skel = rec["true_skel"]
                row = dict(method=rec["method"], repeat=rec["repeat"], **detector_scores(rec),
                           reject_ghat=float(cs[0] > tau), reject_gstar=float(rec["true_score"]["skeleton"] > tau),
                           a_ghat=cs[0], a_gstar=rec["true_score"]["skeleton"],
                           skel_err=_skel_err(rec["cand_marks"][0], t_skel),
                           shd=_shd(rec["cand_adj"][0], rec["true_graph"]))
                std = standard_mask(rec["cand_tags"])
                shds = [_skel_err(M, t_skel) for M in rec["cand_marks"][std]]
                row["spearman_A_skelerr"] = spearman(cs[std], shds)
                if "kappa_g" in rec.get("external", {}):
                    row["kappa_g_truth"] = rec["external"].get("kappa_g_truth", np.nan)
                rows.append(row)
            cache[exp] = pd.DataFrame(rows)
        return cache[exp]

    rows = []
    for scen, exp, ctrl, methods in DET_PAIRS:
        S, C = load(exp), load(ctrl)
        if S.empty or C.empty:
            continue
        dets = [c for c in S.columns if c in C.columns and c not in (
            "method", "repeat", "reject_ghat", "reject_gstar", "a_ghat", "a_gstar", "skel_err", "shd",
            "spearman_A_skelerr", "kappa_g_truth")]
        for m in methods:
            s, c = S[S.method == m], C[C.method == m]
            if s.empty or c.empty:
                continue
            base = dict(scenario=scen, method=m, n_s=len(s), n_c=len(c),
                        skel_err=s.skel_err.mean(), skel_err_ctrl=c.skel_err.mean(),
                        reject_ghat=s.reject_ghat.mean(), reject_ghat_ctrl=c.reject_ghat.mean(),
                        reject_gstar=s.reject_gstar.mean(), reject_gstar_ctrl=c.reject_gstar.mean(),
                        within_auroc=auroc(s.a_ghat.values, s.a_gstar.values),
                        within_auroc_ctrl=auroc(c.a_ghat.values, c.a_gstar.values),
                        spearman=s.spearman_A_skelerr.mean(), spearman_ctrl=c.spearman_A_skelerr.mean())
            for det in dets:
                a = auroc(s[det].values, c[det].values)
                lo, hi = auroc_ci(s[det].values, c[det].values, rng, n_boot=1000)
                pw = power_at_matched_fpr(s[det].values, c[det].values, 0.1)
                rows.append(dict(base, detector=det, auroc=a, auroc_lo=lo, auroc_hi=hi,
                                 power=pw["power"], fpr=pw["fpr"],
                                 native_flag=float(np.mean(s[det].values > 0)) if ("margin" in det or "quantile" in det or "no subsets" in det or "Uncal" in det) else np.nan,
                                 native_flag_ctrl=float(np.mean(c[det].values > 0)) if ("margin" in det or "quantile" in det or "no subsets" in det or "Uncal" in det) else np.nan))
    return save(pd.DataFrame(rows), "detection")


# ---------------------------------------------------------------------------
# knowledge (R9)
# ---------------------------------------------------------------------------

def sec_knowledge():
    rows = []
    exps = ["know_none"] + [f"know_hard_rev{r}" for r in (0, 10, 20, 30, 40, 50)] + ["know_soft_correct", "know_soft_noisy"]
    for exp in exps:
        for rec in records(exp):
            for key in ("skeleton", "calnopen"):
                if key == "calnopen" and "soft" not in exp:
                    continue
                tau = tau_of(rec, key=key)
                cs = np.asarray(rec["cand_scores"]["skeleton"])
                std = standard_mask(rec["cand_tags"])
                adm = std & (cs <= tau)
                row = dict(experiment=exp, calibration=("penalty in $A_b$" if key == "skeleton" else "penalty on candidates only"),
                           method=rec["method"], repeat=rec["repeat"], tau=tau,
                           exceed=float(np.mean(np.asarray(rec["test_scores"][key]) > tau)),
                           admitted=float(adm.sum() / std.sum()), gstar=float(rec["true_score"]["skeleton"] <= tau),
                           reject_ghat=float(cs[0] > tau), margin=cs[0] - tau)
                for kind in ("adj", "dir_def"):
                    T = truth_claims(rec["true_graph"], rec.get("true_skel"), kind)
                    r = claims_row(claim_frequencies(rec["cand_marks"][adm], kind) if adm.any() else None, T, kind, GAMMA)
                    row[f"{kind}_prec"], row[f"{kind}_rec"] = r["prec"], r["rec"]
                    row[f"single_{kind}_prec"] = point_row(
                        __import__("cscd.analysis_v3", fromlist=["claim_matrix"]).claim_matrix(rec["cand_marks"][0], kind), T, kind)["prec"]
                rows.append(row)
    df = save(pd.DataFrame(rows), "knowledge_per_repeat")
    summ = agg(df.drop(columns=["repeat"]), ["experiment", "calibration", "method"])
    # Detection curve: AUROC of the CSCD margin against the no-knowledge control.
    ctrl = df[df.experiment == "know_none"]
    det = []
    for exp in [f"know_hard_rev{r}" for r in (0, 10, 20, 30, 40, 50)]:
        for m in ("pc", "fci"):
            s, c = df[(df.experiment == exp) & (df.method == m)], ctrl[ctrl.method == m]
            if len(s) and len(c):
                det.append(dict(experiment=exp, method=m, auroc=auroc(s.margin.values, c.margin.values),
                                power=power_at_matched_fpr(s.margin.values, c.margin.values)["power"],
                                reject=s.reject_ghat.mean()))
    save(pd.DataFrame(det), "knowledge_detection")
    return save(summ, "knowledge")


# ---------------------------------------------------------------------------
# alpha sweep (R5)
# ---------------------------------------------------------------------------

def sec_alpha():
    rows = []
    for exp, a in (("alpha_N4", 0.2), ("main_v3", 0.1), ("alpha_N19", 0.05)):
        for rec in records(exp):
            for aa in sorted({a, 0.1, 0.2, 0.05}):
                tau = tau_of(rec, aa)
                if not np.isfinite(tau):
                    continue
                cs = np.asarray(rec["cand_scores"]["skeleton"])
                std = standard_mask(rec["cand_tags"])
                adm = std & (cs <= tau)
                T = truth_claims(rec["true_graph"], rec.get("true_skel"), "adj")
                r = claims_row(claim_frequencies(rec["cand_marks"][adm], "adj") if adm.any() else None, T, "adj", GAMMA)
                rows.append(dict(experiment=exp, N=rec["n_cal"], alpha=aa, method=rec["method"], repeat=rec["repeat"],
                                 tau=tau, exceed=float(np.mean(np.asarray(rec["test_scores"]["skeleton"]) > tau)),
                                 admitted=float(adm.sum() / std.sum()), empty=float(not adm.any()),
                                 gstar=float(rec["true_score"]["skeleton"] <= tau), ghat=float(cs[0] <= tau),
                                 adj_prec=r["prec"], adj_rec=r["rec"]))
    df = save(pd.DataFrame(rows), "alpha_per_repeat")
    return save(agg(df.drop(columns=["repeat"]), ["experiment", "N", "alpha", "method"]), "alpha")


# ---------------------------------------------------------------------------
# MCMC diagnostics (R7)
# ---------------------------------------------------------------------------

def sec_mcmc():
    rows = []
    for exp in ("main_v3", "semi_ecoli70", "sachs_obs_v3"):
        for rec in records(exp):
            if "bayes_rhat" in rec:
                rows.append(dict(experiment=exp, method=rec["method"], repeat=rec["repeat"],
                                 rhat=rec["bayes_rhat"], max_edge_diff=rec["bayes_max_edge_diff"],
                                 rhatF=rec.get("bayesF_rhat", np.nan)))
    df = save(pd.DataFrame(rows), "mcmc_per_repeat")
    out = df.groupby("experiment").agg(n=("rhat", "size"), rhat_median=("rhat", "median"),
                                       frac_rhat_lt_1_1=("rhat", lambda x: float(np.mean(x < 1.1))),
                                       max_edge_diff_median=("max_edge_diff", "median"),
                                       frac_diff_lt_0_2=("max_edge_diff", lambda x: float(np.mean(x < 0.2)))).reset_index()
    return save(out, "mcmc")


# ---------------------------------------------------------------------------
# selective prediction (R11)
# ---------------------------------------------------------------------------

def sec_selective():
    pool = []
    for exp in ("main_v3", "hubs_v3", "highdim_v3", "lowdim_n1000_v3", "quadratic_v3", "know_hard_rev50", "know_none"):
        for rec in records(exp):
            if rec.get("true_graph") is None:
                continue
            row = dict(experiment=exp, method=rec["method"], repeat=rec["repeat"], **detector_scores(rec),
                       skel_err=_skel_err(rec["cand_marks"][0], rec["true_skel"]),
                       p=len(rec["var_names"]))
            row["skel_err_rate"] = row["skel_err"] / (row["p"] * (row["p"] - 1) / 2)
            if rec.get("decision"):
                row["effect_err"] = rec["decision"].get("dec2_bias_single", np.nan)
            pool.append(row)
    df = pd.DataFrame(pool)
    save(df, "selective_pool")
    dets = [c for c in df.columns if c not in ("experiment", "method", "repeat", "skel_err", "p", "skel_err_rate", "effect_err")]
    common = ["CSCD margin $A(\\widehat G)-\\tau_\\alpha$", "Raw $A(\\widehat G)$", "Bootstrap compat.",
              "$\\kappa_G$ (Faller et al.)", "Eulig violations (ratio)", "LOVO loss (Schkoda et al.)"]
    rows = []
    for m in ("pc", "ges", "fci"):
        # One pool of runs for which every compared detector is available.
        d = df[df.method == m].dropna(subset=[c for c in common if c in df.columns])
        for det in dets:
            x = d[[det, "skel_err_rate"]].dropna()
            if len(x) < 10:
                continue
            for r in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
                thr = np.quantile(x[det], 1 - r) if r > 0 else np.inf
                keep = x[det] <= thr
                rows.append(dict(method=m, detector=det, abstain=r, n=len(x), err_kept=x.skel_err_rate[keep].mean(),
                                 err_rejected=x.skel_err_rate[~keep].mean() if (~keep).any() else np.nan,
                                 err_all=x.skel_err_rate.mean()))
    return save(pd.DataFrame(rows), "selective")


# ---------------------------------------------------------------------------
# scale, semi-synthetic, Sachs, decision (R8, R10, R11)
# ---------------------------------------------------------------------------

def _basic_rows(exp: str, with_claims=True):
    rows = []
    for rec in records(exp):
        tau = tau_of(rec)
        cs = np.asarray(rec["cand_scores"]["skeleton"])
        std = standard_mask(rec["cand_tags"], n_candidate_graphs=len([t for t in rec["cand_tags"] if t != "full"]))
        adm = std & (cs <= tau)
        row = dict(experiment=exp, method=rec["method"], repeat=rec["repeat"], tau=tau,
                   exceed=float(np.mean(np.asarray(rec["test_scores"]["skeleton"]) > tau)),
                   admitted=float(adm.sum() / std.sum()), n_subsets=len(rec["subsets"]),
                   sep_cover=rec.get("sep_cover", np.nan), wall=rec.get("wall", np.nan),
                   gstar=float(rec["true_score"]["skeleton"] <= tau) if rec.get("true_score") else np.nan)
        for k, v in rec["times"].items():
            row[f"t_{k}"] = v
        if with_claims and rec.get("true_graph") is not None:
            for kind in ("adj", "dir_def", "anc"):
                T = truth_claims(rec["true_graph"], rec.get("true_skel"), kind)
                for name, f in _estimators(rec, kind, std).items():
                    r = claims_row(f, T, kind, GAMMA)
                    row[f"{name}|{kind}|prec"], row[f"{name}|{kind}|rec"] = r["prec"], r["rec"]
                    row[f"{name}|{kind}|p@r0.5"] = r["p@r0.5"]
                for name, C in _points(rec, kind).items():
                    r = point_row(C, T, kind)
                    row[f"{name}|{kind}|prec"], row[f"{name}|{kind}|rec"] = r["prec"], r["rec"]
        rows.append(row)
    return pd.DataFrame(rows)


def sec_scale():
    df = pd.concat([_basic_rows(e, with_claims=False) for e in ("scale_p100_mb", "scale_p200_mb")], ignore_index=True)
    save(df, "scale_per_repeat")
    return save(agg(df.drop(columns=["repeat"]), ["experiment", "method"]), "scale")


def sec_semi():
    df = _basic_rows("semi_ecoli70")
    save(df, "semi_per_repeat")
    return save(agg(df.drop(columns=["repeat"]), ["experiment", "method"]), "semi")


def sec_sachs():
    from cscd.sachs import interventional_effects, validate_claims

    eff = interventional_effects()
    rows = []
    for rec in records("sachs_obs_v3"):
        names = rec["var_names"]
        std = standard_mask(rec["cand_tags"])
        ests = _estimators(rec, "anc", std)
        ests.update({k: v.astype(float) for k, v in _points(rec, "anc").items()})
        tau = tau_of(rec)
        cs = np.asarray(rec["cand_scores"]["skeleton"])
        for name, f in ests.items():
            if f is None:
                continue
            v = validate_claims(f, names, eff, GAMMA)
            rows.append(dict(method=rec["method"], repeat=rec["repeat"], estimator=name, **v,
                             admitted=float(np.mean(cs[std] <= tau)),
                             consensus_admitted=float(rec["true_score"]["skeleton"] <= tau),
                             consensus_score=rec["true_score"]["skeleton"], tau=tau))
    df = save(pd.DataFrame(rows), "sachs_per_repeat")
    save(eff, "sachs_effects")
    return save(agg(df.drop(columns=["repeat"]), ["method", "estimator"]), "sachs")


def sec_decision():
    rows = []
    for rec in records("main_v3"):
        if rec.get("decision"):
            rows.append(dict(method=rec["method"], repeat=rec["repeat"], **rec["decision"]))
    df = pd.DataFrame(rows)
    save(df, "decision_per_repeat")
    return save(agg(df.drop(columns=["repeat"]), ["method"]), "decision")


SECTIONS = dict(validity=sec_validity, inclusion=sec_inclusion, claims=sec_claims, arms=sec_arms,
                detection=sec_detection, knowledge=sec_knowledge, alpha=sec_alpha, mcmc=sec_mcmc,
                selective=sec_selective, scale=sec_scale, semi=sec_semi, sachs=sec_sachs, decision=sec_decision)


def main():
    names = sys.argv[1:] or list(SECTIONS)
    for n in names:
        try:
            SECTIONS[n]()
        except Exception as exc:
            import traceback

            print(f"[{n}] failed: {exc}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
