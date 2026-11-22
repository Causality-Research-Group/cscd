"""Summaries of the wave-2 rebuttal experiments (branch rebuttal/experiments-chris).

Usage:  .venv/bin/python -W ignore -m rebuttal.analyze_chris [section ...]
Sections: nosub trainfrac nscale kscope hubdose lam gamma   (default: all)
Writes CSV files to rebuttal/out/chris/ and a markdown digest
rebuttal/out/chris/<section>.md for each section.
"""

import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd

from analyze_v3 import _skel_err, agg, detector_scores, tau_of
from cscd.analysis_v3 import (
    auroc,
    claim_frequencies,
    claim_matrix,
    claims_row,
    inclusion_diagnostics,
    point_row,
    power_at_matched_fpr,
    standard_mask,
    truth_claims,
)
from cscd.selfcompat import conformal_threshold

R3, RC = "results_v3", os.environ.get("CSCD_CHRIS_ROOT", "results_v3_chris")
OUT = "rebuttal/out/chris"
ALPHA, GAMMA = 0.1, 0.9


def records(exp, root=RC):
    for path in sorted(glob.glob(os.path.join(root, exp, "rec_*.pkl"))):
        with open(path, "rb") as f:
            yield pickle.load(f)


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def md(df, digits=3):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def save(df, name, digits=3, note=""):
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
    with open(os.path.join(OUT, f"{name}.md"), "w") as f:
        f.write(f"# {name}\n\n{note}\n\n{md(df, digits)}\n")
    print(f"[{name}] {len(df)} rows")
    return df


def rate_ci(s):
    k, n = float(np.nansum(s)), int(np.sum(~np.isnan(s)))
    lo, hi = wilson(k, n)
    return f"{k / max(n, 1):.2f} [{lo:.2f}, {hi:.2f}]"


def claims_of(rec, adm, kinds=("adj", "dir_def")):
    out = {}
    for kind in kinds:
        T = truth_claims(rec["true_graph"], rec.get("true_skel"), kind)
        r = claims_row(claim_frequencies(rec["cand_marks"][adm], kind) if adm.any() else None, T, kind, GAMMA)
        s = point_row(claim_matrix(rec["cand_marks"][0], kind), T, kind)
        out.update({f"{kind}_prec": r["prec"], f"{kind}_rec": r["rec"],
                    f"single_{kind}_prec": s["prec"], f"single_{kind}_rec": s["rec"]})
    return out


# ---------------------------------------------------------------------------
# AI-3.1: redefined "Conformal, no subsets"
# ---------------------------------------------------------------------------

def sec_nosub():
    df = pd.read_csv("rebuttal/out/nosub_per_run.csv")
    pairs = [("Hidden hubs", "hubs_v3", "main_v3"), ("p=50, n=1000", "highdim_v3", "lowdim_n1000_v3"),
             ("Quadratic (shared)", "quadratic_v3", "main_v3")]
    rows = []
    for scen, e, c in pairs:
        for m in ("pc", "ges", "fci"):
            s, k = df[(df.experiment == e) & (df.method == m)], df[(df.experiment == c) & (df.method == m)]
            if s.empty or k.empty:
                continue
            for det, col, flag in (("CSCD (subsets + calibration)", "cscd_margin", "cscd_flag"),
                                   ("No subsets, full-data run", "nosub_full_score", "nosub_full_flag"),
                                   ("No subsets, split D_train", "nosub_split_score", "nosub_split_flag")):
                if col == "cscd_margin":
                    sv, kv = s[col].values, k[col].values
                else:
                    sv = s[col].values - s[col.replace("score", "tau")].values
                    kv = k[col].values - k[col.replace("score", "tau")].values
                rows.append(dict(scenario=scen, method=m, detector=det, runs=len(s),
                                 flag_scenario=rate_ci(s[flag].values), flag_control=rate_ci(k[flag].values),
                                 auroc=auroc(sv, kv), power_at_10pct=power_at_matched_fpr(sv, kv, 0.1)["power"],
                                 skel_err_scen=s.skel_err_ghat.mean(), ghat_reproduced=s.ghat_reproduced.mean()))
    return save(pd.DataFrame(rows), "nosub",
                note="Native flag rates (Wilson 95%) with each detector's own conformal threshold at alpha=0.1, "
                     "and AUROC/power of its margin against the control scenario.")


# ---------------------------------------------------------------------------
# Kmr8-Q1: training fraction
# ---------------------------------------------------------------------------

def sec_trainfrac():
    rows = []
    for f in (30, 40, 50, 60, 70, 80):
        for rec in records(f"trainfrac_{f}"):
            d = inclusion_diagnostics(rec, ALPHA)
            tau = d["tau"]
            cs = np.asarray(rec["cand_scores"]["skeleton"])
            std = standard_mask(rec["cand_tags"])
            adm = std & (cs <= tau)
            row = dict(train_frac=f / 100, method=rec["method"], repeat=rec["repeat"], n_train=rec["n_train"],
                       chunk=rec["chunk_size"], tau=tau,
                       exceed=float(np.mean(np.asarray(rec["test_scores"]["skeleton"]) > tau)),
                       gstar=d["included"], rank_A_gstar=d["rank"], delta=d["delta"], pred_incl=1 - d["pred_excl"],
                       eps_train=d.get("eps_train"), eps_chunk=d.get("eps_chunk"),
                       frac_wrong_pairs=d.get("frac_wrong_pairs"), e_right=d.get("e_right"), e_wrong=d.get("e_wrong"),
                       admitted=float(adm.sum() / std.sum()), empty=float(adm.sum() == 0),
                       ghat_flag=float(cs[0] > tau),
                       ghat_skel_err=_skel_err(rec["cand_marks"][0], rec["true_skel"]),
                       **claims_of(rec, adm))
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "trainfrac_per_run.csv"), index=False)
    g = df.groupby(["method", "train_frac"])
    summ = g[["n_train", "chunk", "tau", "exceed", "gstar", "pred_incl", "delta", "eps_train", "eps_chunk",
              "e_right", "e_wrong", "admitted", "empty", "ghat_flag", "ghat_skel_err",
              "adj_prec", "adj_rec", "dir_def_prec", "dir_def_rec"]].mean().reset_index()
    summ.insert(2, "runs", g.size().values)
    summ["gstar_wilson"] = [rate_ci(df[(df.method == m) & (df.train_frac == t)].gstar.values)
                            for m, t in zip(summ.method, summ.train_frac)]
    return save(summ, "trainfrac", note="N=9 chunks, alpha=0.1, gamma=0.9, main setting (p=20, n=3200). "
                "Claims are precision/recall of accepted claims of the passing set; ghat_skel_err in pairs.")


# ---------------------------------------------------------------------------
# d2sj-W4: sample-size scaling
# ---------------------------------------------------------------------------

def sec_nscale():
    rows = []
    exps = [(f"nscale_{sc}_n{n}", sc, n, "0.05") for sc in ("lin", "hubs")
            for n in (400, 800, 1600, 3200, 6400, 12800, 25600)]
    exps += [(f"nscalea_{sc}_n{n}", sc, n, "0.05*1600/n") for sc in ("lin", "hubs") for n in (6400, 25600, 102400)]
    for exp, scen, n, alevel in exps:
        if True:
            for rec in records(exp):
                d = inclusion_diagnostics(rec, ALPHA)
                cal = np.asarray(rec["cal_scores"]["skeleton"])
                tau = d["tau"]
                cs = np.asarray(rec["cand_scores"]["skeleton"])
                T = rec["truth_proj"].astype(bool)
                L = rec["local_train"]
                iu = np.triu_indices(T.shape[1], 1)
                ok = L[:, 0, 0] != -1
                Lk, Tk = (L[ok][:, iu[0], iu[1]] != 0), T[ok][:, iu[0], iu[1]]
                rows.append(dict(scenario=scen, n=n, test_alpha=alevel, method=rec["method"], repeat=rec["repeat"],
                                 extra_slot=float(np.mean(Lk & ~Tk)), missing_slot=float(np.mean(~Lk & Tk)),
                                 tau=tau, tau_zero=float(tau == 0), a_gstar=d["a_star"],
                                 a_gstar_zero=float(d["a_star"] == 0), gstar=d["included"],
                                 exceed=float(np.mean(np.asarray(rec["test_scores"]["skeleton"]) > tau)),
                                 eps_train=d.get("eps_train"), eps_chunk=d.get("eps_chunk"),
                                 frac_wrong_pairs=d.get("frac_wrong_pairs"), delta=d["delta"],
                                 frac_sufficient=d.get("frac_sufficient"),
                                 ghat_flag=float(cs[0] > tau),
                                 ghat_skel_err=_skel_err(rec["cand_marks"][0], rec["true_skel"])))
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "nscale_per_run.csv"), index=False)
    g = df.groupby(["scenario", "test_alpha", "method", "n"])
    summ = g[["eps_train", "eps_chunk", "frac_wrong_pairs", "extra_slot", "missing_slot", "a_gstar_zero", "tau_zero", "gstar", "delta",
              "exceed", "ghat_flag", "ghat_skel_err", "frac_sufficient"]].mean().reset_index()
    summ.insert(4, "runs", g.size().values)
    summ["gstar_wilson"] = [rate_ci(df[(df.scenario == s) & (df.test_alpha == a) & (df.method == m) & (df.n == n)].gstar.values)
                            for s, a, m, n in zip(summ.scenario, summ.test_alpha, summ.method, summ.n)]
    return save(summ, "nscale", note="Score-only runs, N=9, alpha=0.1, 50/50 split. lin: p=20 linear Gaussian, "
                "no hidden variables; hubs: p=26, the 6 largest-out-degree variables hidden. eps_train: fraction of "
                "training subset graphs with any skeleton error vs the true marginal; frac_wrong_pairs: fraction of "
                "(subset, pair) slots wrong; tau_zero: P(tau_0.1 = 0); a_gstar_zero: P(A(G*) = 0).")


# ---------------------------------------------------------------------------
# Rfqt-Q1: scope of knowledge
# ---------------------------------------------------------------------------

def _det_rows(exp, root=RC, label=None):
    rows = []
    for rec in records(exp, root):
        tau = tau_of(rec)
        cs = np.asarray(rec["cand_scores"]["skeleton"])
        std = standard_mask(rec["cand_tags"])
        adm = std & (cs <= tau)
        row = dict(experiment=label or exp, method=rec["method"], repeat=rec["repeat"], tau=tau,
                   gstar=float(rec["true_score"]["skeleton"] <= tau), admitted=float(adm.sum() / std.sum()),
                   ghat_flag=float(cs[0] > tau), margin=cs[0] - tau,
                   ghat_skel_err=_skel_err(rec["cand_marks"][0], rec["true_skel"]),
                   exceed=float(np.mean(np.asarray(rec["test_scores"]["skeleton"]) > tau)),
                   **claims_of(rec, adm))
        ds = detector_scores(rec)
        if "Bootstrap compat. (own quantile)" in ds:
            row["boot_own"] = ds["Bootstrap compat. (own quantile)"]
            row["boot_flag"] = float(ds["Bootstrap compat. (own quantile)"] > 0)
        if "Uncal. threshold" in ds:
            row["uncal"] = ds["Uncal. threshold"]
            row["uncal_flag"] = float(ds["Uncal. threshold"] > 0)
        rows.append(row)
    return rows


def sec_kscope():
    # The "all" and "none" arms are the stored results_v3 records (same seeds and options).
    old = {"kscope_none": "know_none", "kscope_all_rev0": "know_hard_rev0", "kscope_all_rev30": "know_hard_rev30"}
    # The "local" arm (subset graphs only) was stopped after one run: in skeleton mode
    # orientation-only knowledge leaves the subset skeletons, hence the score, unchanged.
    exps = ["kscope_none"] + [f"kscope_{s}_rev{r}" for s in ("all", "candidates") for r in (0, 30)]
    df = pd.DataFrame([r for e in exps for r in (_det_rows(old[e], R3, e) if e in old else _det_rows(e))])
    df = df[df.repeat < 20]
    df.to_csv(os.path.join(OUT, "kscope_per_run.csv"), index=False)
    ctrl = df[df.experiment == "kscope_none"]
    rows = []
    for e in exps:
        for m in ("pc", "fci"):
            s, c = df[(df.experiment == e) & (df.method == m)], ctrl[ctrl.method == m]
            if s.empty:
                continue
            rows.append(dict(arm=e, method=m, runs=len(s), gstar=rate_ci(s.gstar.values),
                             admitted=s.admitted.mean(), ghat_flag=rate_ci(s.ghat_flag.values),
                             boot_flag=s.boot_flag.mean() if "boot_flag" in s else np.nan,
                             auroc_vs_none=auroc(s.margin.values, c.margin.values) if e != "kscope_none" else np.nan,
                             ghat_skel_err=s.ghat_skel_err.mean(), exceed=s.exceed.mean(),
                             adj_prec=s.adj_prec.mean(), adj_rec=s.adj_rec.mean(),
                             dir_prec=s.dir_def_prec.mean(), dir_rec=s.dir_def_rec.mean(),
                             single_dir_prec=s.single_dir_def_prec.mean(), single_dir_rec=s.single_dir_def_rec.mean()))
    return save(pd.DataFrame(rows), "kscope",
                note="Hard knowledge (30% of orderable pairs as forbidden edges; rev30: 30% of them reversed). "
                     "all = every discovery uses K (paper); candidates = full-variable runs only, subset graphs "
                     "without K; local = subset graphs only, candidates without K. ghat_flag = CSCD native flag "
                     "on G-hat; auroc_vs_none = AUROC of the CSCD margin against the no-knowledge runs.")


# ---------------------------------------------------------------------------
# Rfqt-W3 / Kmr8-W2: dose-response in hidden hubs
# ---------------------------------------------------------------------------

def sec_hubdose(prefix="hubdose", deg="3"):
    exps = [f"{prefix}_h{h}" for h in (0, 1, 2, 3, 4, 6)]
    df = pd.DataFrame([r for e in exps for r in _det_rows(e)])
    if df.empty:
        return df
    df.to_csv(os.path.join(OUT, f"{prefix}_per_run.csv"), index=False)
    ctrl = df[df.experiment == f"{prefix}_h0"]
    rows = []
    for e in exps:
        for m in ("pc", "ges", "fci"):
            s, c = df[(df.experiment == e) & (df.method == m)], ctrl[ctrl.method == m]
            if s.empty:
                continue
            rows.append(dict(hidden_hubs=int(e.split("_h")[1]), method=m, runs=len(s),
                             ghat_skel_err=s.ghat_skel_err.mean(), cscd_flag=rate_ci(s.ghat_flag.values),
                             boot_flag=rate_ci(s.boot_flag.values) if "boot_flag" in s else "",
                             uncal_flag=rate_ci(s.uncal_flag.values) if "uncal_flag" in s else "",
                             cscd_auroc_vs_h0=auroc(s.margin.values, c.margin.values) if not e.endswith("_h0") else np.nan,
                             boot_auroc_vs_h0=auroc(s.boot_own.values, c.boot_own.values) if not e.endswith("_h0") else np.nan,
                             gstar=s.gstar.mean(), admitted=s.admitted.mean(), exceed=s.exceed.mean(),
                             adj_prec=s.adj_prec.mean(), single_adj_prec=s.single_adj_prec.mean()))
    out = save(pd.DataFrame(rows), prefix,
               note=f"p=26, expected degree {deg}, the h variables of largest out-degree hidden, n=3200, N=9. "
                    "Flags use each rule's own threshold (Wilson 95%), no control sample.")
    # Gating across the pooled hub-dose runs: skeleton error of kept vs flagged runs.
    g = []
    for m in ("pc", "ges", "fci"):
        s = df[df.method == m]
        for rule, col in (("CSCD native flag", "ghat_flag"), ("Bootstrap own quantile", "boot_flag"),
                          ("Uncal. threshold", "uncal_flag")):
            if col not in s:
                continue
            kept, fl = s[s[col] == 0], s[s[col] == 1]
            g.append(dict(method=m, rule=rule, runs=len(s), flagged=s[col].mean(),
                          skel_err_all=s.ghat_skel_err.mean(), skel_err_kept=kept.ghat_skel_err.mean(),
                          skel_err_flagged=fl.ghat_skel_err.mean() if len(fl) else np.nan,
                          reduction=1 - kept.ghat_skel_err.mean() / s.ghat_skel_err.mean()))
    save(pd.DataFrame(g), f"{prefix}_gating", note="Pooled over 0-6 hidden hubs (skeleton errors in pairs).")
    return out


# ---------------------------------------------------------------------------
# Rfqt-Q1: soft-penalty lambda sweep (offline, results_v3/know_soft_*)
# ---------------------------------------------------------------------------

def _penalty(adj, idx, forbidden):
    """R_K(G): fraction of applicable forbidden edges (a, b) with adj[a, b] == 1."""
    tot = vio = 0
    for a, b in forbidden:
        if a in idx and b in idx:
            tot += 1
            vio += int(adj[idx[a], idx[b]] == 1)
    return vio / tot if tot else 0.0


def sec_lam():
    lams = (0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
    rows = []
    for exp in ("know_soft_correct", "know_soft_noisy"):
        for path in sorted(glob.glob(os.path.join(R3, exp, "rec_*.pkl"))):
            rec = pickle.load(open(path, "rb"))
            names = rec["var_names"]
            idx = {v: i for i, v in enumerate(names)}
            forb = [tuple(e) for e in rec["knowledge_forbidden"]]
            lam0 = 1.0  # cfg.lambda_penalty of the stored runs
            cal_d = np.asarray(rec["cal_scores"]["calnopen"])
            cal_pen = (np.asarray(rec["cal_scores"]["skeleton"]) - cal_d) / lam0
            test_d = np.asarray(rec["test_scores"]["calnopen"])
            test_pen = (np.asarray(rec["test_scores"]["skeleton"]) - test_d) / lam0
            cand_pen = np.array([_penalty(A, idx, forb) for A in rec["cand_adj"]])
            cand_d = np.asarray(rec["cand_scores"]["skeleton"]) - lam0 * cand_pen
            true_pen = _penalty(rec["true_graph"], idx, forb)
            true_d = rec["true_score"]["skeleton"] - lam0 * true_pen
            std = standard_mask(rec["cand_tags"])
            for lam in lams:
                for mode in ("penalty in A_b", "candidates only"):
                    cal = cal_d + (lam * cal_pen if mode == "penalty in A_b" else 0)
                    test = test_d + (lam * test_pen if mode == "penalty in A_b" else 0)
                    tau = conformal_threshold(cal, ALPHA)
                    cs = cand_d + lam * cand_pen
                    adm = std & (cs <= tau)
                    rows.append(dict(knowledge=exp.split("_")[-1], calibration=mode, lam=lam, method=rec["method"],
                                     repeat=rec["repeat"], tau=tau, exceed=float(np.mean(test > tau)),
                                     gstar=float(true_d + lam * true_pen <= tau), admitted=float(adm.sum() / std.sum()),
                                     ghat_flag=float(cs[0] > tau), **claims_of(rec, adm)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "lam_per_run.csv"), index=False)
    g = df.groupby(["knowledge", "method", "calibration", "lam"])
    summ = g[["tau", "exceed", "gstar", "admitted", "ghat_flag", "adj_prec", "adj_rec", "dir_def_prec",
              "dir_def_rec"]].mean().reset_index()
    summ.insert(4, "runs", g.size().values)
    return save(summ, "lam", note="Offline recomputation from results_v3/know_soft_{correct,noisy} (20 runs, "
                "local discovery without knowledge, soft constraints: 30% of orderable pairs; noisy: 30% reversed). "
                "'penalty in A_b' adds lambda*R_K to the calibration scores (paper); 'candidates only' does not.")


# ---------------------------------------------------------------------------
# d2sj-W1 / Kmr8-W1: claim error as a function of gamma (offline, main_v3)
# ---------------------------------------------------------------------------

def sec_gamma():
    rows = []
    for path in sorted(glob.glob(os.path.join(R3, "main_v3", "rec_*.pkl"))):
        rec = pickle.load(open(path, "rb"))
        tau = tau_of(rec)
        cs = np.asarray(rec["cand_scores"]["skeleton"])
        std = standard_mask(rec["cand_tags"])
        adm = std & (cs <= tau)
        for kind in ("adj", "dir_def", "anc"):
            T = truth_claims(rec["true_graph"], rec.get("true_skel"), kind)
            fc = claim_frequencies(rec["cand_marks"][adm], kind) if adm.any() else None
            fp = claim_frequencies(rec["cand_marks"][std], kind)
            fb = claim_frequencies(rec["boot_marks"], kind) if rec.get("boot_marks") is not None else None
            for gamma in (0.6, 0.7, 0.8, 0.9, 0.95, 1.0):
                for est, F in (("CSCD", fc), ("Pool, no filter", fp), ("Bootstrap freq., D_tr", fb)):
                    r = claims_row(F, T, kind, gamma)
                    rows.append(dict(kind=kind, method=rec["method"], repeat=rec["repeat"], estimator=est,
                                     gamma=gamma, n_acc=r["n_acc"], fdp=1 - r["prec"] if r["n_acc"] else np.nan,
                                     rec=r["rec"], abstain=float(r["n_acc"] == 0)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "gamma_per_run.csv"), index=False)
    g = df.groupby(["kind", "method", "estimator", "gamma"])
    summ = g[["fdp", "rec", "n_acc", "abstain"]].mean().reset_index()
    summ["p_fdp_pos"] = g.fdp.apply(lambda s: float(np.mean(s.dropna() > 0))).values
    return save(summ, "gamma", note="main_v3, 30 runs. FDP over runs with at least one accepted claim; "
                "recall 0 for abstaining runs. Bootstrap frequencies on D_train (same data as CSCD).")


def sec_nscand():
    rows = []
    for scen in ("lin", "hubs"):
        for n in (1600, 6400, 25600):
            for rec in records(f"nscand_{scen}_n{n}"):
                tau = tau_of(rec)
                cs = np.asarray(rec["cand_scores"]["skeleton"])
                std = standard_mask(rec["cand_tags"])
                adm = std & (cs <= tau)
                T = rec["true_skel"].astype(bool)
                iu = np.triu_indices(len(T), 1)
                sk = (rec["cand_marks"] != 0)
                sk = sk | np.transpose(sk, (0, 2, 1))
                fa = np.array([np.sum(S[iu] & ~T[iu]) for S in sk])     # false adjacencies
                mi = np.array([np.sum(~S[iu] & T[iu]) for S in sk])     # missing adjacencies
                row = dict(scenario=scen, n=n, method=rec["method"], repeat=rec["repeat"], tau=tau,
                           tau_zero=float(tau == 0), gstar=float(rec["true_score"]["skeleton"] <= tau),
                           admitted=float(adm.sum() / std.sum()), empty=float(adm.sum() == 0),
                           fa_ghat=fa[0], fa_pool_mean=fa[std].mean(),
                           fa_adm_mean=fa[adm].mean() if adm.any() else np.nan,
                           fa_adm_max=fa[adm].max() if adm.any() else np.nan,
                           adm_zero_fa=float(np.mean(fa[adm] == 0)) if adm.any() else np.nan,
                           miss_adm_mean=mi[adm].mean() if adm.any() else np.nan,
                           ghat_flag=float(cs[0] > tau), **claims_of(rec, adm))
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "nscand_per_run.csv"), index=False)
    g = df.groupby(["scenario", "method", "n"])
    summ = g[["tau", "tau_zero", "gstar", "admitted", "empty", "fa_ghat", "fa_pool_mean", "fa_adm_mean",
              "fa_adm_max", "adm_zero_fa", "miss_adm_mean", "ghat_flag", "adj_prec", "adj_rec"]].mean().reset_index()
    summ.insert(3, "runs", g.size().values)
    return save(summ, "nscand", note="Standard candidate family (101 graphs), N=9, alpha=0.1. fa = false adjacencies "
                "(pairs adjacent in the candidate, non-adjacent in the true marginal over V); miss = missing adjacencies.")


SECTIONS = dict(nscand=sec_nscand, nosub=sec_nosub, trainfrac=sec_trainfrac, nscale=sec_nscale, kscope=sec_kscope,
                hubdose=sec_hubdose, hubdose2=lambda: sec_hubdose("hubdose2", "2"), lam=sec_lam, gamma=sec_gamma)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for s in (sys.argv[1:] or list(SECTIONS)):
        try:
            SECTIONS[s]()
        except Exception as exc:  # keep going: partial results are useful
            import traceback
            print(f"[{s}] FAILED: {exc}")
            traceback.print_exc()
