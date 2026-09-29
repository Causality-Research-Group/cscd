"""Markdown tables for report.md from results_v2/summary/*.csv.

Usage:  python make_report_tables.py > results_v2/report_tables.md

Every cell is mean ± standard error over repeats (alpha = 0.1, gamma = 0.9,
first distance) unless the table says otherwise.  Missing experiments are
reported as "not run".
"""

import os
import sys

import numpy as np
import pandas as pd

S = "results_v2/summary"
M = {"pc": "PC", "ges": "GES", "fci": "FCI", "lingam": "LiNGAM"}


def load(exp):
    p = os.path.join(S, f"{exp}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def cell(row, key, dec=3):
    m, s = row.get(f"{key}_mean", np.nan), row.get(f"{key}_se", np.nan)
    if pd.isna(m):
        return "–"
    return f"{m:.{dec}f} ± {s:.{dec}f}" if not pd.isna(s) else f"{m:.{dec}f}"


def pick(df, alpha=0.1, gamma=0.9, distance=None):
    d = df[(np.isclose(df.alpha, alpha)) & (np.isclose(df.gamma, gamma))]
    if distance is not None:
        d = d[d.distance == distance]
    else:
        d = d[d.distance == d.distance.iloc[0]] if len(d) else d
    return d


def table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def section(title):
    print(f"\n### {title}\n")


def main_tables():
    df = load("main_v2")
    section("E1/E3/E5 — Main experiment: claims against baselines (p=20, n=3200, N=9 chunks)")
    if df is None:
        print("not run"); return
    d = pick(df)
    est = [
        ("CSCD accepted (γ=0.9)", "cscd"), ("Candidate pool, no filter", "nofilter"),
        ("Uncalibrated self-comp.", "uncal"), ("Single run (D_train)", "single"),
        ("Single run (full D)", "singleF"), ("Stability sel. 0.6 (D_train)", "stab"),
        ("Stability sel. 0.6 (full D)", "stabF"), ("Stability, MB bound E[V]≤1 (D_train)", "stabMB"),
        ("Bootstrap freq. ≥ γ (D_train)", "stabfreq"),
        ("Order MCMC ≥ γ (D_train)", "bayes"), ("Order MCMC ≥ γ (full D)", "bayesF"),
        ("Bootstrap compat., median", "bcmed"), ("Bootstrap compat., (1−α) quantile", "bcq"),
        ("Conformal, no subsets", "nosub"),
    ]
    for _, r in d.iterrows():
        print(f"\n**{M.get(r.method, r.method)}** ({int(r.n_repeats)} repeats)\n")
        rows = []
        for label, k in est:
            rows.append([label, cell(r, f"{k}_precision"), cell(r, f"{k}_recall"),
                         cell(r, f"{k}_avg_precision"), cell(r, f"{k}_prec_at_rec0.7")])
        print(table(rows, ["Estimator", "Precision", "Recall", "Avg. precision (curve)", "Prec. @ recall 0.7"]))
    section("E3 — Adjacency-level claims (pair accepted if either direction is; truth = MAG skeleton)")
    adj = [("CSCD accepted (γ=0.9)", "cscd_skprec", "cscd_skrec", "cscd_skap"),
           ("Candidate pool, no filter", "nofilter_skprec", "nofilter_skrec", "nofilter_skap"),
           ("Uncalibrated self-comp.", "uncal_skprec", "uncal_skrec", "uncal_skap"),
           ("Single run (D_train)", "single_skprec", "single_skrec", None),
           ("Stability sel. 0.6 (D_train)", "stab_skprec", "stab_skrec", None),
           ("Stability sel. 0.6 (full D)", "stabF_skprec", "stabF_skrec", None),
           ("Bootstrap freq. ≥ γ (D_train)", "stabfreq_skprec", "stabfreq_skrec", "stabfreq_skap"),
           ("Bootstrap freq. ≥ γ (full D)", "stabFfreq_skprec", "stabFfreq_skrec", "stabFfreq_skap"),
           ("Order MCMC ≥ γ (D_train)", "bayes_skprec", "bayes_skrec", "bayes_skap"),
           ("Order MCMC ≥ γ (full D)", "bayesF_skprec", "bayesF_skrec", "bayesF_skap")]
    for _, r in d.iterrows():
        print(f"\n**{M.get(r.method, r.method)}**\n")
        rows = [[lab, cell(r, a), cell(r, b), cell(r, c) if c else "–"] for lab, a, b, c in adj]
        print(table(rows, ["Estimator", "Adjacency precision", "Adjacency recall", "Adjacency AP"]))
    section("E1 — Calibration and selectivity (main experiment)")
    rows = []
    for _, r in d.iterrows():
        ex = {a: pick(df, alpha=a)[pick(df, alpha=a).method == r.method].iloc[0] for a in (0.05, 0.1, 0.2)}
        rows.append([M.get(r.method), cell(ex[0.05], "tau_finite", 2), cell(ex[0.1], "exceedance"), cell(ex[0.2], "exceedance"),
                     cell(r, "tau"), cell(r, "true_score"), cell(r, "admission"), cell(r, "true_included", 2),
                     cell(r, "full_admitted", 2), cell(r, "eps_train", 2), cell(r, "eps_chunk", 2)])
    print(table(rows, ["Method", "τ finite @α=.05", "Exceed. @α=.1", "Exceed. @α=.2", "τ", "A(G*)",
                       "Admitted frac.", "G* incl.", "Ĝ_full incl.", "ε train", "ε chunk"]))
    print("\nWith N = 9 chunks, α = 0.05 needs N ≥ 19, so τ = ∞ there (column 2 shows the fraction of finite thresholds).")
    section("E13 — Sensitivity to γ (main experiment, α = 0.1)")
    rows = []
    for g in (0.6, 0.7, 0.8, 0.9, 1.0):
        dg = pick(df, gamma=g)
        for _, r in dg.iterrows():
            rows.append([M.get(r.method), g, cell(r, "cscd_precision"), cell(r, "cscd_recall"), cell(r, "cscd_unresolved"),
                         cell(r, "nofilter_precision"), cell(r, "nofilter_recall")])
    print(table(rows, ["Method", "γ", "CSCD prec.", "CSCD rec.", "CSCD unresolved", "No-filter prec.", "No-filter rec."]))
    section("E11 — Downstream effect estimation (local IDA, held-out data; mean |bias|)")
    keys = [("oracle", "Oracle"), ("naive", "No adjustment"), ("single", "Single run"), ("cscd_set", "CSCD set avg."),
            ("pool", "Pool avg."), ("bootstrap_avg", "Bootstrap avg."), ("bma", "BMA (order MCMC)")]
    rows = []
    for _, r in d.iterrows():
        rows.append([M.get(r.method)] + [cell(r, f"dec2_bias_{k}") for k, _ in keys])
    print(table(rows, ["Method"] + [l for _, l in keys]))
    rows = []
    for _, r in d.iterrows():
        rows.append([M.get(r.method)] + [cell(r, f"dec2_cover_{k}", 2) for k, _ in keys[2:]])
    print("\nCoverage of the true effect by the range of IDA estimates over the set:\n")
    print(table(rows, ["Method"] + [l for _, l in keys[2:]]))
    section("E12 — Runtime per repeat (seconds, one core, all baselines included)")
    rows = [[M.get(r.method), cell(r, "time_total", 0)] for _, r in d.iterrows()]
    print(table(rows, ["Method", "Wall time (s)"]))


def design_tables():
    section("E2/E8 — Chunk design and sample size")
    rows = []
    specs = [("design_N20_c40", "n=3200, N=20×40 (old)"), ("design_N19", "n=3200, N=19"),
             ("design_N9", "n=3200, N=9"), ("design_sym_N9", "n=3200, symmetric N=9 (n_tr=n_c=320)")]
    specs += [(f"n{n}_N9", f"n={n}, N=9") for n in (200, 400, 800, 1600, 6400, 12800)]
    for exp, label in specs:
        df = load(exp)
        if df is None:
            rows.append([label, "not run"] + [""] * 9); continue
        for _, r in pick(df).iterrows():
            rows.append([label, M.get(r.method), f"{r.chunk_size_mean:.0f}", cell(r, "tau"), cell(r, "exceedance"),
                         cell(r, "admission"), cell(r, "true_included", 2), cell(r, "cscd_precision"),
                         cell(r, "cscd_recall"), cell(r, "cscd_avg_precision"), cell(r, "nofilter_avg_precision")])
    print(table(rows, ["Design", "Method", "n_c", "τ", "Exceed.", "Admitted", "G* incl.", "CSCD prec.", "CSCD rec.",
                       "CSCD AP", "No-filter AP"]))
    section("E8 — Low-sample safeguards")
    rows = []
    for n in (200, 400, 800, 1600):
        df = load(f"n{n}_N9")
        if df is None:
            continue
        for _, r in pick(df).iterrows():
            rows.append([n, M.get(r.method), cell(r, "tau_finite", 2), cell(r, "empty_set", 2), cell(r, "cscd_unresolved"),
                         cell(r, "cscd_n_accepted", 1), cell(r, "cscd_precision"), cell(r, "single_precision"),
                         cell(r, "eps_chunk", 2)])
    print(table(rows, ["n", "Method", "τ finite", "Empty set", "Unresolved", "# accepted", "CSCD prec.", "Single prec.", "ε chunk"]))


def sys_tables():
    section("E4/E5 — Systematic errors: is the (wrong) full-data graph admitted, and is G* kept?")
    rows = []
    for exp, label in [("sys_control", "Control (linear Gaussian)"), ("sys_latent_hubs", "(a) Hidden hubs"),
                       ("sys_highdim", "(b) p=50, n=1000"), ("sys_wrongK", "(c) Wrong hard knowledge"),
                       ("sys_quadratic", "(d) Quadratic edges (neg. control)")]:
        df = load(exp)
        if df is None:
            rows.append([label, "not run"] + [""] * 8); continue
        for _, r in pick(df).iterrows():
            rows.append([label, M.get(r.method), cell(r, "full_skel_errors", 1),
                         cell(r, "full_admitted", 2), cell(r, "true_included", 2),
                         cell(r, "bcq_full_admitted", 2), cell(r, "bcq_true_included", 2),
                         cell(r, "bcmed_full_admitted", 2), cell(r, "nosub_full_admitted", 2), cell(r, "nosub_true_included", 2)])
    print(table(rows, ["Scenario", "Method", "Skeleton errors of Ĝ_full", "CSCD: Ĝ_full admitted", "CSCD: G* incl.",
                       "BootComp q: Ĝ_full adm.", "BootComp q: G* incl.", "BootComp med: Ĝ_full adm.",
                       "No subsets: Ĝ_full adm.", "No subsets: G* incl."]))


def subset_tables():
    section("E6 — Subset design (PC, FCI; p=20, n=3200)")
    rows = []
    exps = [f"sub_k{k}_m{m}" for k in (0.3, 0.5, 0.7) for m in (10, 25, 50)]
    exps += ["sub_k0.5_m171", "sub_k0.7_m33", "sub_mb_k0.3", "sub_mb_k0.5", "sub_k0.3_m10_nocover"]
    for exp in exps:
        df = load(exp)
        if df is None:
            rows.append([exp, "not run"] + [""] * 8); continue
        for _, r in pick(df).iterrows():
            rows.append([exp.replace("sub_", ""), M.get(r.method), cell(r, "n_subsets", 0), cell(r, "sep_cover", 3),
                         cell(r, "admission"), cell(r, "true_included", 2), cell(r, "cscd_precision"), cell(r, "cscd_recall"),
                         cell(r, "cscd_unresolved"), cell(r, "time_total", 0)])
    print(table(rows, ["Design", "Method", "# subsets", "Separation cover", "Admitted", "G* incl.", "Prec.", "Rec.",
                       "Unresolved", "Time (s)"]))


def distance_tables():
    section("E7 — Distance choice (PC, FCI)")
    df = load("distances")
    if df is None:
        print("not run"); return
    rows = []
    for dist in ("skeleton", "pag", "skel_extra2", "skel_miss2"):
        for _, r in pick(df, distance=dist).iterrows():
            rows.append([dist, M.get(r.method), cell(r, "exceedance"), cell(r, "tau"), cell(r, "admission"),
                         cell(r, "true_included", 2), cell(r, "cscd_precision"), cell(r, "cscd_skel_precision"),
                         cell(r, "cscd_recall"), cell(r, "cscd_avg_precision")])
    print(table(rows, ["Distance", "Method", "Exceed.", "τ", "Admitted", "G* incl.", "Directed prec.", "Skeleton prec.",
                       "Recall", "AP"]))


def latent_tables():
    section("E9 — PC vs FCI under hidden confounding (6 hidden hubs, degree 3)")
    rows = []
    for exp, n in [("latent_n1600", 1600), ("sys_latent_hubs", 3200), ("latent_n6400", 6400), ("latent_n25600", 25600)]:
        df = load(exp)
        if df is None:
            rows.append([n, "not run"] + [""] * 5); continue
        for _, r in pick(df).iterrows():
            if r.method not in ("pc", "fci"):
                continue
            rows.append([n, M.get(r.method), cell(r, "eps_train", 3), cell(r, "eps_chunk", 3), cell(r, "true_score"),
                         cell(r, "tau"), cell(r, "true_included", 2)])
    print(table(rows, ["n", "Method", "ε (train, per subset)", "ε (chunk)", "A(G*)", "τ", "G* incl."]))


def robust_tables():
    section("E10 — Robustness and real data")
    rows = []
    specs = [("knowledge_none", "No knowledge"), ("knowledge_correct_hard", "Correct, hard"),
             ("knowledge_correct_soft", "Correct, soft"), ("knowledge_noisy_hard", "Noisy, hard"),
             ("knowledge_noisy_soft", "Noisy, soft"), ("large_p50", "p = 50"), ("nongauss_laplace", "Laplace noise"),
             ("latent_random", "5 of 25 hidden (random)"), ("sachs_obs", "Sachs, observational (853)"),
             ("sachs_pooled", "Sachs, pooled (7466)"), ("bnlearn_asia", "Asia"), ("bnlearn_child", "Child"),
             ("bnlearn_insurance", "Insurance"), ("bnlearn_alarm", "Alarm"), ("kci_quadratic", "KCI, quadratic, p=8")]
    for exp, label in specs:
        df = load(exp)
        if df is None:
            rows.append([label, "not run"] + [""] * 8); continue
        for _, r in pick(df).iterrows():
            rows.append([label, M.get(r.method), cell(r, "exceedance"), cell(r, "admission"), cell(r, "true_included", 2),
                         cell(r, "cscd_precision"), cell(r, "cscd_recall"), cell(r, "single_precision"),
                         cell(r, "stab_precision"), cell(r, "nofilter_precision")])
    print(table(rows, ["Setting", "Method", "Exceed.", "Admitted", "G* incl.", "CSCD prec.", "CSCD rec.",
                       "Single prec.", "Stab. prec.", "No-filter prec."]))
    print("\nFor Sachs and bnlearn, 'G* incl.' refers to the reference network (Sachs: 17-edge consensus).")


if __name__ == "__main__":
    for f in (main_tables, design_tables, sys_tables, subset_tables, distance_tables, latent_tables, robust_tables):
        try:
            f()
        except Exception as exc:  # keep the report buildable while runs are incomplete
            print(f"\n(table {f.__name__} failed: {exc})\n", file=sys.stdout)
