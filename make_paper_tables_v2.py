"""LaTeX tables for the revised results section, from results_v2/summary/*.csv.

Usage:  python make_paper_tables_v2.py --out <paper>/tables/v2

Each table is a bare tabular (the paper wraps it in table/adjustbox).  Cells
are mean ± standard error over repeats at alpha = 0.1, gamma = 0.9.
"""

import argparse
import os

import numpy as np
import pandas as pd

S = "results_v2/summary"
M = {"pc": "PC", "ges": "GES", "fci": "FCI", "lingam": "LiNGAM"}
ORDER = ["pc", "ges", "fci", "lingam"]


def load(exp):
    p = os.path.join(S, f"{exp}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def rows(df, alpha=0.1, gamma=0.9, distance=None):
    d = df[np.isclose(df.alpha, alpha) & np.isclose(df.gamma, gamma)].copy()
    d = d[d.distance == (distance or d.distance.iloc[0])]
    d = d.assign(_o=d.method.map({m: i for i, m in enumerate(ORDER)})).sort_values("_o")
    return [r for _, r in d.iterrows()]


def c(r, k, dec=2, se=True):
    m = r.get(f"{k}_mean", np.nan)
    if pd.isna(m):
        return "--"
    s = r.get(f"{k}_se", np.nan)
    if se and not pd.isna(s):
        return f"${m:.{dec}f}\\pm{s:.{dec}f}$"
    return f"${m:.{dec}f}$"


def write(out, name, lines):
    with open(os.path.join(out, name), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", name)


def t_calibration(out):
    main = load("main_v2")
    lines = [r"\begin{tabular}{lccccccc}", r"\toprule",
             r" & \multicolumn{2}{c}{Exceedance} & & & & & \\ \cmidrule(lr){2-3}",
             r"Method & $\alpha{=}.1$ & $\alpha{=}.2$ & $\tau_{0.1}$ & $A(G^\ast)$ & $|\mathcal{C}_\alpha|/|\mathcal{C}|$ & $G^\ast\in\mathcal{C}_\alpha$ & $\widehat G\in\mathcal{C}_\alpha$ \\", r"\midrule"]
    r2 = {r.method: r for r in rows(main, alpha=0.2)}
    for r in rows(main):
        lines.append(f"{M[r.method]} & {c(r,'exceedance',3)} & {c(r2[r.method],'exceedance',3)} & {c(r,'tau',3)} & "
                     f"{c(r,'true_score',3)} & {c(r,'admission')} & {c(r,'true_included')} & {c(r,'full_admitted')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, "calibration.tex", lines)


def t_chunk(out):
    lines = [r"\begin{tabular}{llccccc}", r"\toprule",
             r"Design & Method & $n_c$ & $\tau_{0.1}$ & Exceed. & Admitted & $G^\ast\in\mathcal{C}_\alpha$ \\", r"\midrule"]
    for e, lab in [("design_N20_c40", "$N{=}20$"), ("design_N19", "$N{=}19$"), ("design_N9", "$N{=}9$"),
                   ("design_sym_N9", r"$N{=}9$, $n_{\mathrm{tr}}{=}n_c$")]:
        df = load(e)
        if df is None:
            continue
        for i, r in enumerate(rows(df)):
            lines.append(f"{lab if i == 0 else ''} & {M[r.method]} & {r.chunk_size_mean:.0f} & {c(r,'tau',3)} & "
                         f"{c(r,'exceedance',3)} & {c(r,'admission')} & {c(r,'true_included')} \\\\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    write(out, "chunk_design.tex", lines)


def t_systematic(out):
    from sklearn.metrics import roc_auc_score

    def per(e):
        p = f"results_v2/{e}/metrics.csv"
        if not os.path.exists(p):
            return None
        d = pd.read_csv(p)
        return d[np.isclose(d.alpha, 0.1) & np.isclose(d.gamma, 0.9)]

    ctrl = per("sys_control")
    lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
             r" & & & \multicolumn{2}{c}{Reject $\widehat G$} & \multicolumn{2}{c}{AUROC} & \\ \cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r"Scenario & Method & Skel.\ err. & CSCD & Boot. & CSCD & Boot. & $G^\ast\in\mathcal{C}_\alpha$ \\", r"\midrule"]
    for e, lab in [("sys_control", "Control"), ("sys_latent_hubs", "Hidden hubs"), ("sys_highdim", r"$p{=}50$, $n{=}1000$"),
                   ("sys_wrongK", "Wrong knowledge"), ("sys_quadratic", "Quadratic (shared)")]:
        df = load(e)
        s = per(e)
        if df is None:
            continue
        for i, r in enumerate(rows(df)):
            auc_c = auc_b = "--"
            if e != "sys_control" and ctrl is not None:
                a, b = ctrl[ctrl.method == r.method], s[s.method == r.method]
                if len(a) and len(b):
                    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
                    auc_c = f"${roc_auc_score(y, np.r_[(a.full_score - a.tau).values, (b.full_score - b.tau).values]):.2f}$"
                    auc_b = f"${roc_auc_score(y, np.r_[a.bcq_full_score.values, b.bcq_full_score.values]):.2f}$"
            rej_c = 1 - r["full_admitted_mean"]
            rej_b = 1 - r["bcq_full_admitted_mean"]
            lines.append(f"{lab if i == 0 else ''} & {M[r.method]} & {c(r,'full_skel_errors',1,se=False)} & ${rej_c:.2f}$ & ${rej_b:.2f}$ & "
                         f"{auc_c} & {auc_b} & {c(r,'true_included',2,se=False)} \\\\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    write(out, "systematic.tex", lines)


def t_claims(out):
    main = load("main_v2")
    est = [("CSCD (ours)", "cscd_skprec", "cscd_skrec", "cscd_avg_precision"),
           ("Candidate pool, no filter", "nofilter_skprec", "nofilter_skrec", "nofilter_avg_precision"),
           ("Uncalibrated self-comp.", "uncal_skprec", "uncal_skrec", "uncal_avg_precision"),
           ("Single run", "single_skprec", "single_skrec", None),
           ("Stability selection", "stab_skprec", "stab_skrec", None),
           ("Bootstrap frequency", "stabfreq_skprec", "stabfreq_skrec", "stabfreq_avg_precision"),
           ("Order MCMC", "bayes_skprec", "bayes_skrec", "bayes_avg_precision")]
    R = {r.method: r for r in rows(main)}
    ms = [m for m in ORDER if m in R]
    head = " & ".join(r"\multicolumn{3}{c}{" + M[m] + "}" for m in ms)
    cm = "".join(f"\\cmidrule(lr){{{2+3*i}-{4+3*i}}}" for i in range(len(ms)))
    lines = [r"\begin{tabular}{l" + "ccc" * len(ms) + "}", r"\toprule", f" & {head} \\\\ {cm}",
             "Estimator & " + " & ".join(["Adj.\\ prec. & Adj.\\ rec. & Dir.\\ AP"] * len(ms)) + r" \\", r"\midrule"]
    for lab, a, b, ap in est:
        cells = []
        for m in ms:
            r = R[m]
            cells += [c(r, a, 3, se=False), c(r, b, 3, se=False), c(r, ap, 3, se=False) if ap else "--"]
        lines.append(lab + " & " + " & ".join(cells) + r" \\")
        if lab.startswith("CSCD"):
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, "claims.tex", lines)


def t_generic(out, name, specs, cols):
    """specs: [(exp, label)], cols: [(key, header, dec)]"""
    lines = [r"\begin{tabular}{ll" + "c" * len(cols) + "}", r"\toprule",
             "Setting & Method & " + " & ".join(h for _, h, _ in cols) + r" \\", r"\midrule"]
    for e, lab in specs:
        df = load(e)
        if df is None:
            lines.append(f"{lab} & \\multicolumn{{{len(cols)+1}}}{{c}}{{not run}} \\\\")
            continue
        for i, r in enumerate(rows(df)):
            lines.append(f"{lab if i == 0 else ''} & {M.get(r.method, r.method)} & " +
                         " & ".join(c(r, k, d) for k, _, d in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, name, lines)


def t_baselines_full(out):
    """Every estimator, directed and adjacency level, D_train and full-data variants."""
    main = load("main_v2")
    est = [("CSCD (ours)", "cscd", "cscd"), ("Candidate pool, no filter", "nofilter", "nofilter"),
           ("Uncalibrated self-comp.", "uncal", "uncal"), ("Single run, $D_{\\mathrm{train}}$", "single", "single"),
           ("Single run, $D$", "singleF", None), ("Stability sel.\\ 0.6, $D_{\\mathrm{train}}$", "stab", "stab"),
           ("Stability sel.\\ 0.6, $D$", "stabF", "stabF"), ("Stability, MB bound $E[V]\\le1$", "stabMB", None),
           ("Bootstrap freq.\\ $\\ge\\gamma$, $D_{\\mathrm{train}}$", "stabfreq", "stabfreq"),
           ("Bootstrap freq.\\ $\\ge\\gamma$, $D$", "stabFfreq", "stabFfreq"),
           ("Order MCMC, $D_{\\mathrm{train}}$", "bayes", "bayes"), ("Order MCMC, $D$", "bayesF", "bayesF"),
           ("Bootstrap compat., median", "bcmed", None), ("Bootstrap compat., $(1-\\alpha)$ q.", "bcq", None),
           ("Conformal, no subsets", "nosub", None)]
    lines = [r"\begin{tabular}{llccccc}", r"\toprule",
             r"Method & Estimator & Dir.\ prec. & Dir.\ rec. & Dir.\ AP & Adj.\ prec. & Adj.\ rec. \\", r"\midrule"]
    for r in rows(main):
        for i, (lab, k, sk) in enumerate(est):
            adjp = c(r, f"{sk}_skprec", 3, se=False) if sk else "--"
            adjr = c(r, f"{sk}_skrec", 3, se=False) if sk else "--"
            if sk in ("stabfreq", "stabFfreq"):
                adjp, adjr = c(r, f"{sk}_skprec", 3, se=False), c(r, f"{sk}_skrec", 3, se=False)
            lines.append(f"{M[r.method] if i == 0 else ''} & {lab} & {c(r, k + '_precision', 3, se=False)} & "
                         f"{c(r, k + '_recall', 3, se=False)} & {c(r, k + '_avg_precision', 3, se=False)} & {adjp} & {adjr} \\\\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    write(out, "baselines_full.tex", lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_v2/tables")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for f in (t_calibration, t_chunk, t_systematic, t_claims, t_baselines_full):
        try:
            f(a.out)
        except Exception as exc:
            print("failed", f.__name__, exc)
    base = [("exceedance", "Exceed.", 3), ("admission", "Admitted", 2), ("true_included", r"$G^\ast$ incl.", 2),
            ("cscd_skprec", "Adj.\\ prec.", 3), ("cscd_skrec", "Adj.\\ rec.", 3), ("cscd_precision", "Dir.\\ prec.", 3)]
    t_generic(a.out, "samplesize.tex", [(f"n{n}_N9", f"$n{{=}}{n}$") for n in (200, 400, 800, 1600)] +
              [("design_N9", "$n{=}3200$"), ("n6400_N9", "$n{=}6400$"), ("n12800_N9", "$n{=}12800$")], base)
    t_generic(a.out, "subsets.tex",
              [(f"sub_k{k}_m{m}", f"$k{{=}}{k}p$, $m{{=}}{m}$") for k in (0.3, 0.5, 0.7) for m in (10, 25, 50)] +
              [("sub_k0.5_m171", "$k{=}0.5p$, $m{=}171$"), ("sub_k0.7_m33", "$k{=}0.7p$, $m{=}33$"),
               ("sub_mb_k0.3", "MB, $k{=}0.3p$"), ("sub_mb_k0.5", "MB, $k{=}0.5p$"),
               ("sub_k0.3_m10_nocover", "$k{=}0.3p$, $m{=}10$, no cover")],
              [("n_subsets", "$|\\mathcal{S}|$", 0), ("sep_cover", "Sep.\\ cover", 3), ("admission", "Admitted", 2),
               ("true_included", r"$G^\ast$ incl.", 2), ("cscd_skprec", "Adj.\\ prec.", 3), ("cscd_skrec", "Adj.\\ rec.", 3),
               ("time_total", "Time (s)", 0)])
    t_generic(a.out, "latent.tex", [("latent_n1600", "$n{=}1600$"), ("sys_latent_hubs", "$n{=}3200$"),
                                    ("latent_n6400", "$n{=}6400$"), ("latent_n25600", "$n{=}25600$")],
              [("eps_train", r"$\hat\varepsilon$ (train)", 2), ("eps_chunk", r"$\hat\varepsilon$ (chunk)", 2),
               ("true_score", "$A(G^\\ast)$", 3), ("tau", r"$\tau_{0.1}$", 3), ("true_included", r"$G^\ast$ incl.", 2)])
    t_generic(a.out, "robustness.tex",
              [("knowledge_none", "No knowledge"), ("knowledge_correct_hard", "Correct, hard"),
               ("knowledge_correct_soft", "Correct, soft"), ("knowledge_noisy_hard", "Noisy, hard"),
               ("knowledge_noisy_soft", "Noisy, soft"), ("large_p50", "$p{=}50$"), ("nongauss_laplace", "Laplace noise"),
               ("latent_random", "5 of 25 hidden"), ("sachs_obs", "Sachs (obs., 853)"), ("sachs_pooled", "Sachs (pooled)"),
               ("bnlearn_asia", "Asia"), ("bnlearn_light_child", "Child$^\\dagger$"),
               ("bnlearn_light_insurance", "Insurance$^\\dagger$"), ("bnlearn_light_alarm", "Alarm$^\\dagger$"),
               ("kci_quadratic", "KCI, quadratic")],
              [("exceedance", "Exceed.", 3), ("admission", "Admitted", 2), ("true_included", r"$G^\ast$ incl.", 2),
               ("cscd_skprec", "Adj.\\ prec.", 3), ("cscd_skrec", "Adj.\\ rec.", 3), ("cscd_precision", "Dir.\\ prec.", 3),
               ("single_precision", "Single dir.\\ prec.", 3)])
    t_generic(a.out, "distances.tex", [], [])  # placeholder overwritten below
    # Distances: one block per distance.
    df = load("distances")
    if df is not None:
        lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
                 r"Distance & Method & Exceed. & $\tau_{0.1}$ & Admitted & $G^\ast$ incl. & Adj.\ prec. & AP \\", r"\midrule"]
        for dist, lab in [("skeleton", "Skeleton"), ("pag", "PAG"), ("skel_extra2", "Extra edges $\\times2$"),
                          ("skel_miss2", "Missing edges $\\times2$")]:
            for i, r in enumerate(rows(df, distance=dist)):
                lines.append(f"{lab if i == 0 else ''} & {M[r.method]} & {c(r,'exceedance',3)} & {c(r,'tau',3)} & "
                             f"{c(r,'admission')} & {c(r,'true_included')} & {c(r,'cscd_skprec',3)} & {c(r,'cscd_avg_precision',3)} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        write(a.out, "distances.tex", lines)
    # Decision task.
    main = load("main_v2")
    keys = [("oracle", "Oracle"), ("naive", "None"), ("single", "Single"), ("cscd_set", "CSCD set"),
            ("bootstrap_avg", "Bootstrap"), ("bma", "BMA")]
    lines = [r"\begin{tabular}{l" + "c" * len(keys) + "}", r"\toprule",
             "Method & " + " & ".join(l for _, l in keys) + r" \\", r"\midrule",
             r"\multicolumn{" + str(len(keys) + 1) + r"}{l}{\emph{Mean absolute bias}} \\"]
    for r in rows(main):
        lines.append(M[r.method] + " & " + " & ".join(c(r, f"dec2_bias_{k}", 3) for k, _ in keys) + r" \\")
    lines.append(r"\midrule \multicolumn{" + str(len(keys) + 1) + r"}{l}{\emph{Coverage of the true effect by the range of estimates}} \\")
    for r in rows(main):
        lines.append(M[r.method] + " & -- & -- & " + " & ".join(c(r, f"dec2_cover_{k}", 2) for k, _ in keys[2:]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(a.out, "decision.tex", lines)
    # gamma sweep
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r"Method & $\gamma$ & Adj.\ prec. & Adj.\ rec. & Dir.\ prec. & Dir.\ rec. & Unresolved \\", r"\midrule"]
    for g in (0.6, 0.7, 0.8, 0.9, 1.0):
        for r in rows(main, gamma=g):
            lines.append(f"{M[r.method]} & {g} & {c(r,'cscd_skprec',3)} & {c(r,'cscd_skrec',3)} & {c(r,'cscd_precision',3)} & "
                         f"{c(r,'cscd_recall',3)} & {c(r,'cscd_unresolved',3)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(a.out, "gamma.tex", lines)
