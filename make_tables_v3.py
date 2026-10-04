"""LaTeX tables for the second revision, from results_v3/summary/*.csv.

Usage:
    python make_tables_v3.py --paper_root ../aistats2027

Each table is a bare tabular written to <paper_root>/tables/rev/<name>.tex;
the paper wraps it in a table environment.
"""

import argparse
import os

import numpy as np
import pandas as pd

SUMMARY = "results_v3/summary"
M = {"pc": "PC", "ges": "GES", "fci": "FCI"}
ORDER = ["pc", "ges", "fci"]


def rd(name):
    p = os.path.join(SUMMARY, f"{name}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def f(x, d=2):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"${x:.{d}f}$"


def fse(m, s, d=2):
    if m is None or not np.isfinite(m):
        return "--"
    if s is None or not np.isfinite(s):
        return f"${m:.{d}f}$"
    return f"${m:.{d}f}\\pm{s:.{d}f}$"


def write(out, name, body):
    with open(os.path.join(out, f"{name}.tex"), "w") as fh:
        fh.write(body)
    print("wrote", name)


def _row(df, **kw):
    x = df
    for k, v in kw.items():
        x = x[x[k] == v]
    return x.iloc[0] if len(x) else None


# ---------------------------------------------------------------------------

def t_calibration(out):
    val, inc, alpha = rd("validity"), rd("inclusion"), rd("alpha")
    if val is None or inc is None:
        return
    lines = [r"\begin{tabular}{lcccccccc}", r"\toprule",
             r"Method & Exceedance & $\tau_{0.1}$ & $A(G^\ast)$ & Admitted & $G^\ast\in\mathcal{C}_\alpha$ & $\widehat G\in\mathcal{C}_\alpha$ & Rank $0$ & $1-\sum_j\hat\varepsilon_j$ \\",
             r"\midrule"]
    for m in ORDER:
        v = _row(val, experiment="valid_N9", method=m, alpha=0.1)
        i = _row(inc, experiment="valid_N9", method=m)
        im = _row(inc, experiment="main_v3", method=m)
        a = _row(alpha, experiment="main_v3", method=m, alpha=0.1) if alpha is not None else None
        exc = f"${v.exceed:.3f}$ [${v.ci_lo:.3f}$, ${v.ci_hi:.3f}$]" if v is not None else "--"
        lines.append(" & ".join([
            M[m], exc, f(i.tau if i is not None else None, 3), f(i.a_star if i is not None else None, 3),
            fse(a.admitted_mean, a.admitted_se) if a is not None else "--",
            f(i.included if i is not None else None), f(1 - i.full_rank_ge_k if i is not None else None),
            f(i.rank0 if i is not None else None), f(i.union_bound if i is not None else None, 1)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, "calibration", "\n".join(lines) + "\n")


def t_inclusion(out):
    inc = rd("inclusion")
    if inc is None:
        return
    lines = [r"\begin{tabular}{llccccccccc}", r"\toprule",
             r"Setting & Method & $G^\ast\in\mathcal{C}_\alpha$ & $1-\overline{\beta_{N,\alpha}(\hat u)}$ & Rank $0$ & Rank $N$ & $\hat\Delta$ & $\hat\eta$ & Wrong pairs & $\hat e$ (right) & $\hat e$ (wrong) \\",
             r"\midrule"]
    last = None
    for _, r in inc.iterrows():
        lab = r.label if r.label != last else ""
        if r.label != last and last is not None:
            lines.append(r"\addlinespace")
        last = r.label
        lines.append(" & ".join([lab, M.get(r.method, r.method), f(r.included), f(1 - r.pred_excl),
                                 f(r.rank0), f(r.rankN), f(r.delta, 3), f(r.eta),
                                 f(r.frac_wrong_pairs, 3), f(r.e_right, 3), f(r.e_wrong)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, "inclusion", "\n".join(lines) + "\n")


def t_validity(out):
    val = rd("validity")
    if val is None:
        return
    lines = [r"\begin{tabular}{llccccccc}", r"\toprule",
             r"$N$ & Method & $\alpha$ & Nominal & Exceedance [95\% CI] & Repeats & $\mathbb{P}(\text{exc.}>2\alpha)$ & Beta ref. & KS $p$ \\",
             r"\midrule"]
    for N in sorted(val.N.unique()):
        d = val[val.N == N].sort_values(["method", "alpha"])
        for _, r in d.iterrows():
            lines.append(" & ".join([str(N), M[r.method], f"${r.alpha}$", f(r.nominal, 3),
                                     f"${r.exceed:.3f}$ [${r.ci_lo:.3f}$, ${r.ci_hi:.3f}$]", str(int(r.n_repeats)),
                                     f(r.p_gt_2a), f(r.ref_p_gt_2a), f(r.ks_p)]) + r" \\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    write(out, "validity", "\n".join(lines) + "\n")


DET_SHOW = [("CSCD margin $A(\\widehat G)-\\tau_\\alpha$", "CSCD"), ("Bootstrap compat.", "Boot.\\ compat."),
            ("Conformal, no subsets", "No subsets"), ("$\\kappa_G$ (Faller et al.)", "$\\kappa_G$"),
            ("Permutation test (Eulig et al.)", "Perm.\\ test"), ("LOVO loss (Schkoda et al.)", "LOVO")]


def t_detection(out):
    det = rd("detection")
    if det is None:
        return
    head = " & ".join(lab for _, lab in DET_SHOW)
    lines = [r"\begin{tabular}{ll" + "c" * (3 + len(DET_SHOW)) + "}", r"\toprule",
             r" & & & \multicolumn{2}{c}{CSCD rejects} & \multicolumn{" + str(len(DET_SHOW)) + r"}{c}{Power at $10\%$ false-flag rate (AUROC)} \\ \cmidrule(lr){4-5}\cmidrule(lr){6-" + str(5 + len(DET_SHOW)) + "}",
             r"Scenario & Method & Skel.\ err. & $\widehat G$ & $G^\ast$ & " + head + r" \\", r"\midrule"]
    scen_last = None
    for scen in det.scenario.unique():
        for m in ORDER:
            d = det[(det.scenario == scen) & (det.method == m)]
            if d.empty:
                continue
            r0 = d.iloc[0]
            cells = []
            for key, _ in DET_SHOW:
                x = d[d.detector == key]
                cells.append("--" if x.empty else f"${x.power.iloc[0]:.2f}$ (${x.auroc.iloc[0]:.2f}$)")
            lab = scen if scen != scen_last else ""
            if scen != scen_last and scen_last is not None:
                lines.append(r"\addlinespace")
            scen_last = scen
            lines.append(" & ".join([lab, M[m], f"${r0.skel_err:.1f}$", f(r0.reject_ghat), f(r0.reject_gstar)] + cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, "detection", "\n".join(lines) + "\n")


def t_detection_ci(out):
    det = rd("detection")
    if det is None:
        return
    lines = [r"\begin{tabular}{lllcccccc}", r"\toprule",
             r"Scenario & Method & Detector & AUROC [95\% CI] & Power & FPR & Native flag & Native flag (control) \\", r"\midrule"]
    for _, r in det.iterrows():
        lines.append(" & ".join([r.scenario, M[r.method], r.detector.replace("(Faller et al.)", "").replace("(Eulig et al.)", "").replace("(Schkoda et al.)", ""),
                                 f"${r.auroc:.2f}$ [${r.auroc_lo:.2f}$, ${r.auroc_hi:.2f}$]", f(r.power), f(r.fpr),
                                 f(r.native_flag), f(r.native_flag_ctrl)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, "detection_ci", "\n".join(lines) + "\n")


def t_within(out):
    det = rd("detection")
    if det is None:
        return
    d = det.drop_duplicates(["scenario", "method"])
    lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
             r" & & \multicolumn{2}{c}{Reject rate, scenario} & \multicolumn{2}{c}{Reject rate, control} & AUROC $A(\widehat G)$ vs.\ $A(G^\ast)$ & Spearman $A(G)$, skel.\ err. \\ \cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"Scenario & Method & $\widehat G$ & $G^\ast$ & $\widehat G$ & $G^\ast$ & scenario / control & scenario / control \\", r"\midrule"]
    for _, r in d.iterrows():
        lines.append(" & ".join([r.scenario, M[r.method], f(r.reject_ghat), f(r.reject_gstar), f(r.reject_ghat_ctrl),
                                 f(r.reject_gstar_ctrl), f"${r.within_auroc:.2f}$ / ${r.within_auroc_ctrl:.2f}$",
                                 f"${r.spearman:.2f}$ / ${r.spearman_ctrl:.2f}$"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, "within", "\n".join(lines) + "\n")


CLAIM_EST = [("CSCD", "CSCD (ours)"), ("CSCD, MEC-dedup.", "CSCD, MEC-dedup."), ("Pool, no filter", "Pool, no filter"),
             ("Uncal. threshold", "Uncal.\\ threshold"), ("Single run, $D$", "Single run"),
             ("Stability sel., $D$", "Stability sel."), ("Bootstrap freq., $D$", "Bootstrap freq."),
             ("Order MCMC, $D$", "Order MCMC")]
KIND = {"adj": "Adjacency", "dir_def": "Definite orientation", "anc": "Ancestral", "dir_any": "Directed, undirected both ways"}


def t_claims(out, kinds=("adj", "dir_def", "anc"), name="claims", methods=ORDER):
    c = rd("claims")
    if c is None:
        return
    cols = "".join("cc" for _ in kinds)
    lines = [r"\begin{tabular}{ll" + cols + "}", r"\toprule",
             " & & " + " & ".join(r"\multicolumn{2}{c}{" + KIND[k] + "}" for k in kinds) + r" \\ " +
             "".join(r"\cmidrule(lr){" + f"{3 + 2 * i}-{4 + 2 * i}" + "}" for i in range(len(kinds))),
             r"Method & Estimator & " + " & ".join(r"Prec./rec. & P@R$.5$" for _ in kinds) + r" \\", r"\midrule"]
    for m in methods:
        for j, (key, lab) in enumerate(CLAIM_EST):
            cells = []
            for k in kinds:
                x = c[(c.method == m) & (c.kind == k) & (c.estimator == key)]
                if x.empty:
                    cells += ["--", "--"]
                    continue
                x = x.iloc[0]
                pr = "--" if not np.isfinite(x.prec_mean) else f"${x.prec_mean:.2f}$/${x.rec_mean:.2f}$"
                cells += [pr, f(x["p@r0.5_mean"])]
            lines.append(" & ".join([M[m] if j == 0 else "", lab] + cells) + r" \\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    write(out, name, "\n".join(lines) + "\n")


ARM = {"standard": "Standard (paper)", "boot_only": "Bootstrap only", "edit_only": "Edits only", "grid_only": "Grid only",
       "multi_alg": "Multi-algorithm", "mcmc": "Order-MCMC samples", "standard+truth": "Standard $+\\,G^\\ast$",
       "standard+random": "Standard $+$ random DAGs"}


def t_arms(out):
    a = rd("arms")
    if a is None:
        return
    lines = [r"\begin{tabular}{llcccccccc}", r"\toprule",
             r" & & & & \multicolumn{2}{c}{Adjacency} & \multicolumn{2}{c}{Definite orient.} & & \\ \cmidrule(lr){5-6}\cmidrule(lr){7-8}",
             r"Method & Arm & Size & Admitted & Prec. & Rec. & Prec. & Rec. & Jaccard with $\widehat G$ & Empty \\", r"\midrule"]
    for m in ORDER:
        for j, arm in enumerate(ARM):
            x = a[(a.method == m) & (a.arm == arm)]
            if x.empty:
                continue
            x = x.iloc[0]
            lines.append(" & ".join([M[m] if j == 0 else "", ARM[arm], f"${x.size_mean:.0f}$", f(x.admitted_mean),
                                     f(x.adj_prec_mean), f(x.adj_rec_mean), f(x.dir_def_prec_mean), f(x.dir_def_rec_mean),
                                     f(x.jaccard_ghat_mean), f(x.empty_mean)]) + r" \\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    write(out, "arms", "\n".join(lines) + "\n")


def t_alpha(out):
    a = rd("alpha")
    if a is None:
        return
    lines = [r"\begin{tabular}{llcccccccc}", r"\toprule",
             r"Design & Method & $\alpha$ & $\tau_\alpha$ & Exceedance & Admitted & Empty & $G^\ast\in\mathcal{C}_\alpha$ & $\widehat G\in\mathcal{C}_\alpha$ & Adj.\ prec./rec. \\", r"\midrule"]
    for exp, lab in (("alpha_N4", "$N{=}4$, $n_c{=}400$"), ("main_v3", "$N{=}9$, $n_c{=}177$"), ("alpha_N19", "$N{=}19$, $n_c{=}84$")):
        d = a[a.experiment == exp]
        for j, m in enumerate(ORDER):
            for _, x in d[d.method == m].sort_values("alpha", ascending=False).iterrows():
                lines.append(" & ".join([lab if j == 0 and x.alpha == d[d.method == m].alpha.max() else "", M[m], f"${x.alpha}$",
                                         f(x.tau_mean, 3), fse(x.exceed_mean, x.exceed_se), f(x.admitted_mean), f(x.empty_mean),
                                         f(x.gstar_mean), f(x.ghat_mean),
                                         f"${x.adj_prec_mean:.2f}$/${x.adj_rec_mean:.2f}$"]) + r" \\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    write(out, "alpha", "\n".join(lines) + "\n")


def t_knowledge(out):
    k = rd("knowledge")
    if k is None:
        return
    lab = {"know_none": "None", **{f"know_hard_rev{r}": (f"Hard, {r}\\% reversed" if r else "Hard, correct") for r in (0, 10, 20, 30, 40, 50)},
           "know_soft_correct": "Soft, correct", "know_soft_noisy": "Soft, 30\\% reversed"}
    lines = [r"\begin{tabular}{lllccccccc}", r"\toprule",
             r"Knowledge & Calibration & Method & Exceed. & Admitted & $G^\ast\in\mathcal{C}_\alpha$ & Reject $\widehat G$ & Adj.\ prec. & Def.\ orient.\ prec./rec. & Single def.\ prec. \\", r"\midrule"]
    for exp in lab:
        for _, x in k[k.experiment == exp].sort_values(["calibration", "method"]).iterrows():
            cal = "--" if "soft" not in exp else ("$A_b$ with penalty" if "in" in x.calibration else "candidates only")
            lines.append(" & ".join([lab[exp], cal, M[x.method], f(x.exceed_mean, 3), f(x.admitted_mean), f(x.gstar_mean),
                                     f(x.reject_ghat_mean), f(x.adj_prec_mean, 3),
                                     f"${x.dir_def_prec_mean:.2f}$/${x.dir_def_rec_mean:.2f}$", f(x.single_dir_def_prec_mean)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, "knowledge", "\n".join(lines) + "\n")


def t_mcmc(out):
    d = rd("mcmc")
    if d is None:
        return
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"Experiment & Runs & Median $\hat R$ & $\hat R<1.1$ & Median max.\ edge diff. & Max.\ diff.\ $<0.2$ \\", r"\midrule"]
    for _, x in d.iterrows():
        lines.append(" & ".join([x.experiment.replace("_", r"\_"), str(int(x.n)), f(x.rhat_median), f(x.frac_rhat_lt_1_1),
                                 f(x.max_edge_diff_median), f(x.frac_diff_lt_0_2)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, "mcmc", "\n".join(lines) + "\n")


def t_decision(out):
    d = rd("decision")
    if d is None:
        return
    sets = [("oracle", "Oracle"), ("naive", "None"), ("single", "Single"), ("cscd_set", "CSCD set"),
            ("pool", "Pool"), ("bootstrap_avg", "Bootstrap"), ("bma", "BMA")]
    lines = [r"\begin{tabular}{l" + "c" * len(sets) + "}", r"\toprule", "Method & " + " & ".join(s for _, s in sets) + r" \\", r"\midrule",
             r"\multicolumn{" + str(len(sets) + 1) + r"}{l}{\emph{Mean absolute bias}} \\"]
    for m in ORDER:
        x = d[d.method == m]
        if x.empty:
            continue
        x = x.iloc[0]
        lines.append(" & ".join([M[m]] + [fse(x.get(f"dec2_bias_{k}_mean"), x.get(f"dec2_bias_{k}_se"), 3) for k, _ in sets]) + r" \\")
    lines += [r"\midrule", r"\multicolumn{" + str(len(sets) + 1) + r"}{l}{\emph{Coverage of the true effect by the range of estimates}} \\"]
    for m in ORDER:
        x = d[d.method == m]
        if x.empty:
            continue
        x = x.iloc[0]
        lines.append(" & ".join([M[m]] + ["--" if k in ("oracle", "naive") else fse(x.get(f"dec2_cover_{k}_mean"), x.get(f"dec2_cover_{k}_se")) for k, _ in sets]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, "decision", "\n".join(lines) + "\n")


def t_semi(out):
    d = rd("semi")
    if d is None:
        return
    ests = [("CSCD", "CSCD"), ("Pool, no filter", "Pool"), ("Single run, $D_{tr}$", "Single run"),
            ("Stability sel., $D_{tr}$", "Stability sel."), ("Bootstrap freq., $D_{tr}$", "Bootstrap freq."),
            ("Order MCMC, $D_{tr}$", "Order MCMC")]
    lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
             r" & & \multicolumn{2}{c}{Adjacency} & \multicolumn{2}{c}{Definite orientation} & \multicolumn{2}{c}{Ancestral} \\ \cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
             r"Method & Estimator & Prec. & Rec. & Prec. & Rec. & Prec. & Rec. \\", r"\midrule"]
    for m in ("pc", "ges"):
        x = d[d.method == m]
        if x.empty:
            continue
        x = x.iloc[0]
        for j, (key, lab) in enumerate(ests):
            cells = []
            for k in ("adj", "dir_def", "anc"):
                cells += [f(x.get(f"{key}|{k}|prec_mean")), f(x.get(f"{key}|{k}|rec_mean"))]
            lines.append(" & ".join([M[m] if j == 0 else "", lab] + cells) + r" \\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\midrule"
    for m in ("pc", "ges"):
        x = d[d.method == m]
        if len(x):
            x = x.iloc[0]
            lines.append(rf"\multicolumn{{8}}{{l}}{{{M[m]}: exceedance ${x.exceed_mean:.3f}\pm{x.exceed_se:.3f}$, admitted ${x.admitted_mean:.2f}$, $G^\ast\in\mathcal{{C}}_\alpha$ ${x.gstar_mean:.2f}$, separation cover ${x.sep_cover_mean:.3f}$}} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, "semi", "\n".join(lines) + "\n")


def t_sachs(out):
    d = rd("sachs")
    if d is None:
        return
    ests = [("CSCD", "CSCD"), ("Pool, no filter", "Pool"), ("Single run, $D_{tr}$", "Single run"),
            ("Stability sel., $D_{tr}$", "Stability sel."), ("Bootstrap freq., $D_{tr}$", "Bootstrap freq."),
            ("Order MCMC, $D_{tr}$", "Order MCMC")]
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r"Method & Estimator & Accepted & Validated & Recall of effects & Rejected but shifted \\", r"\midrule"]
    for m in ORDER:
        for j, (key, lab) in enumerate(ests):
            x = d[(d.method == m) & (d.estimator == key)]
            if x.empty:
                continue
            x = x.iloc[0]
            lines.append(" & ".join([M[m] if j == 0 else "", lab, f(x.n_accepted_mean, 1), f(x.validated_mean),
                                     f(x.recall_mean), f(x.contradicted_mean)]) + r" \\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    write(out, "sachs", "\n".join(lines) + "\n")


def t_scale(out):
    d = rd("scale")
    if d is None:
        return
    lines = [r"\begin{tabular}{lccccccc}", r"\toprule",
             r"Setting & $|\mathcal{S}|$ & Sep.\ cover & Exceed. & Admitted & $G^\ast\in\mathcal{C}_\alpha$ & Local runs (s) & Total (min) \\", r"\midrule"]
    for _, x in d.iterrows():
        loc = x.get("t_local_train_mean", 0) + x.get("t_chunks_mean", 0)
        lab = {"scale_p100_mb": "$p{=}100$, MB, $k{=}15$", "scale_p200_mb": "$p{=}200$, MB, $k{=}20$, $n{=}5000$"}.get(x.experiment, x.experiment)
        lines.append(" & ".join([lab, f"${x.n_subsets_mean:.0f}$", f(x.sep_cover_mean, 3), fse(x.exceed_mean, x.exceed_se, 3),
                                 f(x.admitted_mean), f(x.gstar_mean), f"${loc:.0f}$", f"${x.wall_mean / 60:.0f}$"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, "scale", "\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper_root", default="../aistats2027")
    args = ap.parse_args()
    out = os.path.join(args.paper_root, "tables", "rev")
    os.makedirs(out, exist_ok=True)
    for t in (t_calibration, t_inclusion, t_validity, t_detection, t_detection_ci, t_within, t_claims, t_arms,
              t_alpha, t_knowledge, t_mcmc, t_decision, t_semi, t_sachs, t_scale):
        try:
            t(out)
        except Exception as exc:
            print("FAILED", t.__name__, exc)
    t_claims(out, kinds=("dir_any",), name="claims_legacy")


if __name__ == "__main__":
    main()
